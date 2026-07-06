"""
TreeText Model — RvNN tree path + decoder's native tokenizer for code text.

Architecture:
    Code -> SemanticExtractorV2 -> CodeBERT -> PixelAdapter -> RecursiveEncoder -> global_vector [1, D]
    Code -> Decoder.get_input_embeddings(code) -> code_embeds [L, D]
    cat([global_vector, code_embeds]) -> [1+L, D] -> Decoder -> loss / generation
"""
import time
import torch
import torch.nn as nn

from model2.semantic_extractor import SemanticExtractorV2
from shared.codebert_encoder import CodeBERTEncoder
from shared.decoder_factory import create_decoder
from model2.recursive_encoder import RecursiveEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TreeTextModel(nn.Module):
    """
    Tree + native-text model for code-to-pseudocode.

    Combines:
    - RvNN tree encoding (global_vector) from the AST
    - Decoder's own tokenizer embeddings for the raw code text

    Trainable: PixelAdapter, RecursiveEncoder
    Frozen: CodeBERTEncoder, Decoder (except LoRA / unfrozen layers)
    """

    def __init__(
        self,
        dropout: float = 0.4,
        decoder_name: str = "qwen",
    ):
        super().__init__()

        # Extractor (tree-aware)
        self.extractor = SemanticExtractorV2()

        # Base Encoder (frozen) — for tree path node embeddings
        self.encoder = CodeBERTEncoder(device=DEVICE)

        # Decoder (created first so we can read hidden_size)
        self.decoder = create_decoder(decoder_name, device=DEVICE)
        dec_dim = self.decoder.hidden_size

        # Adapt CodeBERT (768) to decoder dim for tree path
        self.pixel_adapter = nn.Linear(768, dec_dim).to(DEVICE)

        # Recursive Encoder (tree aggregation)
        self.recursive_encoder = RecursiveEncoder(
            embed_dim=dec_dim,
            max_branching=8,
            hidden_dim=dec_dim * 2,
            dropout=dropout,
        ).to(DEVICE)

    def get_trainable_parameters(self):
        """Return only trainable parameters for optimizer."""
        params = []
        params.extend(self.pixel_adapter.parameters())
        params.extend(self.recursive_encoder.parameters())
        params.extend(self.decoder.get_lora_parameters())
        return params

    def enable_lora(self, **kwargs):
        """Enable LoRA on the decoder."""
        self.decoder.enable_lora(**kwargs)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.decoder.train_mode()
        else:
            self.decoder.eval_mode()
        return self

    def eval(self):
        return self.train(False)

    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.get_trainable_parameters())

    def _build_combined(self, code: str, features: dict = None):
        """Build the combined [global_vector, code_embeds] sequence.

        Returns combined tensor [1+L, dec_dim] or None if tree extraction fails.
        """
        # 1. Extract tree features
        if features is None:
            features = self.extractor(code)

        if not features['texts']:
            return None

        # 2. Tree path: CodeBERT -> PixelAdapter -> RecursiveEncoder -> [1, dec_dim]
        cls_embeddings = self.encoder(features['texts'])
        pixels_for_tree = self.pixel_adapter(cls_embeddings).unsqueeze(0)
        global_vector = self.recursive_encoder.forward_tree(
            features['tree_roots'],
            pixels_for_tree,
        )  # [1, dec_dim]

        # 3. Native text path: decoder tokenizer -> embeddings -> [L, dec_dim]
        embeds, _ = self.decoder.get_input_embeddings(code)  # [1, L, dec_dim]
        code_embeds = embeds.squeeze(0)  # [L, dec_dim]

        # 4. Concatenate: [1+L, dec_dim]
        combined = torch.cat([global_vector, code_embeds], dim=0)
        return combined

    def forward(self, code: str, target: str = None, features: dict = None):
        combined = self._build_combined(code, features)

        if combined is None:
            if target:
                return torch.tensor(0.0, device=DEVICE, requires_grad=True)
            return None

        if target:
            return self.decoder.forward_train(combined, target)
        else:
            return combined

    @torch.no_grad()
    def generate(self, code: str, max_new_tokens: int = 128):
        projected = self.forward(code)
        if projected is None:
            return ""
        return self.decoder.generate(projected, max_new_tokens=max_new_tokens)

    @torch.no_grad()
    def generate_with_metrics(self, code: str, max_new_tokens: int = 128):
        """Generate text and return (text, efficiency_metrics)."""
        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        projected = self.forward(code)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        if projected is None:
            return "", {}

        text, dec_metrics = self.decoder.generate_with_metrics(
            projected, max_new_tokens=max_new_tokens
        )

        peak_vram_mb = 0.0
        if DEVICE == "cuda":
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)

        return text, {
            "encode_time_s": round(t1 - t0, 4),
            "total_time_s": round(time.perf_counter() - t0, 4),
            "peak_vram_mb": round(peak_vram_mb, 1),
            **dec_metrics,
        }
