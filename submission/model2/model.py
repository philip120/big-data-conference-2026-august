"""
Structural Model (Model 2) - Tree-only pipeline

Structural path only:
    Code -> SemanticExtractorV2 -> CodeBERT -> PixelAdapter -> RecursiveEncoder -> Decoder
"""
import time
import torch
import torch.nn as nn

from .semantic_extractor import SemanticExtractorV2
from shared.code_encoder import CodeEncoder
from shared.decoder_factory import create_decoder
from .recursive_encoder import RecursiveEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class StructuralModel(nn.Module):
    """
    Tree-only model for code-to-pseudocode.

    Uses recursive tree encoding of the AST to produce a global vector.

    Trainable: PixelAdapter, RecursiveEncoder
    Frozen: CodeBERTEncoder, QwenDecoder
    """

    def __init__(
        self,
        dropout: float = 0.4,
        decoder_name: str = "qwen",
        encoder_name: str = "codebert",
        max_branching: int = 8,
    ):
        super().__init__()

        # Extractor (tree-aware)
        self.extractor = SemanticExtractorV2()

        # Base Encoder (frozen)
        self.encoder = CodeEncoder(preset=encoder_name, device=DEVICE)
        enc_dim = self.encoder.hidden_size

        # Decoder (created first so we can read hidden_size)
        self.decoder = create_decoder(decoder_name, device=DEVICE)
        dec_dim = self.decoder.hidden_size

        # Adapt pixel (enc_dim) to Recursive dim
        # No LayerNorm: would pin output norm to sqrt(dec_dim),
        # causing the global_vector to dominate the decoder's residual stream.
        self.pixel_adapter = nn.Linear(enc_dim, dec_dim).to(DEVICE)

        # Recursive Encoder (tree aggregation)
        self.recursive_encoder = RecursiveEncoder(
            embed_dim=dec_dim,
            max_branching=max_branching,
            hidden_dim=dec_dim * 2,
            dropout=dropout
        ).to(DEVICE)

    def get_trainable_parameters(self):
        """Return only trainable parameters for optimizer."""
        params = []
        params.extend(self.pixel_adapter.parameters())
        params.extend(self.recursive_encoder.parameters())
        params.extend(self.decoder.get_lora_parameters())
        return params

    def enable_lora(self, **kwargs):
        """Enable LoRA on the Qwen decoder."""
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

    def forward(self, code: str, target: str = None, features: dict = None):
        # 1. Extract features (with tree structure)
        if features is None:
            features = self.extractor(code)

        if not features['texts']:
            if target:
                return torch.tensor(0.0, device=DEVICE, requires_grad=True)
            return None

        # 2. CodeBERT embeddings [N, 768]
        cls_embeddings = self.encoder(features['texts'])

        # 3. Adapt for tree: [N, 768] -> [1, N, dec_dim]
        pixels_for_tree = self.pixel_adapter(cls_embeddings).unsqueeze(0)

        # 5. Recursive tree traversal -> [1, qwen_dim]
        global_vector = self.recursive_encoder.forward_tree(
            features['tree_roots'],
            pixels_for_tree
        )

        # 6. Decode
        if target:
            return self.decoder.forward_train(global_vector, target)
        else:
            return global_vector

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
