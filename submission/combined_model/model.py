# combined_model/model.py
"""
Combined Semantic ViT - Dual pathway model

Fuses both ViT (sequential) and RvNN (structural) paths:

    Code -> CodeBERT
        ├── PatchEmbedder -> Projector       -> [M, dec_dim]  (ViT path)
        └── PixelAdapter  -> RecursiveEncoder -> [1, dec_dim]  (Tree path)
                            ↓
                 cat([global, seq]) -> [M+1, qwen_dim]
                            ↓
                       QwenDecoder
"""
import time
import torch
import torch.nn as nn

from model2.semantic_extractor import SemanticExtractorV2
from shared.codebert_encoder import CodeBERTEncoder
from shared.patch_embedder import PatchEmbedder
from shared.projector import Projector, NonLinearProjector, embedding_rms
from shared.decoder_factory import create_decoder
from model2.recursive_encoder import RecursiveEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CombinedSemanticViT(nn.Module):
    """
    Combined model with both ViT and Tree pathways.

    Trainable:
        - Projector (ViT path)
        - PixelAdapter + RecursiveEncoder (Tree path)

    Frozen:
        - CodeBERTEncoder
        - QwenDecoder
    """

    def __init__(
        self,
        patch_size: int = 4,
        bottleneck_dim: int = 512,
        dropout: float = 0.4,
        decoder_name: str = "qwen",
        projector_arch: str = "linear",  # "linear" | "mlp"
    ):
        super().__init__()

        self.projector_arch = projector_arch

        # Extractor (tree-aware, needed for both paths)
        self.extractor = SemanticExtractorV2()

        # Base Encoder (frozen)
        self.encoder = CodeBERTEncoder(device=DEVICE)

        # --- DECODER (created first so we can read hidden_size) ---
        self.decoder = create_decoder(decoder_name, device=DEVICE)
        dec_dim = self.decoder.hidden_size

        # --- PATH 1: SEQUENTIAL (ViT) ---
        self.patch_embedder = PatchEmbedder(patch_size=patch_size)
        if projector_arch == "mlp":
            self.projector = NonLinearProjector(
                in_dim=patch_size * 768,
                hidden_dim=dec_dim,
                out_dim=dec_dim,
                dropout=dropout,
                target_rms=embedding_rms(self.decoder),
            ).to(DEVICE)
        else:
            self.projector = Projector(
                in_dim=patch_size * 768,
                bottleneck_dim=bottleneck_dim,
                out_dim=dec_dim,
                dropout=dropout
            ).to(DEVICE)

        # --- PATH 2: STRUCTURAL (RvNN) ---
        # No LayerNorm: pins norm to sqrt(dec_dim), causing the global_vector
        # to dominate the decoder's residual stream.
        # Kaiming init on Linear(768, dec_dim) gives output norm close to
        # decoder token norms.
        self.pixel_adapter = nn.Linear(768, dec_dim).to(DEVICE)

        self.recursive_encoder = RecursiveEncoder(
            embed_dim=dec_dim,
            max_branching=8,
            hidden_dim=dec_dim * 2,
            dropout=dropout
        ).to(DEVICE)


    def get_trainable_parameters(self):
        """Return all encoder trainable parameters (excludes unfrozen Qwen layers)."""
        params = []
        # ViT path
        params.extend(self.projector.parameters())
        # Tree path
        params.extend(self.pixel_adapter.parameters())
        params.extend(self.recursive_encoder.parameters())
        # LoRA (if enabled; empty list when using unfreeze)
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

        # --- PATH 1: Sequential ---
        # [N, 768] -> [M, patch*768] -> [M, dec_dim]
        patch_embeddings = self.patch_embedder(cls_embeddings)
        seq_vectors = self.projector(patch_embeddings)

        # --- PATH 2: Structural ---
        # [N, 768] -> [1, N, dec_dim] -> [1, dec_dim]
        pixels_for_tree = self.pixel_adapter(cls_embeddings).unsqueeze(0)
        global_vector = self.recursive_encoder.forward_tree(
            features['tree_roots'],
            pixels_for_tree
        )

        # --- FUSION ---
        # [1, qwen_dim] + [M, qwen_dim] -> [M+1, qwen_dim]
        combined = torch.cat([global_vector, seq_vectors], dim=0)

        # --- DECODE ---
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
