# model/model.py
"""
Semantic ViT (Model 1) - ViT-only pipeline

Sequential path only:
    Code -> SemanticExtractor -> CodeBERT -> PatchEmbedder -> Projector -> Decoder
"""
import time
import torch
import torch.nn as nn

from shared.semantic_extractor import SemanticExtractor
from shared.codebert_encoder import CodeBERTEncoder
from shared.patch_embedder import PatchEmbedder
from shared.projector import Projector, NonLinearProjector, embedding_rms
from shared.decoder_factory import create_decoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SemanticViT(nn.Module):
    """
    ViT-only model for code-to-pseudocode.

    Trainable: Projector
    Frozen: CodeBERTEncoder, QwenDecoder
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

        self.patch_size = patch_size
        self.projector_arch = projector_arch

        # Extractor (not nn.Module)
        self.extractor = SemanticExtractor()

        # CodeBERT (frozen, not nn.Module)
        self.encoder = CodeBERTEncoder(device=DEVICE)

        # Patch embedder
        self.patch_embedder = PatchEmbedder(patch_size=patch_size)

        # Decoder (frozen, not nn.Module — created before projector to read hidden_size)
        self.decoder = create_decoder(decoder_name, device=DEVICE)
        dec_dim = self.decoder.hidden_size

        # Projector (TRAINABLE)
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
                dropout=dropout,
            ).to(DEVICE)

    def get_trainable_parameters(self):
        """Return only trainable parameters for optimizer."""
        params = []
        params.extend(self.projector.parameters())
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
        # 1. Extract semantic features
        if features is None:
            features = self.extractor(code)

        if not features['texts']:
            if target:
                return torch.tensor(0.0, device=DEVICE, requires_grad=True)
            return None

        # 2. CodeBERT embeddings [N, 768]
        cls_embeddings = self.encoder(features['texts'])

        # 3. Patch embeddings [M, patch*768]
        patch_embeddings = self.patch_embedder(cls_embeddings)

        # 5. Project to Qwen space [M, qwen_dim]
        projected = self.projector(patch_embeddings)

        # 6. Decode
        if target:
            return self.decoder.forward_train(projected, target)
        else:
            return projected

    @torch.no_grad()
    def generate(self, code: str, max_new_tokens: int = 128) -> str:
        """Generate pseudocode from MATLAB code."""
        projected = self.forward(code, target=None)
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
        projected = self.forward(code, target=None)
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
