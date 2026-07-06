# model/__init__.py
"""
Semantic ViT Model (ViT-only pipeline)

Structure:
    MATLAB Code
        │
        ▼ SemanticExtractor
    [texts]
        │
        ▼ CodeBERTEncoder
    CLS embeddings [N, 768]
        │
        ▼ PatchEmbedder
    patch_embeddings [N/P, P*768]
        │
        ▼ Projector
    projected [N/P, dec_dim]
        │
        ▼ Decoder
    output text
"""

from shared.semantic_extractor import SemanticExtractor
from shared.codebert_encoder import CodeBERTEncoder
from shared.patch_embedder import PatchEmbedder
from shared.projector import Projector
from shared.decoder_factory import create_decoder
