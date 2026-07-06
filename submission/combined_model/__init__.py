# combined_model/__init__.py
"""
Combined Semantic ViT - Dual pathway model

Fuses:
    - Path 1 (ViT): PatchEmbedder -> Projector -> [M, 1536]
    - Path 2 (Tree): PixelAdapter -> RecursiveEncoder -> [1, 1536]

    cat([global_vector, seq_vectors]) -> [M+1, 1536] -> QwenDecoder
"""

from .model import CombinedSemanticViT
