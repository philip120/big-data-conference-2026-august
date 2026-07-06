# shared/__init__.py
"""
Shared components used by model/, model2/, and combined_model/.

Components:
    - SemanticExtractor: Extracts semantic operations from MATLAB code
    - CodeBERTEncoder: Frozen CodeBERT for CLS embeddings
    - PatchEmbedder: Groups pixels into patches (ViT-style)
    - Projector: Linear projection to decoder embedding space
    - QwenDecoder: Frozen Qwen LLM for text generation
    - GemmaDecoder: Frozen Gemma LLM for text generation
    - create_decoder: Factory to create decoder by name

Constants:
    - TYPE_TO_ID, ID_TO_TYPE, NUM_TYPES, MAX_DEPTH
"""

from .semantic_extractor import SemanticExtractor, TYPE_TO_ID, ID_TO_TYPE, NUM_TYPES, MAX_DEPTH
from .codebert_encoder import CodeBERTEncoder
from .patch_embedder import PatchEmbedder
from .projector import Projector
from .qwen_decoder import QwenDecoder
from .gemma_decoder import GemmaDecoder
from .decoder_factory import create_decoder
