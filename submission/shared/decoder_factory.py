# shared/decoder_factory.py
"""Factory function to create the right decoder from a string name."""

from shared.qwen_decoder import QwenDecoder
from shared.gemma_decoder import GemmaDecoder

DECODER_MAP = {
    "qwen": QwenDecoder,
    "gemma": GemmaDecoder,
}


def create_decoder(decoder_name: str = "qwen", device: str = None):
    """Create a decoder by name. Returns an instance of QwenDecoder or GemmaDecoder."""
    cls = DECODER_MAP.get(decoder_name)
    if cls is None:
        raise ValueError(f"Unknown decoder: {decoder_name!r}. Choose from: {list(DECODER_MAP.keys())}")
    return cls(device=device)
