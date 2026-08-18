# shared/decoder_factory.py
"""Factory function to create the right decoder from a string name."""

from shared.qwen_decoder import QwenDecoder
from shared.gemma_decoder import GemmaDecoder
from shared.structcoder_decoder import StructCoderDecoder

DECODER_MAP = {
    "qwen": QwenDecoder,
    "gemma": GemmaDecoder,
    "structcoder": StructCoderDecoder,
}

DECODER_CHOICES = list(DECODER_MAP)

# Decoder-only vs encoder-decoder. Callers that need to know how conditioning
# is wired (soft prefix vs. encoder input) can check this instead of matching
# on the name.
SEQ2SEQ_DECODERS = {"structcoder"}


def is_seq2seq(decoder_name: str) -> bool:
    return decoder_name in SEQ2SEQ_DECODERS


def create_decoder(decoder_name: str = "qwen", device: str = None,
                   structcoder_ckpt: str = None):
    """Create a decoder by name.

    Args:
        decoder_name: one of DECODER_CHOICES.
        device: torch device string; defaults to cuda when available.
        structcoder_ckpt: path to the released StructCoder weights. Ignored by
            the other decoders. When omitted, StructCoderDecoder also checks
            $STRUCTCODER_CKPT and saved_models/pretrain/pytorch_model.bin
            before falling back to plain CodeT5-base.
    """
    cls = DECODER_MAP.get(decoder_name)
    if cls is None:
        raise ValueError(f"Unknown decoder: {decoder_name!r}. Choose from: {DECODER_CHOICES}")
    if cls is StructCoderDecoder:
        return cls(device=device, checkpoint=structcoder_ckpt)
    return cls(device=device)
