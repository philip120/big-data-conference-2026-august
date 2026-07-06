# shared/code_encoder.py
"""
Pluggable frozen code encoder for pixel/node embeddings.

Presets (select with --code_encoder):

    codebert   microsoft/codebert-base       CLS pooling, 768-d
               (2020, RoBERTa-base, 6 languages, NO MATLAB — the original
               baseline; kept as default for continuity with prior results)
    unixcoder  microsoft/unixcoder-base      CLS pooling, 768-d
               (2022, stronger representations, same dims — drop-in upgrade)
    codesage   codesage/codesage-small-v2    mean pooling, 1024-d
               (embedding model trained on The Stack v2, which INCLUDES
                MATLAB — best match for this paper's motivation)

All downstream dims (pixel_adapter, projector in_dim) must be derived from
`encoder.hidden_size`, never hardcoded to 768.
"""
import torch
from transformers import AutoTokenizer, AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ENCODER_PRESETS = {
    "codebert": {
        "model_name": "microsoft/codebert-base",
        "pooling": "cls",
        "trust_remote_code": False,
    },
    "unixcoder": {
        "model_name": "microsoft/unixcoder-base",
        "pooling": "cls",
        "trust_remote_code": False,
    },
    "codesage": {
        "model_name": "codesage/codesage-small-v2",
        "pooling": "mean",
        "trust_remote_code": True,
    },
}


class CodeEncoder:
    """
    Frozen encoder: list of code-fragment strings -> [N, hidden_size] embeddings.

    With preset="codebert" this reproduces the original CodeBERTEncoder
    behavior exactly (CLS token of microsoft/codebert-base).
    """

    def __init__(self, preset: str = "codebert", device: str = None, max_length: int = 64):
        if preset not in ENCODER_PRESETS:
            raise ValueError(f"Unknown encoder preset '{preset}'. "
                             f"Available: {list(ENCODER_PRESETS)}")
        cfg = ENCODER_PRESETS[preset]
        self.preset = preset
        self.pooling = cfg["pooling"]
        self.max_length = max_length
        self.device = device or DEVICE

        print(f"Loading code encoder '{preset}' ({cfg['model_name']})...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg["model_name"], trust_remote_code=cfg["trust_remote_code"])
        self.model = AutoModel.from_pretrained(
            cfg["model_name"], trust_remote_code=cfg["trust_remote_code"])
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.hidden_size = self.model.config.hidden_size

        # Some tokenizers (e.g. codesage) ship without a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Encoder '{preset}' loaded on {self.device} "
              f"(frozen, hidden_size={self.hidden_size}, pooling={self.pooling})")

    @torch.no_grad()
    def __call__(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            return torch.zeros(0, self.hidden_size, device=self.device)

        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**tokens)
        hidden = outputs.last_hidden_state  # [N, L, H]

        if self.pooling == "cls":
            return hidden[:, 0, :]

        # Mean pooling over non-padding tokens
        mask = tokens.attention_mask.unsqueeze(-1).to(hidden.dtype)  # [N, L, 1]
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        return summed / counts


if __name__ == "__main__":
    texts = ["function y = test(x)", "if x > 0", "y = x * 2"]
    for preset in ["codebert"]:
        enc = CodeEncoder(preset=preset)
        emb = enc(texts)
        print(f"{preset}: {emb.shape}, norm={emb.norm(dim=-1).mean():.3f}")
