# shared/codebert_encoder.py
"""
CodeBERT Encoder

Frozen CodeBERT that extracts CLS token for each pixel text.
"""
import torch
from transformers import AutoTokenizer, AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CodeBERTEncoder:
    """
    Frozen CodeBERT encoder - extracts CLS token for each text.

    Input:  list of text strings (one per pixel)
    Output: [N, 768] tensor of CLS embeddings
    """

    def __init__(self, device: str = None):
        self.device = device or DEVICE

        print("Loading CodeBERT...")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = AutoModel.from_pretrained("microsoft/codebert-base")
        self.model.to(self.device)
        self.model.eval()

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        print(f"CodeBERT loaded on {self.device} (frozen)")

    @torch.no_grad()
    def __call__(self, texts: list[str]) -> torch.Tensor:
        """
        Encode list of texts, return CLS token for each.

        Args:
            texts: list of N strings

        Returns:
            [N, 768] tensor
        """
        if not texts:
            return torch.zeros(0, 768, device=self.device)

        # Tokenize all texts
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt"
        ).to(self.device)

        # Forward through CodeBERT
        outputs = self.model(**tokens)

        # Extract CLS token (first token) for each text
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [N, 768]

        return cls_embeddings


if __name__ == "__main__":
    # Test
    texts = [
        "function y = test(x)",
        "if x > 0",
        "y = x * 2",
    ]

    encoder = CodeBERTEncoder()
    embeddings = encoder(texts)

    print("\nCodeBERTEncoder Test")
    print("=" * 50)
    print(f"Input:  {len(texts)} texts")
    print(f"Output: {embeddings.shape}")
    print(f"\nFirst 5 values of each embedding:")
    for i, text in enumerate(texts):
        print(f"  [{i}] \"{text}\" → {embeddings[i][:5].tolist()}")
