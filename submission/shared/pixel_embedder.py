# shared/pixel_embedder.py
"""
Pixel Embedder

Combines CodeBERT CLS + depth embedding + type embedding.
Trainable: depth_embedding, type_embedding
"""
import torch
import torch.nn as nn


class PixelEmbedder(nn.Module):
    """
    Combines CodeBERT CLS + depth embedding + type embedding.

    pixel_embedding = CLS + depth_emb + type_emb

    Trainable: depth_embedding, type_embedding
    """

    def __init__(self, max_depth: int = 16, num_types: int = 16, embed_dim: int = 768):
        super().__init__()
        self.max_depth = max_depth
        self.num_types = num_types
        self.embed_dim = embed_dim

        # Trainable embeddings
        self.depth_embedding = nn.Embedding(max_depth, embed_dim)
        self.type_embedding = nn.Embedding(num_types, embed_dim)

        # Initialize with small values (so they don't dominate CLS initially)
        nn.init.normal_(self.depth_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.type_embedding.weight, mean=0.0, std=0.02)

    def forward(
        self,
        cls_embeddings: torch.Tensor,  # [N, 768] from CodeBERT
        depth_ids: torch.Tensor,        # [N] depth indices
        type_ids: torch.Tensor,         # [N] type indices
    ) -> torch.Tensor:
        """
        Combine CLS + depth + type embeddings.

        Args:
            cls_embeddings: [N, 768] from CodeBERT
            depth_ids: [N] tensor of depth indices (0 to max_depth-1)
            type_ids: [N] tensor of type indices (0 to num_types-1)

        Returns:
            [N, 768] pixel embeddings
        """
        depth_emb = self.depth_embedding(depth_ids)  # [N, 768]
        type_emb = self.type_embedding(type_ids)      # [N, 768]

        # Combine by addition
        pixel_embeddings = cls_embeddings + depth_emb + type_emb  # [N, 768]

        return pixel_embeddings

    def num_parameters(self) -> int:
        """Return total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Fake CLS embeddings (normally from CodeBERT)
    N = 5
    cls_embeddings = torch.randn(N, 768, device=device)
    depth_ids = torch.tensor([0, 1, 2, 2, 1], device=device)
    type_ids = torch.tensor([0, 1, 9, 10, 3], device=device)

    embedder = PixelEmbedder().to(device)
    pixel_emb = embedder(cls_embeddings, depth_ids, type_ids)

    print("PixelEmbedder Test")
    print("=" * 50)
    print(f"Input CLS:     {cls_embeddings.shape}")
    print(f"Input depths:  {depth_ids.tolist()}")
    print(f"Input types:   {type_ids.tolist()}")
    print(f"Output pixels: {pixel_emb.shape}")
    print(f"\nTrainable params: {embedder.num_parameters():,}")
