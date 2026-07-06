# shared/patch_embedder.py
"""
Patch Embedder

Groups pixel embeddings into patches (like ViT).

Input:  [N, 768] pixel embeddings
Output: [num_patches, patch_size * 768] flattened patches
"""
import torch
import torch.nn as nn


class PatchEmbedder(nn.Module):
    """
    Groups pixels into patches and flattens them.

    Example with patch_size=4:
        Input:  10 pixels [10, 768]
        Pad:    12 pixels [12, 768]  (pad with zeros to make divisible)
        Group:  3 patches [3, 4, 768]
        Flatten: [3, 3072]  (4 * 768 = 3072)
    """

    def __init__(self, patch_size: int = 4, embed_dim: int = 768):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.flat_dim = patch_size * embed_dim  # e.g., 4 * 768 = 3072

    def forward(self, pixel_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Group pixels into patches.

        Args:
            pixel_embeddings: [N, 768] tensor of pixel embeddings

        Returns:
            [num_patches, patch_size * 768] flattened patches
        """
        N, D = pixel_embeddings.shape
        P = self.patch_size

        # Calculate padding needed
        remainder = N % P
        if remainder != 0:
            pad_len = P - remainder
            # Pad with zeros
            padding = torch.zeros(pad_len, D, device=pixel_embeddings.device, dtype=pixel_embeddings.dtype)
            pixel_embeddings = torch.cat([pixel_embeddings, padding], dim=0)
            N = N + pad_len

        # Reshape into patches: [N, D] → [num_patches, P, D]
        num_patches = N // P
        patches = pixel_embeddings.reshape(num_patches, P, D)

        # Flatten each patch: [num_patches, P, D] → [num_patches, P*D]
        flat_patches = patches.reshape(num_patches, P * D)

        return flat_patches

    def get_num_patches(self, num_pixels: int) -> int:
        """Calculate number of patches for given number of pixels."""
        import math
        return math.ceil(num_pixels / self.patch_size)


if __name__ == "__main__":
    # Test
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Fake pixel embeddings
    N = 10  # 10 pixels
    pixel_emb = torch.randn(N, 768, device=device)

    patch_sizes = [2, 4, 8]

    print("PatchEmbedder Test")
    print("=" * 60)
    print(f"Input: {N} pixels, shape {pixel_emb.shape}\n")

    for P in patch_sizes:
        embedder = PatchEmbedder(patch_size=P)
        patches = embedder(pixel_emb)
        print(f"  patch_size={P}:")
        print(f"    → {patches.shape[0]} patches, each {patches.shape[1]} dim")
        print(f"    → shape: {patches.shape}")
        print()
