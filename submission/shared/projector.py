# shared/projector.py
"""
Projectors: patch embeddings -> decoder embedding space.

Two variants:

  Projector          — single Linear (the proven baseline; see its docstring)
  NonLinearProjector — stabilized 2-layer GELU MLP that does NOT explode

Why the naive GELU MLP exploded
-------------------------------
The decoder is frozen, and its token embeddings live at a tiny scale
(Qwen3-4B: per-token L2 norm ~1.09, i.e. per-element RMS ~0.0215).
Cross-entropy always gains from larger logits, so during training nothing
stops the two weight matrices of a plain MLP from co-growing; a 2-layer
map's effective gain is the *product* of its layer gains, so small drifts
compound until the patch tokens dominate the decoder's residual stream.
Gradient clipping cannot prevent this — the growth comes from many small,
consistently-aligned steps, not from gradient spikes.

The single Linear survives because its init happens to land near the right
scale and a 1-layer map drifts linearly, not multiplicatively.

How NonLinearProjector stays stable
-----------------------------------
    x -> LayerNorm(in)                       (a)
      -> skip Linear(in→out)                 (b)
       + MLP: Linear(in→hidden) -> GELU -> Dropout -> Linear(hidden→out, ZERO-init)  (c)
    y -> y / rms(y) * gain,  gain init = decoder embedding RMS            (d)

  (a) CodeBERT CLS vectors are unnormalized and the last patch is
      zero-padded (a patch with 3 zero pixels has 1/4 the variance of a
      full one at patch_size=4); LayerNorm equalizes this before the MLP
      can distort it nonlinearly.
  (b)+(c) ControlNet-style zero-init residual: at init the model IS the
      linear baseline; the nonlinearity fades in only as far as it helps.
      (fc1 receives zero gradient on step 0 — fc2 moves off zero first,
      then gradient flows. This is expected, not a bug.)
  (d) The output is RMS-normalized then scaled by a learned per-channel
      gain *initialized to the measured decoder embedding RMS* (pass
      target_rms; see embedding_rms()). Weight growth in (b)/(c) is now
      radial-invariant — it cannot change the output scale — so the
      runaway direction is projected out, while scale can still adapt
      through the gain, whose gradients are undamped because it starts
      at the correct value.

      This is what the earlier LayerNorm attempt got wrong: default LN
      pins per-element variance to 1 (per-token norm sqrt(out_dim) ≈ 50,
      ~45x too big), and a compensating scalar starting at 1 damps
      gradients by that same 45x. Pinning at the *decoder's* RMS from the
      start has neither problem.

References: LLaVA-1.5 mlp2x_gelu connector (works there because CLIP
features arrive pre-LayerNormed and the LLM is tuned); BLIP-2 / Qwen-VL
input-side LN on the connector; ControlNet zero-init branches; GPT-2
residual-branch downscaling.
"""
import torch
import torch.nn as nn


def embedding_rms(decoder) -> float:
    """Per-element RMS of the decoder's input embedding table.

    Multiply by sqrt(hidden_size) to get the typical per-token L2 norm
    (~1.09 for Qwen3-4B). Used to initialize NonLinearProjector's gain.
    """
    w = decoder.model.get_input_embeddings().weight
    return w.detach().float().pow(2).mean().sqrt().item()


class Projector(nn.Module):
    """
    Single linear projection: patch_dim → qwen_dim

    A bottleneck MLP with a final LayerNorm pins the output norm to
    sqrt(out_dim) ≈ 50, which is 45× larger than Qwen token norms (~1.09),
    causing the projected embeddings to dominate the residual stream and
    bypass the transformer layers. Compensating with a learned scalar damps
    projector gradients by the same 45× factor, collapsing representations.

    A single Linear layer avoids both failure modes:
      - Kaiming init: fan_in=in_dim, so output std ≈ sqrt(2/in_dim) * input_std
        which for unit-normal inputs gives norm ≈ sqrt(out_dim * 2/in_dim)
        ≈ sqrt(2560 * 2/3072) ≈ 1.29 — close to Qwen's ~1.09.
      - No LayerNorm means norm can adapt freely during training.
      - No bottleneck means gradients are not compressed through a narrow layer.
    """

    def __init__(
        self,
        in_dim: int = 3072,        # patch_size * 768
        bottleneck_dim: int = 512,  # unused, kept for API compatibility
        out_dim: int = 2560,        # Qwen embedding dim
        dropout: float = 0.0,       # unused, kept for API compatibility
    ):
        super().__init__()

        self.in_dim = in_dim
        self.bottleneck_dim = bottleneck_dim
        self.out_dim = out_dim

        self.net = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project patch embeddings to Qwen space.

        Args:
            x: [num_patches, in_dim] patch embeddings

        Returns:
            [num_patches, out_dim] projected embeddings
        """
        return self.net(x)

    def num_parameters(self) -> int:
        """Return total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class NonLinearProjector(nn.Module):
    """
    Scale-stabilized non-linear projector (see module docstring for why).

        LN(in) -> [ skip Linear + zero-init GELU MLP ] -> RMS-pin * gain

    At init this is exactly a (scale-corrected) linear projector; the MLP
    branch fades in during training and weight growth cannot change the
    output scale.
    """

    def __init__(
        self,
        in_dim: int = 3072,          # patch_size * 768
        hidden_dim: int = 2560,      # MLP width (LLaVA-style: decoder dim)
        out_dim: int = 2560,         # decoder embedding dim
        dropout: float = 0.1,        # on the MLP branch only
        target_rms: float = 0.0215,  # decoder embedding per-element RMS
    ):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.eps = 1e-6

        self.in_norm = nn.LayerNorm(in_dim)
        self.skip = nn.Linear(in_dim, out_dim)

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        self.gain = nn.Parameter(torch.full((out_dim,), float(target_rms)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [num_patches, in_dim] patch embeddings

        Returns:
            [num_patches, out_dim] embeddings at decoder scale
        """
        x = self.in_norm(x)
        y = self.skip(x) + self.fc2(self.drop(self.act(self.fc1(x))))
        inv_rms = torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return y * inv_rms * self.gain

    def num_parameters(self) -> int:
        """Return total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Self-test: init scale, linear-equivalence at init, runaway immunity.
    torch.manual_seed(0)

    PATCH_SIZE = 4
    in_dim = PATCH_SIZE * 768   # 3072
    out_dim = 2560
    TARGET_RMS = 0.0215         # Qwen3-4B embedding RMS (norm 1.09 / sqrt(2560))
    target_norm = TARGET_RMS * out_dim ** 0.5

    # CodeBERT-like patches: unnormalized (std ~0.4), last patch zero-padded
    x = torch.randn(4, in_dim) * 0.4
    x[-1, 768:] = 0.0  # patch with 1 real pixel + 3 zero pixels

    print("NonLinearProjector Self-Test")
    print("=" * 60)

    proj = NonLinearProjector(in_dim=in_dim, hidden_dim=out_dim,
                              out_dim=out_dim, dropout=0.0,
                              target_rms=TARGET_RMS)
    naive = nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU(),
                          nn.Linear(out_dim, out_dim))

    # 1. Init-time output scale (padded patch included)
    with torch.no_grad():
        y = proj(x)
        y_naive = naive(x)
    norms = y.norm(dim=-1)
    print(f"\n[1] Init output norms (target {target_norm:.3f}):")
    print(f"    stabilized: {[f'{n:.3f}' for n in norms.tolist()]}")
    print(f"    naive MLP : {[f'{n:.3f}' for n in y_naive.norm(dim=-1).tolist()]}")
    assert torch.allclose(norms, torch.full_like(norms, target_norm), rtol=1e-3), \
        "output not pinned to target scale at init"

    # 2. At init the model must equal the RMS-pinned skip path (MLP branch = 0)
    with torch.no_grad():
        s = proj.skip(proj.in_norm(x))
        s = s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + proj.eps) * proj.gain
    assert torch.allclose(y, s, atol=1e-6), "zero-init MLP branch is not silent at init"
    print("\n[2] Init output == scale-corrected linear model: OK")

    # 3. Runaway immunity: weight growth must not change output scale
    with torch.no_grad():
        for m in (proj.skip, proj.fc1, proj.fc2):
            m.weight.mul_(10.0)
        naive[0].weight.mul_(10.0)
        naive[2].weight.mul_(10.0)
        y10 = proj(x)
        y10_naive = naive(x)
    print(f"\n[3] After 10x weight growth, mean output norm:")
    print(f"    stabilized: {y.norm(dim=-1).mean():.3f} -> {y10.norm(dim=-1).mean():.3f}")
    print(f"    naive MLP : {y_naive.norm(dim=-1).mean():.3f} -> {y10_naive.norm(dim=-1).mean():.3f}")
    assert abs(y10.norm(dim=-1).mean() - target_norm) < 1e-2 * target_norm

    # 4. Gradients reach every parameter (fc1 gets grad once fc2 leaves zero)
    proj2 = NonLinearProjector(in_dim=in_dim, hidden_dim=out_dim,
                               out_dim=out_dim, dropout=0.0,
                               target_rms=TARGET_RMS)
    opt = torch.optim.AdamW(proj2.parameters(), lr=1e-4)
    for step in range(2):
        opt.zero_grad()
        proj2(x).sum().backward()
        opt.step()
    grads_ok = {n: (p.grad is not None and p.grad.abs().sum().item() > 0)
                for n, p in proj2.named_parameters()}
    print(f"\n[4] Nonzero grads after 2 steps: {grads_ok}")
    assert all(grads_ok.values()), "some parameters receive no gradient"

    print(f"\nTrainable params: {proj.num_parameters():,} "
          f"(linear baseline: {Projector(in_dim, out_dim=out_dim).num_parameters():,})")
    print("\nAll checks passed.")
