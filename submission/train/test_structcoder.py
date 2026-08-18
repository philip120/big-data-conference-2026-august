# train/test_structcoder.py
"""
Pre-flight check for the StructCoder (seq2seq) decoder.

Run this before spending GPU hours: it exercises every path the training and
evaluation scripts take — checkpoint loading, both adaptation modes, the
checkpoint roundtrip, the loss, and generation — on CPU in about a minute.

    python -m train.test_structcoder
    python -m train.test_structcoder --structcoder_ckpt saved_models/pretrain/pytorch_model.bin

Exit code is nonzero if anything fails.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch

from shared.decoder_factory import create_decoder, DECODER_CHOICES, is_seq2seq
from shared.projector import embedding_rms

CODE = """function y = scale_signal(x, k)
% scale a signal and clip it
y = x * k;
for i = 1:numel(y)
    if y(i) > 1
        y(i) = 1;
    end
end
end
"""
TARGET = ("Step 1: multiply every element of x by k. "
          "Step 2: clip each element to at most 1. Step 3: return the result.")


def check_decoder(ckpt):
    print("=" * 70)
    print("DECODER")
    print("=" * 70)
    print("available decoders:", DECODER_CHOICES)
    assert is_seq2seq("structcoder") and not is_seq2seq("qwen")

    d = create_decoder("structcoder", device="cpu", structcoder_ckpt=ckpt)
    print(f"hidden_size={d.hidden_size}  embedding_rms={embedding_rms(d):.6f}")
    print(f"structcoder weights loaded: {d.structcoder_weights_loaded}")
    if ckpt and not d.structcoder_weights_loaded:
        raise SystemExit(f"FAIL: --structcoder_ckpt {ckpt} given but not loaded")

    # train_full passes --unfreeze_layers 18; CodeT5-base has 12 decoder blocks.
    d.unfreeze_layers(18)
    n_dec = sum(p.numel() for p in d.get_unfrozen_parameters())
    print(f"decoder trainable params: {n_dec:,}")

    state = d.get_unfrozen_state_dict()
    n = d.count_unfrozen_layers(state)
    assert n == d.model.config.num_decoder_layers, f"counted {n} blocks in checkpoint"
    print(f"checkpoint: {len(state)} tensors, {n} decoder blocks")

    # evaluate.py rebuilds the unfrozen set from the saved state before loading
    d2 = create_decoder("structcoder", device="cpu", structcoder_ckpt=ckpt)
    d2.unfreeze_layers(d2.count_unfrozen_layers(state))
    d2.load_unfrozen_state_dict(state)
    assert d2.unfrozen_layers == d.unfrozen_layers
    print("checkpoint roundtrip OK")

    # gradient must reach the projector, or the conditioning path is dead
    proj = torch.nn.Linear(768, 768)
    loss = d.forward_train(proj(torch.randn(3, 768)), TARGET)
    loss.backward()
    gnorm = proj.weight.grad.norm().item()
    print(f"forward_train loss={loss.item():.4f}  projector grad norm={gnorm:.4f}")
    assert gnorm > 0, "no gradient reached the projector"

    text, m = d.generate_text(CODE, max_new_tokens=16)
    print(f"stage-1 generate_text -> {text[:60]!r} ({m['num_generated_tokens']} tokens)")

    # LoRA path
    d3 = create_decoder("structcoder", device="cpu", structcoder_ckpt=ckpt)
    d3.enable_lora(rank=16, alpha=32, dropout=0.1, num_layers=12)
    lp = d3.get_lora_parameters()
    d3.forward_train(torch.randn(3, 768), TARGET).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in lp)
    print(f"LoRA OK: {len(lp)} tensors, {sum(p.numel() for p in lp):,} params")


def check_models(ckpt):
    from train.train_pipeline import create_model

    for model_type in ("vit", "tree", "combined", "tree_text"):
        print("=" * 70)
        print("MODEL:", model_type)
        print("=" * 70)
        if ckpt:
            os.environ["STRUCTCODER_CKPT"] = ckpt
        model = create_model(model_type, patch_size=4, bottleneck_dim=768,
                             dropout=0.05, decoder_name="structcoder",
                             projector_arch="linear", encoder_name="codebert",
                             max_branching=8)
        model.decoder.unfreeze_layers(18)

        n_enc = model.num_trainable_parameters()
        n_dec = sum(p.numel() for p in model.decoder.get_unfrozen_parameters())
        print(f"  encoder trainable {n_enc:,} | decoder trainable {n_dec:,} "
              f"| ratio {n_dec / max(n_enc, 1):.1f}x")

        loss = model(CODE, target=TARGET)
        loss.backward()
        grads = [(n, p.grad.norm().item()) for n, p in model.named_parameters()
                 if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0]
        assert grads, f"{model_type}: no gradient reached the encoder"
        print(f"  loss={loss.item():.4f} | {len(grads)} encoder tensors with grad "
              f"| e.g. {grads[0][0]} {grads[0][1]:.4f}")

        model.eval()
        text, m = model.generate_with_metrics(CODE, max_new_tokens=24)
        print(f"  generated={text[:70]!r}")
        print(f"  in={m['num_input_tokens']} out={m['num_generated_tokens']} "
              f"kv={m['kv_cache_mb']}MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-flight check for the StructCoder decoder")
    parser.add_argument("--structcoder_ckpt", type=str, default=None)
    parser.add_argument("--decoder_only", action="store_true",
                        help="Skip the four stage-2 model classes (skips the CodeBERT download)")
    args = parser.parse_args()

    check_decoder(args.structcoder_ckpt)
    if not args.decoder_only:
        check_models(args.structcoder_ckpt)

    print()
    print("=" * 70)
    print("ALL STRUCTCODER CHECKS PASSED")
    print("=" * 70)
