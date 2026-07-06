# train/train_full.py
"""
Full 2-stage training pipeline. Run overnight.

Stage 1: Fine-tune Qwen decoder on plain text MATLAB→pseudocode.
Stage 2: Train encoder pipeline against the stable Stage 1 decoder.

Usage:
    python -m train.train_full
    python -m train.train_full --s1_epochs 5 --s2_epochs 10
    python -m train.train_full --skip_stage1 --stage1_checkpoint checkpoints_stage1/best_model.pt
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path

from train.train_stage1 import train as train_stage1
from train.train_pipeline import train as train_stage2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full 2-stage training (Stage 1 then Stage 2)")

    # Stage 1 args
    parser.add_argument("--skip_stage1", action="store_true",
                        help="Skip Stage 1 and go straight to Stage 2 (requires --stage1_checkpoint)")
    parser.add_argument("--s1_epochs", type=int, default=5)
    parser.add_argument("--s1_lr", type=float, default=2e-4)
    parser.add_argument("--s1_grad_accum", type=int, default=4)
    parser.add_argument("--s1_save_dir", type=str, default="checkpoints_stage1")

    # Stage 2 args
    parser.add_argument("--s2_epochs", type=int, default=10)
    parser.add_argument("--s2_lr", type=float, default=3e-4)
    parser.add_argument("--s2_lora_lr", type=float, default=1e-4)
    parser.add_argument("--s2_grad_accum", type=int, default=8)
    parser.add_argument("--s2_save_dir", type=str, default="checkpoints_stage2")
    parser.add_argument("--s2_model", type=str, default="combined",
                        choices=["vit", "tree", "combined", "tree_text"])
    parser.add_argument("--bottleneck", type=int, default=768)
    # patch_size default aligned with train_pipeline (was 1 here / 4 there,
    # which silently changed the experiment depending on entry point)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--projector", type=str, default="linear",
                        choices=["linear", "mlp"],
                        help="Patch projector: 'linear' baseline or scale-stabilized GELU 'mlp'")
    parser.add_argument("--code_encoder", type=str, default="codebert",
                        choices=["codebert", "unixcoder", "codesage"],
                        help="Frozen code encoder for pixel embeddings")
    parser.add_argument("--max_branching", type=int, default=8,
                        help="RvNN max children per node before truncation (tree models)")

    # Decoder adaptation — choose one: unfreeze (recommended) or LoRA
    parser.add_argument("--unfreeze_layers", type=int, default=18,
                        help="Unfreeze last N Qwen layers (0 = use LoRA instead)")
    parser.add_argument("--qwen_lr", type=float, default=1e-5,
                        help="LR for unfrozen Qwen layers")

    # LoRA args (used only if --unfreeze_layers 0)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_layers", type=int, default=12)

    # Override: provide an existing Stage 1 checkpoint instead of training one
    parser.add_argument("--stage1_checkpoint", type=str, default=None,
                        help="Path to existing Stage 1 checkpoint (skips Stage 1 training)")

    # Resume args
    parser.add_argument("--s1_resume", type=str, default=None,
                        help="Resume Stage 1 from this checkpoint")
    parser.add_argument("--s2_resume", type=str, default=None,
                        help="Resume Stage 2 from this checkpoint")

    parser.add_argument("--decoder", type=str, default="qwen",
                        choices=["gemma", "qwen"],
                        help="Decoder model: gemma or qwen")
    parser.add_argument("--split", type=str, default="train")

    args = parser.parse_args()

    # Resolve Stage 1 checkpoint path
    s1_best = args.stage1_checkpoint or str(Path(args.s1_save_dir) / "best_model.pt")

    # =========================================================
    # STAGE 1
    # =========================================================
    if not args.skip_stage1 and not args.stage1_checkpoint:
        print("\n" + "#" * 60)
        print("# STAGE 1: Text-only decoder fine-tuning")
        print("#" * 60)
        train_stage1(
            epochs=args.s1_epochs,
            lr=args.s1_lr,
            weight_decay=0.05,
            grad_accum=args.s1_grad_accum,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_layers=args.lora_layers,
            log_every=10,
            save_every=2000,
            save_dir=args.s1_save_dir,
            split=args.split,
            resume=args.s1_resume,
            decoder_name=args.decoder,
            unfreeze_layers=args.unfreeze_layers,
        )
    else:
        print(f"\nSkipping Stage 1. Using checkpoint: {s1_best}")

    # =========================================================
    # STAGE 2
    # =========================================================
    print("\n" + "#" * 60)
    print("# STAGE 2: Encoder training against stable decoder")
    print("#" * 60)
    train_stage2(
        model_type=args.s2_model,
        split=args.split,
        epochs=args.s2_epochs,
        lr=args.s2_lr,
        weight_decay=0.05,
        patch_size=args.patch_size,
        bottleneck_dim=args.bottleneck,
        dropout=args.dropout,
        log_every=10,
        eval_every=50,
        save_every=100,
        save_dir=args.s2_save_dir,
        gradient_accumulation=args.s2_grad_accum,
        lora=args.unfreeze_layers == 0,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_layers=args.lora_layers,
        lora_lr=args.s2_lora_lr,
        unfreeze_layers=args.unfreeze_layers,
        qwen_lr=args.qwen_lr,
        eval_samples=50,
        resume=args.s2_resume,
        stage1_checkpoint=s1_best,
        decoder_name=args.decoder,
        projector_arch=args.projector,
        encoder_name=args.code_encoder,
        max_branching=args.max_branching,
    )

    print("\n" + "#" * 60)
    print("# FULL TRAINING COMPLETE")
    print(f"# Stage 1 checkpoint: {s1_best}")
    print(f"# Stage 2 checkpoints: {args.s2_save_dir}")
    print("#" * 60)
