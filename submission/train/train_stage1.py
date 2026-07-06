# train/train_stage1.py
"""
Stage 1 Training: Fine-tune Qwen on plain text MATLAB→pseudocode pairs.

No encoder. Only QwenDecoder + LoRA trained on (code, target) text pairs.
This gives the decoder a well-formed sense of the task before Stage 2,
where the encoder pipeline is trained against a stable decoder target.

Usage:
    python -m train.train_stage1 \
        --epochs 5 --lora_rank 16 --lora_alpha 128 --lora_layers 12 \
        --grad_accum 4 --lr 2e-4 --save_dir checkpoints_stage1
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
from torch.utils.data import DataLoader
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train.matlab_dataset import MatlabPseudocodeDataset
from shared.decoder_factory import create_decoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True


def train(
    epochs: int = 5,
    lr: float = 2e-4,
    weight_decay: float = 0.05,
    grad_accum: int = 4,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_layers: int = 6,
    log_every: int = 10,
    save_every: int = 100,
    save_dir: str = "checkpoints_stage1",
    split: str = "train",
    resume: str = None,
    decoder_name: str = "qwen",
    unfreeze_layers: int = 0,
):
    """Stage 1 training: text-only decoder fine-tuning."""
    print("=" * 60)
    print(f"STAGE 1 TRAINING: {decoder_name.upper()} text-only MATLAB→pseudocode")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Decoder: {decoder_name}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Weight decay: {weight_decay}")
    print(f"Gradient accumulation: {grad_accum}")
    if unfreeze_layers > 0:
        print(f"Mode: full fine-tune last {unfreeze_layers} Qwen layers")
    else:
        print(f"Mode: LoRA rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}, layers={lora_layers}")
    print(f"Mixed precision (AMP): {DEVICE == 'cuda'}")
    if resume:
        print(f"Resuming from: {resume}")

    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Load dataset
    # model_type="vit" uses the simple extractor; features are pre-computed
    # but only 'code' and 'target' keys are used here.
    print("\n" + "=" * 60)
    print("Loading dataset...")
    dataset = MatlabPseudocodeDataset(split=split, model_type="vit")
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])

    if len(dataset) == 0:
        print("ERROR: No samples found!")
        return

    print(f"Dataset size: {len(dataset)} samples")

    # Load decoder
    print("\n" + "=" * 60)
    print(f"Loading {decoder_name} decoder...")
    decoder = create_decoder(decoder_name, device=DEVICE)

    if unfreeze_layers > 0:
        decoder.unfreeze_layers(unfreeze_layers)
        trainable_params = decoder.get_unfrozen_parameters()
    else:
        decoder.enable_lora(
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            num_layers=lora_layers,
        )
        trainable_params = decoder.get_lora_parameters()

    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    # OneCycleLR scheduler
    # Use ceil to avoid off-by-one: integer division can undercount by 1
    import math
    total_steps = math.ceil(epochs * len(loader) / grad_accum)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=max(total_steps, 1),
        pct_start=0.1,
        anneal_strategy="cos",
    )

    # Mixed precision
    # bfloat16: same dynamic range as float32, no gradient scaling needed (A100 native)
    use_amp = DEVICE == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    # Training state
    global_step = 0
    start_epoch = 0
    accumulation_step = 0
    running_loss = 0.0
    best_loss = float('inf')
    loss_history = []

    # Resume from checkpoint
    if resume:
        print(f"\nResuming from {resume}...")
        ckpt = torch.load(resume, map_location=DEVICE, weights_only=False)

        if unfreeze_layers > 0 and 'qwen_state' in ckpt and ckpt['qwen_state']:
            decoder.load_unfrozen_state_dict(ckpt['qwen_state'])
        elif 'lora_state' in ckpt and ckpt['lora_state']:
            decoder.load_lora_state_dict(ckpt['lora_state'])

        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])

        if 'scheduler' in ckpt:
            ckpt_total = ckpt['scheduler'].get('total_steps', total_steps)
            if ckpt_total == total_steps:
                scheduler.load_state_dict(ckpt['scheduler'])
            else:
                print(f"  Scheduler reset (checkpoint had {ckpt_total} steps, now {total_steps})")

        if 'scaler' in ckpt and use_amp:
            scaler.load_state_dict(ckpt['scaler'])

        global_step = ckpt.get('step', 0)
        start_epoch = ckpt.get('epoch', 0)
        best_loss = ckpt.get('best_loss', float('inf'))
        loss_history = ckpt.get('loss_history', [])
        accumulation_step = global_step * grad_accum

        print(f"  Resumed at epoch {start_epoch + 1}, step {global_step}, best_loss {best_loss:.4f}")

    print("\n" + "=" * 60)
    print("Starting Stage 1 training...")
    print("=" * 60)

    decoder.train_mode()

    for epoch in range(start_epoch, epochs):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1}/{epochs}")
        print(f"{'='*60}")

        epoch_loss = 0.0
        epoch_samples = 0

        for batch_idx, batch in enumerate(loader):
            code = batch['code']
            target = batch['target']

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = decoder.forward_train_text(code, target)

            if loss is None or loss.item() == 0:
                continue

            loss = loss / grad_accum
            scaler.scale(loss).backward()

            accumulation_step += 1
            running_loss += loss.item() * grad_accum
            epoch_loss += loss.item() * grad_accum
            epoch_samples += 1

            if accumulation_step % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % log_every == 0:
                    avg_loss = running_loss / (log_every * grad_accum)
                    lr_now = scheduler.get_last_lr()[0]
                    print(f"  step {global_step:4d} | loss {avg_loss:.4f} | lr {lr_now:.2e}")
                    loss_history.append((global_step, avg_loss))
                    running_loss = 0.0

                if global_step % save_every == 0:
                    # Lite checkpoint: skip qwen_state + optimizer (too large for disk)
                    # Use best_model.pt to resume Qwen weights
                    checkpoint = {
                        'stage': 1,
                        'step': global_step,
                        'epoch': epoch,
                        'decoder_name': decoder_name,
                        'lora_state': decoder.get_lora_state_dict(),
                        'qwen_state': {} if unfreeze_layers > 0 else {},
                        'scheduler': scheduler.state_dict(),
                        'best_loss': best_loss,
                        'loss_history': loss_history,
                    }
                    torch.save(checkpoint, save_path / f"checkpoint_{global_step}.pt")
                    print(f"  Saved checkpoint_{global_step}.pt")

        avg_epoch_loss = epoch_loss / max(epoch_samples, 1)
        print(f"\nEpoch {epoch + 1} complete | avg_loss: {avg_epoch_loss:.4f}")

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            checkpoint = {
                'stage': 1,
                'step': global_step,
                'epoch': epoch,
                'decoder_name': decoder_name,
                'lora_state': decoder.get_lora_state_dict(),
                'qwen_state': decoder.get_unfrozen_state_dict() if unfreeze_layers > 0 else {},
                'scheduler': scheduler.state_dict(),
                'best_loss': best_loss,
                'loss_history': loss_history,
            }
            torch.save(checkpoint, save_path / "best_model.pt")
            print(f"  New best model! Loss: {best_loss:.4f}")

    # Save final weights
    final_checkpoint = {
        'stage': 1,
        'step': global_step,
        'epoch': epochs - 1,
        'decoder_name': decoder_name,
        'lora_state': decoder.get_lora_state_dict(),
        'qwen_state': decoder.get_unfrozen_state_dict() if unfreeze_layers > 0 else {},
        'best_loss': best_loss,
        'loss_history': loss_history,
    }
    torch.save(final_checkpoint, save_path / "final.pt")
    print(f"\nSaved final.pt")

    # Save loss curve
    if loss_history:
        steps, losses = zip(*loss_history)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(steps, losses, linewidth=1.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Stage 1 Training Loss (Text-Only MATLAB→Pseudocode)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_path / "loss_curve.png", dpi=150)
        plt.close(fig)
        print(f"Saved {save_path / 'loss_curve.png'}")

    # Save metrics JSON
    metrics = {
        "stage": 1,
        "loss_history": [{"step": s, "loss": l} for s, l in loss_history],
        "best_loss": best_loss,
        "total_steps": global_step,
        "epochs": epochs,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_layers": lora_layers,
    }
    with open(save_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved {save_path / 'metrics.json'}")

    print("\n" + "=" * 60)
    print("STAGE 1 TRAINING COMPLETE")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 1: Fine-tune decoder on plain text MATLAB→pseudocode"
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_layers", type=int, default=6)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="checkpoints_stage1")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--decoder", type=str, default="qwen",
                        choices=["gemma", "qwen"],
                        help="Decoder model: gemma or qwen")
    parser.add_argument("--unfreeze_layers", type=int, default=0,
                        help="Unfreeze last N decoder layers fully (replaces LoRA)")

    args = parser.parse_args()

    train(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_accum=args.grad_accum,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_layers=args.lora_layers,
        log_every=args.log_every,
        save_every=args.save_every,
        save_dir=args.save_dir,
        split=args.split,
        resume=args.resume,
        decoder_name=args.decoder,
        unfreeze_layers=args.unfreeze_layers,
    )
