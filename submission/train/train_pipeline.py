# train/train_pipeline.py
"""
Training Pipeline for Semantic ViT

Supports four model modes:
    --model vit       : ViT-only (model/)
    --model tree      : Tree-only (model2/)
    --model combined  : Both pathways (combined_model/)
    --model tree_text : RvNN AST vector + native text embeddings (tree_text_model/)

Patch-based modes (vit, combined) additionally take --projector {linear,mlp}:
    linear : single Linear projector (baseline)
    mlp    : scale-stabilized GELU MLP (see shared/projector.py)

Usage:
    python -m train.train_pipeline --model combined --epochs 10 --lr 1e-4
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
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from train.load_dataset import load_matlab_nl_dataset
from train.matlab_dataset import MatlabPseudocodeDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# cuDNN auto-tuner: picks fastest algorithms for the hardware
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True


def create_model(model_type: str, patch_size: int, bottleneck_dim: int, dropout: float,
                 decoder_name: str = "qwen", projector_arch: str = "linear",
                 encoder_name: str = "codebert", max_branching: int = 8):
    """Create model based on type selection."""
    if model_type == "vit":
        from model.model import SemanticViT
        return SemanticViT(
            patch_size=patch_size,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout,
            decoder_name=decoder_name,
            projector_arch=projector_arch,
            encoder_name=encoder_name,
        )
    elif model_type == "tree":
        from model2.model import StructuralModel
        return StructuralModel(
            dropout=dropout,
            decoder_name=decoder_name,
            encoder_name=encoder_name,
            max_branching=max_branching,
        )
    elif model_type == "combined":
        from combined_model.model import CombinedSemanticViT
        return CombinedSemanticViT(
            patch_size=patch_size,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout,
            decoder_name=decoder_name,
            projector_arch=projector_arch,
            encoder_name=encoder_name,
            max_branching=max_branching,
        )
    elif model_type == "tree_text":
        from tree_text_model.model import TreeTextModel
        return TreeTextModel(
            dropout=dropout,
            decoder_name=decoder_name,
            encoder_name=encoder_name,
            max_branching=max_branching,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'vit', 'tree', 'combined', or 'tree_text'.")


# ==============================================================================
# TRAINING LOOP
# ==============================================================================

def train(
    model_type: str = "combined",
    split: str = "train",
    epochs: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 0.05,
    patch_size: int = 4,
    bottleneck_dim: int = 512,
    dropout: float = 0.15,
    log_every: int = 10,
    eval_every: int = 50,
    save_every: int = 100,
    save_dir: str = "checkpoints",
    gradient_accumulation: int = 2,
    lora: bool = False,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_layers: int = 6,
    lora_lr: float = 1e-4,
    eval_samples: int = 50,
    resume: str = None,
    stage1_checkpoint: str = None,
    unfreeze_layers: int = 0,
    qwen_lr: float = 1e-5,
    decoder_name: str = "qwen",
    projector_arch: str = "linear",
    encoder_name: str = "codebert",
    max_branching: int = 8,
):
    """Main training function."""
    print("=" * 60)
    print(f"SEMANTIC VIT TRAINING ({model_type.upper()} MODE)")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Model: {model_type}")
    print(f"Decoder: {decoder_name}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Weight decay: {weight_decay}")
    print(f"Patch size: {patch_size}")
    print(f"Bottleneck: {bottleneck_dim}")
    print(f"Projector: {projector_arch}")
    print(f"Code encoder: {encoder_name}")
    print(f"RvNN max_branching: {max_branching}")
    print(f"Dropout: {dropout}")
    print(f"Gradient accumulation: {gradient_accumulation}")
    print(f"Mixed precision (AMP): {DEVICE == 'cuda'}")
    print(f"LoRA: {lora} (rank={lora_rank}, alpha={lora_alpha}, layers={lora_layers}, lora_lr={lora_lr})")
    if unfreeze_layers > 0:
        print(f"Unfreeze: last {unfreeze_layers} decoder layers (qwen_lr={qwen_lr})")
    if resume:
        print(f"Resuming from: {resume}")

    # Create save directory
    save_path = Path(save_dir) / model_type
    save_path.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("\n" + "=" * 60)
    print("Loading dataset from Hugging Face...")
    # Full train split; evaluation uses the hub's held-out "test" split,
    # which is never seen in stage 1 or stage 2 training.
    dataset = MatlabPseudocodeDataset(split=split, model_type=model_type)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])

    if len(dataset) == 0:
        print("ERROR: No samples found!")
        return

    # Create model
    print("\n" + "=" * 60)
    print("Creating model...")
    model = create_model(model_type, patch_size, bottleneck_dim, dropout, decoder_name,
                         projector_arch, encoder_name, max_branching)

    # Enable LoRA or unfreeze Qwen layers before optimizer construction
    if lora:
        model.enable_lora(
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            num_layers=lora_layers,
        )
    if unfreeze_layers > 0:
        model.decoder.unfreeze_layers(unfreeze_layers)

    # Load Stage 1 weights into decoder before optimizer construction
    if stage1_checkpoint:
        print(f"\nLoading Stage 1 weights from: {stage1_checkpoint}")
        s1_ckpt = torch.load(stage1_checkpoint, map_location=DEVICE, weights_only=False)
        if s1_ckpt.get("qwen_state"):
            # Load Stage 1 decoder weights into the base model regardless
            # of Stage 2 strategy (LoRA or unfreeze). This ensures the
            # decoder starts from the fine-tuned state.
            model.decoder.model.load_state_dict(s1_ckpt["qwen_state"], strict=False)
            print(f"  Loaded {len(s1_ckpt['qwen_state'])} Qwen tensors from Stage 1.")
        elif s1_ckpt.get("lora_state"):
            if not lora:
                print("  Warning: Stage 1 used LoRA but Stage 2 does not. Skipping LoRA state.")
            else:
                model.decoder.load_lora_state_dict(s1_ckpt["lora_state"])
                print(f"  Loaded {len(s1_ckpt['lora_state'])} LoRA tensors from Stage 1.")
        else:
            print("  Warning: Stage 1 checkpoint has no decoder weights to load.")

    print(f"\nTrainable parameters: {model.num_trainable_parameters():,}")

    # Optimizer — separate LR groups for encoder vs decoder adaptation
    if lora:
        lora_ids = set(id(p) for p in model.decoder.get_lora_parameters())
        base_params = [p for p in model.get_trainable_parameters() if id(p) not in lora_ids]
        lora_params = list(model.decoder.get_lora_parameters())
        optimizer = torch.optim.AdamW([
            {'params': base_params, 'lr': lr, 'weight_decay': weight_decay},
            {'params': lora_params, 'lr': lora_lr, 'weight_decay': weight_decay},
        ])
        print(f"\nParam groups: encoder ({len(base_params)} tensors, lr={lr}) "
              f"+ LoRA ({len(lora_params)} tensors, lr={lora_lr})")
    elif unfreeze_layers > 0:
        encoder_params = model.get_trainable_parameters()
        qwen_params = model.decoder.get_unfrozen_parameters()
        optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': lr, 'weight_decay': weight_decay},
            {'params': qwen_params, 'lr': qwen_lr, 'weight_decay': weight_decay},
        ])
        print(f"\nParam groups: encoder ({len(encoder_params)} tensors, lr={lr}) "
              f"+ Qwen unfrozen ({len(qwen_params)} tensors, lr={qwen_lr})")
    else:
        optimizer = torch.optim.AdamW(
            model.get_trainable_parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    # Learning rate scheduler — warmup + cosine decay
    # Use ceil to avoid off-by-one: integer division can undercount by 1
    import math
    total_steps = math.ceil(epochs * len(loader) / gradient_accumulation)
    if lora:
        max_lrs = [lr, lora_lr]
    elif unfreeze_layers > 0:
        max_lrs = [lr, qwen_lr]
    else:
        max_lrs = lr
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lrs,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    # Mixed precision (AMP) — only on CUDA
    # bfloat16: same dynamic range as float32, no gradient scaling needed (A100 native)
    use_amp = DEVICE == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

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

        # Restore trainable parameters
        if 'model_state' in ckpt:
            model_params = dict(model.named_parameters())
            loaded = 0
            for name, data in ckpt['model_state'].items():
                if name in model_params and model_params[name].requires_grad:
                    model_params[name].data.copy_(data)
                    loaded += 1
            print(f"  Restored {loaded} parameter tensors")

        # Restore LoRA or unfrozen Qwen state
        if lora and 'lora_state' in ckpt and ckpt['lora_state']:
            model.decoder.load_lora_state_dict(ckpt['lora_state'])
        if unfreeze_layers > 0 and 'qwen_state' in ckpt and ckpt['qwen_state']:
            model.decoder.load_unfrozen_state_dict(ckpt['qwen_state'])

        # Restore optimizer, scheduler, scaler
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            ckpt_total = ckpt['scheduler'].get('total_steps', total_steps)
            if ckpt_total == total_steps:
                scheduler.load_state_dict(ckpt['scheduler'])
            else:
                print(f"  Scheduler reset (checkpoint had {ckpt_total} total steps, now {total_steps})")
        if 'scaler' in ckpt and use_amp:
            scaler.load_state_dict(ckpt['scaler'])

        # Restore progress
        global_step = ckpt.get('step', 0)
        start_epoch = ckpt.get('epoch', 0)
        best_loss = ckpt.get('best_loss', ckpt.get('loss', float('inf')))
        loss_history = ckpt.get('loss_history', [])
        accumulation_step = global_step * gradient_accumulation

        print(f"  Resumed at epoch {start_epoch + 1}, step {global_step}, best_loss {best_loss:.4f}")

    model.train()

    for epoch in range(start_epoch, epochs):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1}/{epochs}")
        print(f"{'='*60}")

        epoch_loss = 0.0
        epoch_samples = 0

        for batch_idx, batch in enumerate(loader):
            code = batch['code']
            target = batch['target']
            features = batch.get('features')

            # Forward pass (mixed precision)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = model(code, target=target, features=features)

            if loss is None or loss.item() == 0:
                continue

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation
            scaler.scale(loss).backward()

            accumulation_step += 1
            running_loss += loss.item() * gradient_accumulation
            epoch_loss += loss.item() * gradient_accumulation
            epoch_samples += 1

            # Update weights after accumulation
            if accumulation_step % gradient_accumulation == 0:
                scaler.unscale_(optimizer)
                all_trainable = [p for p in model.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % log_every == 0:
                    avg_loss = running_loss / (log_every * gradient_accumulation)
                    lr_now = scheduler.get_last_lr()[0]
                    print(f"  step {global_step:4d} | loss {avg_loss:.4f} | lr {lr_now:.2e}")
                    loss_history.append((global_step, avg_loss))
                    running_loss = 0.0

                # Evaluation
                if global_step % eval_every == 0:
                    model.eval()

                    print(f"\n  --- Eval at step {global_step} ---")
                    print(f"  Code: {code[:80]}...")
                    print(f"  Target: {target[:80]}...")

                    generated = model.generate(code, max_new_tokens=64)
                    print(f"  Generated: {generated[:80]}...")

                    # Projector variance diagnostic
                    with torch.no_grad():
                        projected = model(code)  # [M+1, D]
                        proj_var = projected.var(dim=0).mean().item()
                        proj_norm = projected.norm(dim=-1).mean().item()
                    print(f"  proj_var={proj_var:.6f}  proj_norm={proj_norm:.4f}")
                    print()

                    model.train()

                # Save checkpoint
                if global_step % save_every == 0:
                    # Lite checkpoint: encoder params only, skip qwen_state + optimizer
                    # (qwen_state ~3.6GB + optimizer ~14GB = disk full in 2 saves)
                    checkpoint = {
                        'step': global_step,
                        'epoch': epoch,
                        'model_type': model_type,
                        'decoder_name': decoder_name,
                        'projector_arch': projector_arch,
                        'encoder_name': encoder_name,
                        'max_branching': max_branching,
                        'patch_size': patch_size,
                        'model_state': {
                            name: param.data
                            for name, param in model.named_parameters()
                            if param.requires_grad and 'decoder' not in name
                        },
                        'scheduler': scheduler.state_dict(),
                        'loss': epoch_loss / max(epoch_samples, 1),
                        'best_loss': best_loss,
                        'loss_history': loss_history,
                        'lora_state': model.decoder.get_lora_state_dict() if lora else {},
                        'qwen_state': {},
                    }
                    torch.save(checkpoint, save_path / f"checkpoint_{global_step}.pt")
                    print(f"  Saved checkpoint_{global_step}.pt")

        # End of epoch
        avg_epoch_loss = epoch_loss / max(epoch_samples, 1)
        print(f"\nEpoch {epoch + 1} complete | avg_loss: {avg_epoch_loss:.4f}")

        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            # Skip optimizer state: ~17GB for 2.2B unfrozen Qwen params
            checkpoint = {
                'step': global_step,
                'epoch': epoch,
                'model_type': model_type,
                'decoder_name': decoder_name,
                'projector_arch': projector_arch,
                'encoder_name': encoder_name,
                'max_branching': max_branching,
                'patch_size': patch_size,
                'model_state': {
                    name: param.data
                    for name, param in model.named_parameters()
                    if param.requires_grad
                },
                'scheduler': scheduler.state_dict(),
                'loss': best_loss,
                'best_loss': best_loss,
                'loss_history': loss_history,
                'lora_state': model.decoder.get_lora_state_dict() if lora else {},
                'qwen_state': model.decoder.get_unfrozen_state_dict() if unfreeze_layers > 0 else {},
            }
            torch.save(checkpoint, save_path / "best_model.pt")
            print(f"  New best model! Loss: {best_loss:.4f}")

    # ==================================================================
    # POST-TRAINING: BLEU evaluation on test set
    # ==================================================================
    print("\n" + "=" * 60)
    print("EVALUATING ON TEST SET (BLEU)...")
    print("=" * 60)

    model.eval()
    test_dataset = MatlabPseudocodeDataset(split="test", model_type=model_type)
    smoother = SmoothingFunction().method1
    bleu_scores = []
    efficiency_metrics = []
    proj_vars = []
    proj_norms = []

    for i, sample in enumerate(test_dataset):
        if i >= eval_samples:
            break

        code = sample['code']
        reference = sample['target']
        features = sample.get('features')

        try:
            generated, eff = model.generate_with_metrics(code, max_new_tokens=128)
        except Exception as e:
            print(f"  [sample {i}] generation failed: {e}")
            continue

        # Projector variance diagnostic per sample
        with torch.no_grad():
            projected = model(code)  # [M+1, D]
            pv = projected.var(dim=0).mean().item()
            pn = projected.norm(dim=-1).mean().item()
        proj_vars.append(pv)
        proj_norms.append(pn)

        ref_tokens = reference.split()
        gen_tokens = generated.split()
        score = sentence_bleu(
            [ref_tokens], gen_tokens, smoothing_function=smoother
        )
        bleu_scores.append(score)
        efficiency_metrics.append(eff)

        if i < 5:
            print(f"  Sample {i}: BLEU={score:.4f}  proj_var={pv:.6f}  proj_norm={pn:.4f}")
            print(f"    ref:  {reference[:80]}...")
            print(f"    gen:  {generated[:80]}...")
            print(f"    encode={eff.get('encode_time_s',0):.3f}s  "
                  f"generate={eff.get('generate_time_s',0):.3f}s  "
                  f"tok/s={eff.get('tokens_per_sec',0):.1f}  "
                  f"kv_cache={eff.get('kv_cache_mb',0):.1f}MB")

    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    avg_proj_var = sum(proj_vars) / len(proj_vars) if proj_vars else 0.0
    avg_proj_norm = sum(proj_norms) / len(proj_norms) if proj_norms else 0.0
    print(f"\nAverage BLEU ({len(bleu_scores)} samples): {avg_bleu:.4f}")
    print(f"Average proj_var: {avg_proj_var:.6f}  proj_norm: {avg_proj_norm:.4f}")

    # Efficiency summary
    if efficiency_metrics:
        def avg_key(key):
            vals = [m[key] for m in efficiency_metrics if key in m]
            return sum(vals) / len(vals) if vals else 0.0

        print(f"\nInference Efficiency ({len(efficiency_metrics)} samples):")
        print(f"  Avg encode time:    {avg_key('encode_time_s'):.4f}s")
        print(f"  Avg generate time:  {avg_key('generate_time_s'):.4f}s")
        print(f"  Avg total time:     {avg_key('total_time_s'):.4f}s")
        print(f"  Avg tokens/sec:     {avg_key('tokens_per_sec'):.1f}")
        print(f"  Avg KV cache:       {avg_key('kv_cache_mb'):.2f} MB")
        print(f"  Avg peak VRAM:      {avg_key('peak_vram_mb'):.1f} MB")
        print(f"  Avg proj_var:       {avg_proj_var:.6f}")
        print(f"  Avg proj_norm:      {avg_proj_norm:.4f}")

    # ==================================================================
    # SAVE GRAPHS
    # ==================================================================
    print("\nSaving graphs...")

    # --- Loss curve ---
    if loss_history:
        steps, losses = zip(*loss_history)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(steps, losses, linewidth=1.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title(f"Training Loss ({model_type})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_path / "loss_curve.png", dpi=150)
        plt.close(fig)
        print(f"  Saved {save_path / 'loss_curve.png'}")

    # --- BLEU histogram ---
    if bleu_scores:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(bleu_scores, bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(avg_bleu, color="red", linestyle="--", linewidth=1.5,
                   label=f"Mean BLEU = {avg_bleu:.4f}")
        ax.set_xlabel("BLEU Score")
        ax.set_ylabel("Count")
        ax.set_title(f"Per-Sample BLEU Distribution ({model_type})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_path / "bleu_scores.png", dpi=150)
        plt.close(fig)
        print(f"  Saved {save_path / 'bleu_scores.png'}")

    # --- Raw metrics JSON ---
    metrics = {
        "loss_history": [{"step": s, "loss": l} for s, l in loss_history],
        "bleu_scores": bleu_scores,
        "avg_bleu": avg_bleu,
        "best_loss": best_loss,
        "total_steps": global_step,
        "epochs": epochs,
        "model_type": model_type,
        "projector_arch": projector_arch,
        "encoder_name": encoder_name,
        "max_branching": max_branching,
        "patch_size": patch_size,
        "rvnn_truncation": (model.recursive_encoder.truncation_stats()
                            if hasattr(model, "recursive_encoder") else None),
        "efficiency": efficiency_metrics,
        "avg_proj_var": avg_proj_var,
        "avg_proj_norm": avg_proj_norm,
        "proj_vars": proj_vars,
        "proj_norms": proj_norms,
    }
    with open(save_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved {save_path / 'metrics.json'}")

    if hasattr(model, "recursive_encoder"):
        ts = model.recursive_encoder.truncation_stats()
        print(f"\nRvNN truncation (max_branching={ts['max_branching']}): "
              f"{ts['truncated_aggregations']}/{ts['total_aggregations']} aggregations "
              f"({ts['truncation_rate']:.1%}) dropped {ts['dropped_children']} children total")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Avg BLEU:  {avg_bleu:.4f}")
    print(f"Checkpoints saved to: {save_path}")
    print("=" * 60)


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Semantic ViT")

    parser.add_argument("--model", type=str, default="combined",
                        choices=["vit", "tree", "combined", "tree_text"],
                        help="Model type: vit, tree, or combined")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split (train/test)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--bottleneck", type=int, default=512)
    parser.add_argument("--projector", type=str, default="linear",
                        choices=["linear", "mlp"],
                        help="Patch projector: 'linear' baseline or scale-stabilized GELU 'mlp'")
    parser.add_argument("--code_encoder", type=str, default="codebert",
                        choices=["codebert", "unixcoder", "codesage"],
                        help="Frozen code encoder for pixel embeddings")
    parser.add_argument("--max_branching", type=int, default=8,
                        help="RvNN max children per node before truncation (tree models)")
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--grad_accum", type=int, default=2)

    # Decoder
    parser.add_argument("--decoder", type=str, default="qwen",
                        choices=["gemma", "qwen"],
                        help="Decoder model: gemma (Gemma-2B) or qwen (Qwen3-4B)")

    # LoRA
    parser.add_argument("--lora", action="store_true", help="Enable LoRA on decoder")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_layers", type=int, default=6, help="Number of last decoder layers to apply LoRA")
    parser.add_argument("--lora_lr", type=float, default=1e-4, help="Separate LR for LoRA params (lower than base)")
    parser.add_argument("--eval_samples", type=int, default=50, help="Number of samples for BLEU eval")

    # Resume
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--stage1_checkpoint", type=str, default=None,
                        help="Path to Stage 1 checkpoint. Loads into decoder before Stage 2 encoder training.")

    # Full fine-tuning (alternative to LoRA)
    parser.add_argument("--unfreeze_layers", type=int, default=0,
                        help="Unfreeze last N decoder layers fully (replaces LoRA)")
    parser.add_argument("--qwen_lr", type=float, default=1e-5,
                        help="LR for unfrozen decoder layers (much lower than encoder lr to avoid forgetting)")

    args = parser.parse_args()

    train(
        model_type=args.model,
        split=args.split,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patch_size=args.patch_size,
        bottleneck_dim=args.bottleneck,
        dropout=args.dropout,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
        save_dir=args.save_dir,
        gradient_accumulation=args.grad_accum,
        lora=args.lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_layers=args.lora_layers,
        lora_lr=args.lora_lr,
        eval_samples=args.eval_samples,
        resume=args.resume,
        stage1_checkpoint=args.stage1_checkpoint,
        unfreeze_layers=args.unfreeze_layers,
        qwen_lr=args.qwen_lr,
        decoder_name=args.decoder,
        projector_arch=args.projector,
        encoder_name=args.code_encoder,
        max_branching=args.max_branching,
    )
