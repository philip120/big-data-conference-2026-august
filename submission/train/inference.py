# train/inference.py
"""
Inference — load a Stage 2 checkpoint and generate pseudocode.

Quickest use: edit CODE below, then run:
    python -m train.inference --checkpoint checkpoints_stage2/combined/best_model.pt

Other modes:
    python -m train.inference --checkpoint ... --code "function y = f(x) ..."
    python -m train.inference --checkpoint ... --file path/to/file.m
    python -m train.inference --checkpoint ... --eval --num_samples 20
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from pathlib import Path

from train.load_dataset import load_matlab_nl_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────
# Edit this string to test your own MATLAB code
# ─────────────────────────────────────────────────────────
CODE = """
function y = relu(x)
    if x > 0
        y = x;
    else
        y = 0;
    end
end
"""
# ─────────────────────────────────────────────────────────


def load_model(checkpoint_path: str, lora_rank: int, lora_alpha: int,
               lora_dropout: float, lora_layers: int,
               patch_size: int, bottleneck_dim: int, dropout: float,
               decoder_name: str = None):
    """Load model + weights from a Stage 2 checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    model_type = ckpt.get("model_type", "combined")
    # Auto-detect decoder from checkpoint, with CLI override
    decoder_name = decoder_name or ckpt.get("decoder_name", "qwen")
    projector_arch = ckpt.get("projector_arch", "linear")
    encoder_name = ckpt.get("encoder_name", "codebert")
    max_branching = ckpt.get("max_branching", 8)
    patch_size = ckpt.get("patch_size", patch_size)
    print(f"Model type: {model_type}, Decoder: {decoder_name}, Projector: {projector_arch}, "
          f"Encoder: {encoder_name}")

    # Build model
    if model_type == "vit":
        from model.model import SemanticViT
        model = SemanticViT(patch_size=patch_size, bottleneck_dim=bottleneck_dim,
                            dropout=dropout, decoder_name=decoder_name,
                            projector_arch=projector_arch, encoder_name=encoder_name)
    elif model_type == "tree":
        from model2.model import StructuralModel
        model = StructuralModel(dropout=dropout, decoder_name=decoder_name,
                                encoder_name=encoder_name, max_branching=max_branching)
    elif model_type == "tree_text":
        from tree_text_model.model import TreeTextModel
        model = TreeTextModel(dropout=dropout, decoder_name=decoder_name,
                              encoder_name=encoder_name, max_branching=max_branching)
    else:
        from combined_model.model import CombinedSemanticViT
        model = CombinedSemanticViT(patch_size=patch_size, bottleneck_dim=bottleneck_dim,
                                    dropout=dropout, decoder_name=decoder_name,
                                    projector_arch=projector_arch,
                                    encoder_name=encoder_name, max_branching=max_branching)

    # Restore trainable encoder weights
    if "model_state" in ckpt:
        model_params = dict(model.named_parameters())
        loaded = 0
        for name, data in ckpt["model_state"].items():
            if name in model_params:
                model_params[name].data.copy_(data)
                loaded += 1
        print(f"Restored {loaded} encoder parameter tensors")

    # Restore LoRA or unfrozen Qwen weights
    if "qwen_state" in ckpt and ckpt["qwen_state"]:
        # Infer how many layers were unfrozen from the state dict keys
        unfrozen_idxs = set()
        for k in ckpt["qwen_state"]:
            # keys look like "model.layers.24.self_attn.q_proj.weight"
            parts = k.split(".")
            if len(parts) > 2 and parts[0] == "model" and parts[1] == "layers":
                unfrozen_idxs.add(int(parts[2]))
        num_unfrozen = len(unfrozen_idxs)
        model.decoder.unfreeze_layers(num_unfrozen)
        model.decoder.load_unfrozen_state_dict(ckpt["qwen_state"])
        print(f"Restored {len(ckpt['qwen_state'])} unfrozen Qwen tensors ({num_unfrozen} layers)")
    elif "lora_state" in ckpt and ckpt["lora_state"]:
        model.enable_lora(
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            num_layers=lora_layers,
        )
        model.decoder.load_lora_state_dict(ckpt["lora_state"])
        print(f"Restored {len(ckpt['lora_state'])} LoRA tensors")

    model.eval()

    step = ckpt.get("step", "?")
    best_loss = ckpt.get("best_loss", ckpt.get("loss", 0))
    print(f"Checkpoint: step={step}, best_loss={best_loss:.4f}")
    print("=" * 60)

    return model


def run(model, code: str, max_tokens: int = 128):
    """Generate pseudocode from a MATLAB code string."""
    code = code.strip()
    if not code:
        print("Empty input.")
        return

    print("INPUT CODE:")
    print("─" * 60)
    print(code)
    print("─" * 60)
    print("\nGenerating...")

    output = model.generate(code, max_new_tokens=max_tokens)

    print("\nGENERATED PSEUDOCODE:")
    print("─" * 60)
    print(output)
    print("─" * 60)
    return output


def eval_on_dataset(model, split: str, num_samples: int, max_tokens: int):
    """Run on dataset samples and print side-by-side comparisons."""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smoother = SmoothingFunction().method1

    print(f"\nEvaluating on {num_samples} samples (split='{split}')...")
    data = load_matlab_nl_dataset(split)
    bleu_scores = []

    for i, sample in enumerate(data[:num_samples]):
        code = sample["code"]
        target = sample["nl"]

        print(f"\n{'='*60}")
        print(f"Sample {i+1}/{num_samples}")
        print(f"{'='*60}")
        print(f"CODE:\n{code[:300]}{'...' if len(code) > 300 else ''}")
        print(f"\nTARGET:\n{target}")

        generated = model.generate(code, max_new_tokens=max_tokens)
        print(f"\nGENERATED:\n{generated}")

        score = sentence_bleu([target.split()], generated.split(), smoothing_function=smoother)
        bleu_scores.append(score)
        print(f"\nBLEU: {score:.4f}")

    if bleu_scores:
        avg = sum(bleu_scores) / len(bleu_scores)
        print(f"\n{'='*60}")
        print(f"Average BLEU over {len(bleu_scores)} samples: {avg:.4f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference from Stage 2 checkpoint")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to Stage 2 best_model.pt")
    parser.add_argument("--code", type=str, default=None,
                        help="Inline MATLAB code string")
    parser.add_argument("--file", type=str, default=None,
                        help="Path to a .m file")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate on dataset samples")
    parser.add_argument("--split", type=str, default="train[80%:]",
                        help="Dataset split for --eval")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=128)

    # Must match the values used during Stage 2 training
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_layers", type=int, default=12)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--bottleneck", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--decoder", type=str, default=None,
                        choices=["gemma", "qwen"],
                        help="Decoder model (auto-detected from checkpoint if not specified)")

    args = parser.parse_args()

    model = load_model(
        checkpoint_path=args.checkpoint,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_layers=args.lora_layers,
        patch_size=args.patch_size,
        bottleneck_dim=args.bottleneck,
        dropout=args.dropout,
        decoder_name=args.decoder,
    )

    if args.eval:
        eval_on_dataset(model, split=args.split,
                        num_samples=args.num_samples, max_tokens=args.max_tokens)
    elif args.file:
        code = Path(args.file).read_text()
        run(model, code, max_tokens=args.max_tokens)
    elif args.code:
        run(model, args.code, max_tokens=args.max_tokens)
    else:
        # Default: use the CODE string at the top of this file
        run(model, CODE, max_tokens=args.max_tokens)
