# train/evaluate.py
"""
Unified evaluation script for all model variants.

Outputs JSON in the same format as codepatch-paligemma's evaluate_codepatch.py
so results are directly comparable.

Usage:
    # Stage 2 models (vit, tree, combined, tree_text)
    python -m train.evaluate --model_type tree_text \
        --checkpoint checkpoints_stage2/tree_text/best_model.pt \
        --num_samples 20

    # Stage 1 baseline (pure Qwen fine-tuned, no encoder)
    python -m train.evaluate --model_type stage1 \
        --checkpoint checkpoints_stage1/best_model.pt \
        --num_samples 20
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import torch
import argparse
from pathlib import Path

from datasets import load_dataset
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.chrf_score import sentence_chrf

# Import decoder factory directly to avoid shared/__init__.py pulling in ANTLR
import importlib
_decoder_factory = importlib.import_module("shared.decoder_factory")
create_decoder = _decoder_factory.create_decoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_stage2_model(checkpoint_path, lora_rank, lora_alpha, lora_dropout,
                      lora_layers, patch_size, bottleneck_dim, dropout,
                      decoder_name=None):
    """Load a Stage 2 model (vit/tree/combined/tree_text)."""
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model_type = ckpt.get("model_type", "combined")
    decoder_name = decoder_name or ckpt.get("decoder_name", "qwen")
    projector_arch = ckpt.get("projector_arch", "linear")
    encoder_name = ckpt.get("encoder_name", "codebert")
    max_branching = ckpt.get("max_branching", 8)
    patch_size = ckpt.get("patch_size", patch_size)

    # Infer patch_size and bottleneck_dim from saved projector weights if present
    if "model_state" in ckpt:
        print(f"Checkpoint model_state keys: {list(ckpt['model_state'].keys())}")
        for name, data in ckpt["model_state"].items():
            if "projector" in name and "weight" in name and data.dim() == 2:
                # First projector weight: input is patch_size * 768
                in_dim = data.shape[1]
                if in_dim % 768 == 0:
                    patch_size = in_dim // 768
                    bottleneck_dim = data.shape[0]
                    print(f"Inferred from '{name}': patch_size={patch_size}, bottleneck_dim={bottleneck_dim}")
                break

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

    # Restore encoder weights
    if "model_state" in ckpt:
        model_params = dict(model.named_parameters())
        loaded = 0
        skipped = []
        for name, data in ckpt["model_state"].items():
            if name in model_params:
                if model_params[name].shape != data.shape:
                    skipped.append(f"{name}: model={model_params[name].shape} ckpt={data.shape}")
                    continue
                model_params[name].data.copy_(data)
                loaded += 1
        print(f"Restored {loaded} encoder parameter tensors")
        if skipped:
            print(f"WARNING: Skipped {len(skipped)} shape-mismatched tensors:")
            for s in skipped:
                print(f"  {s}")

    # Restore LoRA or unfrozen decoder weights
    if "qwen_state" in ckpt and ckpt["qwen_state"]:
        unfrozen_idxs = set()
        for k in ckpt["qwen_state"]:
            parts = k.split(".")
            if len(parts) > 2 and parts[0] == "model" and parts[1] == "layers":
                unfrozen_idxs.add(int(parts[2]))
        model.decoder.unfreeze_layers(len(unfrozen_idxs))
        model.decoder.load_unfrozen_state_dict(ckpt["qwen_state"])
        print(f"Restored {len(ckpt['qwen_state'])} unfrozen decoder tensors")
    elif "lora_state" in ckpt and ckpt["lora_state"]:
        model.enable_lora(rank=lora_rank, alpha=lora_alpha,
                          dropout=lora_dropout, num_layers=lora_layers)
        model.decoder.load_lora_state_dict(ckpt["lora_state"])
        print(f"Restored {len(ckpt['lora_state'])} LoRA tensors")

    model.eval()
    return model, model_type


def load_stage1_decoder(checkpoint_path, lora_rank, lora_alpha, lora_dropout,
                        lora_layers, decoder_name=None, unfreeze_layers=0):
    """Load a Stage 1 fine-tuned decoder (no encoder)."""
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    decoder_name = decoder_name or ckpt.get("decoder_name", "qwen")

    decoder = create_decoder(decoder_name, device=DEVICE)

    if "qwen_state" in ckpt and ckpt["qwen_state"]:
        unfrozen_idxs = set()
        for k in ckpt["qwen_state"]:
            parts = k.split(".")
            if len(parts) > 2 and parts[0] == "model" and parts[1] == "layers":
                unfrozen_idxs.add(int(parts[2]))
        decoder.unfreeze_layers(len(unfrozen_idxs))
        decoder.load_unfrozen_state_dict(ckpt["qwen_state"])
        print(f"Restored {len(ckpt['qwen_state'])} unfrozen decoder tensors")
    elif "lora_state" in ckpt and ckpt["lora_state"]:
        decoder.enable_lora(rank=lora_rank, alpha=lora_alpha,
                            dropout=lora_dropout, num_layers=lora_layers)
        decoder.load_lora_state_dict(ckpt["lora_state"])
        print(f"Restored {len(ckpt['lora_state'])} LoRA tensors")

    decoder.eval_mode()
    return decoder, decoder_name


@torch.no_grad()
def generate_stage1(decoder, code: str, max_new_tokens: int = 128):
    """Generate pseudocode using Stage 1 decoder (text-only, no encoder).
    Returns (text, efficiency_metrics)."""
    prompt = f"Convert the following MATLAB code to step-by-step pseudocode:\n{code}\nPseudocode:"
    tokens = decoder.tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512
    ).to(decoder.device)

    num_input_tokens = tokens.input_ids.shape[1]

    if decoder.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    # Greedy decoding: evaluation must be deterministic
    output_ids = decoder.model.generate(
        input_ids=tokens.input_ids,
        attention_mask=tokens.attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    if decoder.device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    # Decode only the new tokens
    new_tokens = output_ids[0, num_input_tokens:]
    text = decoder.tokenizer.decode(new_tokens, skip_special_tokens=True)
    num_generated = len(new_tokens)

    # KV cache size: 2 (K+V) × layers × kv_heads × head_dim × seq_len × 2 bytes
    config = decoder.model.config
    num_layers = config.num_hidden_layers
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads
    total_seq_len = num_input_tokens + num_generated
    kv_cache_bytes = 2 * num_layers * num_kv_heads * head_dim * total_seq_len * 2

    peak_vram_mb = 0.0
    if decoder.device == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)

    gen_time = t1 - t0

    return text, {
        "encode_time_s": 0.0,  # no encoder
        "generate_time_s": round(gen_time, 4),
        "total_time_s": round(gen_time, 4),
        "num_input_tokens": num_input_tokens,
        "num_generated_tokens": num_generated,
        "tokens_per_sec": round(num_generated / gen_time, 1) if gen_time > 0 else 0,
        "kv_cache_mb": round(kv_cache_bytes / (1024**2), 2),
        "peak_vram_mb": round(peak_vram_mb, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Unified evaluation for all model variants")
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["vit", "tree", "combined", "tree_text", "stage1"],
                        help="Model variant to evaluate")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--split", type=str, default="test",
                        help="HF dataset split to evaluate on (default: held-out test)")
    parser.add_argument("--max_code_chars", type=int, default=None,
                        help="Skip code samples longer than this (in characters)")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory for results JSON (point at Drive in Colab)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Explicit results JSON path (overrides --results_dir)")

    # Model params (must match training)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_layers", type=int, default=12)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--bottleneck", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--decoder", type=str, default=None,
                        choices=["gemma", "qwen"])

    args = parser.parse_args()
    if args.output_path:
        output_path = args.output_path
    else:
        from pathlib import Path
        Path(args.results_dir).mkdir(parents=True, exist_ok=True)
        output_path = str(Path(args.results_dir) / f"eval_{args.model_type}.json")

    print(f"Evaluating: {args.model_type}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {DEVICE}")

    # Load model
    if args.model_type == "stage1":
        decoder, decoder_name = load_stage1_decoder(
            args.checkpoint, args.lora_rank, args.lora_alpha,
            args.lora_dropout, args.lora_layers, args.decoder)
        model = None
    else:
        model, detected_type = load_stage2_model(
            args.checkpoint, args.lora_rank, args.lora_alpha,
            args.lora_dropout, args.lora_layers, args.patch_size,
            args.bottleneck, args.dropout, args.decoder)
        decoder = None
        print(f"Detected model type from checkpoint: {detected_type}")

    # Load eval data from the held-out test split (never seen in training)
    print(f"\nLoading eval samples from HuggingFace (split={args.split})...")
    from train.load_dataset import load_matlab_nl_dataset
    hf_data = load_matlab_nl_dataset(args.split)
    eval_samples = []
    for item in hf_data:
        code = item.get("code", "")
        nl = item.get("nl", "")
        if not code or not nl:
            continue
        if args.max_code_chars and len(code) > args.max_code_chars:
            continue
        eval_samples.append(item)
        if len(eval_samples) >= args.num_samples:
            break
    print(f"Selected {len(eval_samples)} samples (max_code_chars={args.max_code_chars})")

    # Eval
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smoother = SmoothingFunction().method1
    results = []

    print(f"\nEvaluating {len(eval_samples)} samples...")
    for i, item in enumerate(eval_samples):
        code = item["code"]
        reference = item["nl"]

        if not code or not reference:
            continue

        try:
            if args.model_type == "stage1":
                generated, eff_metrics = generate_stage1(decoder, code, args.max_tokens)
            else:
                generated, eff_metrics = model.generate_with_metrics(code, max_new_tokens=args.max_tokens)
        except Exception as e:
            print(f"  [{i}] generation failed: {e}")
            continue

        # ROUGE
        scores = scorer.score(reference, generated)

        # BLEU
        bleu = sentence_bleu(
            [reference.split()], generated.split(),
            smoothing_function=smoother
        )

        # chrF (character n-gram F-score; more robust than BLEU for prose)
        try:
            chrf = sentence_chrf(reference, generated)
        except (ValueError, ZeroDivisionError):
            chrf = 0.0

        results.append({
            "code": code,
            "reference_description": reference,
            "generated_description": generated,
            "metrics": {
                "rouge1": scores['rouge1'].fmeasure,
                "rouge2": scores['rouge2'].fmeasure,
                "rougeL": scores['rougeL'].fmeasure,
                "bleu": bleu,
                "chrf": chrf,
            },
            "efficiency": eff_metrics,
        })

        print(f"  [{i}/{len(eval_samples)}] ROUGE-L={scores['rougeL'].fmeasure:.4f}  BLEU={bleu:.4f}  "
              f"time={eff_metrics.get('total_time_s', 0):.1f}s  "
              f"gen_tok={eff_metrics.get('num_generated_tokens', 0)}")

        # Save incrementally every 10 samples
        if len(results) % 10 == 0:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=4)

    # Summary
    if results:
        n = len(results)
        avg_r1 = sum(r["metrics"]["rouge1"] for r in results) / n
        avg_r2 = sum(r["metrics"]["rouge2"] for r in results) / n
        avg_rl = sum(r["metrics"]["rougeL"] for r in results) / n
        avg_bleu = sum(r["metrics"]["bleu"] for r in results) / n
        avg_chrf = sum(r["metrics"]["chrf"] for r in results) / n

        # Efficiency averages
        def eff_avg(key):
            vals = [r["efficiency"].get(key, 0) for r in results if "efficiency" in r]
            return sum(vals) / len(vals) if vals else 0

        print(f"\n{'='*60}")
        print(f"RESULTS: {args.model_type} ({n} samples)")
        print(f"{'='*60}")
        print(f"  ROUGE-1:           {avg_r1:.4f}")
        print(f"  ROUGE-2:           {avg_r2:.4f}")
        print(f"  ROUGE-L:           {avg_rl:.4f}")
        print(f"  BLEU:              {avg_bleu:.4f}")
        print(f"  chrF:              {avg_chrf:.4f}")
        print(f"  --- Efficiency ---")
        print(f"  Encode time:       {eff_avg('encode_time_s'):.3f}s")
        print(f"  Generate time:     {eff_avg('generate_time_s'):.3f}s")
        print(f"  Total time:        {eff_avg('total_time_s'):.3f}s")
        print(f"  Input tokens:      {eff_avg('num_input_tokens'):.0f}")
        print(f"  Generated tokens:  {eff_avg('num_generated_tokens'):.0f}")
        print(f"  Tokens/sec:        {eff_avg('tokens_per_sec'):.1f}")
        print(f"  KV cache:          {eff_avg('kv_cache_mb'):.2f} MB")
        print(f"  Peak VRAM:         {eff_avg('peak_vram_mb'):.0f} MB")

    # Save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
