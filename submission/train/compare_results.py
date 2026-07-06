# train/compare_results.py
"""
Aggregate evaluation results into one comparison table.

Reads all eval_*.json (from evaluate.py) and exec_eval_*.json (from
evaluate_exec.py) in a results directory and prints a markdown table of
ROUGE / BLEU / chrF / exec-match / efficiency per model, plus paired
bootstrap significance tests against a chosen baseline model.

Usage:
    python -m train.compare_results --results_dir results
    python -m train.compare_results --results_dir results --baseline stage1
"""
import argparse
import json
import random
from pathlib import Path

METRICS = ["rouge1", "rouge2", "rougeL", "bleu", "chrf"]


def load_eval_files(results_dir: Path) -> dict:
    """model_name -> list of per-sample result dicts."""
    runs = {}
    for path in sorted(results_dir.glob("eval_*.json")):
        name = path.stem[len("eval_"):]
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list) and data:
            runs[name] = data
    return runs


def load_exec_summaries(results_dir: Path) -> dict:
    """model_name -> exec summary dict."""
    out = {}
    for path in sorted(results_dir.glob("exec_eval_*.json")):
        name = path.stem[len("exec_eval_"):]
        with open(path) as f:
            data = json.load(f)
        if "summary" in data:
            out[name] = data["summary"]
    return out


def mean_metric(samples: list, metric: str) -> float:
    vals = [s["metrics"][metric] for s in samples if metric in s.get("metrics", {})]
    return sum(vals) / len(vals) if vals else float("nan")


def mean_eff(samples: list, key: str) -> float:
    vals = [s["efficiency"].get(key, 0) for s in samples if s.get("efficiency")]
    return sum(vals) / len(vals) if vals else float("nan")


def paired_bootstrap(a: list, b: list, n_boot: int = 10000, seed: int = 0) -> float:
    """P(model A <= model B) under paired bootstrap of per-sample differences.

    Small values (< 0.05) mean A is significantly better than B.
    """
    diffs = [x - y for x, y in zip(a, b)]
    if not diffs:
        return float("nan")
    rng = random.Random(seed)
    n = len(diffs)
    worse = 0
    for _ in range(n_boot):
        s = sum(rng.choice(diffs) for _ in range(n))
        if s <= 0:
            worse += 1
    return worse / n_boot


def main():
    parser = argparse.ArgumentParser(description="Compare evaluation results across models")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--baseline", type=str, default=None,
                        help="Model name for significance comparison (e.g. stage1)")
    parser.add_argument("--n_boot", type=int, default=10000)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    runs = load_eval_files(results_dir)
    execs = load_exec_summaries(results_dir)

    if not runs:
        raise SystemExit(f"No eval_*.json files found in {results_dir}")

    # --- Main table ---
    header = (["model", "n"] + METRICS
              + ["exec_match", "run_ok", "gen_tok", "total_s"])
    rows = []
    for name, samples in runs.items():
        ex = execs.get(name, {})
        rows.append([
            name,
            str(len(samples)),
            *(f"{mean_metric(samples, m):.4f}" for m in METRICS),
            f"{ex.get('exec_match_rate', float('nan')):.3f}" if ex else "-",
            f"{ex.get('run_success_rate', float('nan')):.3f}" if ex else "-",
            f"{mean_eff(samples, 'num_generated_tokens'):.0f}",
            f"{mean_eff(samples, 'total_time_s'):.2f}",
        ])

    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]
    print("\n| " + " | ".join(h.ljust(w) for h, w in zip(header, widths)) + " |")
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for r in rows:
        print("| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |")

    # --- Significance vs baseline (paired on common code samples) ---
    if args.baseline:
        if args.baseline not in runs:
            raise SystemExit(f"Baseline '{args.baseline}' not among {list(runs)}")
        base = {s["code"]: s for s in runs[args.baseline]}
        print(f"\nPaired bootstrap vs baseline '{args.baseline}' "
              f"(p = P(model <= baseline); p < 0.05 => significantly better):")
        for name, samples in runs.items():
            if name == args.baseline:
                continue
            paired = [(s, base[s["code"]]) for s in samples if s["code"] in base]
            if len(paired) < 5:
                print(f"  {name:<12} not enough overlapping samples ({len(paired)})")
                continue
            line = f"  {name:<12} n={len(paired):<4}"
            for m in ["rougeL", "bleu", "chrf"]:
                a = [p[0]["metrics"][m] for p in paired]
                b = [p[1]["metrics"][m] for p in paired]
                p = paired_bootstrap(a, b, n_boot=args.n_boot)
                line += f"  {m}: p={p:.4f}"
            print(line)


if __name__ == "__main__":
    main()
