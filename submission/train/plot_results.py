# train/plot_results.py
"""
Regenerate the paper figures from evaluation results.

Reads the eval_*.json / exec_eval_*.json written by evaluate.py and
evaluate_exec.py and emits the three figures the paper references:

    plot_quality_bars.png   Fig. 3  ROUGE-1/2/L, BLEU, chrF (+ exec-match)
    plot_efficiency.png     Fig. 4  input tokens, KV cache, peak VRAM
    combined_KV_Cache2.png  Fig. 5  tokens/s, latency, verbosity, KV cache

Optionally also plots the ablation sweeps from the metrics.json files that
train_pipeline.py writes under an --ablation_dir.

Every figure is written as .png (drafts) and .pdf (vector, for LaTeX), plus a
.csv of the plotted numbers so the values are checkable without reading bars.

Usage:
    python -m train.plot_results --results_dir results --out_dir images
    python -m train.plot_results --results_dir $RUN/results --out_dir images \
        --ablation_dir $RUN/ablations --column double
"""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Model registry: color follows the model, never its rank in the chart, so a
# variant keeps its hue across every figure. Hatches are the secondary encoding
# for grayscale print and CVD readers.
# ---------------------------------------------------------------------------
MODELS = [
    ("tree_text", "Tree+Text", "#2a78d6", ""),
    ("stage1",    "Text-only", "#eb6834", "///"),
    ("combined",  "Combined",  "#1baf7a", "\\\\\\"),
    ("vit",       "ViT-only",  "#eda100", "..."),
    ("tree",      "Tree-only", "#e87ba4", "xxx"),
]

QUALITY_METRICS = [
    ("rouge1", "ROUGE-1"),
    ("rouge2", "ROUGE-2"),
    ("rougeL", "ROUGE-L"),
    ("bleu",   "BLEU"),
    ("chrf",   "chrF"),
]

# (efficiency key, axis label, scale factor applied to the raw value)
FIG4_PANELS = [
    ("num_input_tokens", "Input tokens", 1.0),
    ("kv_cache_mb",      "KV cache (MB)", 1.0),
    ("peak_vram_mb",     "Peak VRAM (GB)", 1 / 1024),
]
FIG5_PANELS = [
    ("tokens_per_sec",       "Tokens / s", 1.0),
    ("total_time_s",         "Total latency (s)", 1.0),
    ("num_generated_tokens", "Generated tokens", 1.0),
    ("kv_cache_mb",          "KV cache (MB)", 1.0),
]

# IEEE-ish: serif text, thin recessive axes, no chartjunk.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "figure.dpi": 150,
})

SINGLE_COL = 3.5   # inches, IEEE single column
DOUBLE_COL = 7.16  # inches, IEEE full width


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_runs(results_dir: Path) -> dict:
    """model_name -> list of per-sample dicts, in MODELS order."""
    runs = {}
    for path in sorted(results_dir.glob("eval_*.json")):
        name = path.stem[len("eval_"):]
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list) and data:
            runs[name] = data
        else:
            print(f"  Skipping {path.name}: empty or not a sample list")
    ordered = {k: runs[k] for k, _, _, _ in MODELS if k in runs}
    for k in runs:  # anything unrecognised still gets plotted, just last
        ordered.setdefault(k, runs[k])
    return ordered


def load_exec(results_dir: Path) -> dict:
    """model_name -> exec summary dict."""
    out = {}
    for path in sorted(results_dir.glob("exec_eval_*.json")):
        with open(path) as f:
            data = json.load(f)
        if "summary" in data:
            out[path.stem[len("exec_eval_"):]] = data["summary"]
    return out


def style(name: str) -> tuple:
    """(label, color, hatch) for a model key, with a neutral fallback."""
    for key, label, color, hatch in MODELS:
        if key == name:
            return label, color, hatch
    return name, "#8a8a8a", "++"


def mean_sem(values: list) -> tuple:
    """(mean, standard error). SEM, not std: these are sample means."""
    vals = [v for v in values if v is not None]
    if not vals:
        return float("nan"), 0.0
    n = len(vals)
    mu = sum(vals) / n
    if n < 2:
        return mu, 0.0
    var = sum((v - mu) ** 2 for v in vals) / (n - 1)
    return mu, math.sqrt(var / n)


def metric_series(samples: list, metric: str) -> list:
    return [s["metrics"][metric] for s in samples if metric in s.get("metrics", {})]


def eff_series(samples: list, key: str, scale: float = 1.0) -> list:
    return [s["efficiency"][key] * scale for s in samples
            if key in s.get("efficiency", {})]


# ---------------------------------------------------------------------------
# Shared drawing helpers
# ---------------------------------------------------------------------------

def label_bars(ax, bars, values, fmt: str):
    """Direct value labels — required relief for the low-contrast hues."""
    span = ax.get_ylim()[1] or 1.0
    for bar, val in zip(bars, values):
        if math.isnan(val):
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + span * 0.02,
                fmt.format(val), ha="center", va="bottom",
                rotation=90, fontsize=5.5, color="#3d3d3a")


def save(fig, out_dir: Path, stem: str, table: list):
    """Write png + pdf + the csv table view of the same numbers."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.4)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    with open(out_dir / f"{stem}.csv", "w", newline="") as f:
        csv.writer(f).writerows(table)
    print(f"  Saved {stem}.png / .pdf / .csv")


# ---------------------------------------------------------------------------
# Fig. 3 — quality metrics
# ---------------------------------------------------------------------------

def plot_quality(runs: dict, execs: dict, out_dir: Path, width: float):
    groups = list(QUALITY_METRICS)
    if execs:
        groups = groups + [("exec_match", "Exec-match")]

    fig, ax = plt.subplots(figsize=(width, 2.6))
    n = len(runs)
    slot = 0.8 / max(n, 1)

    table = [["model"] + [lab for _, lab in groups]]
    for i, (name, samples) in enumerate(runs.items()):
        label, color, hatch = style(name)
        means, errs = [], []
        for key, _ in groups:
            if key == "exec_match":
                means.append(execs.get(name, {}).get("exec_match_rate", float("nan")))
                errs.append(0.0)  # a rate over the run, not a per-sample mean
            else:
                mu, sem = mean_sem(metric_series(samples, key))
                means.append(mu)
                errs.append(sem)
        # 2px-equivalent gap between adjacent bars: bars sit at 92% of the slot
        xs = [g + i * slot - 0.4 + slot / 2 for g in range(len(groups))]
        bars = ax.bar(xs, means, slot * 0.92, yerr=errs, label=label,
                      color=color, hatch=hatch, edgecolor="white", linewidth=0.5,
                      error_kw={"elinewidth": 0.6, "capsize": 1.5, "ecolor": "#52514e"})
        label_bars(ax, bars, means, "{:.3f}")
        table.append([label] + [f"{m:.4f}" for m in means])

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([lab for _, lab in groups])
    ax.set_ylabel("Score")
    ax.set_ylim(0, min(1.0, max(0.1, ax.get_ylim()[1] * 1.18)))
    ax.grid(axis="y", alpha=0.25, color="#c3c2b7")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=min(len(runs), 4), loc="upper center",
              bbox_to_anchor=(0.5, 1.16), columnspacing=1.2, handlelength=1.4)
    save(fig, out_dir, "plot_quality_bars", table)


# ---------------------------------------------------------------------------
# Figs. 4 & 5 — efficiency panels (one measure per axis, never a dual axis)
# ---------------------------------------------------------------------------

def plot_panels(runs: dict, panels: list, out_dir: Path, stem: str,
                width: float, stacked: bool):
    k = len(panels)
    if stacked:  # single column: panels stack vertically
        fig, axes = plt.subplots(k, 1, figsize=(width, 1.5 * k))
    else:
        fig, axes = plt.subplots(1, k, figsize=(width, 2.1))
    axes = axes if hasattr(axes, "__len__") else [axes]

    table = [["model"] + [lab for _, lab, _ in panels]]
    rows = {name: [] for name in runs}

    for ax, (key, axis_label, scale) in zip(axes, panels):
        means, errs, colors, hatches, labels = [], [], [], [], []
        for name, samples in runs.items():
            label, color, hatch = style(name)
            mu, sem = mean_sem(eff_series(samples, key, scale))
            means.append(mu)
            errs.append(sem)
            colors.append(color)
            hatches.append(hatch)
            labels.append(label)
            rows[name].append(mu)

        xs = range(len(means))
        bars = ax.bar(xs, means, 0.68, yerr=errs, color=colors,
                      edgecolor="white", linewidth=0.5,
                      error_kw={"elinewidth": 0.6, "capsize": 1.5, "ecolor": "#52514e"})
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        ax.set_ylim(0, max([m for m in means if not math.isnan(m)] or [1]) * 1.28)
        label_bars(ax, bars, means, "{:.0f}" if max(means or [0]) > 20 else "{:.2f}")
        ax.set_ylabel(axis_label)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25, color="#c3c2b7")
        ax.set_axisbelow(True)

    for name, vals in rows.items():
        table.append([style(name)[0]] + [f"{v:.3f}" for v in vals])
    save(fig, out_dir, stem, table)


# ---------------------------------------------------------------------------
# Ablations — reads the metrics.json train_pipeline writes per run
# ---------------------------------------------------------------------------

def load_ablations(ablation_dir: Path) -> dict:
    """run dir name -> metrics.json contents (save_dir/<model_type>/metrics.json)."""
    found = {}
    for path in sorted(ablation_dir.glob("*/*/metrics.json")):
        with open(path) as f:
            found[path.parent.parent.name] = json.load(f)
    return found


def plot_ablations(ablation_dir: Path, out_dir: Path, width: float,
                   eval_dir: Path = None):
    runs = load_ablations(ablation_dir)
    if not runs:
        print(f"  No */*/metrics.json under {ablation_dir}, skipping ablation figure")
        return

    # Prefer held-out test scores (evaluate.py --output_path eval_<run>.json)
    # over each run's in-training BLEU. All-or-nothing: mixing the two inside
    # one figure would put incomparable numbers on a shared axis.
    evals, ylabel, score_key = {}, "BLEU (in-training eval)", None
    if eval_dir:
        for path in Path(eval_dir).glob("eval_*.json"):
            name = path.stem[len("eval_"):]
            if name in runs:
                with open(path) as f:
                    evals[name] = json.load(f)
        missing = [n for n in runs if n not in evals]
        if missing:
            print(f"  No eval_*.json for {', '.join(missing)} — falling back to "
                  f"in-training BLEU for every panel")
            evals = {}
        else:
            ylabel, score_key = "ROUGE-L (held-out test)", "rougeL"

    def score(name):
        """(mean, sem) for one ablation run."""
        if score_key:
            return mean_sem(metric_series(evals[name], score_key))
        return runs[name].get("avg_bleu", float("nan")), 0.0

    # (panel title, dir prefix, x-tick key, extra runs to include as reference)
    # The projector panel is linear-vs-MLP, so it pulls in the patch_4 run —
    # that IS the linear arm of the comparison.
    specs = [
        ("Patch size (ViT)", "patch_", "patch_size", ()),
        ("Projector (patch 4)", "projector_", "projector_arch", ("patch_4",)),
        ("RvNN branching (trunc. rate)", "branch_", "max_branching", ()),
        ("Code encoder", "enc_", "encoder_name", ()),
    ]
    present = [(t, p, k, extra) for t, p, k, extra in specs
               if any(n.startswith(p) for n in runs)]
    if not present:
        print("  No recognised ablation dirs (patch_*, branch_*, enc_*), skipping")
        return

    fig, axes = plt.subplots(1, len(present), figsize=(width, 2.2))
    axes = axes if hasattr(axes, "__len__") else [axes]
    table = [["panel", "setting", "score", "note"]]

    def sort_key(name, prefix, xkey):
        """Numeric settings sort numerically; 4, 8, 16 — not 16, 4, 8."""
        val = runs[name].get(xkey, name[len(prefix):])
        try:
            return (0, float(val), "")
        except (TypeError, ValueError):
            return (1, 0.0, str(val))

    for ax, (title, prefix, xkey, extra) in zip(axes, present):
        names = sorted(({n for n in runs if n.startswith(prefix)}
                        | {n for n in extra if n in runs}),
                       key=lambda n: sort_key(n, prefix, xkey))
        xs, ys, errs = [], [], []
        for n in names:
            m = runs[n]
            setting = str(m.get(xkey, n[len(prefix):]))
            mu, sem = score(n)
            ys.append(mu)
            errs.append(sem)
            # Truncation rate rides along in the tick label — inside the bar it
            # was dark ink on a saturated fill. Only the branching panel: it is
            # constant across the other sweeps, where it just collides.
            trunc = (m.get("rvnn_truncation") or {}).get("truncation_rate")
            note = "" if (trunc is None or xkey != "max_branching") else f"{trunc:.0%}"
            xs.append(setting if not note else f"{setting}\n{note}")
            table.append([title, setting, f"{mu:.4f}", note])

        bars = ax.bar(range(len(xs)), ys, 0.6, yerr=errs, color="#2a78d6",
                      edgecolor="white", linewidth=0.5,
                      error_kw={"elinewidth": 0.6, "capsize": 1.5, "ecolor": "#52514e"})
        ax.set_ylim(0, max([y for y in ys if not math.isnan(y)] or [1]) * 1.3)
        label_bars(ax, bars, ys, "{:.3f}")
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25, color="#c3c2b7")
        ax.set_axisbelow(True)

    save(fig, out_dir, "plot_ablations", table)


def main():
    parser = argparse.ArgumentParser(description="Regenerate paper figures from eval results")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory holding eval_*.json / exec_eval_*.json")
    parser.add_argument("--out_dir", type=str, default="images")
    parser.add_argument("--ablation_dir", type=str, default=None,
                        help="Directory of ablation runs (adds plot_ablations)")
    parser.add_argument("--ablation_results_dir", type=str, default=None,
                        help="Dir of eval_<run>.json for the ablation checkpoints; "
                             "use held-out test scores instead of in-training BLEU")
    parser.add_argument("--column", type=str, default="single",
                        choices=["single", "double"],
                        help="Figure width: IEEE single (3.5in) or full width (7.16in)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    width = SINGLE_COL if args.column == "single" else DOUBLE_COL

    runs = load_runs(results_dir)
    if not runs:
        raise SystemExit(f"No eval_*.json files found in {results_dir}")
    execs = load_exec(results_dir)

    print(f"Models: {', '.join(f'{style(n)[0]} (n={len(s)})' for n, s in runs.items())}")
    print(f"Exec-match available for: {', '.join(execs) or 'none'}")

    plot_quality(runs, execs, out_dir, width)
    plot_panels(runs, FIG4_PANELS, out_dir, "plot_efficiency", width,
                stacked=args.column == "single")
    plot_panels(runs, FIG5_PANELS, out_dir, "combined_KV_Cache2", width,
                stacked=args.column == "single")
    if args.ablation_dir:
        plot_ablations(Path(args.ablation_dir), out_dir, width,
                       eval_dir=Path(args.ablation_results_dir)
                       if args.ablation_results_dir else None)

    print(f"\nFigures written to {out_dir}")


if __name__ == "__main__":
    main()
