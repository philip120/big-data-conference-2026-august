# train/evaluate_exec.py
"""
Execution-match evaluation: does the model's pseudocode actually work?

For each sample in an evaluate.py results JSON:
    1. Take the MODEL-generated pseudocode.
    2. Ask Gemini (fixed model, low temperature) to regenerate MATLAB code
       from that pseudocode alone.
    3. Run the regenerated code in Octave (scripts directly; functions via the
       test harness stored in the dataset).
    4. Compare the output against the dataset's stored `orig_output`.

The final metric, exec-match rate, measures functional correctness of the
pseudocode — if the pseudocode is faithful enough to reconstruct a program
with identical behavior, the model demonstrably captured the code's meaning.
This complements ROUGE/BLEU/chrF, which only measure surface overlap with
LLM-authored references.

The regeneration prompt, Octave runner, and output normalization mirror
dataset/dataset_creation_v3.py exactly, so this metric is consistent with how
the dataset itself was validated.

Deliberately decoupled from the GPU stage: evaluate.py stores the generated
pseudocode in its results JSON, so this script needs no GPU and no model —
run it later, on any machine with Octave + a GEMINI_API_KEY.

Usage:
    python -m train.evaluate_exec --results results/eval_tree_text.json
    python -m train.evaluate_exec --results results/eval_*.json   (shell glob)

Requires: octave on PATH, GEMINI_API_KEY in env or ../.env, pip install google-genai
"""
import argparse
import json
import os
import random
import re
import subprocess
import tempfile
import time
from pathlib import Path

from datasets import load_dataset

HF_DATASET = "philip120/sc-matlab-validated"
GEMINI_MODEL = "gemini-flash-lite-latest"
OCTAVE_TIMEOUT = 15
MAX_OUTPUT_TOKENS = 2048

# Same prompt as dataset_creation_v3.py — keeps the metric consistent with
# how the dataset labels were validated.
CODE_GEN_PROMPT = """\
You are given pseudocode describing a MATLAB algorithm.
Write MATLAB/Octave code that implements it exactly.

Rules:
- Output only MATLAB code, no explanations or markdown fences.
- If the pseudocode describes a function, write a function definition preserving the original name and signature.
- If it describes a script, write a flat script.
- Display computed results using disp() or fprintf(), matching the verbosity implied by the pseudocode.
- Do not use toolboxes beyond base MATLAB/Octave.
- Do not read from files or external inputs.

Pseudocode:
<<<PSEUDOCODE>>>
"""


# ----------------------------------------------------------------------
# Helpers mirrored from dataset/dataset_creation_v3.py
# ----------------------------------------------------------------------

def parse_function_info(code: str):
    for line in code.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        m = re.match(
            r"^function\s+(?:\[([^\]]*)\]\s*=\s*|(\w+)\s*=\s*)?(\w+)\s*\(([^)]*)\)",
            stripped, re.IGNORECASE,
        )
        if m:
            return m.group(3)
        return None
    return None


def extract_matlab_code(text: str) -> str:
    text = re.sub(r"```(?:matlab|octave|m)?\n?(.*?)```", r"\1", text, flags=re.DOTALL)
    return text.strip()


def normalize_output(text: str) -> str:
    if not text:
        return ""
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"\b(\d+\.\d{5,})\b", lambda m: f"{float(m.group()):.4f}", line)
        lines.append(line)
    return "\n".join(lines)


def outputs_match(a: str, b: str) -> bool:
    if not a or not b:
        return False
    return normalize_output(a) == normalize_output(b)


def run_octave_script(code: str) -> tuple[str, bool]:
    with tempfile.NamedTemporaryFile(suffix=".m", mode="w", delete=False) as f:
        f.write(code)
        fname = f.name
    try:
        result = subprocess.run(
            ["octave", "--no-gui", "--quiet", fname],
            capture_output=True, text=True, timeout=OCTAVE_TIMEOUT,
        )
        return result.stdout.strip(), result.returncode == 0
    except subprocess.TimeoutExpired:
        return "", False
    finally:
        try:
            os.unlink(fname)
        except OSError:
            pass


def run_with_harness(func_code: str, func_name: str, harness_body: str) -> tuple[str, bool]:
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / f"{func_name}.m").write_text(func_code)
        wrapper = f"addpath('{tmpdir}');\n{harness_body}"
        wrapper_file = Path(tmpdir) / "run_it.m"
        wrapper_file.write_text(wrapper)
        try:
            result = subprocess.run(
                ["octave", "--no-gui", "--quiet", str(wrapper_file)],
                capture_output=True, text=True, timeout=OCTAVE_TIMEOUT,
            )
            return result.stdout.strip(), result.returncode == 0
        except subprocess.TimeoutExpired:
            return "", False


# ----------------------------------------------------------------------
# Gemini
# ----------------------------------------------------------------------

def make_gemini_client():
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists() and not os.environ.get("GEMINI_API_KEY"):
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'").strip('"'))
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not set (env or repo-root .env)")
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
    cfg = types.GenerateContentConfig(temperature=0.2, max_output_tokens=MAX_OUTPUT_TOKENS)
    return client, cfg


def call_gemini(client, cfg, prompt: str, retries: int = 5) -> str:
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL, contents=prompt, config=cfg)
            return (response.text or "").strip()
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower() or "quota" in str(e).lower():
                time.sleep(2 ** attempt + random.uniform(0, 1))
            else:
                print(f"  Gemini error: {e}")
                return ""
    return ""


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def load_test_lookup(split: str) -> dict:
    """Map code string -> {kind, harness, orig_output} from the HF test split."""
    ds = load_dataset(HF_DATASET, split=split)
    return {
        row["code"]: {
            "kind": row.get("kind", ""),
            "harness": row.get("harness", ""),
            "orig_output": row.get("orig_output", ""),
        }
        for row in ds
    }


def evaluate_file(results_path: Path, lookup: dict, client, cfg, out_dir: Path) -> dict:
    with open(results_path) as f:
        samples = json.load(f)

    stats = {"total": 0, "no_dataset_match": 0, "gen_fail": 0,
             "run_fail": 0, "mismatch": 0, "match": 0}
    detailed = []

    print(f"\n=== {results_path.name}: {len(samples)} samples ===")
    for i, sample in enumerate(samples):
        code = sample.get("code", "")
        pseudocode = sample.get("generated_description", "")
        if not code or not pseudocode:
            continue
        stats["total"] += 1

        ref = lookup.get(code)
        if ref is None:
            stats["no_dataset_match"] += 1
            continue

        regen_raw = call_gemini(client, cfg,
                                CODE_GEN_PROMPT.replace("<<<PSEUDOCODE>>>", pseudocode))
        if not regen_raw:
            stats["gen_fail"] += 1
            continue
        regen_code = extract_matlab_code(regen_raw)

        func_name = parse_function_info(regen_code)
        if ref["kind"] == "function" and ref["harness"] and func_name:
            orig_name = parse_function_info(code) or func_name
            harness = (ref["harness"].replace(orig_name, func_name)
                       if func_name != orig_name else ref["harness"])
            regen_out, ok = run_with_harness(regen_code, func_name, harness)
        else:
            regen_out, ok = run_octave_script(regen_code)

        if not ok or not regen_out:
            stats["run_fail"] += 1
            verdict = "run_fail"
        elif outputs_match(ref["orig_output"], regen_out):
            stats["match"] += 1
            verdict = "match"
        else:
            stats["mismatch"] += 1
            verdict = "mismatch"

        detailed.append({
            "code": code,
            "pseudocode": pseudocode,
            "regen_code": regen_code,
            "regen_output": regen_out,
            "orig_output": ref["orig_output"],
            "verdict": verdict,
        })
        print(f"  [{i}] {verdict}")
        time.sleep(0.3)

    evaluated = stats["match"] + stats["mismatch"] + stats["run_fail"]
    summary = {
        "source": results_path.name,
        "stats": stats,
        "exec_match_rate": stats["match"] / evaluated if evaluated else 0.0,
        "run_success_rate": (stats["match"] + stats["mismatch"]) / evaluated if evaluated else 0.0,
    }

    out_path = out_dir / f"exec_{results_path.stem}.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "samples": detailed}, f, indent=2)

    print(f"  exec_match_rate  : {summary['exec_match_rate']:.3f} "
          f"({stats['match']}/{evaluated})")
    print(f"  run_success_rate : {summary['run_success_rate']:.3f}")
    print(f"  saved -> {out_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Execution-match evaluation of generated pseudocode")
    parser.add_argument("--results", nargs="+", required=True,
                        help="One or more evaluate.py results JSON files")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output dir (default: alongside each input file)")
    args = parser.parse_args()

    try:
        subprocess.run(["octave", "--version"], capture_output=True, timeout=10)
    except FileNotFoundError:
        raise SystemExit("Octave not found. Colab/Linux: apt-get install -y octave; macOS: brew install octave")

    client, cfg = make_gemini_client()
    lookup = load_test_lookup(args.split)
    print(f"Loaded {len(lookup)} reference samples from {HF_DATASET} ({args.split})")

    summaries = []
    for path_str in args.results:
        path = Path(path_str)
        out_dir = Path(args.out_dir) if args.out_dir else path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        summaries.append(evaluate_file(path, lookup, client, cfg, out_dir))

    print("\n===== EXEC-MATCH SUMMARY =====")
    for s in summaries:
        print(f"  {s['source']:<32} match={s['exec_match_rate']:.3f} "
              f"run_ok={s['run_success_rate']:.3f}")


if __name__ == "__main__":
    main()
