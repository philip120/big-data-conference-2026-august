import os
import re
import shutil
import time
import random
import subprocess
import tempfile
from pathlib import Path
from google import genai
from google.genai import types
from datasets import load_dataset

# ======================
# CONFIG
# ======================

MODEL = "gemini-flash-lite-latest"
HF_DATASET = "semran1/yulan-code-MNBVC-matlab"
OUT_DIR = Path(__file__).parent / "sc_matlab_validated"
PSEUDOCODE_PROMPT_FILE = Path(__file__).parent.parent / "pseudocode.txt"

MAX_SAMPLES = 8000
MAX_OUTPUT_TOKENS = 2048
OCTAVE_TIMEOUT = 15
AUDIT_SIZE = 50

MIN_LINES = 5
MAX_LINES = 100
MAX_CHAR_LENGTH = 4000
SKIP_SOURCE_ITEMS = 0

# Prompt: ask Gemini to write a self-contained test harness for a function
TEST_HARNESS_PROMPT = """\
Write a minimal Octave test script that calls the function below with appropriate inputs.

Rules:
- Output only runnable Octave code — no explanations, no markdown fences.
- Start with rng(42) for reproducibility.
- Create small numeric inputs (scalars, small vectors/matrices).
- Call the function with those inputs and display every return value with disp().
- DO NOT define, implement, or stub any helper functions — only generate inputs and make the call.
- Do not read files or use toolboxes.

Function:
<<<MATLAB_CODE>>>
"""


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

# Patterns that indicate external I/O, GUI, or known toolbox calls — skip these
_EXTERNAL_DEP_RE = re.compile(
    r"\b(fopen|fwrite|fread|imread|imwrite|xlsread|csvread|xlswrite|csvwrite"
    r"|figure|plot|subplot|uicontrol|guidata|waitfor"
    r"|system|eval|feval|addpath|rmpath"
    r"|xml_read|xml_write|xmlread|xmlwrite"
    r"|spm_\w+|vl_\w+|cv\.\w+)\b",
    re.IGNORECASE,
)

# ======================
# INIT
# ======================

env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip("'").strip('"')

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY not set")
    exit(1)

client = genai.Client(api_key=api_key)
gen_cfg = types.GenerateContentConfig(temperature=0.2, max_output_tokens=MAX_OUTPUT_TOKENS)

OUT_DIR.mkdir(exist_ok=True)

try:
    PSEUDOCODE_PROMPT = PSEUDOCODE_PROMPT_FILE.read_text()
except FileNotFoundError:
    PSEUDOCODE_PROMPT = "Convert the following MATLAB code to step-by-step pseudocode:\n\n<<<MATLAB_CODE>>>"

# ======================
# HELPERS
# ======================

def clean_matlab_code(code: str) -> str:
    if not code:
        return ""
    cleaned = []
    for line in code.split("\n"):
        line = line.strip()
        if line.startswith("%"):
            continue
        if "%" in line:
            parts = line.split("%", 1)
            if parts[0].count("'") % 2 == 0:
                line = parts[0].strip()
        if line:
            cleaned.append(line)
    return "\n".join(cleaned)


def is_high_quality(code: str) -> bool:
    if not code or not code.strip():
        return False
    lines = code.split("\n")
    if not (MIN_LINES <= len(lines) <= MAX_LINES):
        return False
    if len(code) > MAX_CHAR_LENGTH:
        return False
    if len(set(lines)) < len(lines) * 0.5:
        return False
    return True


def has_external_deps(code: str) -> bool:
    return bool(_EXTERNAL_DEP_RE.search(code))


def parse_function_info(code: str) -> tuple[str, int, int] | None:
    """If code starts with a function definition, return (func_name, n_inputs, n_outputs). Else None."""
    for line in code.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        m = re.match(
            r"^function\s+(?:\[([^\]]*)\]\s*=\s*|(\w+)\s*=\s*)?(\w+)\s*\(([^)]*)\)",
            stripped, re.IGNORECASE,
        )
        if m:
            outputs_bracket, output_single, func_name, inputs_str = m.groups()
            n_inputs = len([x for x in inputs_str.split(",") if x.strip()]) if inputs_str.strip() else 0
            if outputs_bracket:
                n_outputs = len([x for x in outputs_bracket.split(",") if x.strip()])
            elif output_single:
                n_outputs = 1
            else:
                n_outputs = 0
            return func_name, n_inputs, n_outputs
        return None
    return None


def has_likely_output(code: str) -> bool:
    print_calls = re.search(r"\b(disp|fprintf|printf|display)\s*\(", code)
    unsuppressed = any(
        line.strip()
        and not line.strip().endswith(";")
        and not line.strip().startswith("%")
        and not line.strip().startswith("if")
        and not line.strip().startswith("for")
        and not line.strip().startswith("while")
        and not line.strip().startswith("end")
        for line in code.split("\n")
    )
    return bool(print_calls or unsuppressed)


def run_octave_script(code: str) -> tuple[str, bool]:
    with tempfile.NamedTemporaryFile(suffix=".m", mode="w", delete=False, dir="/tmp") as f:
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
    except FileNotFoundError:
        raise RuntimeError("Octave not found. Install with: brew install octave")
    finally:
        try:
            os.unlink(fname)
        except OSError:
            pass


def run_with_harness(func_code: str, func_name: str, harness_body: str) -> tuple[str, bool]:
    """Write func_name.m + harness to isolated tmpdir, run harness."""
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
        except FileNotFoundError:
            raise RuntimeError("Octave not found. Install with: brew install octave")


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


def extract_matlab_code(text: str) -> str:
    text = re.sub(r"```(?:matlab|octave|m)?\n?(.*?)```", r"\1", text, flags=re.DOTALL)
    return text.strip()


def call_gemini(prompt: str, retries: int = 5) -> str:
    for attempt in range(retries):
        try:
            response = client.models.generate_content(model=MODEL, contents=prompt, config=gen_cfg)
            return response.text.strip()
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower() or "quota" in str(e).lower():
                sleep_time = 2 ** attempt + random.uniform(0, 1)
                print(f"  Rate limited, sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
            else:
                print(f"  Generation error: {e}")
                return ""
    return ""


def save_pair(
    index: int, code: str, pseudocode: str, regen_code: str, kind: str,
    orig_out: str, regen_out: str, harness: str = "",
):
    sample_dir = OUT_DIR / f"sample_{index}"
    sample_dir.mkdir(exist_ok=True)
    (sample_dir / "code.m").write_text(code, encoding="utf-8")
    (sample_dir / "pseudocode.txt").write_text(pseudocode, encoding="utf-8")
    (sample_dir / "regen_code.m").write_text(regen_code, encoding="utf-8")
    (sample_dir / "kind.txt").write_text(kind, encoding="utf-8")
    (sample_dir / "orig_output.txt").write_text(orig_out, encoding="utf-8")
    (sample_dir / "regen_output.txt").write_text(regen_out, encoding="utf-8")
    if harness:
        (sample_dir / "harness.m").write_text(harness, encoding="utf-8")


def save_audit_sample(n: int):
    audit_dir = OUT_DIR.parent / "audit"
    audit_dir.mkdir(exist_ok=True)
    all_samples = sorted(OUT_DIR.glob("sample_*"))
    chosen = random.sample(all_samples, min(n, len(all_samples)))
    for src in chosen:
        dst = audit_dir / src.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    print(f"Saved {len(chosen)} audit samples to {audit_dir}")


# ======================
# MAIN LOOP
# ======================

def main():
    try:
        r = subprocess.run(["octave", "--version"], capture_output=True, text=True, timeout=5)
        print(f"Octave: {r.stdout.split(chr(10))[0].strip()}")
    except FileNotFoundError:
        print("ERROR: Octave not installed. Run:  brew install octave")
        exit(1)

    print(f"Loading dataset stream: {HF_DATASET}...")
    dataset = load_dataset(HF_DATASET, split="train", streaming=True)

    existing = [d for d in OUT_DIR.iterdir() if d.is_dir() and d.name.startswith("sample_")]
    count = len(existing)
    if count:
        print(f"Resuming from {count} existing samples")

    stats = dict(
        quality=0,
        external_deps=0,
        no_output_heuristic=0,
        no_output_func=0,
        harness_fail=0,
        orig_no_output=0,
        gen_fail=0,
        regen_run_fail=0,
        mismatch=0,
        accepted_script=0,
        accepted_function=0,
    )
    total_seen = 0

    for i, sample in enumerate(dataset):
        if i < SKIP_SOURCE_ITEMS:
            if i % 500 == 0:
                print(f"Skipping source item {i}...", end="\r")
            continue

        if count >= MAX_SAMPLES:
            print(f"\nTarget of {MAX_SAMPLES} validated samples reached.")
            break

        total_seen += 1
        raw_code = sample.get("content") or sample.get("text") or sample.get("code") or ""
        if not raw_code:
            print(f"Unknown dataset keys: {list(sample.keys())}")
            break

        clean_code = clean_matlab_code(raw_code)

        if not is_high_quality(clean_code):
            stats["quality"] += 1
            continue

        if has_external_deps(clean_code):
            stats["external_deps"] += 1
            continue

        func_info = parse_function_info(clean_code)

        # ---- Flat script path ----
        if func_info is None:
            if not has_likely_output(clean_code):
                stats["no_output_heuristic"] += 1
                continue

            print(f"\n[{count+1}/{MAX_SAMPLES}] Sample {total_seen} (script, {len(clean_code.splitlines())} lines)")

            orig_out, orig_ok = run_octave_script(clean_code)
            if not orig_ok or not orig_out:
                stats["orig_no_output"] += 1
                print(f"  Skip: original produced no valid output (ok={orig_ok})")
                continue

            pseudocode = call_gemini(PSEUDOCODE_PROMPT.replace("<<<MATLAB_CODE>>>", clean_code))
            if not pseudocode:
                stats["gen_fail"] += 1
                print("  Skip: pseudocode generation failed")
                continue

            regen_raw = call_gemini(CODE_GEN_PROMPT.replace("<<<PSEUDOCODE>>>", pseudocode))
            if not regen_raw:
                stats["gen_fail"] += 1
                print("  Skip: code regeneration failed")
                continue
            regen_code = extract_matlab_code(regen_raw)

            regen_out, regen_ok = run_octave_script(regen_code)
            if not regen_ok or not regen_out:
                stats["regen_run_fail"] += 1
                print(f"  Skip: regenerated script failed to run (ok={regen_ok})")
                continue

            if not outputs_match(orig_out, regen_out):
                stats["mismatch"] += 1
                print(f"  Skip: outputs differ")
                print(f"    orig:  {orig_out[:100]!r}")
                print(f"    regen: {regen_out[:100]!r}")
                continue

            save_pair(count, clean_code, pseudocode, regen_code, "script", orig_out, regen_out)
            stats["accepted_script"] += 1

        # ---- Function path ----
        else:
            func_name, n_inputs, n_outputs = func_info

            # Functions with no outputs and no disp can't produce comparable output
            if n_outputs == 0 and not has_likely_output(clean_code):
                stats["no_output_func"] += 1
                continue


            print(f"\n[{count+1}/{MAX_SAMPLES}] Sample {total_seen} "
                  f"(function '{func_name}', {n_inputs} in, {n_outputs} out, {len(clean_code.splitlines())} lines)")

            # Ask Gemini to generate a test harness with appropriate inputs
            harness_raw = call_gemini(TEST_HARNESS_PROMPT.replace("<<<MATLAB_CODE>>>", clean_code))
            if not harness_raw:
                stats["harness_fail"] += 1
                print("  Skip: harness generation failed")
                continue
            harness_body = extract_matlab_code(harness_raw)

            orig_out, orig_ok = run_with_harness(clean_code, func_name, harness_body)
            if not orig_ok or not orig_out:
                stats["harness_fail"] += 1
                print(f"  Skip: harness did not produce output (ok={orig_ok})")
                continue

            pseudocode = call_gemini(PSEUDOCODE_PROMPT.replace("<<<MATLAB_CODE>>>", clean_code))
            if not pseudocode:
                stats["gen_fail"] += 1
                print("  Skip: pseudocode generation failed")
                continue

            regen_raw = call_gemini(CODE_GEN_PROMPT.replace("<<<PSEUDOCODE>>>", pseudocode))
            if not regen_raw:
                stats["gen_fail"] += 1
                print("  Skip: code regeneration failed")
                continue
            regen_code = extract_matlab_code(regen_raw)

            # Use regenerated function name if different from original
            regen_info = parse_function_info(regen_code)
            regen_func_name = regen_info[0] if regen_info else func_name
            adapted_harness = harness_body.replace(func_name, regen_func_name) if regen_func_name != func_name else harness_body

            regen_out, regen_ok = run_with_harness(regen_code, regen_func_name, adapted_harness)
            if not regen_ok or not regen_out:
                stats["regen_run_fail"] += 1
                print(f"  Skip: regenerated function failed to run (ok={regen_ok})")
                continue

            if not outputs_match(orig_out, regen_out):
                stats["mismatch"] += 1
                print(f"  Skip: outputs differ")
                print(f"    orig:  {orig_out[:100]!r}")
                print(f"    regen: {regen_out[:100]!r}")
                continue

            save_pair(count, clean_code, pseudocode, regen_code, "function", orig_out, regen_out, harness_body)
            stats["accepted_function"] += 1

        print(f"  VALID -> saved sample_{count}")
        count += 1
        time.sleep(0.3)

    print(f"\n=== Funnel Summary ===")
    print(f"Total seen              : {total_seen}")
    print(f"Accepted (scripts)      : {stats['accepted_script']}")
    print(f"Accepted (functions)    : {stats['accepted_function']}")
    print(f"Accepted (total)        : {count}")
    print(f"---")
    print(f"Dropped quality         : {stats['quality']}")
    print(f"Dropped external deps   : {stats['external_deps']}")
    print(f"Dropped no-output hint  : {stats['no_output_heuristic']}")
    print(f"Dropped fn no output    : {stats['no_output_func']}")
    print(f"Dropped harness fail    : {stats['harness_fail']}")
    print(f"Dropped orig no output  : {stats['orig_no_output']}")
    print(f"Dropped gen fail        : {stats['gen_fail']}")
    print(f"Dropped regen run fail  : {stats['regen_run_fail']}")
    print(f"Dropped mismatch        : {stats['mismatch']}")

    if count > 0:
        save_audit_sample(AUDIT_SIZE)


if __name__ == "__main__":
    main()
