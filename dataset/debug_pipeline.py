import os
import re
import subprocess
import tempfile
from pathlib import Path
from google import genai
from google.genai import types

# ======================
# CONFIG
# ======================

MODEL = "gemini-flash-lite-latest"
OCTAVE_TIMEOUT = 30  # seconds; Octave can be slow on first run

DATA_DIR = Path(__file__).parent / "data"
PSEUDOCODE_PROMPT_FILE = "pseudocode.txt"

CODE_GEN_PROMPT = """\
You are given pseudocode describing a MATLAB algorithm.
Write MATLAB/Octave code that implements it exactly.

Rules:
- Output only MATLAB code, no explanations or markdown fences.
- Display computed results using disp() or fprintf(), matching the level of verbosity implied by the pseudocode.
- Do not use toolboxes beyond base MATLAB/Octave.
- Do not read from files or external inputs.

Pseudocode:
<<<PSEUDOCODE>>>
"""

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
    print("ERROR: GEMINI_API_KEY not set")
    exit(1)

client = genai.Client(api_key=api_key)

try:
    PSEUDOCODE_PROMPT = Path(PSEUDOCODE_PROMPT_FILE).read_text()
except FileNotFoundError:
    PSEUDOCODE_PROMPT = "Convert the following MATLAB code to step-by-step pseudocode:\n\n<<<MATLAB_CODE>>>"

# ======================
# HELPERS
# ======================

def run_octave(code: str) -> tuple[str, str, bool]:
    """Returns (stdout, stderr, success)."""
    with tempfile.NamedTemporaryFile(suffix=".m", mode="w", delete=False, dir="/tmp") as f:
        f.write(code)
        fname = f.name
    try:
        result = subprocess.run(
            ["octave", "--no-gui", "--quiet", fname],
            capture_output=True, text=True, timeout=OCTAVE_TIMEOUT,
        )
        return result.stdout.strip(), result.stderr.strip(), result.returncode == 0
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT", False
    except FileNotFoundError:
        return "", "Octave not found — install with: brew install octave", False
    finally:
        try:
            os.unlink(fname)
        except OSError:
            pass


def extract_matlab_code(text: str) -> str:
    text = re.sub(r"```(?:matlab|octave|m)?\n?(.*?)```", r"\1", text, flags=re.DOTALL)
    return text.strip()


def normalize_output(text: str) -> str:
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"\b(\d+\.\d{5,})\b", lambda m: f"{float(m.group()):.4f}", line)
        lines.append(line)
    return "\n".join(lines)


def sep(title: str):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print('='*55)


# ======================
# PIPELINE (single case)
# ======================

def run_case(name: str, code: str, gen_cfg) -> bool:
    """Run one .m file through the full pipeline. Returns True if outputs match."""
    sep(f"CASE: {name}")
    print(f"--- Original code ---\n{code}")

    # Step 1: run original
    orig_out, orig_err, orig_ok = run_octave(code)
    print(f"--- Octave output (original) ---\n{orig_out or '(none)'}")
    if orig_err:
        print(f"[stderr] {orig_err}")

    if not orig_ok or not orig_out:
        print("SKIP: original produced no output or errored")
        return False

    # Step 2: generate pseudocode
    prompt1 = PSEUDOCODE_PROMPT.replace("<<<MATLAB_CODE>>>", code)
    r1 = client.models.generate_content(model=MODEL, contents=prompt1, config=gen_cfg)
    pseudocode = r1.text.strip()
    print(f"--- Pseudocode ---\n{pseudocode}")

    # Step 3: regenerate MATLAB from pseudocode
    prompt2 = CODE_GEN_PROMPT.replace("<<<PSEUDOCODE>>>", pseudocode)
    r2 = client.models.generate_content(model=MODEL, contents=prompt2, config=gen_cfg)
    regen_code = extract_matlab_code(r2.text.strip())
    print(f"--- Regenerated MATLAB ---\n{regen_code}")

    # Step 4: run regenerated code
    regen_out, regen_err, regen_ok = run_octave(regen_code)
    print(f"--- Octave output (regenerated) ---\n{regen_out or '(none)'}")
    if regen_err:
        print(f"[stderr] {regen_err}")

    # Step 5: compare
    n_orig  = normalize_output(orig_out)
    n_regen = normalize_output(regen_out)
    match = (n_orig == n_regen)
    print(f"--- MATCH: {match} ---")
    if not match:
        print(f"  expected: {n_orig[:200]!r}")
        print(f"  got:      {n_regen[:200]!r}")
    return match


# ======================
# MAIN
# ======================

def main():
    m_files = sorted(DATA_DIR.glob("*.m"))
    if not m_files:
        print(f"No .m files found in {DATA_DIR}")
        return

    print(f"Found {len(m_files)} .m files in {DATA_DIR}")
    gen_cfg = types.GenerateContentConfig(temperature=0.2, max_output_tokens=1024)

    results = []
    for path in m_files:
        code = path.read_text(encoding="utf-8")
        matched = run_case(path.stem, code, gen_cfg)
        results.append((path.stem, matched))

    sep("SUMMARY")
    passed = sum(1 for _, m in results if m)
    for name, matched in results:
        status = "PASS" if matched else "FAIL"
        print(f"  [{status}] {name}")
    print(f"\n  {passed}/{len(results)} cases matched")


if __name__ == "__main__":
    main()
