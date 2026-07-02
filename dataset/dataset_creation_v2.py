import os
import time
import random
import re
from pathlib import Path
import google.generativeai as genai
from datasets import load_dataset

# ======================
# CONFIG
# ======================

MODEL = "gemini-flash-lite-latest"  
# alternatives: "gemini-1.5-flash", "gemini-1.0-pro"

HF_DATASET = "averoo/sc_MATLAB"
OUT_DIR = "dataset/sc_matlab_clean"
PROMPT_FILE = "pseudocode.txt"

MAX_SAMPLES = 8000  # How many pairs we want
MAX_OUTPUT_TOKENS = 2048

# Code filtering criteria
MIN_LINES = 5
MAX_LINES = 100
MAX_CHAR_LENGTH = 4000

# ======================
# INIT
# ======================

# Load .env file manually
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip().strip("'").strip('"')

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("❌ Error: GEMINI_API_KEY environment variable is not set.")
    print(f"Checked for .env at: {env_path.absolute()}")
    exit(1)

genai.configure(api_key=api_key)

model = genai.GenerativeModel(
    MODEL,
    generation_config={
        "temperature": 0.2,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }
)

os.makedirs(OUT_DIR, exist_ok=True)

try:
    with open(PROMPT_FILE, "r") as f:
        PROMPT_TEMPLATE = f.read()
except FileNotFoundError:
    print(f"⚠️ Warning: {PROMPT_FILE} not found. Using default prompt.")
    PROMPT_TEMPLATE = "Convert the following MATLAB code to step-by-step pseudocode:\n\n<<<MATLAB_CODE>>>"

# ======================
# HELPERS
# ======================

def clean_matlab_code(code: str) -> str:
    """
    Cleans MATLAB code:
    1. Removes full-line comments (%)
    2. Removes inline comments (code % comment) -> (code)
    3. Removes empty lines
    """
    if not code:
        return ""

    lines = code.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Skip full line comments
        if line.startswith('%'):
            continue
            
        # Remove inline comments
        # Be careful not to break strings like '100%' or fprintf('%d')
        # Simple heuristic: split by % but ignore if inside quotes (too complex for regex, doing simple split for now)
        # Better simple approach: only strip if % is preceded by space, and not inside a string
        # For robustness, let's just use simple regex for standard comments
        
        # This regex looks for % not inside quotes (simplified)
        # Assuming most inline comments are distinct
        if '%' in line:
            # Check if it's likely a comment or a format string
            parts = line.split('%', 1)
            # If the part before % has an odd number of quotes, it's likely inside a string
            if parts[0].count("'") % 2 == 0:
                line = parts[0].strip()
        
        if line:
            cleaned_lines.append(line)
            
    return "\n".join(cleaned_lines)

def is_high_quality(code: str) -> bool:
    """Check if code is worth processing."""
    if not code or not code.strip():
        return False
        
    lines = code.split('\n')
    if len(lines) < MIN_LINES or len(lines) > MAX_LINES:
        return False
        
    if len(code) > MAX_CHAR_LENGTH:
        return False
        
    # Skip extremely repetitive files (often data files)
    if len(set(lines)) < len(lines) * 0.5:
        return False
        
    return True

def build_prompt(matlab_code: str) -> str:
    return PROMPT_TEMPLATE.replace("<<<MATLAB_CODE>>>", matlab_code)

def save_pair(index: int, code: str, pseudocode: str):
    """Save pair to individual files (better for debugging)"""
    sample_dir = Path(OUT_DIR) / f"sample_{index}"
    sample_dir.mkdir(exist_ok=True)
    
    (sample_dir / "code.m").write_text(code, encoding='utf-8')
    (sample_dir / "pseudocode.txt").write_text(pseudocode, encoding='utf-8')

def call_gemini_with_retry(prompt, retries=5):
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            return response.text.strip()

        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower() or "quota" in str(e).lower():
                sleep_time = 2 ** attempt + random.uniform(0, 1)
                print(f"  ⏳ Rate limited, sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
            else:
                print(f"  ❌ Generation Error: {e}")
                return "" # Skip this sample

    return ""

# ======================
# MAIN LOOP
# ======================

def main():
    print(f"Loading dataset stream: {HF_DATASET}...")
    # Stream the dataset so we don't download 400k examples at once
    dataset = load_dataset(HF_DATASET, split="train", streaming=True)
    
    count = 0
    total_processed = 0
    
    # Check existing count to resume
    existing_dirs = [d for d in Path(OUT_DIR).iterdir() if d.is_dir() and d.name.startswith("sample_")]
    if existing_dirs:
        count = len(existing_dirs)
        print(f"Resuming from sample count: {count}")

    print("Starting processing loop...")
    
    # Simple manual offset for streaming datasets
    # Since streaming datasets don't support random access easily without skipping
    # We will just skip the first N *processed* items if needed, but since we rely on 'count' for file naming,
    # the existing logic actually handles the naming correctly (starts at sample_50).
    # BUT, we need to skip the *source* samples we already processed to avoid re-processing the same files!
    
    # Heuristic: We don't know exactly how many source samples we used to get 50 good ones.
    # So we should probably skip 'total_processed' if we tracked it, but we didn't save it.
    # Alternative: Check if the specific code content already exists? Too slow.
    
    # BEST APPROACH for now: Just skip a fixed number if you know it, OR just rely on 'count' 
    # and re-process (but that wastes API calls).
    
    # Since the user asked to "start from 50th instance", they imply skipping the first 50 source items.
    SKIP_SOURCE_ITEMS = 5000 
    
    for i, sample in enumerate(dataset):
        if i < SKIP_SOURCE_ITEMS:
            if i % 10 == 0: print(f"Skipping source item {i}...", end='\r')
            continue

        if count >= MAX_SAMPLES:
            print(f"Target of {MAX_SAMPLES} samples reached!")
            break
            
        total_processed += 1
        
        # NOTE: Key is often 'content' or 'text' or 'code'. Check dataset features.
        # For averoo/sc_MATLAB, checking keys
        if 'content' in sample:
            raw_code = sample['content']
        elif 'text' in sample:
            raw_code = sample['text']
        elif 'code' in sample:
            raw_code = sample['code']
        else:
            # Inspect first sample keys if unknown
            print(f"Unknown keys in dataset: {sample.keys()}")
            break
            
        # 1. Clean
        clean_code = clean_matlab_code(raw_code)
        
        # 2. Filter
        if not is_high_quality(clean_code):
            continue
            
        # 3. Check if already exists (skip if resuming naive)
        # We just use count, simpler
        
        print(f"[{count+1}/{MAX_SAMPLES}] Generating pseudocode for sample {total_processed} ({len(clean_code.splitlines())} lines)...")
        
        # 4. Generate
        prompt = build_prompt(clean_code)
        pseudocode = call_gemini_with_retry(prompt)
        
        if not pseudocode:
            print("  ⏩ Skipped (generation failed)")
            continue
            
        # 5. Save
        save_pair(count, clean_code, pseudocode)
        print("  ✅ Saved")
        
        count += 1
        
        # Small sleep to be nice to API if not hitting rate limits
        time.sleep(0.5)

if __name__ == "__main__":
    main()
