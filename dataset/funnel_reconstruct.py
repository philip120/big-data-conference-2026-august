"""Reconstruct the static-filter funnel of dataset_creation_v3.py.

Re-runs every static (non-LLM) filter stage, in pipeline order, over the exact
source-corpus range consumed by the real run (source_index 0..MAX_SRC). The
LLM/execution stages cannot be replayed without API calls, so their combined
drop count is derived as: (items passing all static filters) - (accepted).
"""
import os, sys, json
from pathlib import Path

os.environ.setdefault("GEMINI_API_KEY", "dummy-not-used")
ROOT = Path("/Users/philippaskov/Developer/research/paper 2026 for august submission")
sys.path.insert(0, str(ROOT / "dataset"))
import dataset_creation_v3 as dc  # noqa: E402

OUT = Path(os.environ.get("CLAUDE_JOB_DIR", "/tmp")) / "tmp" / "funnel_result.json"

# accepted samples: source_index -> hash of stored (cleaned) code
accepted = {}
for d in (ROOT / "dataset" / "sc_matlab_validated").iterdir():
    if d.is_dir() and d.name.startswith("sample_"):
        idx = int((d / "source_index.txt").read_text())
        accepted[idx] = dc.code_hash((d / "code.m").read_text(encoding="utf-8"))
MAX_SRC = max(accepted)
print(f"accepted={len(accepted)} max_source_index={MAX_SRC}", flush=True)

from datasets import load_dataset  # noqa: E402
ds = load_dataset(dc.HF_DATASET, split="train", streaming=True)

stats = dict(already_done=0, quality=0, external_deps=0, missing_helpers=0,
             no_output_heuristic=0, no_output_func=0, script_dynamic=0,
             func_dynamic=0, accepted=0, accepted_static_mismatch=0, empty=0)
survivors = []  # (i, clean, candidates, func_info, is_accepted)
seen_hashes = set()

for i, sample in enumerate(ds):
    if i > MAX_SRC:
        break
    if i % 2000 == 0:
        print(f"pass1 {i}/{MAX_SRC} survivors={len(survivors)}", flush=True)
    raw = sample.get("content") or sample.get("text") or sample.get("code") or ""
    clean = dc.clean_matlab_code(raw)
    if not clean:
        stats["empty"] += 1
        continue
    h = dc.code_hash(clean)
    is_acc = i in accepted
    if h in seen_hashes:
        stats["already_done"] += 1
        continue
    if not dc.is_high_quality(clean):
        stats["quality"] += 1
        if is_acc: stats["accepted_static_mismatch"] += 1
        continue
    if dc.has_external_deps(clean):
        stats["external_deps"] += 1
        if is_acc: stats["accepted_static_mismatch"] += 1
        continue
    cands = dc._extract_callees(clean)
    survivors.append((i, clean, cands, is_acc))
    if is_acc:
        seen_hashes.add(h)  # real run added accepted hashes to done_hashes

# batch-resolve all candidate names in chunks
names = sorted({n for _, _, c, _ in survivors for n in c})
print(f"pass2: resolving {len(names)} unique names in octave", flush=True)
CHUNK = 400
resolved = {}
for k in range(0, len(names), CHUNK):
    chunk = names[k:k + CHUNK]
    probe = "\n".join(f"printf('%s %d\\n', '{n}', exist('{n}'));" for n in chunk)
    out, _ = dc.run_octave_script(probe)
    for line in out.split("\n"):
        parts = line.strip().split()
        if len(parts) == 2 and parts[1].lstrip("-").isdigit():
            resolved[parts[0]] = int(parts[1]) > 0
    print(f"  resolved {min(k+CHUNK, len(names))}/{len(names)}", flush=True)

for i, clean, cands, is_acc in survivors:
    missing = {n for n in cands if not resolved.get(n, False)}
    if missing:
        stats["missing_helpers"] += 1
        if is_acc: stats["accepted_static_mismatch"] += 1
        continue
    fi = dc.parse_function_info(clean)
    if fi is None:
        if not dc.has_likely_output(clean):
            stats["no_output_heuristic"] += 1
            if is_acc: stats["accepted_static_mismatch"] += 1
        else:
            stats["script_dynamic"] += 1  # reached execution stage; none accepted
        continue
    _, _, n_out = fi
    if n_out == 0 and not dc.has_likely_output(clean):
        stats["no_output_func"] += 1
        if is_acc: stats["accepted_static_mismatch"] += 1
        continue
    if is_acc:
        stats["accepted"] += 1
    else:
        stats["func_dynamic"] += 1

total = MAX_SRC + 1
stats["total_seen"] = total
stats["dynamic_candidates"] = stats["script_dynamic"] + stats["func_dynamic"] + stats["accepted"]
stats["dynamic_dropped"] = stats["script_dynamic"] + stats["func_dynamic"]
OUT.write_text(json.dumps(stats, indent=2))
print(json.dumps(stats, indent=2), flush=True)
print("DONE", flush=True)
