"""Upload sc_matlab_validated samples to the Hugging Face Hub.

Samples are split into train/test deterministically by a content hash of the
code, so the same sample always lands in the same split even as the dataset
grows across pushes. The test split is never seen during training.
"""

import argparse
import hashlib
from pathlib import Path

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

DATA_DIR = Path(__file__).parent / "sc_matlab_validated"
DEFAULT_REPO_ID = "philip120/sc-matlab-validated"

FILE_MAP = {
    "code.m": "code",
    "pseudocode.txt": "pseudocode",
    "regen_code.m": "regen_code",
    "kind.txt": "kind",
    "orig_output.txt": "orig_output",
    "regen_output.txt": "regen_output",
    "harness.m": "harness",
}

DATASET_CARD = """\
---
license: mit
task_categories:
  - text-generation
language:
  - en
tags:
  - code
  - matlab
  - pseudocode
  - program-synthesis
size_categories:
  - n<1K
---

# SC MATLAB Validated

Validated MATLAB/Octave code–pseudocode pairs for program comprehension and synthesis research.

Each sample was filtered from [semran1/yulan-code-MNBVC-matlab](https://huggingface.co/datasets/semran1/yulan-code-MNBVC-matlab),
converted to pseudocode with Gemini, regenerated back to MATLAB, and kept only when Octave execution output matched the original.

## Fields

| Column | Description |
|--------|-------------|
| `sample_id` | Numeric sample index |
| `code` | Original MATLAB/Octave source |
| `pseudocode` | LLM-generated pseudocode from the original code |
| `regen_code` | MATLAB/Octave code regenerated from pseudocode |
| `kind` | `function` or `script` |
| `orig_output` | Octave output from running the original code |
| `regen_output` | Octave output from running the regenerated code |
| `harness` | Test harness used for function samples (empty for scripts) |

## Usage

```python
from datasets import load_dataset

train = load_dataset("philip120/sc-matlab-validated", split="train")
test = load_dataset("philip120/sc-matlab-validated", split="test")
print(train[0]["pseudocode"])
```

The train/test split is deterministic by content hash of `code`, so samples
never migrate between splits as the dataset grows.
"""


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def load_samples(data_dir: Path) -> list[dict]:
    sample_dirs = sorted(
        (p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("sample_")),
        key=lambda p: int(p.name.split("_", 1)[1]),
    )
    if not sample_dirs:
        raise FileNotFoundError(f"No sample_* directories found in {data_dir}")

    records = []
    for sample_dir in sample_dirs:
        record = {"sample_id": int(sample_dir.name.split("_", 1)[1])}
        for filename, field in FILE_MAP.items():
            record[field] = _read_text(sample_dir / filename)
        records.append(record)
    return records


def assign_split(code: str, test_fraction: float) -> str:
    """Deterministic train/test assignment from a content hash of the code.

    Stable across pushes: a sample keeps its split as the dataset grows,
    so the test set is never contaminated by retraining on a re-push.
    """
    digest = hashlib.sha256(code.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "test" if bucket < test_fraction else "train"


def push_dataset(
    repo_id: str,
    data_dir: Path,
    *,
    private: bool = False,
    dry_run: bool = False,
    test_fraction: float = 0.1,
) -> None:
    records = load_samples(data_dir)
    kinds = {}
    for row in records:
        kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1

    train_records = []
    test_records = []
    for row in records:
        (test_records if assign_split(row["code"], test_fraction) == "test"
         else train_records).append(row)

    print(f"Loaded {len(records)} samples from {data_dir}")
    print(f"Kinds: {kinds}")
    print(f"Split: {len(train_records)} train / {len(test_records)} test "
          f"(test_fraction={test_fraction}, content-hash deterministic)")

    if dry_run:
        print(f"Dry run: would push to https://huggingface.co/datasets/{repo_id}")
        return

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    DatasetDict({
        "train": Dataset.from_list(train_records),
        "test": Dataset.from_list(test_records),
    }).push_to_hub(
        repo_id,
        commit_message=f"Upload {len(records)} validated MATLAB samples "
                       f"({len(train_records)} train / {len(test_records)} test)",
    )

    api.upload_file(
        path_or_fileobj=DATASET_CARD.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add dataset card",
    )

    visibility = "private" if private else "public"
    print(f"Pushed {len(records)} samples to https://huggingface.co/datasets/{repo_id} ({visibility})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Push sc_matlab_validated to Hugging Face")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hub dataset repo, e.g. user/name")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Local validated dataset directory")
    parser.add_argument("--private", action="store_true", help="Create/update as a private dataset")
    parser.add_argument("--dry-run", action="store_true", help="Load and summarize without uploading")
    parser.add_argument("--test-fraction", type=float, default=0.1,
                        help="Fraction of samples routed to the held-out test split")
    args = parser.parse_args()

    push_dataset(args.repo_id, args.data_dir, private=args.private, dry_run=args.dry_run,
                 test_fraction=args.test_fraction)


if __name__ == "__main__":
    main()
