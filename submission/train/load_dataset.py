# train/load_dataset.py
from datasets import load_dataset

HF_DATASET = "philip120/sc-matlab-validated"


def load_matlab_nl_dataset(split="train"):
    """
    Loads the execution-validated MATLAB/pseudocode pairs from Hugging Face
    and returns a list of {code, nl} dicts.

    The hub dataset stores the target text in the `pseudocode` column;
    it is exposed here as `nl` for the training pipeline.
    Supports HF split slicing, e.g. split="train[80%:]".
    """

    ds = load_dataset(HF_DATASET, split=split)

    examples = []
    skipped_classdef = 0

    for row in ds:
        code = row.get("code")
        nl = row.get("pseudocode") or row.get("nl")

        if not code or not nl:
            continue

        if code.lstrip().startswith("classdef"):
            skipped_classdef += 1
            continue

        examples.append({
            "code": code,
            "nl": nl
        })

    if skipped_classdef:
        print(f"  Filtered {skipped_classdef} classdef samples")

    print(f"  Loaded {len(examples)} samples from {HF_DATASET} ({split})")
    return examples
