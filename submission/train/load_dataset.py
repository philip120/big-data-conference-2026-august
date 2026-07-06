# train/load_dataset.py
from datasets import load_dataset

def load_matlab_nl_dataset(split="train"):
    """
    Loads from Hugging Face
    and returns a list of {code, nl} dicts.
    """

    ds = load_dataset(
        "huggingfacedataset",
        split=split
    )

    examples = []
    skipped_classdef = 0

    for row in ds:
        code = row.get("code")
        nl   = row.get("nl")

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

    return examples
