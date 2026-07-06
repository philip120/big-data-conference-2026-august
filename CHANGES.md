# Revision changes (July 2026, for August Big Data submission)

Summary of code changes made while revising after the ECML-PKDD reviews.
Each item notes the reviewer concern it addresses (R1/R2/R3).

## Dataset

- **Execution-validated dataset pipeline** (`dataset/dataset_creation_v3.py`):
  every sample is validated by round-trip — code → Gemini pseudocode →
  Gemini-regenerated code → both run in Octave → kept only when outputs
  match. Replaces spot-checking of LLM labels. *(R2/R3: label quality)*
- **Held-out test split** (`dataset/push_to_hf.py`): deterministic
  train/test split (default 10%) by content hash of the code, stable across
  re-pushes. Training never sees the test split — including stage 1
  (previously stage 1 fine-tuned the decoder on 100% of the data while
  stage 2 evaluated on the last 20% of it — a leak, now fixed).
  *(R2: rigor)*
- `train/load_dataset.py` wired to `philip120/sc-matlab-validated`
  (maps `pseudocode` → `nl`). Placeholder in `train/evaluate.py` fixed too.

## Parser (AST correctness)

- The ANTLR MATLAB parser (`grammars-v4/matlab`, generated with ANTLR 4.13.2)
  is now vendored in the repo and found from either the repo root or
  `submission/`. Previously the import failed silently and **all runs used a
  regex fallback**, not a real AST. A startup line now prints which parser is
  active; the regex fallback warns loudly. `antlr4-python3-runtime` is pinned
  to 4.13.2 to match the generated code. *(R2: reproducibility)*

## Models

- **Non-linear projector** (`shared/projector.py`, `--projector mlp`):
  LayerNorm-in → linear skip + zero-init GELU MLP → RMS-pinned output with
  gain initialized to the decoder's embedding RMS. Fixes the variance
  explosion that made anything beyond a single Linear untrainable at
  patch_size=4. The linear baseline is unchanged and remains the default.
  *(R2: missing fusion/design ablations)*
- **Pluggable code encoder** (`shared/code_encoder.py`, `--code_encoder`):
  `codebert` (default, original behavior), `unixcoder`, `codesage`
  (trained on The Stack v2, which includes MATLAB). All downstream dims
  derive from the encoder's hidden size. *(R1/R3: why MATLAB / baseline choice)*
- **RvNN width** (`--max_branching`, default 8) is now configurable, and the
  encoder counts how often children are silently truncated
  (`truncation_stats()`, reported at end of training and in metrics.json).
  *(R2: width-truncation ablation)*
- `--patch_size` default aligned between `train_pipeline.py` and
  `train_full.py` (was 4 vs 1 — the experiment silently differed by entry
  point). Checkpoints now record `projector_arch`, `encoder_name`,
  `max_branching`, `patch_size` and restore them automatically in
  evaluate/inference.

## Evaluation

- **Greedy decoding everywhere in evaluation** (was temperature-0.7 sampling —
  metrics were non-deterministic). `do_sample=True` still available.
  *(R2: no significance testing possible with noisy eval)*
- **chrF** added alongside ROUGE-1/2/L and BLEU.
- **Execution-match evaluation** (`train/evaluate_exec.py`): the model's
  generated pseudocode is given to Gemini to regenerate MATLAB, which is run
  in Octave against the dataset's stored harness/outputs. Reports
  `exec_match_rate` (functional correctness of the pseudocode) — a direct
  answer to "ROUGE only measures similarity to LLM-authored references".
  GPU-free by design: it consumes evaluate.py's results JSON, so it can run
  after the Colab GPU session ends. *(R1/R2/R3: evaluation proxy)*
- **`train/compare_results.py`**: one table across all models
  (ROUGE/BLEU/chrF/exec-match/efficiency) + paired bootstrap significance
  vs a chosen baseline. *(R2: significance tests)*
- Stage-1 baseline eval prompt aligned with the shared training PROMPT.
- Results are written to `--results_dir` (evaluate) and `--save_dir`
  (training) so everything can live on Google Drive across Colab sessions.

## Planned ablation matrix

| Axis | Values | Flag |
|---|---|---|
| Patch size | 1, 2, 4 | `--patch_size` |
| RvNN width | 4, 8, 16 | `--max_branching` |
| Projector | linear, mlp | `--projector` |
| Encoder | codebert, unixcoder, codesage | `--code_encoder` |
| Seeds | ≥3 per headline config | (rerun) |
