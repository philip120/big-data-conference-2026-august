# Training & evaluation on Google Colab

GPU tier: **A100 40GB** for the default unfreeze-18-layers recipe;
**L4 24GB** works with LoRA (`--unfreeze_layers 0`). Avoid T4 (no bf16).

Everything below runs in notebook cells with `!` (or the Colab Pro terminal,
same commands without `!`).

## 0. Setup (once per runtime)

```bash
!git clone https://github.com/<YOU>/<REPO>.git repo
%cd repo/submission
!pip install -q -r train/requirements.txt
```

Mount Drive and define one place where **everything** (checkpoints, metrics,
plots, eval JSONs) is stored, so nothing is lost when the runtime dies:

```python
from google.colab import drive
drive.mount('/content/drive')
RUN = "/content/drive/MyDrive/matlab_paper"   # the single results root
```

Sanity check — this line MUST say ANTLR, not regex fallback:

```bash
!python -c "import train.semantic_adapter"
# -> [semantic_adapter] AST parser: ANTLR (grammars-v4/matlab)
```

## 1. Stage 1 (once): text-only decoder fine-tune

Trains the decoder on plain code→pseudocode. Also *is* the text-only baseline.

```bash
!python -m train.train_full --s2_model tree_text \
    --s1_save_dir  $RUN/checkpoints_stage1 \
    --s2_save_dir  $RUN/checkpoints_stage2
```

`train_full` runs stage 1 then stage 2 for one variant. For every further
variant, reuse the stage-1 checkpoint (skips stage 1):

```bash
!python -m train.train_full --s2_model vit \
    --stage1_checkpoint $RUN/checkpoints_stage1/best_model.pt \
    --s2_save_dir $RUN/checkpoints_stage2
!python -m train.train_full --s2_model tree      --stage1_checkpoint $RUN/checkpoints_stage1/best_model.pt --s2_save_dir $RUN/checkpoints_stage2
!python -m train.train_full --s2_model combined  --stage1_checkpoint $RUN/checkpoints_stage1/best_model.pt --s2_save_dir $RUN/checkpoints_stage2
```

Checkpoints/metrics/plots land in `$RUN/checkpoints_stage2/<model_type>/`.
Resume an interrupted run with `--s2_resume <checkpoint.pt>`.

## 2. Ablations (reviewer-requested)

```bash
# patch size (vit): does the failure come from grouping or from CLS pixels?
for P in 1 2 4; do
  python -m train.train_pipeline --model vit --patch_size $P \
      --stage1_checkpoint $RUN/checkpoints_stage1/best_model.pt \
      --save_dir $RUN/ablations/patch_$P
done

# non-linear projector at patch 4
!python -m train.train_pipeline --model vit --patch_size 4 --projector mlp \
    --stage1_checkpoint $RUN/checkpoints_stage1/best_model.pt \
    --save_dir $RUN/ablations/projector_mlp

# RvNN width (tree_text) — end-of-training log reports truncation rate
for B in 4 8 16; do
  python -m train.train_pipeline --model tree_text --max_branching $B \
      --stage1_checkpoint $RUN/checkpoints_stage1/best_model.pt \
      --save_dir $RUN/ablations/branch_$B
done

# encoder swap (tree_text)
for E in codebert unixcoder codesage; do
  python -m train.train_pipeline --model tree_text --code_encoder $E \
      --stage1_checkpoint $RUN/checkpoints_stage1/best_model.pt \
      --save_dir $RUN/ablations/enc_$E
done
```

## 3. Evaluation (GPU, deterministic, held-out test split)

Generation is greedy and runs on the hub's `test` split, which no stage of
training ever saw. Model config (encoder, projector, patch size, branching)
is restored from the checkpoint automatically.

```bash
for M in stage1 vit tree combined tree_text; do
  CKPT=$RUN/checkpoints_stage2/$M/best_model.pt
  [ "$M" = stage1 ] && CKPT=$RUN/checkpoints_stage1/best_model.pt
  python -m train.evaluate --model_type $M --checkpoint $CKPT \
      --num_samples 1000 --results_dir $RUN/results
done
```

Writes `$RUN/results/eval_<model>.json` with per-sample generations +
ROUGE-1/2/L, BLEU, chrF and efficiency metrics.

## 4. Execution-match evaluation (NO GPU — run any time, even locally)

Needs Octave and a Gemini key; consumes the JSONs from step 3, so the GPU
runtime can be long gone.

```bash
!apt-get -qq install -y octave          # (macOS: brew install octave)
%env GEMINI_API_KEY=...
!python -m train.evaluate_exec --results $RUN/results/eval_*.json
```

Writes `$RUN/results/exec_eval_<model>.json` with `exec_match_rate`:
the fraction of generated pseudocode faithful enough that regenerated code
reproduces the original program's output.

## 5. Compare everything

```bash
!python -m train.compare_results --results_dir $RUN/results --baseline stage1
```

Prints a markdown table (ROUGE/BLEU/chrF/exec-match/efficiency per model) and
paired-bootstrap p-values vs the text-only baseline.

## Re-pushing the dataset (local, after the 8k generation finishes)

```bash
cd dataset && python push_to_hf.py          # 90/10 train/test, content-hash split
```

The split is deterministic by code hash: samples never migrate between
train and test across pushes.
