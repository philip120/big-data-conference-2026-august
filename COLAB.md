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

## 1b. StructCoder decoder (`--decoder structcoder`)

An encoder-decoder alternative to Qwen. Same four stage-2 variants, same
scripts; only the decoder changes. Worth running because it removes the two
structural problems behind the negative Qwen result: the soft tokens become
the *encoder input* that every decoder layer cross-attends to (rather than a
~5-token prefix on a decoder-only stream that the prompt never refers to), and
the decoder is 138M trainable params instead of 2.2B — measured
decoder-to-encoder trainable ratio drops from ~150x to ~9-11x. The T5
tokenizer also appends `</s>`, so the missing-EOS bug does not apply here.

Note what does and does not transfer from StructCoder. Its structure-aware
*encoder* needs a tree-sitter grammar for the source language and there is no
MATLAB one in the released parser blobs — and in this pipeline the encoder is
our own ViT/tree/combined model anyway. Its auxiliary heads predict the AST and
data flow of the *target*, and our target is English pseudocode. So what is
loaded is the half that the structure-based denoising objective pretrained:
`shared` + `decoder` + `lm_head`, on a stock CodeT5-base skeleton. Their
`modeling_structcoder.py` is not imported; it subclasses `T5Attention` against
the 2022 transformers API and only the unused encoder stack depends on it.

Fetch the weights once (Google Drive, ~1 GB) and point the flag at them:

```python
!pip install -q gdown
!mkdir -p $RUN/structcoder
!gdown --id 10Jee9uv4-XuqecWTlKvo1CeNQh1hOXEs -O $RUN/structcoder/structcoder_pretrain.bin
import os; os.environ["STRUCTCODER_CKPT"] = f"{RUN}/structcoder/structcoder_pretrain.bin"
```

Without a checkpoint everything still runs, on plain `Salesforce/codet5-base`,
and the decoder prints a loud warning. That is a legitimate ablation — it
isolates what the structure-aware pretraining is worth — but it is not
StructCoder, so do not report it as one.

Pre-flight before spending GPU hours (CPU, ~1 min, nonzero exit on failure):

```bash
!python -m train.test_structcoder --structcoder_ckpt $STRUCTCODER_CKPT
```

Then stage 1 and the four variants. LoRA on a 220M seq2seq buys nothing, so
full fine-tune the decoder stack: `--unfreeze_layers 12` is all CodeT5 decoder
blocks + `final_layer_norm` + `lm_head` (passing 18, as the Qwen recipe does,
just clamps to 12 with a note). Everything goes under `$RUN/structcoder` so it
cannot overwrite the Qwen checkpoints or the `eval_*.json` the paper figures
read.

```bash
!python -m train.train_stage1 \
  --decoder structcoder --structcoder_ckpt $STRUCTCODER_CKPT \
  --unfreeze_layers 12 \
  --epochs 6 --lr 1e-4 --grad_accum 8 --weight_decay 0.05 \
  --save_dir $RUN/structcoder/checkpoints_stage1

!python -m train.train_full --s2_model vit \
    --decoder structcoder --structcoder_ckpt $STRUCTCODER_CKPT \
    --unfreeze_layers 12 \
    --stage1_checkpoint $RUN/structcoder/checkpoints_stage1/best_model.pt \
    --s2_save_dir $RUN/structcoder/checkpoints_stage2
```

…and the same for `tree`, `combined`, `tree_text`. Evaluate into a separate
results dir (`--results_dir $RUN/results_structcoder --decoder structcoder`)
so the two decoders' numbers stay side by side rather than on top of each
other. `submission/colab_structcoder.py` has every cell ready to paste.

Optional control for the capacity-asymmetry claim: a fully frozen decoder,
projector only. `train_full` always switches to LoRA when `--unfreeze_layers`
is 0, so go through `train_pipeline`, where `--lora` is opt-in:

```bash
!python -m train.train_pipeline --model vit \
    --decoder structcoder --structcoder_ckpt $STRUCTCODER_CKPT \
    --unfreeze_layers 0 \
    --epochs 10 --lr 3e-4 --bottleneck 768 --dropout 0.05 --grad_accum 8 \
    --stage1_checkpoint $RUN/structcoder/checkpoints_stage1/best_model.pt \
    --save_dir $RUN/structcoder/frozen/vit
```

GPU: an L4 is plenty — CodeT5-base is 220M against Qwen3-4B.

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
