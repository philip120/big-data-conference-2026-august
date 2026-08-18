# -*- coding: utf-8 -*-
"""
StructCoder decoder run — Colab cells.

Drop-in replacement for the qwen cells in ieee_big_data_colab.py. Each block
below is one notebook cell. Everything is written under $RUN/structcoder so a
StructCoder run never overwrites the Qwen checkpoints or the eval_*.json the
paper figures are built from.

What changes vs. the Qwen recipe
--------------------------------
Qwen is decoder-only: the encoder pipeline conditions it by prepending ~5 soft
tokens to the prompt, and 2.2B trainable decoder params sit downstream of that
prefix. StructCoder is a CodeT5-base-shaped seq2seq: the soft tokens are the
*encoder input*, every decoder layer cross-attends to them, and the whole
decoder is 138M trainable params — a 16x cut, so every variant's
decoder-to-encoder trainable ratio improves by 16x (combined 149x -> 9.3x,
tree/tree_text 178x -> 11.1x, vit 934x -> 58.4x).
The T5 tokenizer also appends </s>, so the missing-EOS bug does not apply.
"""

# ============================================================
# CELL 1 — setup (unchanged from the qwen notebook)
# ============================================================
"""
!git clone https://github.com/philip120/big-data-conference-2026-august.git
%cd /content/big-data-conference-2026-august/submission
!pip install -q -r train/requirements.txt

from google.colab import drive
drive.mount('/content/drive')
RUN = "/content/drive/MyDrive/matlab_paper"

!python -c "import train.semantic_adapter"
# -> [semantic_adapter] AST parser: ANTLR (grammars-v4/matlab)

!pip uninstall -y torchao
"""

# ============================================================
# CELL 2 — fetch the StructCoder weights (once per Drive, ~1 GB)
# ============================================================
"""
import os, glob, subprocess

SC_DIR = f"{RUN}/structcoder"
os.makedirs(SC_DIR, exist_ok=True)

# https://github.com/reddy-lab-code-research/StructCoder -> pretrained weights
FILE_ID = "10Jee9uv4-XuqecWTlKvo1CeNQh1hOXEs"

found = sorted(glob.glob(f"{SC_DIR}/*.bin") + glob.glob(f"{SC_DIR}/*.pt")
               + glob.glob(f"{SC_DIR}/*.pth"))
if not found:
    subprocess.run(["pip", "install", "-q", "gdown"], check=True)
    subprocess.run(["gdown", "--id", FILE_ID, "-O", f"{SC_DIR}/structcoder_pretrain.bin"],
                   check=True)
    found = sorted(glob.glob(f"{SC_DIR}/*.bin"))

SC_CKPT = found[0] if found else None

# SC_FLAG is what the training cells interpolate. Defining it here means a
# missing checkpoint degrades to "run on plain CodeT5-base with a warning"
# instead of producing an argparse error three cells later. Never write
# $STRUCTCODER_CKPT in a ! line: an unset shell var expands to nothing and
# argparse reports "expected one argument", which does not name the cause.
SC_FLAG = f"--structcoder_ckpt {SC_CKPT}" if SC_CKPT else ""

if SC_CKPT:
    os.environ["STRUCTCODER_CKPT"] = SC_CKPT
    print("StructCoder checkpoint:", SC_CKPT,
          round(os.path.getsize(SC_CKPT) / 1e6), "MB")
else:
    print("NO StructCoder checkpoint — will run on plain Salesforce/codet5-base.")
    print("That is a valid ablation but it is NOT StructCoder.")

# If the Drive link is rate-limited, download it by hand in a browser and
# upload to $RUN/structcoder/ — the glob above picks it up on the next run.
# Without it everything still runs, on plain Salesforce/codet5-base, and the
# decoder prints a loud warning. That is a valid ablation, not StructCoder.
"""

# ============================================================
# CELL 3 — pre-flight (CPU, ~1 min). Do this before burning GPU hours.
# ============================================================
"""
!python -m train.test_structcoder {SC_FLAG}
# Confirms: the checkpoint loads, both adaptation modes work, the checkpoint
# roundtrip evaluate.py performs works, and gradient reaches the encoder in
# all four stage-2 models. Nonzero exit = do not start training.
"""

# ============================================================
# CELL 4 — Stage 1: text-only decoder fine-tune
# ============================================================
"""
# LoRA on a 220M seq2seq buys nothing, so full fine-tune the decoder stack.
# --unfreeze_layers 12 = all CodeT5 decoder blocks + final_layer_norm + lm_head
# (passing 18, as the qwen recipe does, just clamps to 12 with a note).
# lr 1e-4: T5 tolerates a higher LR than a 4B decoder-only model.
!python -m train.train_stage1 \
  --decoder structcoder {SC_FLAG} \
  --unfreeze_layers 12 \
  --epochs 6 --lr 1e-4 --grad_accum 8 --weight_decay 0.05 \
  --save_dir $RUN/structcoder/checkpoints_stage1
"""

# ============================================================
# CELL 5 — Stage 2: the four encoder variants
# ============================================================
"""
S1 = f"{RUN}/structcoder/checkpoints_stage1/best_model.pt"
S2 = f"{RUN}/structcoder/checkpoints_stage2"

for m in ("vit", "tree", "combined", "tree_text"):
    !python -m train.train_full --s2_model {m} \
        --decoder structcoder {SC_FLAG} \
        --unfreeze_layers 12 --qwen_lr 1e-5 \
        --stage1_checkpoint {S1} \
        --s2_save_dir {S2}
"""

# The same thing as plain cells, if you prefer them one per variant:
"""
!python -m train.train_full --s2_model vit \
    --decoder structcoder {SC_FLAG} \
    --unfreeze_layers 12 \
    --stage1_checkpoint $RUN/structcoder/checkpoints_stage1/best_model.pt \
    --s2_save_dir $RUN/structcoder/checkpoints_stage2
!python -m train.train_full --s2_model tree      --decoder structcoder {SC_FLAG} --unfreeze_layers 12 --stage1_checkpoint $RUN/structcoder/checkpoints_stage1/best_model.pt --s2_save_dir $RUN/structcoder/checkpoints_stage2
!python -m train.train_full --s2_model combined  --decoder structcoder {SC_FLAG} --unfreeze_layers 12 --stage1_checkpoint $RUN/structcoder/checkpoints_stage1/best_model.pt --s2_save_dir $RUN/structcoder/checkpoints_stage2
!python -m train.train_full --s2_model tree_text \
    --decoder structcoder {SC_FLAG} \
    --unfreeze_layers 12 \
    --stage1_checkpoint $RUN/structcoder/checkpoints_stage1/best_model.pt \
    --s2_save_dir $RUN/structcoder/checkpoints_stage2
"""

# ============================================================
# CELL 5b (optional) — frozen decoder, projector only
# ============================================================
"""
# The capacity-asymmetry control: nothing in the decoder trains, so the only
# way to lower the loss is to make the prefix informative. train_full always
# picks LoRA when --unfreeze_layers is 0, so go through train_pipeline, where
# --lora is opt-in and omitting it leaves the decoder fully frozen.
for m in ("vit", "tree", "combined", "tree_text"):
    !python -m train.train_pipeline --model {m} \
        --decoder structcoder {SC_FLAG} \
        --unfreeze_layers 0 \
        --epochs 10 --lr 3e-4 --bottleneck 768 --dropout 0.05 --grad_accum 8 \
        --stage1_checkpoint $RUN/structcoder/checkpoints_stage1/best_model.pt \
        --save_dir $RUN/structcoder/frozen/{m}
"""

# ============================================================
# CELL 6 — evaluate, into a separate results dir
# ============================================================
"""
import os, subprocess
from pathlib import Path

SC_RESULTS = f"{RUN}/results_structcoder"     # keep away from the qwen eval_*.json

def evaluate(model_type, ckpt):
    out = f"{SC_RESULTS}/eval_{model_type}.json"
    if Path(out).exists():
        print(f"skip {model_type} (already evaluated)", flush=True); return
    os.makedirs(SC_RESULTS, exist_ok=True)
    cmd = ["python", "-m", "train.evaluate", "--model_type", model_type,
           "--checkpoint", ckpt, "--num_samples", "1000",
           "--decoder", "structcoder",
           *(["--structcoder_ckpt", SC_CKPT] if SC_CKPT else []),
           "--results_dir", SC_RESULTS]
    print(">>", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="")
    if p.wait():
        raise RuntimeError(f"eval failed: {model_type}")

evaluate("stage1", f"{RUN}/structcoder/checkpoints_stage1/best_model.pt")
for m in ("tree_text", "vit", "tree", "combined"):
    evaluate(m, f"{RUN}/structcoder/checkpoints_stage2/{m}/best_model.pt")

!python -m train.compare_results --results_dir $RUN/results_structcoder --baseline stage1
"""
