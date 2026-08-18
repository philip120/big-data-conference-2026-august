# shared/structcoder_decoder.py
"""
StructCoder decoder — encoder-decoder (seq2seq) alternative to QwenDecoder.

Why a seq2seq decoder at all
---------------------------
Qwen is decoder-only, so the encoder pipeline conditions it by *prepending*
soft tokens to the prompt: [soft][prompt][target]. Two known failure modes
follow from that shape (see CLAUDE.md):

  * the decoder is 4B params with ~2.2B trainable against a ~5-token prefix,
    so memorizing the unconditional pseudocode prior is cheaper than reading
    the prefix;
  * the prompt says "Convert the following MATLAB code…" and nothing follows
    it, so the instruction points at nothing.

A seq2seq decoder removes both. The soft tokens are the *encoder input*: the
T5 encoder contextualizes them and every decoder layer cross-attends to them.
There is no path that produces the target without reading the encoder side,
and at 220M params (CodeT5-base scale) the capacity asymmetry is ~10x
smaller. As a bonus, the T5 tokenizer appends `</s>`, so the missing-EOS bug
that made all four Qwen variants run to the decode cap does not exist here.

What "StructCoder" means in this file
-------------------------------------
StructCoder (Tipirneni et al., TKDD 2024) is a CodeT5-base-shaped seq2seq
whose *encoder* is structure-aware (AST paths + data-flow graph, built with
tree-sitter) and whose decoder is trained with auxiliary AST/DFG prediction
heads. Only the decoder half is usable here:

  * the structure-aware encoder needs a tree-sitter grammar for the source
    language, and there is no MATLAB grammar in the released parser blobs —
    and in this pipeline the encoder is *our* ViT/tree/combined model anyway;
  * the auxiliary heads predict the AST and data flow of the *target*, and
    our target is English pseudocode, which has neither.

So this loads the released StructCoder checkpoint and keeps its
`shared` / `decoder` / `lm_head` tensors — the half that was pretrained with
the structure-based denoising objective — on top of a stock
`T5ForConditionalGeneration` with the CodeT5-base config. Their custom
`modeling_structcoder.py` is deliberately not imported: it subclasses
`T5Attention` against the 2022 transformers API and only the encoder stack
(which we do not use) depends on it.

Without a checkpoint this falls back to plain `Salesforce/codet5-base` and
says so loudly. That fallback is a legitimate ablation — it isolates how much
the structure-aware pretraining is worth — but it is not StructCoder, and the
banner exists so it never gets reported as if it were.

Checkpoint: https://github.com/reddy-lab-code-research/StructCoder
(weights are a Google Drive download; point --structcoder_ckpt or
$STRUCTCODER_CKPT at the downloaded file).
"""
import os
import time
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from peft import get_peft_model, LoraConfig, TaskType

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Same task prompt as the Qwen path. Here it precedes the soft tokens on the
# encoder side, so unlike the decoder-only path it actually points at them.
PROMPT = "Convert the following MATLAB code to step-by-step pseudocode:\n"

BASE_MODEL = "Salesforce/codet5-base"

# CodeT5's special_tokens_map.json stores additional_special_tokens as bare
# dicts. transformers <5 coerced those to AddedToken; transformers >=5 does
# not, and hands the raw dicts to add_tokens(), which raises
# "Input must be a List[Union[str, AddedToken]]" — so the tokenizer cannot be
# loaded at all on current Colab. Passing the same 100 sentinels as plain
# strings overrides the file and loads on both. They are already in the vocab
# (ids 32000-32099 of 32100), so nothing is added and no embedding resize
# happens.
SENTINEL_TOKENS = [f"<extra_id_{i}>" for i in range(99, -1, -1)]

# Checkpoint tensors worth keeping: everything downstream of the encoder.
# The StructCoder encoder stack, the DFG/AST embeddings and the auxiliary
# heads are dropped — see the module docstring.
_KEEP_PREFIXES = ("shared.", "decoder.", "lm_head.")


def _find_checkpoint(explicit: str = None) -> str:
    """Resolve the StructCoder checkpoint path, or None to fall back."""
    for cand in (explicit, os.environ.get("STRUCTCODER_CKPT"),
                 "saved_models/pretrain/pytorch_model.bin"):
        if cand and os.path.exists(cand):
            return cand
    return None


class StructCoderDecoder:
    """
    Seq2seq decoder with the QwenDecoder duck-type.

    Interface parity with QwenDecoder (what model/*, train/* call):
        hidden_size, tokenizer, model, device
        enable_lora / get_lora_parameters / get_lora_state_dict / load_lora_state_dict
        unfreeze_layers / get_unfrozen_parameters / get_unfrozen_state_dict
            / load_unfrozen_state_dict
        train_mode / eval_mode
        get_input_embeddings
        forward_train / forward_train_text
        generate / generate_with_metrics

    Semantic difference to keep in mind: `projected` is the *encoder input*,
    not a prefix spliced into the decoder stream.
    """

    def __init__(self, model_name: str = BASE_MODEL, device: str = None,
                 checkpoint: str = None):
        self.device = device or DEVICE

        print(f"Loading {model_name} (StructCoder decoder base)...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, additional_special_tokens=SENTINEL_TOKENS
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        )

        ckpt_path = _find_checkpoint(checkpoint)
        if ckpt_path:
            self._load_structcoder_weights(ckpt_path)
        else:
            print("=" * 60)
            print("WARNING: no StructCoder checkpoint found — running plain")
            print(f"         {model_name}. Set --structcoder_ckpt or")
            print("         $STRUCTCODER_CKPT to use the real weights.")
            print("         Do NOT report this run as StructCoder.")
            print("=" * 60)
        self.structcoder_weights_loaded = bool(ckpt_path)

        self.model.to(self.device)
        self.model.eval()
        self.hidden_size = self.model.config.d_model

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        for param in self.model.parameters():
            param.requires_grad = False

        self.lora_enabled = False
        self.unfrozen_enabled = False
        self.unfrozen_layers = []

        n_dec = self.model.config.num_decoder_layers
        print(f"StructCoder decoder loaded on {self.device} (frozen), "
              f"d_model={self.hidden_size}, {n_dec} decoder layers")

    # ------------------------------------------------------------------
    # checkpoint
    # ------------------------------------------------------------------

    def _load_structcoder_weights(self, path: str):
        """Copy the released StructCoder decoder/embedding/lm_head tensors in."""
        print(f"Loading StructCoder weights from {path}...")
        raw = torch.load(path, map_location="cpu", weights_only=False)
        # Their training script saves either a bare state_dict or a dict with
        # the state under a key; accept both.
        if not isinstance(raw, dict):
            raise ValueError(f"Unexpected checkpoint object in {path}: {type(raw)}")
        state = raw
        for key in ("model", "state_dict", "model_state_dict"):
            if key in raw and isinstance(raw[key], dict):
                state = raw[key]
                break
        # DataParallel checkpoints carry a "module." prefix.
        state = {k[len("module."):] if k.startswith("module.") else k: v
                 for k, v in state.items()}

        target = self.model.state_dict()
        copied, skipped = 0, []
        for k, v in state.items():
            if not k.startswith(_KEEP_PREFIXES):
                continue
            if k not in target:
                continue
            if target[k].shape != v.shape:
                skipped.append(f"{k}: model={tuple(target[k].shape)} ckpt={tuple(v.shape)}")
                continue
            target[k].copy_(v.to(target[k].dtype))
            copied += 1
        self.model.load_state_dict(target)

        print(f"  Loaded {copied} StructCoder tensors "
              f"(decoder + shared embeddings + lm_head)")
        if skipped:
            print(f"  WARNING: skipped {len(skipped)} shape-mismatched tensors:")
            for s in skipped[:10]:
                print(f"    {s}")
        if copied == 0:
            raise ValueError(
                f"No usable tensors in {path}. Expected keys prefixed with "
                f"{_KEEP_PREFIXES}; got e.g. {list(state)[:5]}"
            )

    # ------------------------------------------------------------------
    # adaptation: LoRA
    # ------------------------------------------------------------------

    def enable_lora(self, rank: int = 16, alpha: int = 32, dropout: float = 0.05,
                    num_layers: int = 6):
        """LoRA on the last `num_layers` decoder blocks (self- and cross-attn)."""
        total = self.model.config.num_decoder_layers
        num_layers = min(num_layers, total)
        target_layers = list(range(total - num_layers, total))
        target_modules = [
            f"decoder.block.{i}.layer.{sub}.{attn}.{proj}"
            for i in target_layers
            # layer.0 = self-attention, layer.1 = cross-attention
            for sub, attn in ((0, "SelfAttention"), (1, "EncDecAttention"))
            for proj in ("q", "v")
        ]

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
        )
        self.model = get_peft_model(self.model, lora_config)
        self.lora_enabled = True
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"LoRA enabled: {trainable:,} trainable params on decoder blocks {target_layers}")

    def get_lora_parameters(self):
        if not self.lora_enabled:
            return []
        return [p for p in self.model.parameters() if p.requires_grad]

    def get_lora_state_dict(self):
        if not self.lora_enabled:
            return {}
        return {k: v for k, v in self.model.state_dict().items() if "lora_" in k}

    def load_lora_state_dict(self, state_dict):
        if not self.lora_enabled or not state_dict:
            return
        self.model.load_state_dict(state_dict, strict=False)
        print(f"Loaded LoRA state ({len(state_dict)} tensors)")

    # ------------------------------------------------------------------
    # adaptation: full fine-tune
    # ------------------------------------------------------------------

    def unfreeze_layers(self, num_layers: int = 6):
        """
        Unfreeze the last `num_layers` decoder blocks + final norm + lm_head.

        CodeT5-base has 12 decoder blocks, so the Qwen-tuned default of 18 is
        clamped rather than silently doing nothing.
        """
        total = self.model.config.num_decoder_layers
        if num_layers > total:
            print(f"  Note: --unfreeze_layers {num_layers} > {total} decoder blocks; "
                  f"unfreezing all {total}.")
            num_layers = total
        target_layers = list(range(total - num_layers, total))

        unfrozen = 0
        for i in target_layers:
            for p in self.model.decoder.block[i].parameters():
                p.requires_grad = True
                unfrozen += p.numel()
        for p in self.model.decoder.final_layer_norm.parameters():
            p.requires_grad = True
            unfrozen += p.numel()
        for p in self.model.lm_head.parameters():
            p.requires_grad = True
            unfrozen += p.numel()

        self.unfrozen_layers = target_layers
        self.unfrozen_enabled = True
        print(f"Unfrozen decoder blocks {target_layers[0]}-{target_layers[-1]} "
              f"+ final_layer_norm + lm_head: {unfrozen:,} trainable params")

    def get_unfrozen_parameters(self):
        if not self.unfrozen_enabled:
            return []
        return [p for p in self.model.parameters() if p.requires_grad]

    def get_unfrozen_state_dict(self):
        if not self.unfrozen_enabled:
            return {}
        state = {}
        for k, v in self.model.state_dict().items():
            if any(f"decoder.block.{i}." in k for i in self.unfrozen_layers):
                state[k] = v
            elif "decoder.final_layer_norm" in k or k.startswith("lm_head"):
                state[k] = v
        return state

    def load_unfrozen_state_dict(self, state_dict):
        if not self.unfrozen_enabled or not state_dict:
            return
        self.model.load_state_dict(state_dict, strict=False)
        print(f"Loaded unfrozen decoder state ({len(state_dict)} tensors)")

    @staticmethod
    def count_unfrozen_layers(state_dict) -> int:
        """
        How many decoder blocks a saved `qwen_state` covers.

        evaluate.py has to rebuild the same unfrozen set before loading, and
        the key layout differs per decoder — Qwen uses `model.layers.{i}.`,
        T5 uses `decoder.block.{i}.`.
        """
        idxs = set()
        for k in state_dict:
            parts = k.split(".")
            for a, b in zip(parts, parts[1:]):
                if a == "block" and b.isdigit():
                    idxs.add(int(b))
        return len(idxs)

    # ------------------------------------------------------------------
    # modes / embeddings
    # ------------------------------------------------------------------

    def train_mode(self):
        if self.lora_enabled or self.unfrozen_enabled:
            self.model.train()

    def eval_mode(self):
        self.model.eval()

    def get_input_embeddings(self, text: str, max_length: int = 512):
        """Token embeddings for `text` in the encoder's input space."""
        tokens = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=max_length, padding=True,
        ).to(self.device)
        embeds = self.model.get_input_embeddings()(tokens.input_ids)
        return embeds, tokens

    def _encoder_side(self, projected: torch.Tensor):
        """[PROMPT embeds][projected soft tokens] + attention mask."""
        projected = projected.unsqueeze(0)
        prompt_embeds, prompt_tokens = self.get_input_embeddings(PROMPT)
        projected = projected.to(prompt_embeds.dtype)

        inputs_embeds = torch.cat([prompt_embeds, projected], dim=1)
        soft_mask = torch.ones(1, projected.shape[1], device=self.device,
                               dtype=prompt_tokens.attention_mask.dtype)
        attn_mask = torch.cat([prompt_tokens.attention_mask, soft_mask], dim=1)
        return inputs_embeds, attn_mask

    def _labels(self, target_text: str, max_length: int = 512):
        """Target ids with pad masked to -100. The T5 tokenizer appends </s>."""
        tokens = self.tokenizer(
            target_text, return_tensors="pt", truncation=True,
            max_length=max_length, padding=True,
        ).to(self.device)
        labels = tokens.input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return labels

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------

    def forward_train(self, projected: torch.Tensor, target_text: str,
                      max_target_length: int = 512) -> torch.Tensor:
        """
        Stage-2 forward: soft tokens are the encoder input, target is decoded
        by cross-attention. Loss is over target tokens only (T5 shifts the
        decoder inputs internally from `labels`).
        """
        inputs_embeds, attn_mask = self._encoder_side(projected)
        labels = self._labels(target_text, max_length=max_target_length)

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            labels=labels,
        )
        return outputs.loss

    def forward_train_text(self, matlab_code: str, target_text: str,
                           max_source_length: int = 512,
                           max_target_length: int = 512) -> torch.Tensor:
        """Stage-1 forward: raw MATLAB text in the encoder, no soft tokens."""
        source = PROMPT + matlab_code
        tokens = self.tokenizer(
            source, return_tensors="pt", truncation=True,
            max_length=max_source_length, padding=True,
        ).to(self.device)
        labels = self._labels(target_text, max_length=max_target_length)

        outputs = self.model(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
            labels=labels,
        )
        return outputs.loss

    # ------------------------------------------------------------------
    # generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, projected: torch.Tensor, max_new_tokens: int = 128,
                 do_sample: bool = False, temperature: float = 0.7,
                 top_p: float = 0.9) -> str:
        """Greedy by default so evaluation metrics stay deterministic."""
        text, _ = self.generate_with_metrics(
            projected, max_new_tokens=max_new_tokens, do_sample=do_sample,
            temperature=temperature, top_p=top_p,
        )
        return text

    @torch.no_grad()
    def generate_with_metrics(self, projected: torch.Tensor,
                              max_new_tokens: int = 128, do_sample: bool = False,
                              temperature: float = 0.7, top_p: float = 0.9) -> tuple:
        inputs_embeds, attn_mask = self._encoder_side(
            projected.to(self.model.dtype)
        )
        num_input_tokens = inputs_embeds.shape[1]

        if self.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        sampling_kwargs = (
            {"do_sample": True, "temperature": temperature, "top_p": top_p}
            if do_sample else {"do_sample": False}
        )
        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **sampling_kwargs,
        )

        if self.device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Encoder-decoder: `outputs` is decoder-side only, including the
        # start token — unlike the decoder-only path, no prefix to strip.
        num_generated = outputs.shape[1]

        generate_time = t1 - t0
        return text, {
            "num_input_tokens": num_input_tokens,
            "num_generated_tokens": num_generated,
            "generate_time_s": round(generate_time, 4),
            "tokens_per_sec": round(num_generated / generate_time, 1) if generate_time > 0 else 0,
            "kv_cache_mb": round(
                self._kv_cache_bytes(num_input_tokens, num_generated) / (1024 ** 2), 2
            ),
        }

    def _kv_cache_bytes(self, num_input_tokens: int, num_generated: int) -> int:
        """
        Seq2seq KV cache = decoder self-attention (grows with generation)
        + cross-attention (fixed at encoder length, cached once).

        2 (K+V) x layers x heads x d_kv x len x 2 bytes (bf16).
        """
        cfg = self.model.config
        per_token = 2 * cfg.num_decoder_layers * cfg.num_heads * cfg.d_kv * 2
        return per_token * num_generated + per_token * num_input_tokens

    @torch.no_grad()
    def generate_text(self, code: str, max_new_tokens: int = 128,
                      max_source_length: int = 512) -> tuple:
        """
        Stage-1 generation from raw code (no encoder pipeline).

        Exists because evaluate.generate_stage1's decoder-only logic — slice
        off the prompt, read config.num_key_value_heads — is wrong for a
        seq2seq model.
        """
        source = PROMPT + code + "\nPseudocode:"
        tokens = self.tokenizer(
            source, return_tensors="pt", truncation=True, max_length=max_source_length
        ).to(self.device)
        num_input_tokens = tokens.input_ids.shape[1]

        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        outputs = self.model.generate(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        if self.device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        num_generated = outputs.shape[1]
        gen_time = t1 - t0

        peak_vram_mb = 0.0
        if self.device == "cuda":
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        return text, {
            "encode_time_s": 0.0,
            "generate_time_s": round(gen_time, 4),
            "total_time_s": round(gen_time, 4),
            "num_input_tokens": num_input_tokens,
            "num_generated_tokens": num_generated,
            "tokens_per_sec": round(num_generated / gen_time, 1) if gen_time > 0 else 0,
            "kv_cache_mb": round(
                self._kv_cache_bytes(num_input_tokens, num_generated) / (1024 ** 2), 2
            ),
            "peak_vram_mb": round(peak_vram_mb, 1),
        }


if __name__ == "__main__":
    print("StructCoderDecoder Test")
    print("=" * 60)

    decoder = StructCoderDecoder()

    num_patches = 3
    projected = torch.randn(num_patches, decoder.hidden_size, device=DEVICE)
    print(f"\n  Input projected: {projected.shape}")

    print("\n  Testing training forward...")
    target = "This function doubles the input if positive."
    loss = decoder.forward_train(projected, target)
    print(f"  Loss: {loss.item():.4f}")

    print("\n  Testing text-only (stage 1) forward...")
    loss = decoder.forward_train_text("function y = f(x)\ny = 2*x;\nend", target)
    print(f"  Loss: {loss.item():.4f}")

    print("\n  Testing generation...")
    output, metrics = decoder.generate_with_metrics(projected, max_new_tokens=32)
    print(f"  Generated: {output[:100]}...")
    print(f"  Metrics: {metrics}")
