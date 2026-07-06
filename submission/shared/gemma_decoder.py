# shared/gemma_decoder.py
"""
Gemma Decoder

Frozen Gemma LLM that generates text from projected embeddings.

Training: [projected patches] + [PROMPT] + [target text] → loss (only on target)
Inference: [projected patches] + [PROMPT] → generated text

NOTE: Gemma internally scales inputs_embeds by sqrt(hidden_size) ≈ 45.
The projector learns to produce values at the right pre-scaling magnitude.
"""
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Task prompt - same for all samples (train & test)
PROMPT = "Convert the following MATLAB code to step-by-step pseudocode:\n"


class GemmaDecoder:
    """
    Frozen Gemma decoder for text generation.

    Training mode:
        - Concatenates projected embeddings with target text
        - Computes next-token prediction loss (only on text tokens)

    Inference mode:
        - Takes projected embeddings as prompt
        - Generates text autoregressively
    """

    def __init__(self, model_name: str = "google/gemma-2b", device: str = None):
        self.device = device or DEVICE

        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()
        self.hidden_size = self.model.config.hidden_size

        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.lora_enabled = False
        self.unfrozen_enabled = False
        self.unfrozen_layers = []

        print(f"Gemma loaded on {self.device} (frozen)")

    def enable_lora(self, rank: int = 16, alpha: int = 32, dropout: float = 0.05, num_layers: int = 4):
        """Apply LoRA adapters to the last `num_layers` Gemma layers."""
        total_layers = self.model.config.num_hidden_layers  # 18 for Gemma-2B
        target_layers = list(range(total_layers - num_layers, total_layers))
        target_modules = [
            f"model.layers.{i}.self_attn.{proj}"
            for i in target_layers
            for proj in ("q_proj", "v_proj")
        ]

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
        )

        self.model = get_peft_model(self.model, lora_config)
        self.lora_enabled = True
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"LoRA enabled: {trainable:,} trainable params on layers {target_layers}")

    def get_lora_parameters(self):
        """Return LoRA trainable parameters."""
        if not self.lora_enabled:
            return []
        return [p for p in self.model.parameters() if p.requires_grad]

    def get_lora_state_dict(self):
        """Return LoRA adapter state dict for checkpointing."""
        if not self.lora_enabled:
            return {}
        return {
            k: v for k, v in self.model.state_dict().items()
            if "lora_" in k
        }

    def load_lora_state_dict(self, state_dict):
        """Load LoRA adapter weights from checkpoint."""
        if not self.lora_enabled or not state_dict:
            return
        self.model.load_state_dict(state_dict, strict=False)
        print(f"Loaded LoRA state ({len(state_dict)} tensors)")

    def unfreeze_layers(self, num_layers: int = 9):
        """Unfreeze the last `num_layers` Gemma transformer layers for full fine-tuning."""
        total_layers = self.model.config.num_hidden_layers
        target_layers = list(range(total_layers - num_layers, total_layers))

        unfrozen_params = 0
        for i in target_layers:
            for param in self.model.model.layers[i].parameters():
                param.requires_grad = True
                unfrozen_params += param.numel()

        # Also unfreeze final norm and lm_head — critical for generation quality
        for param in self.model.model.norm.parameters():
            param.requires_grad = True
            unfrozen_params += param.numel()
        for param in self.model.lm_head.parameters():
            param.requires_grad = True
            unfrozen_params += param.numel()

        self.unfrozen_layers = target_layers
        self.unfrozen_enabled = True
        print(f"Unfrozen layers {target_layers[0]}-{target_layers[-1]} + norm + lm_head: "
              f"{unfrozen_params:,} trainable Gemma params")

    def get_unfrozen_parameters(self):
        """Return unfrozen Gemma parameters."""
        if not self.unfrozen_enabled:
            return []
        return [p for p in self.model.parameters() if p.requires_grad]

    def get_unfrozen_state_dict(self):
        """Return state dict of unfrozen Gemma components for checkpointing."""
        if not self.unfrozen_enabled:
            return {}
        state = {}
        for k, v in self.model.state_dict().items():
            if any(f'layers.{i}.' in k for i in self.unfrozen_layers):
                state[k] = v
            elif k.startswith('model.norm') or k.startswith('lm_head'):
                state[k] = v
        return state

    def load_unfrozen_state_dict(self, state_dict):
        """Load unfrozen Gemma layer weights from checkpoint."""
        if not self.unfrozen_enabled or not state_dict:
            return
        self.model.load_state_dict(state_dict, strict=False)
        print(f"Loaded unfrozen Gemma state ({len(state_dict)} tensors)")

    def train_mode(self):
        """Set Gemma to train mode (needed for dropout in unfrozen layers / LoRA)."""
        if self.lora_enabled or self.unfrozen_enabled:
            self.model.train()

    def eval_mode(self):
        """Set Gemma to eval mode."""
        self.model.eval()

    def get_input_embeddings(self, text: str, max_length: int = 512) -> torch.Tensor:
        """Get Gemma's embeddings for text tokens."""
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(self.device)

        embeds = self.model.get_input_embeddings()(tokens.input_ids)
        return embeds, tokens

    def forward_train(
        self,
        projected: torch.Tensor,    # [num_patches, D]
        target_text: str,
    ) -> torch.Tensor:
        """
        Training forward pass.

        Args:
            projected: [num_patches, D] from Projector
            target_text: ground truth pseudocode

        Returns:
            loss: cross-entropy loss on target text tokens only
        """
        # Add batch dimension: [num_patches, D] → [1, num_patches, D]
        projected = projected.unsqueeze(0)

        # Get prompt embeddings
        prompt_embeds, prompt_tokens = self.get_input_embeddings(PROMPT)

        # Get target embeddings
        target_embeds, target_tokens = self.get_input_embeddings(target_text)

        # Match dtype
        projected = projected.to(target_embeds.dtype)

        # Concatenate: [patches] + [prompt] + [target text]
        input_embeds = torch.cat([projected, prompt_embeds, target_embeds], dim=1)

        # Create attention mask
        num_patches = projected.shape[1]
        num_prompt = prompt_embeds.shape[1]
        patch_mask = torch.ones(1, num_patches, device=self.device)
        attn_mask = torch.cat([
            patch_mask,
            prompt_tokens.attention_mask,
            target_tokens.attention_mask
        ], dim=1)

        # Create labels
        # -100 = ignore (don't compute loss on patches or prompt)
        # We only predict target text tokens
        patch_labels = torch.full(
            (1, num_patches), -100, dtype=torch.long, device=self.device
        )
        prompt_labels = torch.full(
            (1, num_prompt), -100, dtype=torch.long, device=self.device
        )
        target_labels = target_tokens.input_ids.clone()
        labels = torch.cat([patch_labels, prompt_labels, target_labels], dim=1)

        # Forward through Gemma
        outputs = self.model(
            inputs_embeds=input_embeds,
            attention_mask=attn_mask,
            labels=labels,
        )

        return outputs.loss

    def forward_train_text(self, matlab_code: str, target_text: str) -> torch.Tensor:
        """
        Training forward pass using raw text (no projected embeddings).

        Used in Stage 1 training to fine-tune the decoder on plain
        MATLAB→pseudocode pairs before the encoder pipeline is trained
        against it.

        Args:
            matlab_code: raw MATLAB source code string
            target_text: ground truth pseudocode

        Returns:
            loss: cross-entropy loss on target text tokens only
        """
        # Tokenize MATLAB code with larger max_length to fit full functions
        code_tokens = self.tokenizer(
            matlab_code,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)
        code_embeds = self.model.get_input_embeddings()(code_tokens.input_ids)

        # Get prompt embeddings
        prompt_embeds, prompt_tokens = self.get_input_embeddings(PROMPT)

        # Get target embeddings
        target_embeds, target_tokens = self.get_input_embeddings(target_text)

        # Concatenate: [code] + [prompt] + [target text]
        input_embeds = torch.cat([code_embeds, prompt_embeds, target_embeds], dim=1)

        # Create attention mask
        num_code = code_embeds.shape[1]
        num_prompt = prompt_embeds.shape[1]
        attn_mask = torch.cat([
            code_tokens.attention_mask,
            prompt_tokens.attention_mask,
            target_tokens.attention_mask,
        ], dim=1)

        # Create labels: -100 for code and prompt, ground truth IDs for target
        code_labels = torch.full(
            (1, num_code), -100, dtype=torch.long, device=self.device
        )
        prompt_labels = torch.full(
            (1, num_prompt), -100, dtype=torch.long, device=self.device
        )
        target_labels = target_tokens.input_ids.clone()
        labels = torch.cat([code_labels, prompt_labels, target_labels], dim=1)

        # Forward through Gemma
        outputs = self.model(
            inputs_embeds=input_embeds,
            attention_mask=attn_mask,
            labels=labels,
        )

        return outputs.loss

    @torch.no_grad()
    def generate(
        self,
        projected: torch.Tensor,    # [num_patches, D]
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate text from projected embeddings.

        Greedy by default so evaluation metrics are deterministic;
        pass do_sample=True for sampled generation.

        Args:
            projected: [num_patches, D] from Projector
            max_new_tokens: max tokens to generate
            do_sample: enable nucleus sampling (default: greedy)
            temperature: sampling temperature (ignored when greedy)
            top_p: nucleus sampling threshold (ignored when greedy)

        Returns:
            generated text string
        """
        # Add batch dimension and match dtype
        projected = projected.unsqueeze(0).to(self.model.dtype)

        # Get prompt embeddings
        prompt_embeds, prompt_tokens = self.get_input_embeddings(PROMPT)

        # Concatenate: [patches] + [prompt]
        input_embeds = torch.cat([projected, prompt_embeds], dim=1)

        # Build attention mask for full prefix (all 1s — no padding)
        num_patches = projected.shape[1]
        patch_mask = torch.ones(1, num_patches, device=self.device)
        attention_mask = torch.cat([patch_mask, prompt_tokens.attention_mask], dim=1)

        # Generate
        sampling_kwargs = (
            {"do_sample": True, "temperature": temperature, "top_p": top_p}
            if do_sample else {"do_sample": False}
        )
        outputs = self.model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            **sampling_kwargs,
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    @torch.no_grad()
    def generate_with_metrics(
        self,
        projected: torch.Tensor,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> tuple:
        """Generate text and return (text, metrics_dict). Greedy by default."""
        projected = projected.unsqueeze(0).to(self.model.dtype)
        prompt_embeds, prompt_tokens = self.get_input_embeddings(PROMPT)
        input_embeds = torch.cat([projected, prompt_embeds], dim=1)
        num_input_tokens = input_embeds.shape[1]

        num_patches = projected.shape[1]
        patch_mask = torch.ones(1, num_patches, device=self.device)
        attention_mask = torch.cat([patch_mask, prompt_tokens.attention_mask], dim=1)

        if self.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        sampling_kwargs = (
            {"do_sample": True, "temperature": temperature, "top_p": top_p}
            if do_sample else {"do_sample": False}
        )
        outputs = self.model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            **sampling_kwargs,
        )

        if self.device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        num_generated = outputs.shape[1]

        # KV cache size: 2 (K+V) x layers x kv_heads x head_dim x seq_len x 2 bytes (fp16)
        config = self.model.config
        num_layers = config.num_hidden_layers
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = config.hidden_size // config.num_attention_heads
        total_seq_len = num_input_tokens + num_generated
        kv_cache_bytes = 2 * num_layers * num_kv_heads * head_dim * total_seq_len * 2

        generate_time = t1 - t0

        return text, {
            "num_input_tokens": num_input_tokens,
            "num_generated_tokens": num_generated,
            "generate_time_s": round(generate_time, 4),
            "tokens_per_sec": round(num_generated / generate_time, 1) if generate_time > 0 else 0,
            "kv_cache_mb": round(kv_cache_bytes / (1024**2), 2),
        }


if __name__ == "__main__":
    # Test
    print("GemmaDecoder Test")
    print("=" * 60)

    decoder = GemmaDecoder()

    # Fake projected embeddings (normally from Projector)
    num_patches = 3
    projected = torch.randn(num_patches, decoder.hidden_size, device=DEVICE)

    print(f"\n  Input projected: {projected.shape}")

    # Test training forward
    print("\n  Testing training forward...")
    target = "This function doubles the input if positive."
    loss = decoder.forward_train(projected, target)
    print(f"  Loss: {loss.item():.4f}")

    # Test generation
    print("\n  Testing generation...")
    output = decoder.generate(projected, max_new_tokens=32)
    print(f"  Generated: {output[:100]}...")
