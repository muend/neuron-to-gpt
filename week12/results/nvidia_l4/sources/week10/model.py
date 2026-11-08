"""A small, explicit GPT-2 implementation with checked Hugging Face imports.

The forward pass, causal attention and sampling below are our own PyTorch
implementation. Transformers is used only by ``from_pretrained`` to obtain
the published weights. No Transformer library is required for random models.

Architecture references:
https://huggingface.co/openai-community/gpt2/blob/main/config.json
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    weight_tying: bool = True
    attn_implementation: str = "eager"

    def __post_init__(self):
        for name in ("block_size", "vocab_size", "n_layer", "n_head", "n_embd"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1)")
        if not isinstance(self.weight_tying, bool):
            raise ValueError("weight_tying must be boolean")
        if self.attn_implementation not in ("eager", "sdpa"):
            raise ValueError("attn_implementation must be 'eager' or 'sdpa'")


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.attn_implementation = config.attn_implementation
        # All heads' Q, K and V are calculated by one matrix multiplication.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "causal_mask",
            torch.ones(config.block_size, config.block_size, dtype=torch.bool).tril(),
            persistent=False,
        )

    def forward(self, x):
        batch, time, channels = x.shape
        query, key, value = self.c_attn(x).split(self.n_embd, dim=-1)
        head_size = channels // self.n_head
        query = query.view(batch, time, self.n_head, head_size).transpose(1, 2)
        key = key.view(batch, time, self.n_head, head_size).transpose(1, 2)
        value = value.view(batch, time, self.n_head, head_size).transpose(1, 2)
        if self.attn_implementation == "sdpa":
            attended = F.scaled_dot_product_attention(
                query, key, value, is_causal=True,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            scores = (query @ key.transpose(-2, -1)) * (head_size ** -0.5)
            scores = scores.masked_fill(
                ~self.causal_mask[:time, :time], torch.finfo(scores.dtype).min
            )
            weights = self.attn_dropout(F.softmax(scores, dim=-1))
            attended = weights @ value
        attended = attended.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.resid_dropout(self.c_proj(attended))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=True)
        # GPT-2's gelu_new is the tanh approximation, rather than ReLU.
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=True)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, config: GPTConfig = GPTConfig()):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "wpe": nn.Embedding(config.block_size, config.n_embd),
            "drop": nn.Dropout(config.dropout),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            "ln_f": nn.LayerNorm(config.n_embd, eps=1e-5),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.weight_tying:
            self.lm_head.weight = self.transformer.wte.weight
        self.apply(self._init_weights)
        # Each block adds two branches to the residual stream. The GPT-2 /
        # lecture initialization compensates for their accumulated variance.
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def num_parameters(self) -> int:
        """Count unique parameters; the tied embedding is counted only once."""
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, idx, targets=None):
        """Predict next-token logits; supplied targets are already shifted by one.

        Input and target tensors both have shape (batch, time). Unlike the
        Hugging Face ``labels`` convenience API, this method does not shift
        labels internally, so callers supply x=tokens[:-1], y=tokens[1:].
        """
        if idx.ndim != 2 or not idx.shape[0] or not idx.shape[1]:
            raise ValueError("idx must have nonempty (batch, time) dimensions")
        if idx.dtype not in (torch.int32, torch.int64):
            raise TypeError("idx must contain integer token IDs")
        _, time = idx.shape
        if time > self.config.block_size:
            raise ValueError(f"Input length {time} exceeds block_size {self.config.block_size}")
        if targets is not None and targets.shape != idx.shape:
            raise ValueError("targets must have the same shape as idx")
        positions = torch.arange(time, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(positions))
        for block in self.transformer.h:
            x = block(x)
        logits = self.lm_head(self.transformer.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.config.vocab_size), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def load_hf_weights(self, hf_model):
        """Copy compatible GPT2LMHeadModel weights, checking every name and shape.

        Hugging Face's historical Conv1D stores weights as (input, output).
        nn.Linear stores (output, input), so four projection types per block
        must be transposed, including the square attention output matrix.
        Returns an audit dictionary of copied and transposed state entries.
        """
        source_config = hf_model.config
        for local_name, hf_name in (
            ("vocab_size", "vocab_size"), ("block_size", "n_positions"),
            ("n_layer", "n_layer"), ("n_head", "n_head"), ("n_embd", "n_embd"),
        ):
            if getattr(self.config, local_name) != getattr(source_config, hf_name):
                raise ValueError(f"Hugging Face {hf_name} does not match our {local_name}")
        if (
            getattr(source_config, "model_type", None) != "gpt2"
            or getattr(source_config, "activation_function", None) not in ("gelu_new", "gelu_pytorch_tanh")
            or getattr(source_config, "n_inner", None) not in (None, 4 * self.config.n_embd)
            or getattr(source_config, "layer_norm_epsilon", None) != 1e-5
            or not getattr(source_config, "scale_attn_weights", True)
            or getattr(source_config, "scale_attn_by_inverse_layer_idx", False)
            or getattr(source_config, "reorder_and_upcast_attn", False)
            or getattr(source_config, "add_cross_attention", False)
        ):
            raise ValueError("Unsupported GPT-2 architecture settings in Hugging Face model")
        local = self.state_dict()
        source = {
            name: tensor for name, tensor in hf_model.state_dict().items()
            if not name.endswith((".attn.bias", ".attn.masked_bias"))
        }
        if set(local) != set(source):
            missing = sorted(set(local) - set(source))
            unexpected = sorted(set(source) - set(local))
            raise ValueError(f"State keys differ: missing={missing}, unexpected={unexpected}")
        transposed_suffixes = (
            ".attn.c_attn.weight", ".attn.c_proj.weight",
            ".mlp.c_fc.weight", ".mlp.c_proj.weight",
        )
        # Validate before copying, so a late shape error cannot partially load.
        transposed = []
        for name, destination in local.items():
            tensor = source[name]
            if name.endswith(transposed_suffixes):
                tensor = tensor.t()
                transposed.append(name)
            if destination.shape != tensor.shape:
                raise ValueError(f"Shape differs for {name}: {tuple(tensor.shape)} vs {tuple(destination.shape)}")
        if self.config.weight_tying and not torch.equal(
            source["transformer.wte.weight"], source["lm_head.weight"]
        ):
            raise ValueError("Cannot import distinct input/output weights into a tied model")
        for name, destination in local.items():
            tensor = source[name].t() if name.endswith(transposed_suffixes) else source[name]
            destination.copy_(tensor)
        return {"copied_tensors": len(local), "transposed_tensors": transposed,
                "unique_parameters": self.num_parameters()}

    @classmethod
    def from_pretrained(
        cls, model_name="openai-community/gpt2", *, local_files_only=False,
        cache_dir: str | Path | None = None, revision: str | None = None,
    ):
        """Load published GPT-2 weights through HF; return our model in eval mode."""
        from transformers import GPT2LMHeadModel

        kwargs = {"local_files_only": local_files_only, "attn_implementation": "eager",
                  "torch_dtype": torch.float32, "use_safetensors": True}
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)
        if revision is not None:
            kwargs["revision"] = revision
        reference = GPT2LMHeadModel.from_pretrained(model_name, **kwargs).eval()
        source = reference.config
        dropouts = (source.embd_pdrop, source.attn_pdrop, source.resid_pdrop)
        if len(set(dropouts)) != 1:
            raise ValueError("Our GPTConfig supports a shared dropout rate only")
        model = cls(GPTConfig(
            block_size=source.n_positions, vocab_size=source.vocab_size,
            n_layer=source.n_layer, n_head=source.n_head, n_embd=source.n_embd,
            dropout=source.resid_pdrop, weight_tying=source.tie_word_embeddings,
        ))
        model.load_hf_weights(reference)
        return model.eval()

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=50, generator=None):
        """Sample from the latest context using our own top-k sampling loop.

        Pass a torch.Generator on the model's device for reproducibility.
        Absolute positions restart at zero after cropping, as in the lecture's
        simple sampler; no KV cache or advanced long-context policy is used.
        """
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if top_k is not None and (
            isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1
        ):
            raise ValueError("top_k must be a positive integer or None")
        if idx.ndim != 2 or not idx.shape[0] or not idx.shape[1]:
            raise ValueError("Generation needs nonempty (batch, time) starting tokens")
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                logits, _ = self(idx[:, -self.config.block_size:])
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    values, _ = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1)
                    logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
                probabilities = F.softmax(logits, dim=-1)
                sampled = torch.multinomial(probabilities, num_samples=1, generator=generator)
                idx = torch.cat((idx, sampled), dim=1)
        finally:
            self.train(was_training)
        return idx
