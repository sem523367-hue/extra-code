"""Llama-style decoder-only transformer, sized for one consumer GPU.

~25.3M parameters with the default config (32k vocab, tied embeddings):
12.97M transformer + 12.29M embedding matrix. GQA with 2 KV heads keeps the
KV cache at 6 layers * 2 heads * 64 dim * 2 (K,V) * 2 bytes = 6 KiB per token,
so 1024 tokens of context cost ~6 MiB instead of ~18 MiB with MHA.

Everything here is plain PyTorch: no flash-attn, no xformers, no custom CUDA.
`F.scaled_dot_product_attention` picks the best available kernel per device.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(slots=True)
class ModelConfig:
    vocab_size: int = 32_000
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 2
    d_ff: int = 1536
    max_seq_len: int = 512
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-5
    tie_embeddings: bool = True
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.d_ff <= 0 or self.max_seq_len <= 0:
            raise ValueError("d_ff and max_seq_len must be positive")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        fields = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in fields})

    @classmethod
    def load(cls, path: Path) -> "ModelConfig":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in fp32: bf16 mean-square over 384 dims loses too much.
        dtype = x.dtype
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * self.weight.float()).to(dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, theta: float = 10_000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build(max_seq_len)

    def _build(self, seq_len: int) -> None:
        device = self.inv_freq.device
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.max_seq_len = seq_len

    def forward(self, seq_len: int, offset: int = 0, device: torch.device | None = None,
                dtype: torch.dtype | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len + offset > self.max_seq_len:
            self._build(seq_len + offset)
        cos = self.cos_cached[offset : offset + seq_len]
        sin = self.sin_cached[offset : offset + seq_len]
        if device is not None:
            cos, sin = cos.to(device), sin.to(device)
        if dtype is not None:
            cos, sin = cos.to(dtype), sin.to(dtype)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D); cos/sin: (T, D)
    return (x * cos[None, None]) + (_rotate_half(x) * sin[None, None])


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    batch, heads, seq, dim = x.shape
    x = x[:, :, None, :, :].expand(batch, heads, n_rep, seq, dim)
    return x.reshape(batch, heads * n_rep, seq, dim)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = config.n_heads // config.n_kv_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.kv_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.kv_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)
        self.dropout = config.dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        if past is not None:
            k = torch.cat((past[0], k), dim=2)
            v = torch.cat((past[1], v), dim=2)
        present = (k, v) if use_cache else None

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        causal = attention_mask is None and seq_len > 1
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            is_causal=causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.n_heads * self.head_dim)
        return self.o_proj(out), present


class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attn = Attention(config)
        self.mlp_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        attn_out, present = self.attn(self.attn_norm(x), cos, sin, attention_mask, past, use_cache)
        x = x + attn_out
        x = x + self.mlp(self.mlp_norm(x))
        return x, present


@dataclass(slots=True)
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None


class MiniLLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.rotary = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_theta)
        self.lm_head = None if config.tie_embeddings else nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.gradient_checkpointing = False
        self.apply(self._init_weights)
        if config.tie_embeddings:
            self.tok_embeddings.weight.data.normal_(0.0, 0.02)
        self._scale_residuals()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residuals(self) -> None:
        # GPT-2 style: residual-branch output projections start smaller so the
        # residual stream variance stays ~1 across depth.
        scale = 1.0 / math.sqrt(2 * self.config.n_layers)
        for layer in self.layers:
            nn.init.normal_(layer.attn.o_proj.weight, mean=0.0, std=0.02 * scale)
            nn.init.normal_(layer.mlp.down_proj.weight, mean=0.0, std=0.02 * scale)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool = False,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> ModelOutput:
        batch, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_len {self.config.max_seq_len}")
        offset = past_key_values[0][0].shape[2] if past_key_values else 0
        cos, sin = self.rotary(seq_len, offset, input_ids.device, self.tok_embeddings.weight.dtype)
        x = self.tok_embeddings(input_ids)
        presents: list[tuple[torch.Tensor, torch.Tensor]] = []
        for index, layer in enumerate(self.layers):
            past = past_key_values[index] if past_key_values is not None else None
            x, present = layer(x, cos, sin, attention_mask, past, use_cache)
            if present is not None:
                presents.append(present)
        x = self.norm(x)
        if self.config.tie_embeddings:
            logits = F.linear(x, self.tok_embeddings.weight)
        else:
            logits = self.lm_head(x)  # type: ignore[misc]
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                labels.view(-1),
                ignore_index=-100,
            )
        return ModelOutput(logits, loss, presents if use_cache else None)

    def num_parameters(self, *, include_embeddings: bool = True) -> int:
        total = sum(p.numel() for p in self.parameters())
        if not include_embeddings:
            total -= self.tok_embeddings.weight.numel()
        return total

    def kv_cache_bytes(self, seq_len: int, dtype_bytes: int = 2) -> int:
        per_token = self.config.n_layers * 2 * self.config.kv_dim * dtype_bytes
        return per_token * seq_len

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """Reference sampler for eval scripts. The server uses llama.cpp."""
        self.eval()
        ids = input_ids
        past: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        for _ in range(max_new_tokens):
            if ids.shape[1] > self.config.max_seq_len:
                ids = ids[:, -self.config.max_seq_len :]
                past = None
            out = self.forward(ids if past is None else ids[:, -1:], use_cache=True, past_key_values=past)
            past = out.past_key_values
            logits = out.logits[:, -1, :].float()
            if temperature <= 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k > 0:
                    k = min(top_k, logits.size(-1))
                    values, _ = torch.topk(logits, k)
                    logits[logits < values[:, -1:]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                if 0.0 < top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cumulative = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumulative - sorted_probs > top_p
                    sorted_probs[mask] = 0.0
                    probs = torch.zeros_like(probs).scatter_(1, sorted_idx, sorted_probs)
                    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat((ids, next_id), dim=1)
            if eos_id is not None and int(next_id[0, 0]) == eos_id:
                break
        return ids


def build_model(config: ModelConfig, *, dtype: torch.dtype | None = None) -> MiniLLM:
    model = MiniLLM(config)
    if dtype is not None and dtype != torch.float32:
        model = model.to(dtype)
    return model


if __name__ == "__main__":
    config = ModelConfig()
    model = MiniLLM(config)
    total = model.num_parameters()
    print(
        json.dumps(
            {
                "params": total,
                "params_millions": round(total / 1e6, 2),
                "non_embedding": model.num_parameters(include_embeddings=False),
                "kv_bytes_per_token": config.n_layers * 2 * config.kv_dim * 2,
                "head_dim": config.head_dim,
            },
            indent=2,
        )
    )
