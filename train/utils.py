"""Shared training plumbing: seeding, schedules, batching, checkpoints, logging."""
from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def configure_cpu_threads() -> int | None:
    """Honour TRAIN_NUM_THREADS; torch defaults to half the cores, which leaves
    CPU-only pretraining at roughly half the achievable token rate."""
    raw = os.environ.get("TRAIN_NUM_THREADS", "").strip()
    if not raw:
        return None
    try:
        threads = int(raw)
    except ValueError:
        return None
    if threads <= 0:
        return None
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    return threads


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(name: str, device: torch.device) -> str:
    if name != "auto":
        if name not in {"bf16", "fp16", "fp32"}:
            raise ValueError("dtype must be auto, bf16, fp16 or fp32")
        if name in {"bf16", "fp16"} and device.type != "cuda":
            return "fp32"
        return name
    if device.type != "cuda":
        return "fp32"
    if torch.cuda.is_bf16_supported():
        return "bf16"
    return "fp16"


def autocast(device: torch.device, dtype: str):
    if dtype == "bf16" and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if dtype == "fp16" and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    if dtype == "bf16" and device.type == "cpu":
        return torch.autocast("cpu", dtype=torch.bfloat16)
    return _nullcontext()


class _nullcontext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None


def get_lr(step: int, *, total_steps: int, warmup: int, base_lr: float, min_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    if step >= total_steps:
        return min_lr
    ratio = (step - warmup) / max(1, total_steps - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * ratio))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


class TokenData:
    """Random fixed-length windows from a uint16 memmap."""

    def __init__(self, path: Path, block_size: int) -> None:
        if not path.exists():
            raise FileNotFoundError(f"token file missing: {path}; run data/pack.py first")
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        if len(self.data) < block_size + 2:
            raise ValueError(f"{path} has {len(self.data)} tokens, need at least {block_size + 2}")

    def batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = len(self.data) - self.block_size - 1
        starts = np.random.randint(0, max_start, size=batch_size)
        x = np.stack([np.asarray(self.data[s : s + self.block_size], dtype=np.int64) for s in starts])
        y = np.stack([np.asarray(self.data[s + 1 : s + 1 + self.block_size], dtype=np.int64) for s in starts])
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)

    def __len__(self) -> int:
        return len(self.data)


def make_optimizer(
    model: torch.nn.Module, *, lr: float, weight_decay: float = 0.1, betas: tuple[float, float] = (0.9, 0.95)
) -> torch.optim.Optimizer:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay.append(param)
        else:
            no_decay.append(param)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    fused = torch.cuda.is_available()
    try:
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)
    except (TypeError, RuntimeError):  # older torch or CPU build without fused
        return torch.optim.AdamW(groups, lr=lr, betas=betas)


@dataclass(slots=True)
class TrainLog:
    path: Path

    def write(self, record: dict[str, Any]) -> None:
        record = {"time": round(time.time(), 3), **record}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    best_val: float,
    config: Any,
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "step": step,
        "best_val": best_val,
        "config": config.to_dict(),
        "model": model.state_dict(),
        **(extra or {}),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model") or payload.get("state_dict")
    if state is None:
        raise ValueError(f"{path} has no model weights")
    model.load_state_dict(state, strict=True)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def batches(n: int, size: int) -> Iterator[list[int]]:
    batch: list[int] = []
    for i in range(n):
        batch.append(i)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
