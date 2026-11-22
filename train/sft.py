"""Instruction tuning on chat-format JSONL.

Row formats accepted:
  {"messages": [{"role": "system"|"user"|"assistant", "content": "..."}]}
  {"prompt": "...", "response": "..."}
  {"text": "..."}                      # plain completion, loss on everything

Loss is masked on prompt tokens: only assistant content, the closing
<|im_end|> and the final </s> contribute. Early stopping watches val loss.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import shutil
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset

from model.model import MiniLLM, ModelConfig
from train.utils import (
    autocast,
    get_lr,
    load_checkpoint,
    make_optimizer,
    resolve_device,
    resolve_dtype,
    save_checkpoint,
    seed_everything,
    set_lr,
    TrainLog,
)

log = logging.getLogger("extra_code.sft")

IGNORE = -100
MAX_LEN = 512


def read_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                log.warning("skip malformed line %d", line_no)
                continue
            if "messages" in row or ("prompt" in row and "response" in row) or "text" in row:
                rows.append(row)
            else:
                log.warning("skip row %d: unknown schema", line_no)
    return rows


def to_messages(row: dict) -> list[tuple[str, str]]:
    if "messages" in row:
        out: list[tuple[str, str]] = []
        for msg in row["messages"]:
            role = str(msg.get("role", "user")).lower()
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                out.append((role, content))
        return out
    if "prompt" in row:
        return [("user", str(row["prompt"])), ("assistant", str(row["response"]))]
    return [("user", ""), ("assistant", str(row["text"]))]


class SFTDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer: Tokenizer, max_len: int = MAX_LEN) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.start_id = tokenizer.token_to_id("<|im_start|>")
        self.end_id = tokenizer.token_to_id("<|im_end|>")
        self.eos_id = tokenizer.token_to_id("</s>")
        if None in (self.start_id, self.end_id, self.eos_id):
            raise ValueError("tokenizer is missing ChatML special tokens; retrain with data/tokenizer.py")
        self.examples: list[tuple[list[int], list[int]]] = []
        for row in rows:
            example = self._build(to_messages(row))
            if example is not None:
                self.examples.append(example)
        if not self.examples:
            raise ValueError("no usable SFT examples after formatting/filtering")

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def _build(self, messages: list[tuple[str, str]]) -> tuple[list[int], list[int]] | None:
        ids: list[int] = []
        labels: list[int] = []
        for role, content in messages:
            supervised = role == "assistant"
            ids.append(self.start_id)  # type: ignore[arg-type]
            labels.append(IGNORE)
            prefix = self._encode(f"{role}\n")
            ids.extend(prefix)
            labels.extend([IGNORE] * len(prefix))
            body = self._encode(content)
            ids.extend(body)
            labels.extend(body if supervised else [IGNORE] * len(body))
            ids.append(self.end_id)  # type: ignore[arg-type]
            labels.append(self.end_id if supervised else IGNORE)  # type: ignore[list-item]
        ids.append(self.eos_id)  # type: ignore[arg-type]
        labels.append(self.eos_id)  # type: ignore[list-item]

        if len(ids) > self.max_len:
            ids = ids[-self.max_len :]
            labels = labels[-self.max_len :]
        if all(label == IGNORE for label in labels):
            return None
        return ids, labels

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ids, labels = self.examples[index]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate(batch: list[dict[str, torch.Tensor]], pad_id: int, max_len: int) -> dict[str, torch.Tensor]:
    width = min(max_len, max(item["input_ids"].numel() for item in batch))
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), IGNORE, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for row, item in enumerate(batch):
        ids = item["input_ids"][:width]
        target = item["labels"][:width]
        input_ids[row, : ids.numel()] = ids
        labels[row, : target.numel()] = target
        mask[row, : ids.numel()] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": mask}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    dtype: str,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast(device, dtype):
            out = model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"][:, None, None, :].bool(),
                labels=batch["labels"],
            )
        total += float(out.loss)
        count += 1
    model.train()
    return total / max(1, count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Instruction-tune extra-code")
    parser.add_argument("--model", default="runs/base/final/model.pt")
    parser.add_argument("--config", default="model/config.json")
    parser.add_argument("--data", default="data/sft/train.jsonl")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--out", default="runs/sft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--wandb", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    pad_id = tokenizer.token_to_id("<pad>")
    if pad_id is None:
        raise ValueError("tokenizer has no <pad> token")

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    config = ModelConfig.from_dict(checkpoint.get("config") or ModelConfig.load(Path(args.config)).to_dict())
    config.max_seq_len = max(config.max_seq_len, args.max_len)
    model = MiniLLM(config)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint, strict=True)
    model.to(device)
    model.train()

    rows = read_rows(Path(args.data))
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    val_size = max(1, int(len(rows) * args.val_fraction)) if len(rows) > 20 else max(1, len(rows) // 10)
    val_rows, train_rows = rows[:val_size], rows[val_size:]
    train_ds = SFTDataset(train_rows, tokenizer, args.max_len)
    val_ds = SFTDataset(val_rows, tokenizer, args.max_len)
    log.info("sft rows: train=%d val=%d examples=%d", len(train_rows), len(val_rows), len(train_ds))

    def loader(ds: SFTDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            collate_fn=lambda batch: collate(batch, pad_id, args.max_len),
            drop_last=False,
        )

    train_loader = loader(train_ds, True)
    val_loader = loader(val_ds, False)
    optimizer = make_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    logger = TrainLog(Path(args.out) / "log.jsonl")
    Path(args.out).mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if args.wandb:
        try:
            import wandb  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("--wandb requires `pip install wandb`") from exc
        wandb_run = wandb.init(project=args.wandb, config={**config.to_dict(), **vars(args)})

    best_val = float("inf")
    stale = 0
    global_step = 0
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        seen = 0
        for index, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            lr = get_lr(
                global_step,
                total_steps=max(1, total_steps),
                warmup=args.warmup,
                base_lr=args.lr,
                min_lr=args.min_lr,
            )
            set_lr(optimizer, lr)
            with autocast(device, dtype):
                out = model(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"][:, None, None, :].bool(),
                    labels=batch["labels"],
                )
                loss = out.loss / args.grad_accum
            loss.backward()
            running += float(out.loss.detach())
            seen += 1
            if (index + 1) % args.grad_accum == 0 or index + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 10 == 0:
                    avg = running / seen
                    log.info("epoch %d step %d loss %.4f lr %.2e", epoch + 1, global_step, avg, lr)
                    logger.write({"step": global_step, "loss": round(avg, 5), "lr": lr})
                    if wandb_run is not None:
                        wandb_run.log({"loss": avg, "lr": lr}, step=global_step)
                    running = 0.0
                    seen = 0

        val_loss = evaluate(model, val_loader, device=device, dtype=dtype)
        perplexity = math.exp(min(val_loss, 20.0))
        elapsed = time.perf_counter() - epoch_start
        log.info(
            "epoch %d/%d val_loss=%.4f perplexity=%.2f %.1fs",
            epoch + 1,
            args.epochs,
            val_loss,
            perplexity,
            elapsed,
        )
        logger.write({"step": global_step, "val_loss": round(val_loss, 5), "perplexity": round(perplexity, 3)})
        if wandb_run is not None:
            wandb_run.log({"val_loss": val_loss, "perplexity": perplexity}, step=global_step)
        if val_loss < best_val:
            best_val = val_loss
            stale = 0
            save_checkpoint(
                Path(args.out) / "best.pt",
                model=model,
                optimizer=None,
                step=global_step,
                best_val=best_val,
                config=config,
            )
        else:
            stale += 1
            if stale >= args.patience:
                log.info("early stop: no improvement for %d epochs", stale)
                break

    final_dir = Path(args.out) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config.to_dict()}, final_dir / "model.pt")
    config.save(final_dir / "config.json")
    shutil.copy2(args.tokenizer, final_dir / "tokenizer.json")
    summary = {
        "best_val": best_val,
        "best_perplexity": math.exp(min(best_val, 20.0)),
        "steps": global_step,
        "epochs": args.epochs,
        "train_examples": len(train_ds),
        "val_examples": len(val_ds),
        "params": sum(p.numel() for p in model.parameters()),
    }
    (Path(args.out) / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if wandb_run is not None:
        wandb_run.finish()
    log.info("sft done: %s", final_dir / "model.pt")


if __name__ == "__main__":
    main()
