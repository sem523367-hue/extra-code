"""Pretrain the base model on packed uint16 tokens.

Single-GPU friendly: bf16 autocast, gradient accumulation, cosine schedule,
optional activation checkpointing, resumable checkpoints, JSONL logs and
optional wandb. CPU works too, slowly, which is how the smoke test runs.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import torch

from model.model import MiniLLM, ModelConfig
from train.utils import (
    TokenData,
    autocast,
    configure_cpu_threads,
    count_parameters,
    get_lr,
    human_bytes,
    load_checkpoint,
    make_optimizer,
    resolve_device,
    resolve_dtype,
    save_checkpoint,
    seed_everything,
    set_lr,
    TrainLog,
)

log = logging.getLogger("extra_code.pretrain")


def evaluate(
    model: torch.nn.Module,
    data: TokenData,
    *,
    batches: int,
    device: torch.device,
    dtype: str,
) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(batches):
            x, y = data.batch(8, device)
            with autocast(device, dtype):
                out = model(x, labels=y)
            total += float(out.loss)
    model.train()
    return total / max(1, batches)


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain extra-code")
    parser.add_argument("--config", default="model/config.json")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="runs/base")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=40)
    parser.add_argument("--ckpt-interval", type=int, default=2000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--save-optimizer", action="store_true")
    parser.add_argument("--wandb", default=None, help="wandb project name")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    threads = configure_cpu_threads()
    if threads:
        log.info("cpu threads pinned to %d", threads)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    run_dir = Path(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config)
    config = ModelConfig.load(config_path) if config_path.exists() else ModelConfig()
    if args.block_size > config.max_seq_len:
        raise ValueError(f"block size {args.block_size} exceeds config max_seq_len {config.max_seq_len}")
    model = MiniLLM(config).to(device)
    model.gradient_checkpointing = args.grad_checkpointing
    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is unavailable in this torch build")
        model = torch.compile(model)  # type: ignore[assignment]
        log.info("torch.compile enabled")
    model.train()
    params = count_parameters(unwrap(model))
    log.info("model params=%s device=%s dtype=%s", f"{params/1e6:.2f}M", device, dtype)

    train_data = TokenData(Path(args.data_dir) / "train.bin", args.block_size)
    val_data = TokenData(Path(args.data_dir) / "val.bin", args.block_size)
    meta_path = Path(args.data_dir) / "meta.json"
    data_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    log.info(
        "tokens train=%.2fM val=%.2fM",
        len(train_data) / 1e6,
        len(val_data) / 1e6,
    )

    optimizer = make_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scaler = None
    if dtype == "fp16":
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        except TypeError:  # torch < 2.4
            scaler = torch.cuda.amp.GradScaler(enabled=True)

    start_step = 0
    best_val = float("inf")
    latest = run_dir / "latest.pt"
    if args.resume and latest.exists():
        payload = load_checkpoint(latest, unwrap(model), optimizer if args.save_optimizer else None)
        start_step = int(payload.get("step", 0)) + 1
        best_val = float(payload.get("best_val", best_val))
        log.info("resumed from %s at step %d", latest, start_step)

    logger = TrainLog(run_dir / "log.jsonl")
    wandb_run = None
    if args.wandb:
        try:
            import wandb  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("--wandb requires `pip install wandb`") from exc
        wandb_run = wandb.init(project=args.wandb, config={**config.to_dict(), **vars(args)})

    tokens_per_step = args.batch_size * args.grad_accum * args.block_size
    log.info("tokens/step=%d effective_batch=%d", tokens_per_step, args.batch_size * args.grad_accum)
    interval_start = time.perf_counter()
    interval_tokens = 0
    running_loss = 0.0
    running_count = 0
    step = start_step - 1

    try:
        for step in range(start_step, args.steps):
            lr = get_lr(
                step,
                total_steps=args.steps,
                warmup=args.warmup,
                base_lr=args.lr,
                min_lr=args.min_lr,
            )
            set_lr(optimizer, lr)
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            for _ in range(args.grad_accum):
                x, y = train_data.batch(args.batch_size, device)
                with autocast(device, dtype):
                    out = model(x, labels=y)
                    loss = out.loss / args.grad_accum
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                loss_value += float(out.loss.detach())
                interval_tokens += x.numel()
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            step_loss = loss_value / args.grad_accum
            running_loss += step_loss
            running_count += 1
            if step % args.log_interval == 0:
                now = time.perf_counter()
                elapsed = max(1e-6, now - interval_start)
                tok_s = interval_tokens / elapsed
                avg = running_loss / running_count
                log.info(
                    "step %d/%d loss %.4f lr %.2e tok/s %.0f",
                    step,
                    args.steps,
                    avg,
                    lr,
                    tok_s,
                )
                logger.write(
                    {
                        "step": step,
                        "loss": round(avg, 5),
                        "lr": lr,
                        "tok_s": round(tok_s, 1),
                        "tokens": interval_tokens,
                    }
                )
                if wandb_run is not None:
                    wandb_run.log({"loss": avg, "lr": lr, "tok_s": tok_s}, step=step)
                running_loss = 0.0
                running_count = 0
                interval_start = now
                interval_tokens = 0

            if args.eval_interval and (step + 1) % args.eval_interval == 0:
                val_loss = evaluate(
                    model,
                    val_data,
                    batches=args.eval_batches,
                    device=device,
                    dtype=dtype,
                )
                perplexity = math.exp(min(val_loss, 20.0))
                log.info("eval step=%d val_loss=%.4f perplexity=%.2f", step + 1, val_loss, perplexity)
                logger.write({"step": step + 1, "val_loss": round(val_loss, 5), "perplexity": round(perplexity, 3)})
                if wandb_run is not None:
                    wandb_run.log({"val_loss": val_loss, "perplexity": perplexity}, step=step)
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(
                        run_dir / "best.pt",
                        model=unwrap(model),
                        optimizer=None,
                        step=step + 1,
                        best_val=best_val,
                        config=config,
                    )
                save_checkpoint(
                    run_dir / "latest.pt",
                    model=unwrap(model),
                    optimizer=optimizer if args.save_optimizer else None,
                    step=step + 1,
                    best_val=best_val,
                    config=config,
                )
            elif args.ckpt_interval and (step + 1) % args.ckpt_interval == 0:
                save_checkpoint(
                    run_dir / "latest.pt",
                    model=unwrap(model),
                    optimizer=optimizer if args.save_optimizer else None,
                    step=step + 1,
                    best_val=best_val,
                    config=config,
                )
    except KeyboardInterrupt:
        log.warning("interrupted, saving emergency checkpoint")
        save_checkpoint(
            run_dir / "interrupted.pt",
            model=unwrap(model),
            optimizer=None,
            step=step,
            best_val=best_val,
            config=config,
        )
        raise

    final_dir = run_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": unwrap(model).state_dict(), "config": config.to_dict()}, final_dir / "model.pt")
    config.save(final_dir / "config.json")
    summary = {
        "steps": args.steps,
        "params": params,
        "best_val": best_val,
        "dtype": dtype,
        "device": str(device),
        "tokens_per_step": tokens_per_step,
        "data": data_meta,
        "model_size_fp32_mb": round(params * 4 / 1e6, 2),
        "model_size_bf16_mb": round(params * 2 / 1e6, 2),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if wandb_run is not None:
        wandb_run.finish()
    log.info("done: %s", final_dir / "model.pt")


if __name__ == "__main__":
    main()
