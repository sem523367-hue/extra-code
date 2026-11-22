"""Run the whole pipeline, stage by stage, in dependency order.

    python -m data.run_all --limit 1000 --steps 50000
    python -m data.run_all --skip collect,train --stage pack

Every stage is a subprocess so a crash leaves the previous stage's artifacts
intact and the run report records exactly where it stopped. Stages are
idempotent: rerunning skips nothing silently, it just overwrites the output of
that stage. Training stages are opt-in via `--train` because they are the only
ones measured in hours.

Stages: collect -> prepare -> tokenize -> sft -> pack -> pretrain -> sft-train -> export
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("extra_code.run_all")

STAGE_ORDER = ["collect", "prepare", "tokenize", "sft", "pack", "pretrain", "sft-train", "export"]
TRAIN_STAGES = {"pretrain", "sft-train"}


@dataclass(slots=True)
class StageResult:
    name: str
    ok: bool
    seconds: float
    command: list[str]
    returncode: int = 0
    skipped: bool = False


def run(cmd: list[str], *, cwd: Path) -> int:
    log.info("$ %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=cwd)
    return proc.returncode


def build_stages(args: argparse.Namespace, root: Path) -> dict[str, list[str]]:
    py = sys.executable
    data_dir = Path(args.data_dir)
    raw = data_dir / "raw"
    clean = data_dir / "clean"
    sft_dir = data_dir / "sft"
    tok = Path(args.tokenizer)

    stages: dict[str, list[str]] = {
        "collect": [
            py, "-m", "data.collect.run",
            "--out", str(raw),
            "--limit", str(args.limit),
            "--sources", args.sources,
        ],
        "prepare": [
            py, "-m", "data.prepare",
            "--raw", str(raw),
            "--out", str(clean),
            "--threshold", str(args.dedup_threshold),
        ],
        "tokenize": [
            py, "-m", "data.tokenizer",
            "--train", str(clean / "train.jsonl.gz"),
            "--out", str(tok / "tokenizer.json"),
            "--vocab-size", str(args.vocab_size),
        ],
        "sft": [
            py, "-m", "data.make_sft",
            "--clean", str(clean),
            "--out", str(sft_dir),
            "--per-task", str(args.per_task),
        ],
        "pack": [
            py, "-m", "data.pack",
            "--clean", str(clean),
            "--tokenizer", str(tok / "tokenizer.json"),
            "--out", str(data_dir),
        ],
        "pretrain": [
            py, "-m", "train.pretrain",
            "--config", args.model_config,
            "--data-dir", str(data_dir),
            "--out", str(Path(args.runs_dir) / "base"),
            "--steps", str(args.steps),
            "--batch-size", str(args.batch_size),
            "--grad-accum", str(args.grad_accum),
            "--block-size", str(args.block_size),
            "--dtype", args.dtype,
            "--device", args.device,
        ],
        "sft-train": [
            py, "-m", "train.sft",
            "--model", str(Path(args.runs_dir) / "base" / "final" / "model.pt"),
            "--config", args.model_config,
            "--data", str(sft_dir / "train.jsonl"),
            "--tokenizer", str(tok / "tokenizer.json"),
            "--out", str(Path(args.runs_dir) / "sft"),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.sft_batch_size),
            "--grad-accum", str(args.sft_grad_accum),
            "--max-len", str(args.block_size),
            "--dtype", args.dtype,
            "--device", args.device,
        ],
        "export": [
            py, "-m", "export.to_hf",
            "--checkpoint", str(Path(args.runs_dir) / "sft" / "final" / "model.pt"),
            "--tokenizer", str(tok),
            "--out", str(Path("export") / "hf"),
            "--dtype", args.export_dtype,
        ],
    }
    return stages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Extra code pipeline end to end")
    parser.add_argument("--root", default=".", help="project root")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--tokenizer", default="tokenizer")
    parser.add_argument("--model-config", default="model/config.json")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--sources", default="all")
    parser.add_argument("--limit", type=int, default=1000, help="documents per collector")
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--dedup-threshold", type=float, default=0.8)
    parser.add_argument("--per-task", type=int, default=4000)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--sft-batch-size", type=int, default=4)
    parser.add_argument("--sft-grad-accum", type=int, default=8)
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--export-dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--stage", default=None, help="stop after this stage")
    parser.add_argument("--only", default=None, help="comma list of stages to run")
    parser.add_argument("--skip", default="", help="comma list of stages to skip")
    parser.add_argument("--train", action="store_true", help="include pretrain and sft-train")
    parser.add_argument("--clean-first", action="store_true", help="delete stage outputs before running")
    parser.add_argument("--report", default="data/run_report.json")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = Path(args.root).resolve()
    if not (root / "model" / "model.py").exists():
        print(f"error: {root} does not look like the project root (model/model.py missing)", file=sys.stderr)
        return 2

    stages = build_stages(args, root)
    only = {s.strip() for s in args.only.split(",") if s.strip()} if args.only else None
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    if not args.train:
        skip |= TRAIN_STAGES

    order = STAGE_ORDER
    if args.stage:
        if args.stage not in order:
            print(f"error: unknown --stage {args.stage!r}", file=sys.stderr)
            return 2
        order = order[: order.index(args.stage) + 1]

    if args.clean_first:
        for target in (
            Path(args.data_dir) / "clean",
            Path(args.tokenizer),
            Path(args.runs_dir),
            Path("export") / "hf",
        ):
            if target.exists():
                log.info("removing %s", target)
                shutil.rmtree(target)

    results: list[StageResult] = []
    started_all = time.perf_counter()
    for name in order:
        if only is not None and name not in only:
            continue
        if name in skip:
            results.append(StageResult(name, True, 0.0, [], skipped=True))
            log.info("stage %s skipped", name)
            continue
        cmd = stages[name]
        started = time.perf_counter()
        code = run(cmd, cwd=root)
        elapsed = time.perf_counter() - started
        results.append(StageResult(name, code == 0, elapsed, cmd, returncode=code))
        if code != 0:
            log.error("stage %s failed with exit %d", name, code)
            break

    report = {
        "root": str(root),
        "stages": [
            {
                "name": r.name,
                "ok": r.ok,
                "skipped": r.skipped,
                "seconds": round(r.seconds, 2),
                "returncode": r.returncode,
                "command": r.command,
            }
            for r in results
        ],
        "total_seconds": round(time.perf_counter() - started_all, 2),
        "failed": [r.name for r in results if not r.ok],
    }
    out = root / args.report
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({"stages": report["stages"], "failed": report["failed"]}, indent=2))
    print(f"report: {out}", file=sys.stderr)
    return 0 if not report["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
