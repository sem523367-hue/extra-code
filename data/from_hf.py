"""Pull a HuggingFace dataset and emit extra-code SFT JSONL.

`train/sft.py` reads rows shaped like

    {"messages": [{"role": "system"|"user"|"assistant", "content": "..."}]}

This module is the bridge from the several schemas datasets actually ship with
to that one shape. It streams, so a 600 GB corpus costs only the rows you keep.

    # alpaca-style: instruction / input / output
    python -m data.from_hf --dataset iamtarun/python_code_instructions_18k_alpaca \
        --instruction instruction --input input --output output \
        --out data/sft/hf_python.jsonl --limit 5000

    # pair-style: prompt / response
    python -m data.from_hf --dataset some/repo --prompt-field question \
        --response-field answer --out data/sft/hf_pair.jsonl

    # already chat-shaped: messages
    python -m data.from_hf --dataset some/chat-repo --messages-field messages \
        --out data/sft/hf_chat.jsonl

Rows longer than --max-chars are dropped instead of truncated: a truncated
assistant turn teaches the model to stop mid-sentence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger("extra_code.from_hf")

DEFAULT_SYSTEM = "You are Extra code, a compact local model. Answer briefly and to the point."
ROLES = {"system", "user", "assistant"}


def build_messages(
    row: dict[str, Any],
    *,
    system: str,
    messages_field: str | None,
    instruction_field: str | None,
    input_field: str | None,
    output_field: str | None,
    prompt_field: str | None,
    response_field: str | None,
) -> list[dict[str, str]] | None:
    """Map one dataset row onto ChatML messages, or None when it does not fit."""
    if messages_field:
        raw = row.get(messages_field)
        if not isinstance(raw, list):
            return None
        out: list[dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                return None
            role = str(item.get("role", "")).lower()
            content = item.get("content")
            if role not in ROLES or not isinstance(content, str) or not content.strip():
                return None
            out.append({"role": role, "content": content.strip()})
        if not any(m["role"] == "assistant" for m in out):
            return None
        if not any(m["role"] == "user" for m in out):
            return None
        return out

    if prompt_field and response_field:
        prompt = row.get(prompt_field)
        response = row.get(response_field)
        if not isinstance(prompt, str) or not isinstance(response, str):
            return None
        if not prompt.strip() or not response.strip():
            return None
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt.strip()},
            {"role": "assistant", "content": response.strip()},
        ]

    if instruction_field and output_field:
        instruction = row.get(instruction_field)
        extra = row.get(input_field) if input_field else None
        output = row.get(output_field)
        if not isinstance(instruction, str) or not isinstance(output, str):
            return None
        if not instruction.strip() or not output.strip():
            return None
        user = instruction.strip()
        if isinstance(extra, str) and extra.strip():
            user = f"{user}\n\n{extra.strip()}"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": output.strip()},
        ]

    raise ValueError("no field mapping: pass --messages-field, --prompt-field/--response-field, "
                     "or --instruction/--output")


def assistant_chars(messages: list[dict[str, str]]) -> int:
    return sum(len(m["content"]) for m in messages if m["role"] == "assistant")


def row_id(messages: list[dict[str, str]]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


def iter_parquet(paths: list[str], batch_rows: int = 512) -> Iterator[dict[str, Any]]:
    """Local parquet files, read in batches so memory stays flat."""
    import pyarrow.parquet as pq  # imported late so --help stays fast

    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_rows):
            yield from batch.to_pylist()


def iter_dataset(args: argparse.Namespace) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset  # imported late so --help stays fast

    kwargs: dict[str, Any] = {"split": args.split, "streaming": True}
    if args.config:
        kwargs["name"] = args.config
    if args.revision:
        kwargs["revision"] = args.revision
    dataset = load_dataset(args.dataset, **kwargs)
    if args.shuffle:
        dataset = dataset.shuffle(seed=args.seed, buffer_size=max(args.limit, 1000))
    yield from dataset


def convert(args: argparse.Namespace) -> dict[str, Any]:
    source = iter_parquet(args.parquet) if args.parquet else iter_dataset(args)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    stats = {"scanned": 0, "written": 0, "dropped_shape": 0, "dropped_long": 0, "dropped_short": 0,
             "dropped_duplicate": 0, "assistant_chars": 0}

    with out_path.open("w", encoding="utf-8") as fh:
        for row in source:
            if stats["written"] >= args.limit:
                break
            stats["scanned"] += 1
            if stats["scanned"] > args.max_scan:
                log.warning("hit --max-scan %d, stopping", args.max_scan)
                break
            messages = build_messages(
                row,
                system=args.system,
                messages_field=args.messages_field,
                instruction_field=args.instruction,
                input_field=args.input,
                output_field=args.output,
                prompt_field=args.prompt_field,
                response_field=args.response_field,
            )
            if messages is None:
                stats["dropped_shape"] += 1
                continue
            chars = assistant_chars(messages)
            if chars > args.max_chars:
                stats["dropped_long"] += 1
                continue
            if chars < args.min_chars:
                stats["dropped_short"] += 1
                continue
            digest = row_id(messages)
            if digest in seen:
                stats["dropped_duplicate"] += 1
                continue
            seen.add(digest)
            record = {
                "messages": messages,
                "task": "hf",
                "meta": {"dataset": args.dataset, "split": args.split, "prompt_id": digest},
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["written"] += 1
            stats["assistant_chars"] += chars
            if stats["written"] % 500 == 0:
                log.info("written=%d scanned=%d", stats["written"], stats["scanned"])

    stats["out"] = str(out_path)
    stats["out_bytes"] = out_path.stat().st_size if out_path.exists() else 0
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HuggingFace dataset -> extra-code SFT JSONL")
    parser.add_argument("--dataset", default=None, help="repo id, e.g. iamtarun/python_code_instructions_18k_alpaca")
    parser.add_argument("--parquet", nargs="+", default=None,
                        help="local .parquet file(s) instead of --dataset; no network, no datasets import")
    parser.add_argument("--config", default=None, help="dataset config/subset name")
    parser.add_argument("--split", default="train")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--out", required=True)

    shape = parser.add_argument_group("row shape (pick one mapping)")
    shape.add_argument("--preset", choices=["alpaca", "pair", "chat"], default=None,
                       help="alpaca: instruction/input/output; pair: prompt/response; chat: messages")
    shape.add_argument("--messages-field", default=None, help="column holding [{role, content}, ...]")
    shape.add_argument("--prompt-field", default=None)
    shape.add_argument("--response-field", default=None)
    shape.add_argument("--instruction", default=None)
    shape.add_argument("--input", default=None, help="optional extra context column for alpaca rows")
    shape.add_argument("--output", default=None)

    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument("--limit", type=int, default=5000, help="rows to keep")
    parser.add_argument("--max-scan", type=int, default=2_000_000, help="hard ceiling on rows read")
    parser.add_argument("--max-chars", type=int, default=1600, help="drop rows whose assistant text is longer")
    parser.add_argument("--min-chars", type=int, default=8, help="drop rows whose assistant text is shorter")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)

    if args.preset == "alpaca":
        args.instruction = args.instruction or "instruction"
        args.input = args.input if args.input is not None else "input"
        args.output = args.output or "output"
    elif args.preset == "pair":
        args.prompt_field = args.prompt_field or "prompt"
        args.response_field = args.response_field or "response"
    elif args.preset == "chat":
        args.messages_field = args.messages_field or "messages"

    if args.limit <= 0:
        parser.error("--limit must be positive")
    if not args.dataset and not args.parquet:
        parser.error("pass --dataset or --parquet")
    if args.max_chars <= args.min_chars:
        parser.error("--max-chars must exceed --min-chars")
    if not (args.messages_field or (args.prompt_field and args.response_field)
            or (args.instruction and args.output)):
        parser.error("choose a row mapping: --messages-field, --prompt-field/--response-field, "
                     "or --instruction/--output")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        stats = convert(args)
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0 if stats["written"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
