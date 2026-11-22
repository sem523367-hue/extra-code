"""Sanity-check an SFT JSONL before burning days of training on it.

Reuses the real dataset class so the numbers match what train/sft.py will see:

    python -m tools.verify_sft data/sft/hf_math.jsonl --max-len 256

Reports usable rows, total tokens, supervised (assistant-only) tokens, the
share of rows truncated at --max-len, and a sample assistant turn. Exit 1 when
nothing is usable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    parser = argparse.ArgumentParser(description="verify SFT jsonl for train/sft.py")
    parser.add_argument("path")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tokenizers import Tokenizer  # noqa: E402
    from train.sft import SFTDataset, read_rows  # noqa: E402

    path = Path(args.path)
    if not path.exists():
        print(f"error: {path} does not exist", file=sys.stderr)
        return 2
    tokenizer = Tokenizer.from_file(args.tokenizer)

    rows = read_rows(path)
    dataset = SFTDataset(rows, tokenizer, max_len=args.max_len)
    if not dataset.examples:
        print("no usable rows", file=sys.stderr)
        return 1

    total = sum(len(ids) for ids, _ in dataset.examples)
    supervised = sum(sum(1 for label in labels if label != -100) for _, labels in dataset.examples)
    truncated = sum(1 for ids, _ in dataset.examples if len(ids) >= args.max_len)
    longest = max(len(ids) for ids, _ in dataset.examples)

    report = {
        "file": str(path),
        "rows": len(rows),
        "usable": len(dataset.examples),
        "tokens": total,
        "supervised_tokens": supervised,
        "supervised_share": round(supervised / total, 3) if total else 0.0,
        "avg_len": round(total / len(dataset.examples), 1),
        "longest": longest,
        f"truncated_at_{args.max_len}": truncated,
        "bytes": path.stat().st_size,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    sample = next((r for r in rows if r.get("messages")), None)
    if sample:
        assistant = next((m["content"] for m in sample["messages"] if m["role"] == "assistant"), "")
        preview = assistant[:200].replace("\n", "\\n")
        print(f"\nsample assistant turn:\n{preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
