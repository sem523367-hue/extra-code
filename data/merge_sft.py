"""Merge every SFT source into one training file with code kept dominant.

7771 extractive rows would drown 300 curated code answers, and a 25M model
trained on 0.3 epochs of pretraining needs the code format repeated. This script
deduplicates by question with an explicit source priority, then upsamples the
curated code rows.

    python -m data.merge_sft --out data/sft/all.jsonl --extractive 2500
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

PRIORITY = {
    "code": 4,
    "distilled_code": 3,
    "distilled_clean": 2,
    "distilled_seed": 2,
    "seed": 1,
    "extractive": 0,
}


def question(row: dict) -> str:
    for message in row.get("messages") or []:
        if message.get("role") == "user":
            return " ".join(str(message.get("content", "")).lower().split())
    return ""


def digest(row: dict) -> str:
    text = question(row)
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest() if text else ""


def answer_len(row: dict) -> int:
    for message in row.get("messages") or []:
        if message.get("role") == "assistant":
            return len(str(message.get("content", "")))
    return 0


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("messages"):
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge SFT sources with code upsampling")
    parser.add_argument("--sft-dir", default="data/sft")
    parser.add_argument("--out", default="data/sft/all.jsonl")
    parser.add_argument("--extractive", type=int, default=2500)
    parser.add_argument("--code-repeat", type=int, default=5)
    parser.add_argument("--distilled-repeat", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    base = Path(args.sft_dir)
    extractive = read(base / "train.jsonl")
    random.Random(args.seed).shuffle(extractive)

    groups: list[tuple[str, list[dict], int]] = [
        ("code", read(base / "code_seed.jsonl"), args.code_repeat),
        ("distilled_code", read(base / "distilled_code.jsonl"), args.distilled_repeat),
        ("distilled_clean", read(base / "distilled_clean.jsonl"), 1),
        ("distilled_seed", read(base / "distilled_seed.jsonl"), 1),
        ("seed", read(base / "seed.jsonl"), 1),
        ("extractive", extractive[: args.extractive], 1),
    ]

    chosen: dict[str, tuple[dict, str]] = {}
    counts: dict[str, int] = {}
    for name, rows, _ in groups:
        counts[name] = len(rows)
        for row in rows:
            key = digest(row)
            if not key:
                continue
            existing = chosen.get(key)
            if existing is None:
                chosen[key] = (row, name)
                continue
            better_source = PRIORITY[name] > PRIORITY[existing[1]]
            same_source_longer = PRIORITY[name] == PRIORITY[existing[1]] and answer_len(row) > answer_len(existing[0])
            if better_source or same_source_longer:
                chosen[key] = (row, name)

    merged: list[dict] = []
    by_name: dict[str, list[dict]] = {name: [] for name, _, _ in groups}
    for row, name in chosen.values():
        by_name.setdefault(name, []).append(row)

    repeats = {name: repeat for name, _, repeat in groups}
    for name, rows in by_name.items():
        merged.extend(rows * repeats.get(name, 1))

    random.Random(args.seed).shuffle(merged)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in merged:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    lengths = [answer_len(row) for row in merged]
    per_source = {name: len(rows) * repeats.get(name, 1) for name, rows in by_name.items()}
    print(json.dumps({
        "input_rows": counts,
        "after_dedup": {name: len(rows) for name, rows in by_name.items()},
        "output_rows": len(merged),
        "output_by_source": per_source,
        "avg_answer_chars": round(sum(lengths) / max(1, len(lengths))),
        "max_answer_chars": max(lengths, default=0),
        "out": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
