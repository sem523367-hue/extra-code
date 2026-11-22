"""Extract a text column from parquet into a plain-text corpus for pretraining.

`data/train.bin` is built from raw text; most pretrey-grade HF datasets (wiki,
fineweb, k-12 textbooks) ship a single text column. This bridges the two:

    python -m data.corpus_from_parquet data/raw/history/big.parquet \
        --column text --out data/corpus/history.txt --limit 20000

Only rows whose text passes a minimal sanity gate are kept: non-empty, not
mostly-boilerplate, within length bounds. Output is one document per line block
separated by a blank line, so the existing corpus cleaner can pick it up.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def clean(text: str, *, min_chars: int, max_chars: int) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    if not (min_chars <= len(collapsed) <= max_chars):
        return ""
    letters = sum(ch.isalpha() for ch in collapsed)
    if letters / len(collapsed) < 0.55:  # tables, code dumps, navigation junk
        return ""
    return collapsed


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    parser = argparse.ArgumentParser(description="parquet text column -> corpus .txt")
    parser.add_argument("parquet", nargs="+")
    parser.add_argument("--column", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=100_000, help="documents to keep")
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--max-chars", type=int, default=20_000)
    args = parser.parse_args(argv)

    import pyarrow.parquet as pq

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = kept_chars = dropped = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for path in args.parquet:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=256, columns=[args.column]):
                for row in batch.to_pylist():
                    if written >= args.limit:
                        break
                    raw = row.get(args.column)
                    if not isinstance(raw, str):
                        dropped += 1
                        continue
                    text = clean(raw, min_chars=args.min_chars, max_chars=args.max_chars)
                    if not text:
                        dropped += 1
                        continue
                    fh.write(text + "\n\n")
                    written += 1
                    kept_chars += len(text)
                if written >= args.limit:
                    break
    print(json.dumps({
        "out": str(out_path),
        "documents": written,
        "dropped": dropped,
        "chars": kept_chars,
        "out_bytes": out_path.stat().st_size if out_path.exists() else 0,
    }, ensure_ascii=False, indent=2))
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
