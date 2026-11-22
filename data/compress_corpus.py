# -*- coding: utf-8 -*-
"""Compress every corpus .txt to .txt.gz (level 9) and drop the original.

Text compresses ~4-6x. kb.py reads both forms transparently, so a
half-converted tree still builds. Resume-safe: .gz next to .txt means
skip. Run from extra-code root: python -m data.compress_corpus
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

ROOTS = [
    Path("data/corpus"),
]


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    converted = 0
    saved_before = 0.0
    saved_after = 0.0
    for root in ROOTS:
        for path in sorted(root.rglob("*.txt")):
            gz_path = path.with_suffix(".txt.gz")
            if gz_path.exists():
                continue
            raw = path.read_bytes()
            with gzip.open(gz_path, "wb", compresslevel=9) as fh:
                fh.write(raw)
            # verify the archive actually round-trips before deleting
            with gzip.open(gz_path, "rb") as fh:
                back = fh.read()
            if back != raw:
                print(f"VERIFY FAIL, kept original: {path}", flush=True)
                gz_path.unlink()
                continue
            saved_before += len(raw)
            saved_after += gz_path.stat().st_size
            path.unlink()
            converted += 1
    ratio = (saved_before / saved_after) if saved_after else 0.0
    print(json.dumps({
        "converted": converted,
        "mb_before": round(saved_before / 1e6, 1),
        "mb_after": round(saved_after / 1e6, 1),
        "ratio": round(ratio, 1),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
