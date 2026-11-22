"""Corpus cleaning, deduplication and split.

Input:  data/raw/**/*.jsonl.gz  (written by data/collect/*)
Output: data/clean/train.jsonl.gz, data/clean/val.jsonl.gz, data/clean/report.json

Dedup is exact-hash first, then MinHash/LSH over 5-gram shingles with a
bottom-k sketch. Bottom-k + banding is an approximation: candidates are
confirmed by Jaccard on the sketches, so a pair only dies if it really looks
like the same document at the configured threshold.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import re
import unicodedata
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

log = logging.getLogger("extra_code.prepare")

SKETCH_SIZE = 64
BAND_ROWS = 4
VAL_MODULO = 100  # 1% validation split

ZW = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff\u2060"), None)
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SPACE_RE = re.compile(r"[ \t\u00a0]+")
BLANK_RE = re.compile(r"\n{3,}")
WORD_RE = re.compile(r"[\w]+", re.UNICODE)
CODE_SOURCES = {"github", "local_code"}
LOW_VALUE = (
    "lorem ipsum",
    "this article is a stub",
    "this section is empty",
    "click here to",
    "cookie policy",
    "subscribe to our newsletter",
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").translate(ZW)
    text = CTRL_RE.sub("", text)
    text = SPACE_RE.sub(" ", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = BLANK_RE.sub("\n\n", text)
    return text.strip()


def guess_lang(text: str) -> str:
    sample = text[:20_000]
    cyr = lat = 0
    for ch in sample:
        if not ch.isalpha():
            continue
        low = ch.lower()
        if "а" <= low <= "я" or low == "ё":
            cyr += 1
        elif "a" <= low <= "z":
            lat += 1
    total = cyr + lat
    if total < 20:
        return "und"
    if cyr / total >= 0.3:
        return "ru"
    if lat / total >= 0.7:
        return "en"
    return "und"


@dataclass(slots=True)
class Quality:
    ok: bool
    reason: str = ""


def quality(text: str, source: str) -> Quality:
    n = len(text)
    if n < 200:
        return Quality(False, "too_short")
    if n > 400_000:
        return Quality(False, "too_long")
    if text.count("\ufffd") > 5:
        return Quality(False, "encoding")
    whitespace = sum(ch.isspace() for ch in text)
    if whitespace / n > 0.58:
        return Quality(False, "whitespace")
    alnum = sum(ch.isalnum() for ch in text)
    if alnum / n < 0.18 and source not in CODE_SOURCES:
        return Quality(False, "symbols")
    if len(set(text[:50_000])) < 16 and n > 5_000:
        return Quality(False, "repetitive")
    lower = text[:5_000].lower()
    if any(marker in lower for marker in LOW_VALUE):
        return Quality(False, "boilerplate")
    if source not in CODE_SOURCES:
        lines = text.split("\n", 400)
        avg = sum(len(line) for line in lines) / max(1, len(lines))
        if avg > 900:
            return Quality(False, "minified")
    return Quality(True)


class MinHashDeduper:
    """Bottom-k sketch + LSH banding + exact sketch Jaccard."""

    def __init__(self, *, size: int = SKETCH_SIZE, band_rows: int = BAND_ROWS, threshold: float = 0.8) -> None:
        if size % band_rows:
            raise ValueError("sketch size must divide evenly into bands")
        self.size = size
        self.band_rows = band_rows
        self.threshold = threshold
        self.bands: int = size // band_rows
        self.buckets: list[dict[bytes, list[int]]] = [defaultdict(list) for _ in range(self.bands)]
        self.sketches: list[array] = []
        self.exact: set[bytes] = set()
        self.duplicates = 0

    @staticmethod
    def _shingles(text: str) -> Iterator[bytes]:
        words = WORD_RE.findall(text.lower())[:50_000]
        if len(words) < 5:
            for word in words:
                yield hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
            return
        for i in range(len(words) - 4):
            yield " ".join(words[i : i + 5]).encode("utf-8")

    def sketch(self, text: str) -> array:
        hashes: list[int] = []
        for shingle in self._shingles(text):
            digest = hashlib.blake2b(shingle, digest_size=8).digest()
            hashes.append(int.from_bytes(digest, "big"))
        if not hashes:
            return array("Q")
        smallest = sorted(set(hashes))[: self.size]
        return array("Q", smallest)

    def _band_keys(self, sketch: array) -> Iterator[bytes]:
        for band in range(self.bands):
            chunk = sketch[band * self.band_rows : (band + 1) * self.band_rows]
            if not chunk:
                continue
            yield hashlib.blake2b(chunk.tobytes(), digest_size=8).digest()

    def is_duplicate(self, text: str) -> bool:
        exact = hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()
        if exact in self.exact:
            self.duplicates += 1
            return True
        sketch = self.sketch(text)
        if not sketch:
            return False
        candidates: set[int] = set()
        for band, key in enumerate(self._band_keys(sketch)):
            found = self.buckets[band].get(key)
            if found:
                candidates.update(found)
        mine = set(sketch)
        for index in candidates:
            other = set(self.sketches[index])
            union = len(mine | other)
            if union and len(mine & other) / union >= self.threshold:
                self.duplicates += 1
                return True
        doc_id = len(self.sketches)
        self.sketches.append(sketch)
        for band, key in enumerate(self._band_keys(sketch)):
            self.buckets[band][key].append(doc_id)
        self.exact.add(exact)
        return False


@dataclass(slots=True)
class Report:
    raw_documents: int = 0
    kept: int = 0
    train: int = 0
    val: int = 0
    dropped: Counter = field(default_factory=Counter)
    by_source: Counter = field(default_factory=Counter)
    by_lang: Counter = field(default_factory=Counter)
    bytes_kept: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "raw_documents": self.raw_documents,
            "kept": self.kept,
            "train": self.train,
            "val": self.val,
            "dropped": dict(self.dropped),
            "by_source": dict(self.by_source),
            "by_lang": dict(self.by_lang),
            "bytes_kept": self.bytes_kept,
            "avg_chars": round(self.bytes_kept / max(1, self.kept), 1),
        }


def iter_raw(raw_dir: Path) -> Iterator[dict]:
    for path in sorted(raw_dir.rglob("*.jsonl.gz")):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue
        except (OSError, EOFError) as exc:
            # A killed collector leaves a truncated shard; everything before the
            # cut is still valid, so warn and keep the documents already yielded.
            log.warning("truncated shard %s: %s", path, exc)


def split_for(text: str) -> str:
    digest = hashlib.blake2b(text[:8192].encode("utf-8"), digest_size=8).digest()
    return "val" if int.from_bytes(digest, "big") % VAL_MODULO == 0 else "train"


def prepare(
    raw_dir: Path,
    out_dir: Path,
    *,
    threshold: float = 0.8,
    limit: int | None = None,
    min_chars: int = 200,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = Report()
    deduper = MinHashDeduper(threshold=threshold)
    train_path = out_dir / "train.jsonl.gz"
    val_path = out_dir / "val.jsonl.gz"
    train_fh = gzip.open(train_path, "wt", encoding="utf-8", compresslevel=6)
    val_fh = gzip.open(val_path, "wt", encoding="utf-8", compresslevel=6)
    val_overflow: list[str] = []

    try:
        for row in iter_raw(raw_dir):
            report.raw_documents += 1
            if limit is not None and report.kept >= limit:
                break
            text = row.get("text")
            source = str(row.get("source") or "unknown")
            if not isinstance(text, str):
                report.dropped["missing_text"] += 1
                continue
            text = normalize(text)
            if len(text) < min_chars:
                report.dropped["too_short"] += 1
                continue
            verdict = quality(text, source)
            if not verdict.ok:
                report.dropped[verdict.reason] += 1
                continue
            if deduper.is_duplicate(text):
                report.dropped["duplicate"] += 1
                continue
            lang = row.get("lang") if row.get("lang") in {"ru", "en", "und"} else guess_lang(text)
            record = {
                "text": text,
                "source": source,
                "url": row.get("url", ""),
                "lang": lang,
                "meta": row.get("meta") or {},
            }
            line = json.dumps(record, ensure_ascii=False) + "\n"
            if split_for(text) == "val":
                val_fh.write(line)
                report.val += 1
            else:
                train_fh.write(line)
                report.train += 1
            report.kept += 1
            report.bytes_kept += len(text.encode("utf-8"))
            report.by_source[source] += 1
            report.by_lang[lang] += 1
            if report.kept % 5_000 == 0:
                log.info(
                    "prepare: %d kept / %d seen (dups=%d)",
                    report.kept,
                    report.raw_documents,
                    deduper.duplicates,
                )
    finally:
        train_fh.close()
        val_fh.close()

    # Guarantee a non-empty validation slice on tiny corpora.
    if report.val == 0 and report.train > 20:
        with gzip.open(train_path, "rt", encoding="utf-8") as fh:
            lines = fh.readlines()
        moved = max(1, len(lines) // VAL_MODULO)
        val_overflow = lines[-moved:]
        with gzip.open(train_path, "wt", encoding="utf-8") as fh:
            fh.writelines(lines[:-moved])
        with gzip.open(val_path, "at", encoding="utf-8") as fh:
            fh.writelines(val_overflow)
        report.train -= moved
        report.val += moved

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "threshold": threshold,
        "min_chars": min_chars,
        **report.as_dict(),
    }
    (out_dir / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "train": str(train_path),
                "val": str(val_path),
                "format": "gzip jsonl, utf-8, one document per line",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("prepare done: %s", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean, deduplicate and split the corpus")
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--out", default="data/clean")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-chars", type=int, default=200)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    prepare(
        Path(args.raw),
        Path(args.out),
        threshold=args.threshold,
        limit=args.limit,
        min_chars=args.min_chars,
    )


if __name__ == "__main__":
    main()
