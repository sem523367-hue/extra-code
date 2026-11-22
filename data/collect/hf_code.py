"""HuggingFace dataset collector for the Extra code corpus.

Streams selected datasets (no full download) and turns each row into one
corpus document. This is how verilog, ru-en parallel text and Wikipedia
utterances get mixed into pretraining without pulling multi-gigabyte files.

Default set, override with HF_DATASETS env var (comma-separated repo ids,
2000 docs each):

    python -m data.collect.hf_code --out data/raw
    python -m data.collect.hf_code --out data/raw --limits 3000
    HF_DATASETS="starcoder/data,quickmt/quickmt-train.ru-en-v2" python -m data.collect.hf_code
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from .base import Doc, Sink, run_collector

log = logging.getLogger(__name__)

DEFAULT_DATASETS: list[dict] = [
    {"name": "psch9912/verilog-code-base", "lang": "en", "min_chars": 100},
    {"name": "quickmt/quickmt-train.ru-en-v2", "lang": "ru", "min_chars": 50},
    {"name": "jspaulsen/wikipedia-utterances", "lang": "en", "min_chars": 50},
]

# Column names to try first when a dataset has no obvious "text" field.
PREFERRED = ("text", "content", "code", "problem_statement", "prompt", "response", "utterance")


def slugify(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def extract_text(row: dict) -> str:
    """Best-effort: known columns first, then the longest string in the row."""
    ru = row.get("ru")
    en = row.get("en")
    if isinstance(ru, str) and ru.strip() and isinstance(en, str) and en.strip():
        return f"RU: {ru}\nEN: {en}"
    for col in PREFERRED:
        value = row.get(col)
        if isinstance(value, str) and value.strip():
            return value.strip()
    best = ""
    for value in row.values():
        if isinstance(value, str) and len(value) > len(best):
            best = value
    return best.strip()


def collect(out_dir: Path, *, limit: int = 2000) -> dict[str, object]:
    configured = os.environ.get("HF_DATASETS", "").strip()
    if configured:
        names = [s.strip() for s in configured.split(",") if s.strip()]
        datasets = [{"name": n, "lang": "ru" if "ru" in n else "en", "min_chars": 50} for n in names]
    else:
        datasets = DEFAULT_DATASETS

    for spec in datasets:
        name = spec["name"]
        slug = slugify(name)
        lang = spec.get("lang", "en")
        min_chars = int(spec.get("min_chars", 50))

        def body(sink: Sink, spec: dict = spec, min_chars: int = min_chars) -> None:
            from datasets import load_dataset  # heavy import, only when collecting

            try:
                stream = load_dataset(spec["name"], split="train", streaming=True)
            except Exception as exc:  # noqa: BLE001 - one bad dataset must not kill the batch
                log.warning("hf %s: cannot open train split: %s", spec["name"], exc)
                return
            for i, row in enumerate(stream):
                if sink.count >= limit:
                    return
                text = extract_text(row) if isinstance(row, dict) else ""
                if len(text) < min_chars:
                    continue
                doc = Doc(
                    text=text,
                    source=f"hf_{slug}",
                    url=f"hf://{spec['name']}/{i}",
                    lang=lang,
                    meta={"row": i},
                )
                sink.add(doc)

        try:
            run_collector(out_dir, f"hf_{slug}", limit=limit, body=body)
        except Exception as exc:  # noqa: BLE001 - keep going with the next dataset
            log.warning("hf %s failed: %s", name, exc)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect code/text datasets from HuggingFace")
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--limit", type=int, default=2000, help="docs per dataset")
    args = parser.parse_args()
    collect(Path(args.out), limit=args.limit)
