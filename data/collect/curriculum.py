"""Curriculum collector: turn the code-teaching answers into corpus documents.

The SFT set teaches question/answer shape; this collector feeds the same code
into pretraining so the tokenizer and the base model see C++, Java, C#, asm and
reverse-engineering syntax, not only Python and prose.

    python -m data.collect.curriculum --out data/raw
"""
from __future__ import annotations

import argparse
from pathlib import Path

from data.make_code_sft import TOPICS
from .base import Doc, run_collector


def collect(out_dir: Path, *, limit: int = 2000) -> dict[str, object]:
    stats: dict[str, int] = {}

    def body(sink) -> None:
        for entry in TOPICS:
            if sink.count >= limit:
                return
            text = (
                f"# {entry['topic']}\n\n"
                f"Вопрос: {entry['question']}\n\n"
                f"{entry['answer']}\n"
            )
            doc = Doc(
                text=text,
                source="curriculum",
                url=f"curriculum://{entry['lang']}/{entry['topic'].replace(' ', '-')}",
                lang=entry["lang"],
                meta={"topic": entry["topic"], "level": entry["level"]},
            )
            if sink.add(doc):
                stats[entry["lang"]] = stats.get(entry["lang"], 0) + 1

    manifest = run_collector(out_dir, "curriculum", limit=limit, body=body)
    manifest["by_language"] = stats
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect the code curriculum into the corpus")
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--limit", type=int, default=2000)
    args = parser.parse_args()
    collect(Path(args.out), limit=args.limit)
