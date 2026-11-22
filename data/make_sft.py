"""Build the instruction-tuning set.

Two sources, merged and shuffled:

  seed      hand-written rows in data/sft/seed.jsonl: identity, format
            following, short technical answers, code snippets. These teach the
            model *how* to answer.
  extract   rows derived from the cleaned corpus with tasks that have a correct
            answer by construction: definition extraction, keyword extraction,
            heading/title recovery, docstring-based code explanation. These
            teach the model *what* the corpus contains.

Extractive tasks are used on purpose. Template-generated "answers" that a
teacher model never wrote teach fluent nonsense; extraction from real text
teaches short, verifiable answers, which is what a 25M model can actually fit.
The upgrade path is distillation from a larger model: point --teacher at any
OpenAI-compatible endpoint and the same prompts get real answers.

    python -m data.make_sft --clean data/clean --out data/sft --per-task 4000
    python -m data.make_sft --clean data/clean --out data/sft --teacher http://127.0.0.1:11434

Output: train.jsonl, val.jsonl, report.json (all UTF-8, one JSON object per line).
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

log = logging.getLogger("extra_code.sft_data")

SYSTEM_RU = "Ты Extra code — компактная локальная модель. Отвечай кратко и по делу."
SYSTEM_EN = "You are Extra code, a compact local model. Answer briefly and to the point."

DEF_RU = re.compile(
    r"^\s*([А-ЯЁA-Z][\w\s\-()]{2,50}?)\s+[—–-]\s+(?:это\s+)?([^.\n]{25,240})\.",
    re.MULTILINE,
)
DEF_EN = re.compile(
    r"^\s*([A-Z][\w\s\-()]{2,50}?)\s+(?:is|are)\s+([^.\n]{25,240})\.",
    re.MULTILINE,
)
HEADING = re.compile(r"^\s{0,3}(#{1,3})\s+(.{3,80})\s*$", re.MULTILINE)
PY_DEF = re.compile(r"^\s*def\s+(\w+)\s*\(([^)]*)\)[^:]*:\s*\n\s*(\"\"\"|''')(.+?)\3", re.MULTILINE | re.DOTALL)

STOP_RU = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все", "она",
    "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за", "бы", "по", "только", "ее",
    "мне", "было", "вот", "от", "меня", "еще", "нет", "о", "из", "ему", "теперь", "когда",
    "даже", "ну", "вдруг", "ли", "если", "уже", "или", "ни", "быть", "был", "него", "до",
    "вас", "нибудь", "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей",
    "может", "они", "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя", "их", "чем",
    "была", "сам", "чтоб", "без", "будто", "чего", "раз", "тоже", "себе", "под", "будет",
    "ж", "тогда", "кто", "этот", "того", "потому", "этого", "какой", "совсем", "ним",
    "здесь", "этом", "один", "почти", "мой", "тем", "чтобы", "нее", "сейчас", "были",
    "куда", "зачем", "всех", "никогда", "можно", "при", "наконец", "два", "об", "другой",
    "хоть", "после", "над", "больше", "тот", "через", "эти", "нас", "про", "всего", "них",
    "какая", "много", "разве", "три", "эту", "моя", "впрочем", "хорошо", "свою", "этой",
    "перед", "иногда", "лучше", "чуть", "том", "нельзя", "такой", "им", "более", "всегда",
    "конечно", "всю", "между",
}
STOP_EN = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this", "these",
    "those", "is", "are", "was", "were", "be", "been", "being", "to", "of", "in", "on",
    "for", "with", "as", "by", "at", "from", "it", "its", "we", "you", "they", "he", "she",
    "i", "not", "no", "can", "will", "would", "should", "could", "may", "might", "must",
    "have", "has", "had", "do", "does", "did", "so", "such", "there", "here", "which",
    "who", "whom", "what", "when", "where", "why", "how", "all", "any", "both", "each",
    "more", "most", "other", "some", "only", "own", "same", "too", "very", "just", "also",
}
WORD = re.compile(r"[A-Za-zА-Яа-яЁё_][\w\-]{2,}", re.UNICODE)


def iter_docs(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:  # type: ignore[operator]
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("text")
            if isinstance(text, str) and text.strip():
                yield row


def is_russian(text: str) -> bool:
    cyrillic = sum(1 for ch in text if "а" <= ch.lower() <= "я" or ch.lower() == "ё")
    latin = sum(1 for ch in text if "a" <= ch.lower() <= "z")
    return cyrillic > latin


def keywords(text: str, count: int) -> list[str]:
    stop = STOP_RU | STOP_EN
    words = [w.lower() for w in WORD.findall(text)]
    counts = Counter(w for w in words if w not in stop and len(w) > 3)
    return [word for word, _ in counts.most_common(count)]


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if 20 <= len(p.strip()) <= 400]


# ---------------------------------------------------------------------------
# task builders: each returns a row or None
# ---------------------------------------------------------------------------

def task_definition(text: str) -> dict[str, Any] | None:
    russian = is_russian(text)
    pattern = DEF_RU if russian else DEF_EN
    match = pattern.search(text)
    if not match:
        return None
    term, body = match.group(1).strip(), match.group(2).strip()
    if len(term.split()) > 6 or len(body.split()) < 4:
        return None
    question = f"Что такое {term}?" if russian else f"What is {term}?"
    system = SYSTEM_RU if russian else SYSTEM_EN
    answer = body[0].upper() + body[1:] + "."
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "task": "definition",
    }


def task_keywords(text: str) -> dict[str, Any] | None:
    words = keywords(text, 5)
    if len(words) < 4:
        return None
    russian = is_russian(text)
    question = "Назови пять ключевых слов этого текста через запятую." if russian else (
        "List five keywords from this text, comma separated."
    )
    passage = text.strip()[:1200]
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_RU if russian else SYSTEM_EN},
            {"role": "user", "content": f"{question}\n\n{passage}"},
            {"role": "assistant", "content": ", ".join(words)},
        ],
        "task": "keywords",
    }


def task_heading(text: str) -> dict[str, Any] | None:
    match = HEADING.search(text)
    if not match:
        return None
    heading = match.group(2).strip().rstrip("#").strip()
    if len(heading.split()) < 2:
        return None
    body = text[match.end() :].strip()
    if len(body) < 120:
        return None
    russian = is_russian(text)
    question = (
        "Придумай заголовок для этого текста. Ответь одной строкой без решёток."
        if russian
        else "Give this text a title. Answer in one line, no markdown."
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_RU if russian else SYSTEM_EN},
            {"role": "user", "content": f"{question}\n\n{body[:1200]}"},
            {"role": "assistant", "content": heading},
        ],
        "task": "heading",
    }


def task_code(text: str) -> dict[str, Any] | None:
    match = PY_DEF.search(text)
    if not match:
        return None
    name, params, _, doc = match.groups()
    doc = " ".join(doc.split())
    if len(doc) < 25:
        return None
    first = doc.split(". ")[0].strip().rstrip(".")
    snippet = match.group(0).strip()[:800]
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_RU},
            {"role": "user", "content": f"Что делает эта функция?\n\n{snippet}"},
            {"role": "assistant", "content": first + "."},
        ],
        "task": "code",
    }


def task_summary(text: str) -> dict[str, Any] | None:
    parts = sentences(text)
    if len(parts) < 4:
        return None
    russian = is_russian(text)
    body = " ".join(parts[:8])[:1400]
    # Extractive summary: the two highest-information sentences by keyword density.
    key = set(keywords(body, 8))
    scored = sorted(
        parts[:8],
        key=lambda s: sum(1 for w in WORD.findall(s.lower()) if w in key) / max(1, len(s.split())),
        reverse=True,
    )
    chosen = [s for s in scored[:2]]
    if len(chosen) < 2:
        return None
    summary = " ".join(sorted(chosen, key=lambda s: body.find(s)))
    question = "Сожми текст в два предложения." if russian else "Summarise the text in two sentences."
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_RU if russian else SYSTEM_EN},
            {"role": "user", "content": f"{question}\n\n{body}"},
            {"role": "assistant", "content": summary},
        ],
        "task": "summary",
    }


TASKS = {
    "definition": task_definition,
    "keywords": task_keywords,
    "heading": task_heading,
    "code": task_code,
    "summary": task_summary,
}


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def load_seed(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        log.warning("seed file missing: %s", path)
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                log.warning("%s:%d: bad JSON, skipped", path, line_no)
                continue
            if row.get("messages"):
                row.setdefault("task", "seed")
                rows.append(row)
    return rows


def valid_row(row: dict[str, Any], max_chars: int) -> bool:
    messages = row.get("messages") or []
    if len(messages) < 2:
        return False
    if messages[-1].get("role") != "assistant":
        return False
    answer = messages[-1].get("content") or ""
    if not answer.strip() or len(answer) > max_chars:
        return False
    if any(m.get("role") == "assistant" for m in messages[:-1]):
        return False
    total = sum(len(m.get("content") or "") for m in messages)
    return total <= max_chars * 3


def build(
    clean_dir: Path,
    out_dir: Path,
    *,
    seed_path: Path,
    per_task: int,
    max_chars: int,
    val_fraction: float,
    seed: int,
) -> dict[str, Any]:
    source = clean_dir / "train.jsonl.gz"
    if not source.exists():
        raise FileNotFoundError(f"{source} missing; run data/prepare.py first")
    rng = random.Random(seed)

    rows: list[dict[str, Any]] = load_seed(seed_path)
    seed_count = len(rows)
    produced = Counter({"seed": seed_count})

    docs = list(iter_docs(source))
    rng.shuffle(docs)
    if not docs:
        raise ValueError(f"{source} has no documents")

    limits = {name: per_task for name in TASKS}
    for row in docs:
        if not any(limits.values()):
            break
        text = row["text"]
        if len(text) < 120:
            continue
        for name, builder in TASKS.items():
            if limits[name] <= 0:
                continue
            try:
                built = builder(text)
            except (ValueError, IndexError):
                continue
            if built is None or not valid_row(built, max_chars):
                continue
            rows.append(built)
            produced[name] += 1
            limits[name] -= 1
            if not any(limits.values()):
                break

    kept = [r for r in rows if valid_row(r, max_chars)]
    deduped: dict[str, dict[str, Any]] = {}
    for row in kept:
        fingerprint = json.dumps(row["messages"], ensure_ascii=False, sort_keys=True)
        deduped.setdefault(fingerprint, row)
    rows = list(deduped.values())
    dropped_dupes = len(kept) - len(rows)
    rng.shuffle(rows)

    val_size = max(1, int(len(rows) * val_fraction)) if len(rows) >= 50 else max(1, len(rows) // 20)
    val_rows, train_rows = rows[:val_size], rows[val_size:]
    if not train_rows:
        raise ValueError("no training rows produced; lower --per-task filters or check the corpus")

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in (("train", train_rows), ("val", val_rows)):
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in subset:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "documents_seen": len(docs),
        "seed_rows": seed_count,
        "produced_by_task": dict(produced),
        "rows": len(rows),
        "dropped_duplicates": dropped_dupes,
        "train": len(train_rows),
        "val": len(val_rows),
        "val_fraction": val_fraction,
        "max_chars": max_chars,
        "seed": seed,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(
        "sft rows=%d train=%d val=%d tasks=%s",
        len(rows),
        len(train_rows),
        len(val_rows),
        dict(produced),
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the SFT dataset")
    parser.add_argument("--clean", default="data/clean")
    parser.add_argument("--out", default="data/sft")
    parser.add_argument("--seed-file", default="data/sft/seed.jsonl")
    parser.add_argument("--per-task", type=int, default=4000)
    parser.add_argument("--max-chars", type=int, default=1600)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        report = build(
            Path(args.clean),
            Path(args.out),
            seed_path=Path(args.seed_file),
            per_task=args.per_task,
            max_chars=args.max_chars,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({k: report[k] for k in ("rows", "train", "val", "produced_by_task")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
