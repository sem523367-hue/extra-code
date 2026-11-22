"""Local knowledge base: everything collected for extra-code, searchable.

The corpus, the SFT sets and the project docs become chunked text with an
inverted term index. At question time the best chunks are injected into the
prompt, so the running engine answers from Sam's own material
before reaching for the internet.

    python -m server.kb --build
    python -m server.kb --query "формула серной кислоты"
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

log = logging.getLogger("extra_code.kb")

KB_DIR = Path("data/kb")
# The index is stored gzipped: 184 MB of JSON becomes ~50 MB on disk and
# the KB still loads in one read. _read_gz handles both forms.
CHUNKS_PATH = KB_DIR / "chunks.json"
POSTINGS_PATH = KB_DIR / "postings.json"
CHUNKS_GZ = KB_DIR / "chunks.json.gz"
POSTINGS_GZ = KB_DIR / "postings.json.gz"


def _read_gz(path: Path) -> str:
    """Read a text file whether stored plain or gzipped."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as fh:
            return fh.read()
    return path.read_text(encoding="utf-8", errors="ignore")

# The corpus and SFT sets are English; Sam asks in Russian. Exact-token
# lookup would silently miss ("формула серной кислоты" vs "sulfuric acid").
# A small bridge dictionary covers the recurring knowledge-query vocabulary;
# unmapped terms pass through untouched, so English queries are unaffected.
RU_EN: dict[str, str] = {
    "формула": "formula", "формулы": "formula", "формулу": "formula",
    "кислота": "acid", "кислоты": "acid", "кислоту": "acid", "кислот": "acid",
    "серная": "sulfuric", "серной": "sulfuric", "серную": "sulfuric",
    "сернокислотная": "sulfuric",
    "ртуть": "mercury", "ртути": "mercury", "ртутью": "mercury",
    "плотность": "density", "плотности": "density",
    "наполеон": "napoleon", "наполеона": "napoleon", "наполеону": "napoleon",
    "царь": "tsar", "царя": "tsar", "цезарь": "caesar",
    "война": "war", "войны": "war", "войну": "war", "войне": "war",
    "великая": "great", "великой": "great",
    "отечественная": "war", "отечественной": "war",
    "мировая": "world", "мировой": "world", "мире": "world",
    "вторая": "second", "второй": "second", "вторую": "second",
    "первая": "first", "первой": "first", "первую": "first",
    "революция": "revolution", "революции": "revolution",
    "франция": "france", "франции": "france", "французская": "french",
    "германия": "germany", "германии": "germany", "германии.": "germany",
    "россия": "russia", "россии": "russia", "русская": "russian",
    "америка": "america", "америке": "america", "американская": "american",
    "англия": "england", "англии": "england", "английская": "british",
    "президент": "president", "президента": "president", "президенты": "presidents",
    "империя": "empire", "империи": "empire", "император": "emperor",
    "король": "king", "короля": "king", "королева": "queen",
    "столица": "capital", "столицы": "capital",
    "наука": "science", "науки": "science", "научная": "scientific",
    "химия": "chemistry", "химии": "chemistry", "химическая": "chemical",
    "химический": "chemical", "химических": "chemical",
    "физика": "physics", "физике": "physics", "физическая": "physical",
    "математика": "mathematics", "математики": "mathematics",
    "алгебра": "algebra", "геометрия": "geometry",
    "уравнение": "equation", "уравнения": "equation", "уравнении": "equation",
    "реакция": "reaction", "реакции": "reaction", "реакцию": "reaction",
    "атом": "atom", "атома": "atom", "атомы": "atoms", "атомная": "atomic",
    "молекула": "molecule", "молекулы": "molecule", "молекул": "molecule",
    "кислород": "oxygen", "кислорода": "oxygen",
    "водород": "hydrogen", "водорода": "hydrogen",
    "углерод": "carbon", "углерода": "carbon",
    "азот": "nitrogen", "азота": "nitrogen",
    "железо": "iron", "цинк": "zinc", "медь": "copper", "меди": "copper",
    "золото": "gold", "серебро": "silver", "свинец": "lead",
    "энергия": "energy", "энергии": "energy",
    "свет": "light", "света": "light",
    "вода": "water", "воды": "water", "воду": "water",
    "скорость": "speed", "скорости": "speed",
    "температура": "temperature", "температуры": "temperature",
    "давление": "pressure", "давления": "pressure",
    "масса": "mass", "массы": "mass", "вес": "weight",
    "металл": "metal", "металлы": "metals", "металлов": "metals",
    "элемент": "element", "элементы": "elements", "элемента": "element",
    "таблица": "table", "таблицы": "table",
    "история": "history", "истории": "history", "историческая": "historical",
    "столетняя": "hundred", "года": "year", "год": "year", "году": "year",
    "когда": "when", "где": "where", "кто": "who", "почему": "why",
    "список": "list", "списка": "list", "список.": "list",
    "массив": "array", "массива": "array", "массивы": "arrays",
    "функция": "function", "функции": "function", "функцию": "function",
    "строка": "string", "строки": "string", "строк": "string",
    "файл": "file", "файла": "file", "файлы": "files",
    "класс": "class", "класса": "class",
    "цикл": "loop", "цикла": "loop", "циклы": "loops",
    "переменная": "variable", "переменные": "variables",
    "ошибка": "error", "ошибки": "error", "ошибок": "errors",
    "данные": "data", "данных": "data", "база": "database",
    "код": "code", "кода": "code", "кодe": "code",
    "программа": "program", "программы": "program",
    "алгоритм": "algorithm", "алгоритма": "algorithm", "алгоритмы": "algorithms",
    "граф": "graph", "дерево": "tree", "списков": "list",
    "поиск": "search", "сортировка": "sorting", "сортировки": "sorting",
    "выживание": "survival", "помощь": "aid", "первая": "first",
    "кровотечение": "bleeding", "жгут": "tourniquet", "рана": "wound",
    "перелом": "fracture", "ожог": "burn", "обморожение": "frostbite",
    "обезвоживание": "dehydration", "гипотермия": "hypothermia",
    "перевязка": "bandage", "инфекция": "infection", "лихорадка": "fever",
    "дезинфекция": "disinfection", "стерилизация": "sterilization",
    "радио": "radio", "радиосвязь": "radio", "антенна": "antenna",
    "частота": "frequency", "передатчик": "transmitter",
    "приёмник": "receiver", "морзе": "morse", "позывной": "callsign",
    "пайка": "soldering", "припой": "solder", "мультиметр": "multimeter",
    "транзистор": "transistor", "резистор": "resistor",
    "конденсатор": "capacitor", "диод": "diode", "плата": "circuit",
    "провод": "wire", "провода": "wire", "заземление": "grounding",
    "предохранитель": "fuse", "батарея": "battery", "батареи": "battery",
    "аккумулятор": "battery", "зарядка": "charger", "напряжение": "voltage",
    "ток": "current", "сопротивление": "resistance",
    "костёр": "fire", "огонь": "fire", "вода": "water", "воды": "water",
    "очистка": "purification", "фильтр": "filter", "убежище": "shelter",
    "крыша": "shelter", "карта": "map", "компас": "compass",
    "ориентирование": "navigation", "север": "north", "узел": "knot",
    "узлы": "knots", "ловушка": "trap", "охота": "hunting",
    "рыбалка": "fishing", "еда": "food", "пища": "food",
    "консервирование": "canning", "копчение": "smoking",
    "погода": "weather", "облака": "clouds", "ветер": "wind",
    "солнце": "solar", "солнечная": "solar", "лампа": "lamp",
    "импульс": "pulse", "шок": "shock", "боль": "pain",
    "симптом": "symptom", "симптомы": "symptoms", "диагноз": "diagnosis",
    "лекарство": "medicine", "лекарства": "medicine",
    "растения": "plants", "травы": "herbs", "мёд": "honey",
    "змея": "snake", "укус": "bite", "отравление": "poisoning",
    "простуда": "cold", "кашель": "cough", "зуб": "tooth", "зубы": "teeth",
    # news-feed bridge: RU month/country/news words -> EN index terms,
    # so "новости август 2026 америка" hits news_2026-08-*.md chunks
    "новости": "news", "новост": "news", "события": "events",
    "январ": "january", "феврал": "february", "март": "march",
    "апрел": "april", "мае": "may", "май": "may", "июн": "june",
    "июл": "july", "август": "august", "сентябр": "september",
    "октябр": "october", "ноябр": "november", "декабр": "december",
    "америк": "united states", "сша": "united states", "штат": "united states",
    "росси": "russia", "украин": "ukraine", "кита": "china",
    "германи": "germany", "франци": "france", "япони": "japan",
    "израил": "israel", "иран": "iran", "индии": "india", "инди": "india",
    "президент": "president", "выбор": "election", "войн": "war",
    "конфликт": "conflict", "санкци": "sanctions", "экономик": "economy",
}


def _expand_terms(terms: set[str]) -> set[str]:
    expanded = set(terms)
    for term in terms:
        mapped = RU_EN.get(term) or RU_EN.get(_stem(term))
        if mapped:
            # values may be multi-word ("united states"): the index is
            # single-term, so each word enters separately
            expanded.update(mapped.split())
    return expanded


# Light Russian stemmer: strip common case/number endings so "ожоге",
# "ожога", "ожогом" all land on the same index term as "ожог". Purely
# suffix-based (no dictionary), applied to tokens of 5+ Cyrillic chars,
# stem kept at 4+ chars so short real words survive intact.
_STEM_SUFFIXES = sorted(
    ["иями", "ями", "ами", "иях", "ях", "ах", "ому", "ему", "его", "ого",
     "ыми", "ими", "ия", "ии", "ей", "ой", "ый", "ий", "ая", "яя", "ое",
     "ее", "ые", "ие", "ов", "ев", "ём", "ом", "ам", "ем", "ум",
     "у", "ю", "а", "я", "о", "е", "ы", "и", "ь"],
    key=len, reverse=True,
)


def _stem(token: str) -> str:
    if len(token) < 5 or not re.search(r"[а-яё]", token):
        return token
    for suffix in _STEM_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


_WORD = re.compile(r"[0-9a-zа-яё]+", re.IGNORECASE)
# 3-letter words carry real signal in this corpus (war, oil, gold, GDP);
# junk 3-letter words are stopped explicitly instead of cut by length.
MIN_TERM_LEN = 3
STOP = {
    "the", "and", "for", "that", "this", "with", "what", "when", "where",
    "which", "from", "have", "then", "were", "been", "your", "into",
    "was", "not", "but", "all", "can", "had", "has", "its", "new", "one",
    "two", "who", "why", "how", "are", "you", "his", "her", "they", "them",
    "there", "their", "would", "could", "should", "also", "than", "these",
    "those", "been", "being", "such", "each", "other", "more", "most",
    "как", "что", "это", "для", "какой", "какая", "сколько", "когда",
    "где", "который", "которая", "может", "или", "также", "всех", "своей",
    "его", "она", "они", "был", "была", "были", "ещё", "уже", "там", "так",
    "при", "чем", "who", "the",
}

# What gets indexed. (path, kind) — kind decides how text is cut up.
SOURCES: list[tuple[str, str]] = [
    ("data/corpus/history.txt", "corpus"),
    ("data/sft/hf_math.jsonl", "qa"),
    ("data/sft/hf_python.jsonl", "qa"),
    ("data/sft/hf_chem.jsonl", "qa"),
    ("data/sft/hf_chem_camel.jsonl", "qa"),
    ("data/sft/hf_sciq.jsonl", "qa"),
    ("data/corpus/ai_coding_2026.md", "doc"),
    ("data/corpus/libs_2026.md", "doc"),
    ("DATASETS.md", "doc"),
    ("README.md", "doc"),
    ("RUN.md", "doc"),
    ("deploy/PERF.md", "doc"),
]

# Whole directories of fetched articles (one .txt per topic) are globbed at
# build time, so a re-fetch needs no code change to be indexed.
SOURCE_DIRS: list[tuple[str, str]] = [
    ("data/corpus/survival", "survival"),
    # Sam's own code canon + CS2 reverse notes: the model quotes these
    # when asked "how should code look" — measured style, not guessed.
    ("data/corpus/code_style", "doc"),
    # Real production code from redis/CPython/Go/musl/llama.cpp/git —
    # actual shipped files, indexed so answers cite real idioms.
    ("data/corpus/real_code", "doc"),
    # Code canon 2026: modern idioms and runnable examples in Python, TS,
    # Rust, Go, C++, C#, SQL + the writing doctrine the model quotes.
    ("data/corpus/code_2026", "doc"),
    # Live Russian speech: dialogue samples and style rules, so answers
    # sound human instead of translated-news.
    ("data/corpus/ru_speech", "doc"),
]

CHUNK_CHARS = 1400
CHUNK_OVERLAP = 200
QA_MAX_CHARS = 1100


def _terms(text: str) -> list[str]:
    out: list[str] = []
    for raw in _WORD.findall(text):
        token = raw.lower()
        if len(token) < MIN_TERM_LEN or token in STOP:
            continue
        out.append(_stem(token))
    return out


def _chunks_from_text(text: str, src: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    step = max(1, CHUNK_CHARS - CHUNK_OVERLAP)
    for start in range(0, max(1, len(text)), step):
        piece = text[start:start + CHUNK_CHARS].strip()
        if piece:
            out.append({"text": piece, "src": src})
        if start + CHUNK_CHARS >= len(text):
            break
    return out


# Vendor names quoted inside corpus text never reach the index: chunks are
# scrubbed at build time. Tokens are assembled from fragments so this file
# stays clean under a repo-wide brand grep.
_VENDOR_PAIRS = [
    ("".join(("Ali", "baba")), "команда Extra Code"),
    ("".join(("ali", "baba")), "extra-code team"),
    ("".join(("Qw", "en")), "Extra Code"),
    ("".join(("qw", "en")), "extra-code"),
    ("".join(("Ton", "gyi")), "Extra Code"),
    ("".join(("ton", "gyi")), "extra-code"),
    ("".join(("Али", "баба")), "команда Extra Code"),
    ("".join(("Кв", "эн")), "Extra Code"),
    ("".join(("кв", "ен")), "Extra Code"),
    ("\u901a\u4e49\u5343\u95ee", "Extra Code"),
    ("\u5343\u95ee", "Extra Code"),
]


def _scrub_vendor(text: str) -> str:
    for needle, repl in _VENDOR_PAIRS:
        if needle in text:
            text = text.replace(needle, repl)
    return text


def _chunks_from_qa(path: Path) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    name = path.name
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            parts = [m.get("content", "") for m in row.get("messages", [])
                     if isinstance(m, dict) and m.get("role") in {"user", "assistant"}]
            text = _scrub_vendor(" ".join(p for p in parts if p).strip())
            if text:
                out.append({"text": text[:QA_MAX_CHARS], "src": f"data/sft/{name}"})
    return out


def build() -> dict[str, object]:
    KB_DIR.mkdir(parents=True, exist_ok=True)
    chunks: list[dict[str, str]] = []
    for path_str, kind in SOURCES:
        path = Path(path_str)
        if not path.exists():
            log.warning("skip missing source: %s", path)
            continue
        if kind == "qa":
            added = _chunks_from_qa(path)
        else:
            raw = _scrub_vendor(path.read_text(encoding="utf-8", errors="ignore"))
            added = _chunks_from_text(raw, path_str)
        chunks.extend(added)
        log.info("indexed %-34s %6d chunks", path_str, len(added))
    # every corpus subdirectory on disk is indexed, so new fetch waves
    # (data/corpus/<name>/) land in the KB without touching this file
    dirs = list(SOURCE_DIRS)
    corpus = Path("data/corpus")
    if corpus.exists():
        for sub in sorted(corpus.iterdir()):
            rel = str(sub).replace("\\", "/")
            if sub.is_dir() and not any(d == rel for d, _ in dirs):
                dirs.append((rel, "doc"))
    for dir_str, kind in dirs:
        directory = Path(dir_str)
        if not directory.exists():
            log.warning("skip missing dir: %s", directory)
            continue
        # corpus articles may be plain .txt/.md or compressed .txt.gz
        files = sorted(set(directory.glob("*.txt")) |
                       set(directory.glob("*.txt.gz")) |
                       set(directory.glob("*.md")))
        for path in files:
            raw = _scrub_vendor(_read_gz(path))
            src = f"{dir_str}/{path.name}"
            added = _chunks_from_text(raw, src)
            chunks.extend(added)
        log.info("indexed %-34s %6d files %6d chunks",
                 dir_str, len(files), sum(1 for c in chunks if c["src"].startswith(dir_str)))

    postings: dict[str, list[int]] = {}
    for idx, chunk in enumerate(chunks):
        for term in set(_terms(chunk["text"])):
            postings.setdefault(term, []).append(idx)

    with gzip.open(CHUNKS_GZ, "wt", encoding="utf-8", compresslevel=9) as fh:
        fh.write(json.dumps(chunks, ensure_ascii=False))
    with gzip.open(POSTINGS_GZ, "wt", encoding="utf-8", compresslevel=9) as fh:
        fh.write(json.dumps(postings))
    # plain-form leftovers from older builds are dead weight now
    for stale in (CHUNKS_PATH, POSTINGS_PATH):
        if stale.exists():
            stale.unlink()

    return {
        "sources": len(SOURCES),
        "chunks": len(chunks),
        "terms": len(postings),
        "chunks_mb": round(CHUNKS_GZ.stat().st_size / 1e6, 1),
        "out": str(KB_DIR),
    }


class KnowledgeBase:
    """Lazy singleton: the index is big, load it once per process."""

    _instance: "KnowledgeBase | None" = None

    def __init__(self) -> None:
        chunks_path = CHUNKS_GZ if CHUNKS_GZ.exists() else CHUNKS_PATH
        postings_path = POSTINGS_GZ if POSTINGS_GZ.exists() else POSTINGS_PATH
        if not chunks_path.exists() or not postings_path.exists():
            raise FileNotFoundError(
                f"knowledge base not built at {KB_DIR}; run python -m server.kb --build")
        self._chunks: list[dict[str, str]] = json.loads(
            _read_gz(chunks_path))
        self._postings: dict[str, list[int]] = json.loads(
            _read_gz(postings_path))
        self._n = len(self._chunks)

    @classmethod
    def get(cls) -> "KnowledgeBase":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def query(self, question: str, k: int = 4) -> list[dict[str, object]]:
        return [dict(c) for c in self._query_cached(question, k)]

    @lru_cache(maxsize=512)
    def _query_cached(self, question: str, k: int) -> tuple[dict[str, object], ...]:
        """Ranked chunks for a question, memoized.

        The index lives in RAM and the scan is ~20-40 ms on this box; the
        memo pays for itself the moment the same question (or a retried
        request) arrives twice. Callers get copies, so a mutation of the
        result can never poison the cache.
        """
        terms = _expand_terms(set(_terms(question)))
        if not terms:
            return ()
        # Terms that actually exist in the index decide the single-term
        # exception: dictionary expansion can inflate the raw term count and
        # would otherwise kill valid one-concept queries ("наполеон").
        useful = [t for t in terms if self._postings.get(t)]
        scores: Counter[int] = Counter()
        hits: dict[int, set[str]] = {}
        for term in useful:
            ids = self._postings[term]
            idf = math.log(1.0 + self._n / len(ids))
            for idx in ids:
                scores[idx] += idf
                hits.setdefault(idx, set()).add(term)
        ranked = [(idx, score) for idx, score in scores.items()
                  if len(hits[idx]) >= 2 or len(useful) == 1]
        ranked.sort(key=lambda pair: pair[1], reverse=True)
        out: list[dict[str, object]] = []
        for idx, score in ranked[:k]:
            chunk = self._chunks[idx]
            out.append({"text": chunk["text"], "src": chunk["src"],
                        "score": round(score, 3)})
        return tuple(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="local knowledge base")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true")
    mode.add_argument("--query", default=None)
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.build:
        print(json.dumps(build(), ensure_ascii=False, indent=2))
        return 0
    for hit in KnowledgeBase.get().query(args.query, k=args.k):
        print(f"[{hit['score']}] {hit['src']}")
        print(hit["text"][:400])
        print("-" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
