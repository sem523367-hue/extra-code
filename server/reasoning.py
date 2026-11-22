"""Brain-first answering: own answer -> search -> verify -> final.

The model's own knowledge is the PRIMARY source; the internet and the local
KB only reinforce it. Three phases:

  Phase 1 (own answer): the model answers from its own weights, no context,
                     and lists what it is unsure about.
  Phase 2 (search):    python retrieves KB + web ONLY for the uncertain
                     points (or the whole question when the model flags it).
  Phase 3 (verify):    the model compares its draft with the findings and
                     writes the final answer: keep what matched, fix what
                     the sources contradict, mark what stays unconfirmed.

Any phase failing degrades to the previous phase's output, never to an error.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# One canonical prompt file, shared with the API and the CLI: editing it
# changes the identity in every phase on the next restart.
_IDENTITY_FILE = (
    Path(__file__).resolve().parent.parent / "tools" / "extra-code-system.txt"
)
_FALLBACK_IDENTITY = (
    "Ты — Extra Code v1, локальная ИИ-система команды Extra Code. "
    "Ты не продукт внешних компаний. На вопрос «кто ты» отвечай только: "
    "Extra Code v1."
)


def _load_identity() -> str:
    try:
        text = _IDENTITY_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return _FALLBACK_IDENTITY
    return text or _FALLBACK_IDENTITY


IDENTITY = _load_identity()

_DRAFT_TASK = (
    "## Задача этого прохода\n"
    "Ответь на вопрос СВОИМИ знаниями, без поиска и без источников.\n"
    "Формат строго из двух меток:\n"
    "ОТВЕТ: <твой ответ по существу, кратко и точно>\n"
    "НЕУВЕРЕН: <1-3 конкретных момента, в которых ты не уверен — факты, "
    "числа, даты; напиши «ничего», если уверен во всём>"
)

_VERIFY_TASK = (
    "## Задача этого прохода\n"
    "У тебя есть ЧЕРНОВИК твоего ответа и НАЙДЕННЫЕ источники. Работай "
    "строго по шагам:\n"
    "1. Сначала ВЫПИШИ из источников конкретные факты по вопросу — "
    "формулы, числа, названия. Не выдумывай: есть в источнике — пиши, "
    "нет — не пиши.\n"
    "2. Сравни факты с черновиком. Факт опровергает черновик — исправь. "
    "Подтверждает — оставь. В источниках нет — оставь из черновика с "
    "пометкой «(не подтверждено)».\n"
    "Формат строго из трёх меток:\n"
    "ФАКТЫ: <конкретные факты из источников по вопросу>\n"
    "ОТВЕТ: <итоговый ответ, исправленный по фактам>\n"
    "ПРОВЕРКА: <одной строкой: что подтвердилось, что исправлено, что "
    "не подтверждено>"
)

DRAFT_SYSTEM = f"{IDENTITY}\n\n{_DRAFT_TASK}"
VERIFY_SYSTEM = f"{IDENTITY}\n\n{_VERIFY_TASK}"


@dataclass
class ReasoningTrace:
    """Everything the phases produced, for display and debugging."""

    question: str
    draft: str = ""
    uncertain: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    kb_chunks: list[dict] = field(default_factory=list)
    web_pages: list[dict] = field(default_factory=list)
    verification: str = ""
    answer: str = ""

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "draft": self.draft,
            "uncertain": self.uncertain,
            "queries": self.queries,
            "kb_chunks": self.kb_chunks,
            "web_pages": self.web_pages,
            "verification": self.verification,
            "answer": self.answer,
        }


def split_two_labels(raw: str, first: str, second: str) -> tuple[str, str]:
    """Tolerant two-label split; missing second label means all text is first
    (losing the answer is worse than losing the split)."""
    text = raw.strip()
    m1 = re.search(re.escape(first) + r"\s*[:：]", text)
    m2 = re.search(re.escape(second) + r"\s*[:：]", text)
    if not m1:
        return text, ""
    if not m2:
        return text[:m1.start()].strip() or text, ""
    if m2.end() <= m1.end():
        first_text = text[m2.end():m1.start()].strip()
        second_text = text[m1.end():].strip()
        return second_text, first_text
    first_text = text[m1.end():m2.start()].strip()
    second_text = text[m2.end():].strip()
    return first_text, second_text


def parse_draft(raw: str) -> tuple[str, list[str]]:
    """Split the draft response into (answer, uncertain points)."""
    answer, unsure = split_two_labels(raw, "ОТВЕТ", "НЕУВЕРЕН")
    points = []
    if unsure and unsure.lower() not in {"ничего", "ничего.", "нет"}:
        for part in re.split(r"[;\n•]", unsure):
            part = part.strip(" -•*")
            if part and part.lower() not in {"ничего", "ничего."}:
                points.append(part)
    if not answer:
        answer = raw.strip()
    return answer.strip(), points[:3]


def build_queries(question: str, uncertain: list[str]) -> list[str]:
    """Queries come from the uncertain points when present, else the question.
    English retrieves best: both the local KB and web indexes skew English,
    and the model writes English queries more reliably than Russian."""
    if not uncertain:
        return [question]
    out = [question]
    for point in uncertain:
        out.append(point)
    return out[:3]


def retrieve(
    queries: list[str],
    *,
    use_kb: bool,
    use_web: bool,
    pages_per_query: int = 2,
    kb_per_query: int = 3,
    max_chars: int = 1400,
    web_builder=None,
) -> tuple[list[dict], list[dict]]:
    """Run every query against the KB and the web, deduplicate by source.

    web_builder is injected so the module stays importable without network."""
    kb_chunks: list[dict] = []
    web_pages: list[dict] = []
    seen_kb: set[str] = set()
    seen_web: set[str] = set()

    if use_kb:
        try:
            from server.kb import KnowledgeBase

            kb = KnowledgeBase.get()
            for query in queries:
                for hit in kb.query(query, k=kb_per_query):
                    key = f"{hit['src']}:{hash(hit['text']) & 0xFFFFFFFF}"
                    if key not in seen_kb:
                        seen_kb.add(key)
                        kb_chunks.append(hit)
        except FileNotFoundError:
            pass

    if use_web and web_builder is not None:
        for query in queries:
            try:
                context = web_builder(query, pages=pages_per_query,
                                      max_chars=max_chars)
            except Exception:
                continue
            if not context or not context.text:
                continue
            for source in context.sources:
                if source not in seen_web:
                    seen_web.add(source)
                    web_pages.append({"src": source, "text": context.text})

    return kb_chunks, web_pages


def build_verify_messages(
    question: str,
    draft: str,
    kb_chunks: list[dict],
    web_pages: list[dict],
) -> list[dict[str, str]]:
    """Assemble the verification prompt: draft + findings -> final answer.

    A worked example rides along because small models follow shown format
    far more reliably than they follow described format."""
    parts: list[str] = [
        "Пример, как работать (это пример, не ответ на вопрос):",
        "ЧЕРНОВИК: Пирит имеет формулу FeO.",
        "[источник] Пирит — минерал FeS2, золотистого цвета.",
        "ФАКТЫ: пирит — FeS2",
        "ОТВЕТ: Формула пирита — FeS2. (исправлено: в черновике было FeO)",
        "ПРОВЕРКА: черновик опровергнут и исправлен по источнику.",
        "",
        f"Вопрос: {question}",
        "",
        f"ЧЕРНОВИК:\n{draft}",
        "",
    ]
    if kb_chunks:
        joined = "\n\n".join(
            f"[{c['src']}]\n{c['text']}" for c in kb_chunks[:5]
        )
        parts.append(f"Источники — база знаний Extra Code:\n\n{joined}")
    if web_pages:
        joined = "\n\n".join(
            f"[{p['src']}]\n{p['text'][:1400]}" for p in web_pages[:3]
        )
        parts.append(f"Источники — интернет:\n\n{joined}")
    if len(parts) == 4:
        parts.append("Источники: поиск ничего не нашёл. Отвечай своими знаниями.")
    return [
        {"role": "system", "content": VERIFY_SYSTEM},
        {"role": "user", "content": "\n".join(parts)},
    ]


def parse_verify(raw: str) -> tuple[str, str]:
    """Split the verify response into (final answer, check note).

    ФАКТЫ is folded into the check note: it exists to force the model to
    read the sources, not to be shown separately."""
    text = raw.strip()
    facts_m = re.search(r"ФАКТЫ\s*[:：]", text)
    # ОТВЕТ/ОТВЕТЬ/ОТВЕТЫ — small models misspell the label.
    answer_m = re.search(r"ОТВЕТ\w{0,2}\s*[:：]", text)
    check_m = re.search(r"ПРОВЕРК\w{0,2}\s*[:：]", text)

    # Default: everything is the answer — losing the answer is worse.
    answer, check = text, ""

    if answer_m and check_m and check_m.end() > answer_m.end():
        answer = text[answer_m.end():check_m.start()].strip()
        check = text[check_m.end():].strip()
    elif answer_m:
        answer = text[answer_m.end():].strip()

    if facts_m and answer_m and facts_m.start() < answer_m.start():
        facts = text[facts_m.end():answer_m.start()].strip()
        if facts:
            check = f"факты: {facts}" + (f" | {check}" if check else "")

    # Label echo: the model sometimes opens its reply with the draft's
    # own label instead of the prescribed ones.
    answer = re.sub(r"^ЧЕРНОВИК\s*[:：]\s*", "", answer)
    # Source echo: the model sometimes repeats the prompt's source block
    # after its answer — the user already gets sources separately.
    answer = re.split(r"\n?\s*Источники\s*[—-]\s", answer)[0].rstrip()
    return answer, check


def build_draft_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": DRAFT_SYSTEM},
        {"role": "user", "content": question},
    ]

