"""Ask a question with live web context, answered by any OpenAI-compatible server.

This is what "the model can use the internet" actually means at 25M parameters:
the searching happens in Python, the model only ever sees a short block of
retrieved text plus the question. There is no tool-call protocol to get wrong and
no context budget to blow, because the retrieval layer already decided what is
worth looking at.

    python -m tools.ask "кто изобрёл динамит"
    python -m tools.ask "GIL в Python" --base-url http://127.0.0.1:8080/v1 --no-web
    python -m tools.ask "плотность ртути" --context-only

Sources are always printed. When retrieval finds nothing, the model is told so
explicitly instead of being left to invent an answer.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx

from server.web import build_context
from server.identity import scrub as identity_scrub
from server import reasoning as reasoning_mod

# One canonical prompt file: the same text the API injects, so the CLI and
# the HTTP door answer with the same identity.
_SYSTEM_FILE = Path(__file__).resolve().parent.parent / "tools" / "extra-code-system.txt"


def _load_system() -> str:
    try:
        text = _SYSTEM_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        text = ""
    return text or (
        "Ты — Extra Code v1, локальная ИИ-система команды Extra Code. "
        "На вопрос о том, кто ты, отвечай только: Extra Code v1. "
        "Никогда не называй внешние компании и сторонние модели."
    )


SYSTEM = _load_system()
NO_CONTEXT = "Фрагменты не найдены. Отвечай только тем, что знаешь сам, и пометь это."


def build_messages(
    question: str, kb_text: str, web_text: str, *, use_kb: bool, use_web: bool
) -> list[dict[str, str]]:
    blocks: list[str] = []
    if use_kb:
        blocks.append(
            f"База знаний Extra Code:\n\n{kb_text.strip()}"
            if kb_text.strip()
            else "База знаний Extra Code: подходящего материала не найдено."
        )
    if use_web:
        blocks.append(
            f"Фрагменты из интернета:\n\n{web_text.strip()}"
            if web_text.strip()
            else NO_CONTEXT
        )
    body = ("\n\n".join(blocks) + f"\n\nВопрос: {question}") if blocks else question
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}]


def _chat(base_url: str, api_key: str | None, payload: dict[str, Any],
          timeout: float) -> str:
    """One chat call against the OpenAI-compatible endpoint; returns content.

    Sampling is pinned here because quantized small models fall into
    repetition loops with plain greedy defaults; these values hold long
    Russian answers without hurting precision."""
    payload = {
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.18,
        "presence_penalty": 0.3,
        **payload,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    with httpx.Client(headers=headers, timeout=httpx.Timeout(timeout, connect=10.0)) as client:
        response = client.post(f"{base_url.rstrip('/')}/chat/completions", json=payload)
        response.raise_for_status()
        body = response.json()
    return (((body.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()


def ask_reasoning(
    question: str,
    *,
    base_url: str,
    model: str,
    api_key: str | None,
    web: bool,
    local: bool,
    pages: int,
    max_chars: int,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> tuple[reasoning_mod.ReasoningTrace, list[str]]:
    """Own answer -> search the uncertain points -> verify -> final.

    The model's brain is primary; KB and internet only reinforce it."""
    trace = reasoning_mod.ReasoningTrace(question=question)

    # Phase 1: own answer, straight from the weights. Failure degrades to
    # search-only flow (draft left empty, verification still runs).
    try:
        raw_draft = _chat(base_url, api_key, {
            "model": model,
            "messages": reasoning_mod.build_draft_messages(question),
            "max_tokens": max(2 * max_tokens, 320),
            "temperature": temperature,
        }, timeout)
        trace.draft, trace.uncertain = reasoning_mod.parse_draft(raw_draft)
    except httpx.HTTPError:
        trace.draft, trace.uncertain = "", []

    # Phase 2: search — targeted at what the model itself flagged uncertain.
    trace.queries = reasoning_mod.build_queries(question, trace.uncertain)
    kb_chunks, web_pages = reasoning_mod.retrieve(
        trace.queries,
        use_kb=local,
        use_web=web,
        pages_per_query=max(1, pages - 1),
        max_chars=max_chars,
        web_builder=build_context,
    )
    trace.kb_chunks = kb_chunks
    trace.web_pages = web_pages

    # Phase 3: verify the draft against findings -> final answer.
    messages = reasoning_mod.build_verify_messages(
        question, trace.draft, kb_chunks, web_pages
    )
    try:
        raw = _chat(base_url, api_key, {
            "model": model,
            "messages": messages,
            "max_tokens": max(2 * max_tokens, 384),
            "temperature": temperature,
        }, timeout)
        trace.answer, trace.verification = reasoning_mod.parse_verify(raw)
    except httpx.HTTPError:
        trace.answer, trace.verification = trace.draft, "(фаза проверки не прошла — черновик)"
    if not trace.answer:
        trace.answer = trace.draft

    sources = [f"kb:{c['src']}" for c in kb_chunks]
    sources += [p["src"] for p in web_pages]
    return trace, sources


def ask(
    question: str,
    *,
    base_url: str,
    model: str,
    api_key: str | None,
    web: bool,
    local: bool,
    pages: int,
    max_chars: int,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> tuple[str, list[str]]:
    kb_text = ""
    kb_sources: list[str] = []
    if local:
        try:
            from server.kb import KnowledgeBase

            hits = KnowledgeBase.get().query(question, k=3)
            if hits:
                kb_text = "\n\n".join(str(hit["text"]) for hit in hits)
                kb_sources = [f"kb:{hit['src']}" for hit in hits]
        except FileNotFoundError:
            pass  # base not built yet: fall through to a plain or web answer
    context = build_context(question, pages=pages, max_chars=max_chars) if web else None
    text = context.text if context else ""
    sources = context.sources if context else []
    messages = build_messages(
        question, kb_text, text, use_kb=local, use_web=web
    )

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        # Same anti-repetition pin as _chat: long Russian answers loop
        # into one sentence forever with plain greedy sampling.
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.18,
        "presence_penalty": 0.3,
    }
    with httpx.Client(headers=headers, timeout=httpx.Timeout(timeout, connect=10.0)) as client:
        response = client.post(f"{base_url.rstrip('/')}/chat/completions", json=payload)
        response.raise_for_status()
        body = response.json()
    answer = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    return answer.strip(), sources


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    parser = argparse.ArgumentParser(description="web-grounded question answering")
    parser.add_argument("question")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="extra-code")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--pages", type=int, default=3)
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--no-web", dest="web", action="store_false")
    parser.add_argument("--no-local", dest="local", action="store_false",
                        help="disable the local knowledge base (on by default)")
    parser.add_argument("--no-reasoning", dest="reasoning", action="store_false",
                        help="single-pass answer instead of draft->search->verify")
    parser.add_argument("--show-trace", action="store_true",
                        help="print the draft and uncertain points too")
    parser.add_argument("--context-only", action="store_true",
                        help="print the retrieved context and skip the model call")
    args = parser.parse_args(argv)

    if args.context_only:
        context = build_context(args.question, pages=args.pages, max_chars=args.max_chars)
        print(json.dumps(context.as_dict(), indent=2, ensure_ascii=False))
        return 0 if context.text else 1

    try:
        if args.reasoning:
            trace, sources = ask_reasoning(
                args.question,
                base_url=args.base_url,
                model=args.model,
                api_key=args.api_key,
                web=args.web,
                local=args.local,
                pages=args.pages,
                max_chars=args.max_chars,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
            )
            answer = trace.answer
            if args.show_trace:
                print("--- черновик (своим умом) ---")
                print(identity_scrub(trace.draft) or "(пусто)")
                if trace.uncertain:
                    print("\n--- не уверен ---")
                    for point in trace.uncertain:
                        print(f"- {point}")
                if trace.queries:
                    print("\n--- поисковые запросы ---")
                    for query in trace.queries:
                        print(f"- {query}")
                print()
        else:
            answer, sources = ask(
                args.question,
                base_url=args.base_url,
                model=args.model,
                api_key=args.api_key,
                web=args.web,
                local=args.local,
                pages=args.pages,
                max_chars=args.max_chars,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
            )
    except httpx.HTTPError as exc:
        print(f"server error: {exc}", file=sys.stderr)
        return 2

    print(identity_scrub(answer or "") or "(empty answer)")
    if sources:
        print("\n--- sources ---")
        for url in sources:
            print(f"- {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
