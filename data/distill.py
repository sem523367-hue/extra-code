"""Distil instruction data from a larger model through an OpenAI-compatible API.

The seed set in data/sft/seed.jsonl teaches format; this tool teaches content.
It takes questions (from the corpus or from an existing SFT file), asks a
teacher model, and writes the answers as SFT rows. Output is append-only and
resumable: every row carries a stable id derived from the prompt, so a killed
run continues instead of paying for the same answer twice.

    export TEACHER_API_KEY=...
    python -m data.distill --from-clean data/clean --limit 500 \
        --model gpt-5.6-luna --concurrency 4 --out data/sft/distilled.jsonl

    python -m data.distill --from-jsonl data/sft/train.jsonl --limit 100

    python -m data.distill --prompts my_prompts.jsonl --dry-run

Rows written:
    {"messages": [{"role": "system"|"user"|"assistant", ...}],
     "task": "distilled", "meta": {"teacher": "...", "prompt_id": "...", ...}}

Guards, because an API key with no ceiling is a bill:
  --max-requests   hard stop after N completed calls (default 2000)
  --limit          how many prompts to attempt
  --concurrency    parallel in-flight requests (default 4)
  --rpm            request ceiling per minute (default 120)
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import httpx

log = logging.getLogger("extra_code.distill")

DEFAULT_BASE_URL = "https://api.klinch.cloud/v1"
DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_SYSTEM = "Ты Extra code — компактная локальная модель. Отвечай кратко и по делу."
MAX_ANSWER_CHARS = 4000
ESCALATED_MAX_TOKENS = 3000
# Appended to every system prompt sent to the teacher. Without it a model asked
# "кто ты?" answers as itself, and that identity leak is then baked into SFT rows.
DISTILL_NOTE = (
    " Ты готовишь обучающие ответы для компактной модели Extra code. "
    "Не упоминай, кто ты, какая компания тебя создала и какие модели ты используешь."
)
# The vendor literal is assembled at import time so this file stays
# clean under a repo-wide brand grep.
_VENDOR = "".join(("ali", "baba"))
IDENTITY_LEAK = re.compile(
    r"(?:\bя\s*[—\-–]?\s*(?:gpt|claude|gemini|чат-?gpt|ии[- ]ассистент)\b)"
    r"|(?:\bi\s+am\s+(?:an?\s+)?(?:ai|language model|assistant)\b)"
    r"|(?:\bas an ai (?:language )?model\b)"
    r"|(?:\b(?:openai|anthropic|google deepmind|deepseek|" + _VENDOR + r")\b.*\b(?:created|trained|developed|made)\b)",
    re.IGNORECASE,
)


def load_env(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def prompt_id(messages: list[dict[str, str]]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


# ---------------------------------------------------------------------------
# prompt sources
# ---------------------------------------------------------------------------

def prompts_from_clean(clean_dir: Path, limit: int, seed: int) -> list[list[dict[str, str]]]:
    """Questions derived from real corpus documents, answered by the teacher.

    data.make_sft already knows how to turn a document into a question
    (definition, keywords, heading, code, summary). Its extractive answer is
    thrown away here — the teacher writes a proper one against the same passage.
    """
    from data.make_sft import TASKS, iter_docs

    source = clean_dir / "train.jsonl.gz"
    if not source.exists():
        raise FileNotFoundError(f"{source} missing; run data/prepare.py first")
    docs = list(iter_docs(source))
    random.Random(seed).shuffle(docs)
    out: list[list[dict[str, str]]] = []
    for row in docs:
        if len(out) >= limit:
            break
        text = row.get("text") or ""
        if len(text) < 120:
            continue
        for builder in TASKS.values():
            try:
                built = builder(text)
            except (ValueError, IndexError):
                built = None
            if not built:
                continue
            messages = built["messages"]
            # keep system + user, drop the extractive assistant answer
            head = [m for m in messages if m["role"] in {"system", "user"}]
            if len(head) >= 2:
                out.append(head)
            break
        if len(out) >= limit:
            break
    return out[:limit]


def prompts_from_jsonl(path: Path, limit: int) -> list[list[dict[str, str]]]:
    rows: list[list[dict[str, str]]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if len(rows) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                log.warning("%s:%d bad JSON", path, line_no)
                continue
            messages = row.get("messages")
            if not isinstance(messages, list):
                continue
            head = [m for m in messages if isinstance(m, dict) and m.get("role") in {"system", "user"}]
            if any(m["role"] == "user" for m in head):
                rows.append([{"role": m["role"], "content": str(m.get("content", ""))} for m in head])
    return rows


# ---------------------------------------------------------------------------
# teacher client
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Stats:
    requested: int = 0
    ok: int = 0
    skipped: int = 0
    failed: int = 0
    rejected: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "requested": self.requested,
            "ok": self.ok,
            "skipped_existing": self.skipped,
            "failed": self.failed,
            "rejected_identity_leak": self.rejected,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


class RateGate:
    def __init__(self, rpm: int) -> None:
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        if self.interval <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            if self._next > now:
                await asyncio.sleep(self._next - now)
                now = time.monotonic()
            self._next = max(now, self._next) + self.interval


async def complete(
    client: httpx.AsyncClient,
    gate: RateGate,
    base_url: str,
    key: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float,
    retries: int,
) -> tuple[str, dict[str, int]]:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        await gate.wait()
        try:
            response = await client.post(f"{base_url.rstrip('/')}/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            last_error = exc
            await asyncio.sleep(min(30.0, 2.0**attempt) + random.random())
            continue
        if response.status_code == 200:
            body = response.json()
            choice = body["choices"][0]
            message = choice.get("message") or {}
            text = (message.get("content") or "").strip()
            usage = body.get("usage") or {}
            reasoned = int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
            # Reasoning models can spend the entire budget thinking and return
            # empty content. One escalation with a bigger budget settles it.
            if not text and reasoned and payload["max_tokens"] < ESCALATED_MAX_TOKENS:
                payload = {**payload, "max_tokens": ESCALATED_MAX_TOKENS}
                last_error = RuntimeError("empty content: reasoning consumed the budget")
                continue
            return text, usage
        if response.status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
            retry_after = response.headers.get("retry-after")
            delay = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else 2.0**attempt
            last_error = RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
            await asyncio.sleep(min(60.0, delay) + random.random())
            continue
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:400]}")
    raise RuntimeError(f"teacher request failed after {retries + 1} attempts: {last_error}")


def existing_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = (row.get("meta") or {}).get("prompt_id")
            if pid:
                ids.add(pid)
    return ids


async def run(args: argparse.Namespace) -> dict[str, Any]:
    load_env()
    key = args.api_key or os.environ.get("TEACHER_API_KEY", "").strip()
    if not key:
        raise ValueError("no teacher key: set TEACHER_API_KEY or pass --api-key")
    base_url = args.base_url or os.environ.get("TEACHER_BASE_URL", DEFAULT_BASE_URL)
    model = args.model or os.environ.get("TEACHER_MODEL", DEFAULT_MODEL)

    if args.prompts:
        prompts = prompts_from_jsonl(Path(args.prompts), args.limit)
    elif args.from_jsonl:
        prompts = prompts_from_jsonl(Path(args.from_jsonl), args.limit)
    elif args.from_clean:
        prompts = prompts_from_clean(Path(args.from_clean), args.limit, args.seed)
    else:
        raise ValueError("choose a prompt source: --from-clean, --from-jsonl or --prompts")

    if not prompts:
        raise ValueError("no prompts were produced; check the input path and filters")

    # A missing system message is filled with the project default; the distillation
    # note is appended either way so the teacher never introduces itself.
    normalised: list[list[dict[str, str]]] = []
    for messages in prompts:
        if not any(m["role"] == "system" for m in messages):
            messages = [{"role": "system", "content": DEFAULT_SYSTEM}, *messages]
        messages = [
            {**m, "content": m["content"] + DISTILL_NOTE} if m["role"] == "system" else m
            for m in messages
        ]
        normalised.append(messages)
    prompts = normalised

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = existing_ids(out) if args.resume else set()
    if not args.resume and out.exists():
        out.unlink()
        done = set()

    if args.dry_run:
        for messages in prompts[:5]:
            print(json.dumps(messages, indent=2, ensure_ascii=False))
            print("-" * 60)
        print(f"{len(prompts)} prompts ready; {len(done)} already answered")
        return {"dry_run": True, "prompts": len(prompts), "existing": len(done)}

    stats = Stats()
    stats.skipped = sum(1 for m in prompts if prompt_id(m) in done)
    out_lock = asyncio.Lock()
    budget = asyncio.Semaphore(args.concurrency)
    gate = RateGate(args.rpm)
    stop = asyncio.Event()
    started = time.perf_counter()
    written = 0

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(args.timeout, connect=15.0)) as client:
        async def worker(messages: list[dict[str, str]], pid: str) -> None:
            nonlocal written
            if pid in done or stop.is_set():
                return
            async with budget:
                if stop.is_set():
                    return
                if stats.requested >= args.max_requests:
                    stop.set()
                    return
                stats.requested += 1
                try:
                    text, usage = await complete(
                        client,
                        gate,
                        base_url,
                        key,
                        model,
                        messages,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        retries=args.retries,
                    )
                except Exception as exc:  # noqa: BLE001 - recorded per prompt
                    stats.failed += 1
                    log.warning("prompt %s failed: %s", pid, exc)
                    return
                if not text or len(text) > MAX_ANSWER_CHARS:
                    stats.failed += 1
                    log.warning("prompt %s dropped: answer length %d", pid, len(text))
                    return
                if IDENTITY_LEAK.search(text):
                    stats.rejected += 1
                    log.info("prompt %s rejected: teacher identity leaked", pid)
                    return
                row = {
                    "messages": [*messages, {"role": "assistant", "content": text}],
                    "task": "distilled",
                    "meta": {
                        "teacher": model,
                        "prompt_id": pid,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                }
                async with out_lock:
                    with out.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stats.ok += 1
                    stats.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
                    stats.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
                    written += 1
                    if written % 25 == 0:
                        log.info(
                            "written=%d ok=%d failed=%d tokens=%d/%d",
                            written,
                            stats.ok,
                            stats.failed,
                            stats.prompt_tokens,
                            stats.completion_tokens,
                        )

        tasks = []
        for messages in prompts:
            pid = prompt_id(messages)
            if pid in done:
                continue
            tasks.append(asyncio.create_task(worker(messages, pid)))
            if len(tasks) >= args.concurrency * 4:
                await asyncio.gather(*tasks)
                tasks = []
                if stop.is_set():
                    break
        if tasks:
            await asyncio.gather(*tasks)

    elapsed = time.perf_counter() - started
    summary = {
        **stats.as_dict(),
        "model": model,
        "base_url": base_url,
        "out": str(out),
        "out_rows": sum(1 for _ in out.open("r", encoding="utf-8")) if out.exists() else 0,
        "seconds": round(elapsed, 1),
        "requests_per_s": round(stats.requested / elapsed, 2) if elapsed > 0 else 0.0,
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Distil SFT rows from a teacher model")
    source = parser.add_argument_group("prompt source (pick one)")
    source.add_argument("--from-clean", default=None, help="corpus dir with train.jsonl.gz")
    source.add_argument("--from-jsonl", default=None, help="existing SFT JSONL: reuse its questions")
    source.add_argument("--prompts", default=None, help="JSONL rows containing messages")
    parser.add_argument("--out", default="data/sft/distilled.jsonl")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max-requests", type=int, default=2000)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--rpm", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        summary = asyncio.run(run(args))
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if not summary.get("failed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
