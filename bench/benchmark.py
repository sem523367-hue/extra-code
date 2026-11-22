"""Latency and throughput benchmark for an OpenAI-compatible endpoint.

Four measurements, one run, one JSON report:

  stream       sequential streaming requests: TTFT, inter-token latency, tok/s
  batch        sequential non-streaming requests: end-to-end latency
  prefix       a long shared prefix, cold run then N warm runs: the TTFT ratio
               is the prefix-cache number the acceleration spec asks for
  concurrency  N parallel streaming requests: aggregate tok/s and per-request
               TTFT, to see continuous batching hold up under load

Token counts come from the `usage` block when the server returns one (set
`stream_options.include_usage`), otherwise from a character estimate that is
labelled `estimated: true` so nobody mistakes it for a measurement.

    python -m bench.benchmark --base-url http://127.0.0.1:8000 --api-key "$API_KEY"
    python -m bench.benchmark --base-url http://127.0.0.1:8080 --api-key "$LLAMA_API_KEY" \
        --llama-server --prefix-repeat 5 --concurrency 4

Exit code: 0 normally, 1 when --min-prefix-speedup is given and missed.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

SHARED_PREFIX = (
    "Ты компактная локальная модель Extra code. Ты помогаешь писать код, "
    "объясняешь технические темы и отвечаешь кратко. Ниже — правила, которые "
    "нужно соблюдать в каждом ответе.\n\n"
    "1. Отвечай по-русски, если вопрос задан по-русски.\n"
    "2. Код давай без пояснений, если не попросили объяснить.\n"
    "3. Не выдумывай факты: если не знаешь, скажи об этом прямо.\n"
    "4. Держи ответ в пределах трёх абзацев.\n"
    "5. Не повторяй вопрос перед ответом.\n\n"
    "Контекст проекта: сервер на FastAPI, движок llama.cpp, модель Extra code "
    "на 25 миллионов параметров, квантование Q4_K_M, контекст 1024 токена. "
    "Все запросы проходят авторизацию по Bearer-ключу и ограничены по RPM и TPM. "
    "Стриминг идёт через SSE с терминатором [DONE].\n\n"
) * 4


@dataclass(slots=True)
class Sample:
    ttft: float = 0.0
    latency: float = 0.0
    completion_tokens: int = 0
    prompt_tokens: int = 0
    chars: int = 0
    estimated: bool = False
    finish_reason: str | None = None
    error: str | None = None
    inter_token: list[float] = field(default_factory=list)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 4),
        "p50": round(_percentile(values, 0.50), 4),
        "p95": round(_percentile(values, 0.95), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


class Client:
    def __init__(self, base_url: str, api_key: str | None, timeout: float) -> None:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=10.0),
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
        )

    def close(self) -> None:
        self._client.close()

    def models(self) -> list[str]:
        response = self._client.get("/v1/models")
        response.raise_for_status()
        return [item["id"] for item in response.json().get("data", [])]

    def chat_stream(self, model: str, messages: list[dict[str, str]], max_tokens: int) -> Sample:
        sample = Sample()
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        started = time.perf_counter()
        try:
            with self._client.stream("POST", "/v1/chat/completions", json=payload) as response:
                if response.status_code != 200:
                    body = response.read().decode("utf-8", "replace")[:300]
                    sample.error = f"HTTP {response.status_code}: {body}"
                    sample.latency = time.perf_counter() - started
                    return sample
                usage: dict[str, int] = {}
                last = started
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw = line[6:].strip()
                    if raw == "[DONE]":
                        break
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if event.get("usage"):
                        usage = event["usage"]
                    for choice in event.get("choices") or []:
                        piece = (choice.get("delta") or {}).get("content")
                        now = time.perf_counter()
                        if piece:
                            if sample.ttft == 0.0:
                                sample.ttft = now - started
                            else:
                                sample.inter_token.append(now - last)
                            sample.chars += len(piece)
                        last = now
                        if choice.get("finish_reason"):
                            sample.finish_reason = choice["finish_reason"]
        except httpx.HTTPError as exc:
            sample.error = f"{type(exc).__name__}: {exc}"
            sample.latency = time.perf_counter() - started
            return sample

        sample.latency = time.perf_counter() - started
        if usage:
            sample.completion_tokens = int(usage.get("completion_tokens", 0))
            sample.prompt_tokens = int(usage.get("prompt_tokens", 0))
        if sample.completion_tokens == 0:
            sample.completion_tokens = max(1, round(sample.chars / 3.6))
            sample.estimated = True
        return sample

    def chat_once(self, model: str, messages: list[dict[str, str]], max_tokens: int) -> Sample:
        sample = Sample()
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
        }
        started = time.perf_counter()
        try:
            response = self._client.post("/v1/chat/completions", json=payload)
            sample.latency = time.perf_counter() - started
            if response.status_code != 200:
                sample.error = f"HTTP {response.status_code}: {response.text[:300]}"
                return sample
            body = response.json()
            choice = body["choices"][0]
            text = choice["message"].get("content") or ""
            usage = body.get("usage") or {}
            sample.chars = len(text)
            sample.finish_reason = choice.get("finish_reason")
            sample.completion_tokens = int(usage.get("completion_tokens", 0))
            sample.prompt_tokens = int(usage.get("prompt_tokens", 0))
            if sample.completion_tokens == 0:
                sample.completion_tokens = max(1, round(len(text) / 3.6))
                sample.estimated = True
        except httpx.HTTPError as exc:
            sample.error = f"{type(exc).__name__}: {exc}"
            sample.latency = time.perf_counter() - started
        return sample


def _messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def bench_stream(client: Client, model: str, requests: int, max_tokens: int) -> dict[str, Any]:
    prompts = [
        "Объясни, чем отличается поток от процесса.",
        "Напиши функцию на Python для бинарного поиска.",
        "Что делает флаг --parallel в llama-server?",
        "Перечисли три способа снизить TTFT.",
        "Как проверить, что модель не изменила выход после квантизации?",
    ]
    samples: list[Sample] = []
    for index in range(requests):
        sample = client.chat_stream(model, _messages(prompts[index % len(prompts)]), max_tokens)
        samples.append(sample)
        status = sample.error or f"ttft={sample.ttft * 1000:.0f}ms tok/s={_tok_s(sample):.1f}"
        print(f"  stream[{index + 1}/{requests}] {status}", file=sys.stderr)
    ok = [s for s in samples if not s.error]
    errors = [s.error for s in samples if s.error]
    ttfts = [s.ttft for s in ok if s.ttft > 0]
    tps = [_tok_s(s) for s in ok if _tok_s(s) > 0]
    inter = [x for s in ok for x in s.inter_token]
    return {
        "requests": requests,
        "errors": errors,
        "estimated_tokens": any(s.estimated for s in ok),
        "ttft_s": _summary(ttfts),
        "inter_token_s": _summary(inter),
        "tokens_per_s": _summary(tps),
        "completion_tokens": _summary([float(s.completion_tokens) for s in ok]),
        "latency_s": _summary([s.latency for s in ok]),
    }


def bench_batch(client: Client, model: str, requests: int, max_tokens: int) -> dict[str, Any]:
    prompts = [
        "Что такое KV-кэш? Ответь одним предложением.",
        "Назови три HTTP-кода ошибок и их смысл.",
        "Как посчитать perplexity?",
        "Зачем нужен gradient accumulation?",
    ]
    samples = [client.chat_once(model, _messages(prompts[i % len(prompts)]), max_tokens) for i in range(requests)]
    ok = [s for s in samples if not s.error]
    return {
        "requests": requests,
        "errors": [s.error for s in samples if s.error],
        "estimated_tokens": any(s.estimated for s in ok),
        "latency_s": _summary([s.latency for s in ok]),
        "tokens_per_s": _summary([_tok_s(s) for s in ok if _tok_s(s) > 0]),
    }


def bench_prefix(client: Client, model: str, repeats: int, max_tokens: int) -> dict[str, Any]:
    """Cold vs warm TTFT for one long prefix.

    The prefix is sent verbatim every time; a server with prefix caching keeps
    the computed KV and only prefills the short tail. Without caching every
    request pays the full prefill again, so the ratio stays near 1.0.
    """
    tail = "\n\nВопрос: как проверить, что кэш префикса включён? Ответь коротко."
    messages = _messages(SHARED_PREFIX + tail)
    samples = [client.chat_stream(model, messages, max_tokens) for _ in range(max(repeats, 1))]
    ok = [s for s in samples if not s.error]
    cold = ok[0].ttft if ok and ok[0].ttft > 0 else 0.0
    warm = [s.ttft for s in ok[1:] if s.ttft > 0]
    warm_median = statistics.median(warm) if warm else 0.0
    speedup = cold / warm_median if cold > 0 and warm_median > 0 else 0.0
    return {
        "repeats": repeats,
        "errors": [s.error for s in samples if s.error],
        "prefix_chars": len(SHARED_PREFIX),
        "prompt_tokens": ok[0].prompt_tokens if ok else 0,
        "cold_ttft_s": round(cold, 4),
        "warm_ttft_s": _summary(warm),
        "speedup": round(speedup, 2),
        "estimated_tokens": any(s.estimated for s in ok),
    }


def bench_concurrency(client: Client, model: str, concurrency: int, max_tokens: int) -> dict[str, Any]:
    prompts = [
        "Что такое continuous batching?",
        "Объясни разницу между TTFT и tok/s.",
        "Зачем нужен draft-модель при спекулятивном декодировании?",
        "Как работает PagedAttention?",
        "Что даёт --cache-ram в llama-server?",
        "Почему FLASH attention экономит память?",
        "Как оценить потерю качества после Q4_K_M?",
        "Что делать, если очередь запросов переполнена?",
    ]
    started = time.perf_counter()
    results: list[Sample] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(client.chat_stream, model, _messages(prompts[i % len(prompts)]), max_tokens)
            for i in range(concurrency)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    wall = time.perf_counter() - started
    ok = [s for s in results if not s.error]
    total_tokens = sum(s.completion_tokens for s in ok)
    return {
        "concurrency": concurrency,
        "errors": [s.error for s in results if s.error],
        "wall_s": round(wall, 3),
        "aggregate_tokens_per_s": round(total_tokens / wall, 2) if wall > 0 else 0.0,
        "per_request_tokens_per_s": _summary([_tok_s(s) for s in ok if _tok_s(s) > 0]),
        "ttft_s": _summary([s.ttft for s in ok if s.ttft > 0]),
        "latency_s": _summary([s.latency for s in ok]),
        "estimated_tokens": any(s.estimated for s in ok),
    }


def _tok_s(sample: Sample) -> float:
    if sample.completion_tokens <= 0 or sample.ttft <= 0:
        return 0.0
    decode_time = sample.latency - sample.ttft
    if decode_time <= 0:
        return 0.0
    return sample.completion_tokens / decode_time


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark an OpenAI-compatible endpoint")
    # Environment fallbacks let CI run the same benchmark without embedding a
    # host or a key in the command line.
    parser.add_argument("--base-url", default=os.environ.get("BENCH_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--api-key", default=os.environ.get("BENCH_API_KEY") or None)
    parser.add_argument("--model", default=os.environ.get("BENCH_MODEL") or None,
                        help="defaults to the first id from /v1/models")
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--prefix-repeat", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--skip", default="", help="comma list: stream,batch,prefix,concurrency")
    parser.add_argument("--min-prefix-speedup", type=float, default=None)
    parser.add_argument("--out", default=os.environ.get("BENCH_OUT", "bench/results.json"))
    args = parser.parse_args(argv)

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    client = Client(args.base_url, args.api_key, args.timeout)
    report: dict[str, Any] = {
        "base_url": args.base_url,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "max_tokens": args.max_tokens,
    }
    try:
        report["models"] = client.models()
        model = args.model or report["models"][0]
        report["model"] = model
        print(f"endpoint {args.base_url} model {model}", file=sys.stderr)

        if "batch" not in skip:
            print("batch...", file=sys.stderr)
            report["batch"] = bench_batch(client, model, args.requests, args.max_tokens)
        if "stream" not in skip:
            print("stream...", file=sys.stderr)
            report["stream"] = bench_stream(client, model, args.requests, args.max_tokens)
        if "prefix" not in skip:
            print("prefix...", file=sys.stderr)
            report["prefix"] = bench_prefix(client, model, args.prefix_repeat, args.max_tokens)
        if "concurrency" not in skip:
            print("concurrency...", file=sys.stderr)
            report["concurrency"] = bench_concurrency(client, model, args.concurrency, args.max_tokens)
    except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        client.close()
        return 2
    finally:
        client.close()

    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(_highlights(report), indent=2, ensure_ascii=False))
    print(f"written: {out}", file=sys.stderr)

    if args.min_prefix_speedup is not None:
        got = report.get("prefix", {}).get("speedup", 0.0)
        if got < args.min_prefix_speedup:
            print(
                f"FAIL: prefix speedup {got:.2f}x < required {args.min_prefix_speedup:.2f}x",
                file=sys.stderr,
            )
            return 1
    return 0


def _highlights(report: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"model": report.get("model")}
    if "stream" in report:
        out["stream_ttft_p50_ms"] = round(report["stream"]["ttft_s"]["p50"] * 1000, 1)
        out["stream_ttft_p95_ms"] = round(report["stream"]["ttft_s"]["p95"] * 1000, 1)
        out["stream_tok_s_p50"] = round(report["stream"]["tokens_per_s"]["p50"], 1)
    if "batch" in report:
        out["batch_latency_p50_ms"] = round(report["batch"]["latency_s"]["p50"] * 1000, 1)
    if "prefix" in report:
        out["prefix_cold_ttft_ms"] = round(report["prefix"]["cold_ttft_s"] * 1000, 1)
        out["prefix_warm_ttft_ms"] = round(report["prefix"]["warm_ttft_s"]["p50"] * 1000, 1)
        out["prefix_speedup"] = report["prefix"]["speedup"]
    if "concurrency" in report:
        out["concurrency_aggregate_tok_s"] = report["concurrency"]["aggregate_tokens_per_s"]
        out["concurrency_errors"] = len(report["concurrency"]["errors"])
    return out


if __name__ == "__main__":
    raise SystemExit(main())
