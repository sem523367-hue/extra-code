"""Byte-identical output check between two OpenAI-compatible endpoints.

This is the acceptance test that separates a lossless optimisation from a
plausible-looking one: run the same prompts with temperature 0 and a fixed seed
against the reference server and the accelerated server, then compare the text
character for character. Speculative decoding, FlashAttention, prefix caching
and PagedAttention must all produce the same greedy output; quantization is
expected to differ and is gated by perplexity instead.

    python -m eval.equivalence \
        --a http://127.0.0.1:8000 --b http://127.0.0.1:8080 \
        --a-key "$API_KEY" --b-key "$LLAMA_API_KEY" \
        --prompts eval/prompts.jsonl --out eval/equivalence.json

Exit code: 0 when every prompt matches, 1 on any mismatch, 2 on transport or
protocol failure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx


def load_prompts(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({exc})") from exc
            if not row.get("messages"):
                raise ValueError(f"{path}:{line_no}: row has no messages")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def complete(
    client: httpx.Client,
    base_url: str,
    api_key: str | None,
    model: str,
    row: dict[str, Any],
    *,
    timeout: float,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": row["messages"],
        "max_tokens": int(row.get("max_tokens", 64)),
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "stream": False,
    }
    response = client.post(
        f"{base_url.rstrip('/')}/v1/chat/completions", json=payload, headers=headers, timeout=timeout
    )
    if response.status_code != 200:
        raise RuntimeError(f"{base_url} returned {response.status_code}: {response.text[:400]}")
    body = response.json()
    choice = body["choices"][0]
    return {
        "text": choice["message"].get("content") or "",
        "finish_reason": choice.get("finish_reason"),
        "usage": body.get("usage") or {},
    }


def run(
    a_url: str,
    b_url: str,
    *,
    a_key: str | None,
    b_key: str | None,
    model_a: str,
    model_b: str,
    prompts: Path,
    timeout: float = 180.0,
) -> dict[str, Any]:
    rows = load_prompts(prompts)
    results: list[dict[str, Any]] = []
    mismatches = 0
    with httpx.Client() as client:
        for row in rows:
            left = complete(client, a_url, a_key, model_a, row, timeout=timeout)
            right = complete(client, b_url, b_key, model_b, row, timeout=timeout)
            same = left["text"] == right["text"]
            if not same:
                mismatches += 1
                prefix = 0
                for x, y in zip(left["text"], right["text"]):
                    if x != y:
                        break
                    prefix += 1
            else:
                prefix = len(left["text"])
            results.append(
                {
                    "id": row.get("id", "prompt"),
                    "match": same,
                    "common_prefix_chars": prefix,
                    "a": left,
                    "b": right,
                }
            )
            print(
                f"{'OK  ' if same else 'DIFF'} {row.get('id', 'prompt')}: "
                f"{len(left['text'])} vs {len(right['text'])} chars",
                file=sys.stderr,
            )
    return {
        "a": {"base_url": a_url, "model": model_a},
        "b": {"base_url": b_url, "model": model_b},
        "prompts": len(rows),
        "mismatches": mismatches,
        "identical": mismatches == 0,
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare greedy output of two endpoints")
    parser.add_argument("--a", required=True, help="reference endpoint, e.g. http://127.0.0.1:8000")
    parser.add_argument("--b", required=True, help="candidate endpoint, e.g. http://127.0.0.1:8080")
    parser.add_argument("--a-key", default=None)
    parser.add_argument("--b-key", default=None)
    parser.add_argument("--a-model", default=None, help="defaults to /v1/models on the endpoint")
    parser.add_argument("--b-model", default=None)
    parser.add_argument("--prompts", default="eval/prompts.jsonl")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--out", default="eval/equivalence.json")
    args = parser.parse_args(argv)

    try:
        if not args.a_model:
            args.a_model = _first_model(args.a, args.a_key, args.timeout)
        if not args.b_model:
            args.b_model = _first_model(args.b, args.b_key, args.timeout)
        payload = run(
            args.a,
            args.b,
            a_key=args.a_key,
            b_key=args.b_key,
            model_a=args.a_model,
            model_b=args.b_model,
            prompts=Path(args.prompts),
            timeout=args.timeout,
        )
    except (httpx.HTTPError, KeyError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("prompts", "mismatches", "identical")}, indent=2))
    print(f"written: {out}", file=sys.stderr)
    return 0 if payload["identical"] else 1


def _first_model(base_url: str, api_key: str | None, timeout: float) -> str:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    response = httpx.get(f"{base_url.rstrip('/')}/v1/models", headers=headers, timeout=timeout)
    response.raise_for_status()
    data = response.json().get("data") or []
    if not data:
        raise RuntimeError(f"{base_url}/v1/models returned no models")
    return str(data[0]["id"])


if __name__ == "__main__":
    raise SystemExit(main())
