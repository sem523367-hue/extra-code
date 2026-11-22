"""Async client for the OpenAI-compatible API.

The bot owns no model: every answer comes from /v1/chat/completions. SSE chunks
are parsed incrementally so the Telegram message can be edited while the model
is still generating.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import httpx

log = logging.getLogger("extra_code.bot.client")

DeltaCallback = Callable[[str], Awaitable[None]]


class ApiError(RuntimeError):
    def __init__(self, status: int, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


@dataclass(slots=True)
class ChatResult:
    text: str
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)


class ApiClient:
    def __init__(self, base_url: str, api_key: str, *, timeout_s: float = 120.0) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=httpx.Timeout(timeout_s, connect=10.0),
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def health(self) -> dict:
        response = await self._client.get("/health")
        response.raise_for_status()
        return response.json()

    async def models(self) -> list[str]:
        response = await self._client.get("/v1/models")
        if response.status_code != 200:
            raise ApiError(response.status_code, _error_message(response))
        return [item["id"] for item in response.json().get("data", [])]

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stream: bool,
        on_delta: DeltaCallback | None = None,
    ) -> ChatResult:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }
        if not stream:
            response = await self._client.post("/v1/chat/completions", json=payload)
            if response.status_code != 200:
                raise ApiError(response.status_code, _error_message(response))
            body = response.json()
            choice = body["choices"][0]
            return ChatResult(
                text=choice["message"].get("content") or "",
                finish_reason=choice.get("finish_reason") or "stop",
                usage=body.get("usage") or {},
            )

        payload["stream_options"] = {"include_usage": True}
        chunks: list[str] = []
        finish = "stop"
        usage: dict[str, int] = {}
        async with self._client.stream("POST", "/v1/chat/completions", json=payload) as response:
            if response.status_code != 200:
                await response.aread()
                raise ApiError(response.status_code, _error_message(response))
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[6:]
                if raw == "[DONE]":
                    break
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if "error" in event:
                    error = event["error"]
                    raise ApiError(response.status_code, str(error.get("message", "api error")), error.get("code"))
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        chunks.append(piece)
                        if on_delta is not None:
                            await on_delta(piece)
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
        return ChatResult(text="".join(chunks), finish_reason=finish, usage=usage)


def _error_message(response: httpx.Response) -> str:
    try:
        body = response.json()
        return str(body.get("error", {}).get("message") or body)
    except (json.JSONDecodeError, ValueError):
        return response.text[:500] or f"HTTP {response.status_code}"
