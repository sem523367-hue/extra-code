"""Inference backends and prompt assembly.

Two implementations behind one blocking iterator interface:

* `LlamaCppEngine` — the production path, GGUF through llama-cpp-python.
* `OpenAIEngine` — fronts a running llama-server (or any OpenAI-compatible
  endpoint) over HTTP, so the API layer can ride a model it cannot load itself.
* `MockEngine` — deterministic, dependency-free backend for CI and protocol
  work. It never pretends to be a language model: /health reports it.

Blocking-by-design: llama.cpp is not thread-safe per handle and gives no async
API, so the HTTP layer runs `generate()` in a worker thread and bridges chunks
through an asyncio queue. Parallelism lives above this file.
"""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Protocol, Sequence

import httpx

Message = tuple[str, str]

_ROLE_ALIASES = {"tool": "user", "function": "user"}


def _role(role: str) -> str:
    role = role.lower()
    return role if role in {"system", "user", "assistant"} else _ROLE_ALIASES.get(role, "user")


def render_chat(messages: Sequence[Message], template: str) -> str:
    """Render a conversation and open the assistant turn.

    The trailing assistant header is part of the prompt so the model continues
    an answer instead of starting a new role block. Keep this function and the
    GGUF chat template in sync — `export/to_gguf.py` writes the same preset into
    tokenizer_config.json.
    """
    if template == "chatml":
        body = "".join(
            f"<|im_start|>{_role(r)}\n{text}<|im_end|>\n" for r, text in messages
        )
        return body + "<|im_start|>assistant\n"
    if template == "llama3":
        body = "".join(
            f"<|start_header_id|>{_role(r)}<|end_header_id|>\n\n{text}<|eot_id|>"
            for r, text in messages
        )
        return body + "<|start_header_id|>assistant<|end_header_id|>\n\n"
    if template == "plain":
        body = "".join(f"### {_role(r)}\n{text}\n\n" for r, text in messages)
        return body + "### assistant\n"
    raise ValueError(f"unknown chat template {template!r}")


@dataclass(slots=True)
class GenerationParams:
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repeat_penalty: float
    stop: list[str] = field(default_factory=list)
    seed: int | None = None


@dataclass(slots=True)
class GenStats:
    finish_reason: str = "stop"


@dataclass(slots=True)
class FittedChat:
    prompt: str
    max_tokens: int
    prompt_tokens: int = 0
    dropped_messages: int = 0
    truncated: bool = False


class Engine(Protocol):
    name: str

    def tokenize(self, text: str) -> list[int]: ...
    def detokenize(self, ids: Sequence[int]) -> str: ...
    def count_tokens(self, text: str) -> int: ...
    def generate(
        self, prompt: str, params: GenerationParams, stats: GenStats, cancel: threading.Event
    ) -> Iterator[str]: ...
    def describe(self) -> dict[str, object]: ...


# --------------------------------------------------------------------------
# context fitting
# --------------------------------------------------------------------------

def fit_chat(
    engine: Engine,
    messages: Sequence[Message],
    *,
    n_ctx: int,
    max_tokens: int,
    template: str,
) -> FittedChat:
    """Shrink a conversation until prompt + completion fits the context.

    Strategy, in order: drop the oldest non-system message, then trim the
    largest remaining body token-by-token. Raises ValueError when even a bare
    prompt cannot fit, which the HTTP layer maps to 400.
    """
    msgs = list(messages)
    dropped = 0
    truncated = False
    budget = n_ctx - max_tokens
    if budget < 16:
        max_tokens = max(1, n_ctx // 2)
        budget = n_ctx - max_tokens

    for _ in range(512):
        prompt = render_chat(msgs, template)
        prompt_tokens = engine.count_tokens(prompt)
        excess = prompt_tokens - budget
        if excess <= 0:
            return FittedChat(prompt, max_tokens, prompt_tokens, dropped, truncated)

        non_system = [i for i, (role, _) in enumerate(msgs) if role != "system"]
        if len(non_system) > 1:
            msgs.pop(non_system[0])
            dropped += 1
            continue

        # Only the system message and one turn remain: cut text by tokens.
        idx = max(range(len(msgs)), key=lambda i: len(msgs[i][1])) if msgs else -1
        if idx < 0:
            break
        role, text = msgs[idx]
        ids = engine.tokenize(text)
        if not ids:
            break
        keep = len(ids) - excess - 8
        if keep < 1:
            keep = 1
        new_text = engine.detokenize(ids[-keep:])
        if new_text == text:
            break
        msgs[idx] = (role, new_text)
        truncated = True

    raise ValueError("prompt does not fit the model context")


def fit_prompt(engine: Engine, prompt: str, *, n_ctx: int, max_tokens: int) -> tuple[str, int]:
    """Raw-completion equivalent: keep the tail of the prompt."""
    if max_tokens >= n_ctx:
        max_tokens = max(1, n_ctx // 2)
    budget = n_ctx - max_tokens
    ids = engine.tokenize(prompt)
    if len(ids) <= budget:
        return prompt, max_tokens
    return engine.detokenize(ids[-budget:]), max_tokens


# --------------------------------------------------------------------------
# production backend
# --------------------------------------------------------------------------

class LlamaCppEngine:
    def __init__(
        self,
        model_path: Path,
        *,
        n_ctx: int,
        n_threads: int,
        n_gpu_layers: int,
        n_batch: int,
        verbose: bool = False,
    ) -> None:
        try:
            from llama_cpp import Llama  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on install
            raise RuntimeError(
                "llama-cpp-python is not installed; `pip install llama-cpp-python` "
                "or set BACKEND=mock for protocol work"
            ) from exc
        if not model_path.exists():
            raise FileNotFoundError(f"model file not found: {model_path}")
        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            verbose=verbose,
            logits_all=False,
            embedding=False,
        )
        self._lock = threading.Lock()
        self._n_ctx = n_ctx
        self._path = model_path
        self.name = "llama_cpp"

    def tokenize(self, text: str) -> list[int]:
        return list(self._llm.tokenize(text.encode("utf-8"), add_bos=True, special=True))

    def detokenize(self, ids: Sequence[int]) -> str:
        return self._llm.detokenize(list(ids)).decode("utf-8", errors="replace")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    def generate(
        self, prompt: str, params: GenerationParams, stats: GenStats, cancel: threading.Event
    ) -> Iterator[str]:
        with self._lock:  # one inference at a time per handle
            stream = self._llm.create_completion(
                prompt=prompt,
                max_tokens=params.max_tokens,
                temperature=params.temperature,
                top_p=params.top_p,
                top_k=params.top_k,
                repeat_penalty=params.repeat_penalty,
                stop=params.stop or None,
                seed=params.seed,
                stream=True,
                echo=False,
            )
            for chunk in stream:
                if cancel.is_set():
                    stats.finish_reason = "cancelled"
                    return
                choice = chunk["choices"][0]
                piece = choice.get("text") or ""
                if piece:
                    yield piece
                reason = choice.get("finish_reason")
                if reason:
                    stats.finish_reason = str(reason)

    def describe(self) -> dict[str, object]:
        meta = getattr(self._llm, "metadata", None) or {}
        size = self._path.stat().st_size if self._path.exists() else 0
        return {
            "backend": self.name,
            "model_file": str(self._path),
            "model_bytes": size,
            "n_ctx": self._n_ctx,
            "n_threads": getattr(self._llm, "n_threads", None),
            "gpu_layers": getattr(self._llm, "n_gpu_layers", None),
            "architecture": meta.get("general.architecture"),
            "parameters": meta.get("general.parameter_count"),
        }


# --------------------------------------------------------------------------
# openai-compatible remote backend (llama-server, vLLM, llama.cpp server)
# --------------------------------------------------------------------------

class OpenAIEngine:
    """Fronts a running `llama-server` (or any OpenAI-compatible endpoint).

    The extra-code FastAPI layer keeps its auth, rate limits, streaming and
    web retrieval; the heavy lifting happens in the external server process,
    which already owns the GGUF, the GPU queue and its own tokenizer.
    """

    def __init__(
        self,
        base_url: str,
        *,
        model: str,
        api_key: str | None,
        n_ctx: int,
        timeout_s: float = 120.0,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._model = model
        self._n_ctx = n_ctx
        self._timeout = timeout_s
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.Client(headers=self._headers, timeout=timeout_s)
        self.name = "openai"
        # Fail fast: a dead upstream should surface at boot, not mid-request.
        probe = self._http.get(f"{self._base}/models")
        probe.raise_for_status()
        listed = probe.json().get("data") or []
        if listed:
            self._model = str(listed[0].get("id") or model)

    # -- tokenizer ------------------------------------------------------
    # Token math is delegated: /tokenize on llama-server is exact. A plain
    # char-based estimate is the fallback for endpoints without it.
    def tokenize(self, text: str) -> list[int]:
        try:
            body = self._http.post(f"{self._base}/tokenize", json={"content": text}).json()
            tokens = body.get("tokens")
            if isinstance(tokens, list):
                return [int(t) for t in tokens]
        except (httpx.HTTPError, ValueError, KeyError):
            pass
        return list(range(0, max(1, len(text) // 3), 1))

    def detokenize(self, ids: Sequence[int]) -> str:
        try:
            body = self._http.post(f"{self._base}/detokenize", json={"tokens": list(ids)}).json()
            text = body.get("content")
            if isinstance(text, str):
                return text
        except (httpx.HTTPError, ValueError, KeyError):
            pass
        return ""

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    # -- generation -----------------------------------------------------
    def generate(
        self, prompt: str, params: GenerationParams, stats: GenStats, cancel: threading.Event
    ) -> Iterator[str]:
        payload: dict[str, object] = {
            "model": self._model,
            "prompt": prompt,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "stop": params.stop or None,
            "stream": True,
        }
        if params.top_k:
            payload["top_k"] = params.top_k
        if params.seed:
            payload["seed"] = params.seed
        with self._http.stream(
            "POST", f"{self._base}/completions", json=payload,
            headers={"Accept": "text/event-stream"},
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if cancel.is_set():
                    stats.finish_reason = "cancelled"
                    return
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    return
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choice = (chunk.get("choices") or [{}])[0]
                piece = choice.get("text") or ""
                if piece:
                    yield piece
                reason = choice.get("stop_reason") or choice.get("finish_reason")
                if reason:
                    stats.finish_reason = str(reason)

    def generate_chat(
        self, messages: Sequence[Message], params: GenerationParams, stats: GenStats,
        cancel: threading.Event,
    ) -> Iterator[str]:
        """Chat path: the endpoint applies its own template from the GGUF.

        A manually rendered prompt and the model's built-in template drift in
        ways that make an instruct model echo instead of answer, so for chat
        requests the messages are forwarded and the server does the rendering.
        """
        payload: dict[str, object] = {
            "model": self._model,
            "messages": [{"role": role, "content": text} for role, text in messages],
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "stream": True,
        }
        if params.stop:
            payload["stop"] = params.stop
        if params.top_k:
            payload["top_k"] = params.top_k
        if params.seed:
            payload["seed"] = params.seed
        with self._http.stream(
            "POST", f"{self._base}/chat/completions", json=payload,
            headers={"Accept": "text/event-stream"},
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if cancel.is_set():
                    stats.finish_reason = "cancelled"
                    return
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    return
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choice = (chunk.get("choices") or [{}])[0]
                piece = (choice.get("delta") or {}).get("content") or ""
                if piece:
                    yield piece
                reason = choice.get("finish_reason")
                if reason:
                    stats.finish_reason = str(reason)

    def describe(self) -> dict[str, object]:
        return {
            "backend": self.name,
            "base_url": self._base,
            "model": self._model,
            "n_ctx": self._n_ctx,
        }


# --------------------------------------------------------------------------
# test backend
# --------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\s+|[^\W\d_]+|\d+|[^\w\s]", re.UNICODE)


class MockEngine:
    """Deterministic stand-in used by unit tests and UI integration.

    It answers with a fixed sentence plus a digest of the prompt, streamed token
    by token, honouring max_tokens and stop strings. Nothing about the output
    claims to be model quality; /health exposes backend="mock".
    """

    name = "mock"

    def __init__(self, *, delay_ms: int = 0, vocab_hint: int = 0) -> None:
        self._delay = max(0, delay_ms) / 1000.0
        self._vocab: list[str] = []
        self._index: dict[str, int] = {}

    # -- tokenizer -----------------------------------------------------
    def tokenize(self, text: str) -> list[int]:
        out: list[int] = []
        for piece in _TOKEN_RE.findall(text):
            idx = self._index.get(piece)
            if idx is None:
                idx = len(self._vocab)
                self._vocab.append(piece)
                self._index[piece] = idx
            out.append(idx)
        return out

    def detokenize(self, ids: Sequence[int]) -> str:
        return "".join(self._vocab[i] for i in ids if 0 <= i < len(self._vocab))

    def count_tokens(self, text: str) -> int:
        return len(_TOKEN_RE.findall(text))

    # -- generation ----------------------------------------------------
    def generate(
        self, prompt: str, params: GenerationParams, stats: GenStats, cancel: threading.Event
    ) -> Iterator[str]:
        reply = self._reply(prompt)
        ids = self.tokenize(reply)
        emitted = ""
        for token_id in ids[: params.max_tokens]:
            if cancel.is_set():
                stats.finish_reason = "cancelled"
                return
            piece = self.detokenize([token_id])
            emitted += piece
            hit = next((s for s in params.stop if s and emitted.endswith(s)), None)
            if hit is not None:
                trimmed = emitted[: -len(hit)]
                if trimmed:
                    yield trimmed
                stats.finish_reason = "stop"
                return
            yield piece
            if self._delay:
                time.sleep(self._delay)
        stats.finish_reason = "length" if len(ids) > params.max_tokens else "stop"

    @staticmethod
    def _reply(prompt: str) -> str:
        tail = prompt.strip().splitlines()[-1][:120] if prompt.strip() else ""
        return (
            "mock backend online. protocol check passed. "
            f"context tail: {tail}. "
            "replace me with a GGUF model for real answers.\n"
        )

    def describe(self) -> dict[str, object]:
        return {"backend": self.name, "model": "mock", "n_ctx": None, "parameters": 0}


def load_engine(settings) -> Engine:  # type: ignore[no-untyped-def]
    if settings.backend == "mock":
        return MockEngine(delay_ms=settings.mock_delay_ms)
    if settings.backend == "openai":
        return OpenAIEngine(
            settings.openai_base_url,
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            n_ctx=settings.n_ctx,
            timeout_s=settings.generation_timeout_s + 30.0,
        )
    return LlamaCppEngine(
        settings.model_path,
        n_ctx=settings.n_ctx,
        n_threads=settings.n_threads,
        n_gpu_layers=settings.n_gpu_layers,
        n_batch=settings.n_batch,
    )
