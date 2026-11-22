"""Pydantic models for the OpenAI wire format.

The request models are deliberately lenient: OpenAI clients send fields this
server does not use (`presence_penalty`, `logit_bias`, ...). Accepting them
keeps drop-in compatibility; silently pretending to honour them would not.
Unused fields are listed in README under "compat".
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Role = Literal["system", "user", "assistant", "tool", "function"]


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def now() -> int:
    return int(time.time())


class ChatMessage(BaseModel):
    role: Role = "user"
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None

    def as_text(self) -> str:
        """Flatten OpenAI content parts to plain text.

        Image/audio parts cannot be consumed by a text-only model, so they are
        dropped rather than rendered as junk tokens.
        """
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        chunks: list[str] = []
        for part in self.content:
            if isinstance(part, dict) and part.get("type") in {"text", "input_text"}:
                text = part.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunks)


class StreamOptions(BaseModel):
    include_usage: bool = False


class _Sampling(BaseModel):
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=4.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0)
    stop: str | list[str] | None = None
    seed: int | None = None
    stream: bool = False
    n: int = Field(default=1, ge=1)
    # accepted for client compatibility, not implemented by llama.cpp here
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logit_bias: dict[str, float] | None = None
    user: str | None = None

    @field_validator("stop")
    @classmethod
    def _stops(cls, value: str | list[str] | None) -> str | list[str] | None:
        if isinstance(value, list):
            if len(value) > 4:
                raise ValueError("at most 4 stop sequences are supported")
            if any(not s for s in value):
                raise ValueError("stop sequences must be non-empty")
        if isinstance(value, str) and not value:
            raise ValueError("stop must be non-empty")
        return value

    def resolved_max_tokens(self, default: int, cap: int) -> int:
        wanted = self.max_completion_tokens or self.max_tokens or default
        return max(1, min(wanted, cap))

    def stops(self) -> list[str]:
        if self.stop is None:
            return []
        return [self.stop] if isinstance(self.stop, str) else list(self.stop)

    @model_validator(mode="after")
    def _single_choice(self) -> "_Sampling":
        if self.n != 1:
            raise ValueError("n > 1 is not supported by this server")
        return self


class ChatCompletionRequest(_Sampling):
    model: str
    messages: list[ChatMessage] = Field(min_length=1, max_length=512)
    stream_options: StreamOptions | None = None
    # Retrieval is a server concern, not a model concern: a small model cannot
    # emit a reliable tool call, so the search happens here and the model only
    # sees the ranked snippets.
    web: bool = False
    web_pages: int = Field(default=2, ge=1, le=5)
    web_max_chars: int = Field(default=1200, ge=200, le=6000)
    # Local knowledge base: everything collected for extra-code (corpus,
    # SFT sets, project docs) indexed and injected before web search.
    local: bool = False
    local_chunks: int = Field(default=3, ge=1, le=8)
    # Understanding pass: the model first restates the request as a
    # checklist (what to do / what NOT to do), then answers against it.
    # Costs one short generation, kills "answered something else" drift.
    reason: bool = True

    @field_validator("messages")
    @classmethod
    def _has_content(cls, messages: list[ChatMessage]) -> list[ChatMessage]:
        if not any(m.role != "system" and m.as_text().strip() for m in messages):
            raise ValueError("at least one non-system message with text content is required")
        return messages


class CompletionRequest(_Sampling):
    model: str
    prompt: str | list[str]
    echo: bool = False
    suffix: str | None = None
    stream_options: StreamOptions | None = None

    @field_validator("prompt")
    @classmethod
    def _prompt(cls, value: str | list[str]) -> str | list[str]:
        if isinstance(value, str):
            if not value:
                raise ValueError("prompt must be non-empty")
            if len(value) > 1_000_000:
                raise ValueError("prompt exceeds 1,000,000 characters")
            return value
        if not value or any(not p for p in value):
            raise ValueError("prompt list must be non-empty and contain no empty strings")
        if len(value) > 16:
            raise ValueError("at most 16 prompts per request")
        return value


class TokenizeRequest(BaseModel):
    model: str | None = None
    text: str = Field(min_length=1, max_length=1_000_000)


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ErrorBody(BaseModel):
    message: str
    type: str = "invalid_request_error"
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
