"""Telegram bot settings, all from the environment."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional at runtime
    pass


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}") from exc


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class BotConfig:
    token: str
    api_base_url: str
    api_key: str
    default_model: str
    max_context_messages: int = 10
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    system_prompt: str = "Ты компактная локальная модель Extra code. Отвечай кратко и по делу."
    db_path: Path = Path("bot/data/chats.db")
    rate_limit_per_min: int = 12
    stream_edits: bool = True
    request_timeout_s: float = 120.0
    group_mode: str = "mention"

    @classmethod
    def from_env(cls) -> "BotConfig":
        token = os.environ.get("BOT_TOKEN", "").strip()
        api_key = os.environ.get("API_KEY", "").strip()
        if not token:
            raise ValueError("BOT_TOKEN is required")
        if not api_key:
            raise ValueError("API_KEY is required (the bot authenticates to the API)")
        group_mode = os.environ.get("GROUP_MODE", "mention").strip().lower()
        if group_mode not in {"mention", "all", "off"}:
            raise ValueError("GROUP_MODE must be mention, all or off")
        return cls(
            token=token,
            api_base_url=os.environ.get("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/"),
            api_key=api_key,
            default_model=os.environ.get("DEFAULT_MODEL", "extra-code-30m"),
            max_context_messages=_int("MAX_CONTEXT_MESSAGES", 10),
            max_tokens=_int("MAX_TOKENS", 512),
            temperature=_float("TEMPERATURE", 0.7),
            top_p=_float("TOP_P", 0.9),
            system_prompt=os.environ.get(
                "SYSTEM_PROMPT",
                "Ты компактная локальная модель Extra code. Отвечай кратко и по делу.",
            ),
            db_path=Path(os.environ.get("BOT_DB_PATH", "bot/data/chats.db")),
            rate_limit_per_min=_int("RATE_LIMIT_PER_MIN", 12),
            stream_edits=_bool("STREAM_EDITS", True),
            request_timeout_s=_float("REQUEST_TIMEOUT_S", 120.0),
            group_mode=group_mode,
        )
