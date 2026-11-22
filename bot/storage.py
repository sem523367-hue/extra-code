"""Per-chat history in SQLite.

History is JSON in one row per chat. The bot trims by message count and by a
character budget because it cannot know the model tokenizer; the server does
the exact token-level fitting afterwards.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    chat_id   INTEGER PRIMARY KEY,
    history   TEXT NOT NULL,
    updated_at REAL NOT NULL
);
"""


class ChatStore:
    def __init__(self, path: Path, *, max_messages: int = 10, max_chars: int = 24_000) -> None:
        self.path = path
        self.max_messages = max_messages
        self.max_chars = max_chars
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def history(self, chat_id: int) -> list[dict[str, str]]:
        with self._connect() as conn:
            row = conn.execute("SELECT history FROM chats WHERE chat_id = ?", (chat_id,)).fetchone()
        if row is None:
            return []
        try:
            history = json.loads(row["history"])
        except json.JSONDecodeError:
            return []
        return [m for m in history if isinstance(m, dict) and m.get("content")]

    def save(self, chat_id: int, history: list[dict[str, str]]) -> None:
        history = self._trim(history)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO chats(chat_id, history, updated_at) VALUES (?, ?, ?)"
                " ON CONFLICT(chat_id) DO UPDATE SET history=excluded.history, updated_at=excluded.updated_at",
                (chat_id, json.dumps(history, ensure_ascii=False), time.time()),
            )

    def clear(self, chat_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chats WHERE chat_id = ?", (chat_id,))

    def _trim(self, history: list[dict[str, str]]) -> list[dict[str, str]]:
        trimmed = history[-self.max_messages :]
        while len(trimmed) > 1 and sum(len(m["content"]) for m in trimmed) > self.max_chars:
            trimmed.pop(0)
        return trimmed
