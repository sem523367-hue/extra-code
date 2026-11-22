"""API keys and per-key limits.

Keys are stored only as sha256(pepper + key); the plaintext exists for exactly
one moment — when `python -m server.auth add` prints it. Lookup is O(1) on the
hash index.

The limiter is an in-process sliding window: two deques per key (requests and
settled tokens) plus a pending reservation so a long generation cannot be
started twice over the same token budget. It is correct for a single-process
deployment; with multiple workers each worker enforces its own share, so set
`MAX_CONCURRENT` and `--workers` with that in mind.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import secrets
import sqlite3
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

WINDOW_S = 60.0
_SCHEMA = """
CREATE TABLE IF NOT EXISTS api_keys (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    key_hash    TEXT NOT NULL UNIQUE,
    rpm         INTEGER NOT NULL,
    tpm         INTEGER NOT NULL,
    max_tokens  INTEGER NOT NULL,
    enabled     INTEGER NOT NULL DEFAULT 1,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
"""


@dataclass(frozen=True, slots=True)
class ApiKey:
    id: int
    name: str
    rpm: int
    tpm: int
    max_tokens: int


def hash_key(pepper: str, raw: str) -> str:
    return hashlib.sha256(f"{pepper}:{raw}".encode("utf-8")).hexdigest()


def generate_key() -> str:
    return "sk-local-" + secrets.token_urlsafe(32)


class KeyStore:
    def __init__(self, path: Path, pepper: str) -> None:
        self._path = path
        self._pepper = pepper
        self._lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def init(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(_SCHEMA)

    def bootstrap(self, raw: str, *, rpm: int, tpm: int, max_tokens: int) -> str:
        """Ensure the env-provided key exists. Returns its name.

        Idempotent: restarting with the same API_KEY reuses the row instead of
        creating duplicates or rotating the secret.
        """
        key_hash = hash_key(self._pepper, raw)
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT name FROM api_keys WHERE key_hash = ?", (key_hash,)
            ).fetchone()
            if row is not None:
                return str(row["name"])
            name = "bootstrap"
            while conn.execute("SELECT 1 FROM api_keys WHERE name = ?", (name,)).fetchone():
                name = f"bootstrap-{secrets.token_hex(2)}"
            conn.execute(
                "INSERT INTO api_keys(name, key_hash, rpm, tpm, max_tokens, enabled, created_at)"
                " VALUES (?, ?, ?, ?, ?, 1, ?)",
                (name, key_hash, rpm, tpm, max_tokens, time.time()),
            )
            return name

    def lookup(self, raw: str) -> ApiKey | None:
        key_hash = hash_key(self._pepper, raw)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, name, rpm, tpm, max_tokens, enabled FROM api_keys WHERE key_hash = ?",
                (key_hash,),
            ).fetchone()
        if row is None or not row["enabled"]:
            return None
        return ApiKey(
            id=int(row["id"]),
            name=str(row["name"]),
            rpm=int(row["rpm"]),
            tpm=int(row["tpm"]),
            max_tokens=int(row["max_tokens"]),
        )

    def add(self, name: str, *, rpm: int, tpm: int, max_tokens: int) -> tuple[int, str]:
        raw = generate_key()
        key_hash = hash_key(self._pepper, raw)
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO api_keys(name, key_hash, rpm, tpm, max_tokens, enabled, created_at)"
                " VALUES (?, ?, ?, ?, ?, 1, ?)",
                (name, key_hash, rpm, tpm, max_tokens, time.time()),
            )
            return int(cur.lastrowid), raw

    def revoke(self, name: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute("UPDATE api_keys SET enabled = 0 WHERE name = ?", (name,))
            return cur.rowcount > 0

    def list(self) -> list[dict[str, object]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, name, rpm, tpm, max_tokens, enabled, created_at FROM api_keys ORDER BY id"
            ).fetchall()
        return [dict(r) for r in rows]


class RateLimiter:
    """Sliding-window limiter with token reservations.

    `admit` checks request count and settled+pending tokens, then reserves an
    estimate. `settle` replaces the reservation with the real usage. Callers
    must settle in a finally block or the reservation leaks until window expiry.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: dict[int, deque[float]] = defaultdict(deque)
        self._tokens: dict[int, deque[tuple[float, int]]] = defaultdict(deque)
        self._pending: dict[int, int] = defaultdict(int)

    def _prune(self, key_id: int, now: float) -> None:
        req = self._requests[key_id]
        while req and now - req[0] >= WINDOW_S:
            req.popleft()
        tok = self._tokens[key_id]
        while tok and now - tok[0][0] >= WINDOW_S:
            tok.popleft()

    def admit(
        self, key: ApiKey, reserve_tokens: int
    ) -> tuple[bool, float, str]:
        """Returns (allowed, retry_after_seconds, reason)."""
        now = time.monotonic()
        with self._lock:
            self._prune(key.id, now)
            req = self._requests[key.id]
            if len(req) >= key.rpm:
                retry = WINDOW_S - (now - req[0]) if req else 1.0
                return False, max(0.1, retry), "requests_per_minute"
            used = sum(t for _, t in self._tokens[key.id]) + self._pending[key.id]
            if key.tpm > 0 and used + reserve_tokens > key.tpm:
                oldest = self._tokens[key.id][0][0] if self._tokens[key.id] else now
                return False, max(0.1, WINDOW_S - (now - oldest)), "tokens_per_minute"
            req.append(now)
            self._pending[key.id] += reserve_tokens
            return True, 0.0, ""

    def settle(self, key_id: int, reserved: int, actual_tokens: int) -> None:
        now = time.monotonic()
        with self._lock:
            self._pending[key_id] = max(0, self._pending[key_id] - reserved)
            self._tokens[key_id].append((now, max(0, actual_tokens)))


def _cli() -> None:
    parser = argparse.ArgumentParser(prog="python -m server.auth", description="API key admin")
    parser.add_argument("--db", default=os.environ.get("API_DB_PATH", "server/data/keys.db"))
    parser.add_argument("--pepper", default=os.environ.get("API_KEY_PEPPER", "change-me-in-production"))
    sub = parser.add_subparsers(dest="cmd", required=True)

    add = sub.add_parser("add", help="create a key and print it once")
    add.add_argument("name")
    add.add_argument("--rpm", type=int, default=int(os.environ.get("DEFAULT_RPM", "60")))
    add.add_argument("--tpm", type=int, default=int(os.environ.get("DEFAULT_TPM", "20000")))
    add.add_argument("--max-tokens", type=int, default=int(os.environ.get("DEFAULT_KEY_MAX_TOKENS", "512")))

    rev = sub.add_parser("revoke", help="disable a key by name")
    rev.add_argument("name")

    sub.add_parser("list", help="list keys (hashes never printed)")

    args = parser.parse_args()
    store = KeyStore(Path(args.db), args.pepper)
    store.init()
    if args.cmd == "add":
        key_id, raw = store.add(args.name, rpm=args.rpm, tpm=args.tpm, max_tokens=args.max_tokens)
        print(f"id={key_id} name={args.name} key={raw}")
        print("store it now, the server keeps only the hash")
    elif args.cmd == "revoke":
        print("revoked" if store.revoke(args.name) else "not found")
    elif args.cmd == "list":
        for row in store.list():
            print(row)


if __name__ == "__main__":
    _cli()
