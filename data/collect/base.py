"""Shared plumbing for corpus collectors.

Contract every collector implements:

    def collect(out_dir: Path, *, limit: int) -> dict[str, object]

It writes one gzip-JSONL file per run plus a manifest and returns the manifest
dict. Documents are plain dicts:

    {"text": str, "source": str, "url": str, "lang": str, "meta": {...}}

Collectors are I/O-bound and polite: token-bucket rate limits, bounded retries
with jittered exponential backoff, hard caps on downloaded bytes.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import httpx

# Wikimedia rejects generic UAs with a 403 and asks for a contactable identity.
# Override with CORPUS_USER_AGENT when running a big crawl under your own name.
USER_AGENT = os.environ.get(
    "CORPUS_USER_AGENT",
    "extra-code-corpus/1.0 (https://github.com/extra-code/corpus; contact: corpus@extra-code.local) httpx",
)

# Refuse to swallow a 2 GB HTML dump as one "document".
MAX_DOC_CHARS = 200_000
MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024


class RateLimiter:
    """Thread-safe token bucket."""

    def __init__(self, rate_per_s: float, burst: int = 1) -> None:
        if rate_per_s <= 0:
            raise ValueError("rate_per_s must be positive")
        self._rate = rate_per_s
        self._capacity = max(1, burst)
        self._tokens = float(self._capacity)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, n: int = 1) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self._capacity, self._tokens + (now - self._last) * self._rate)
                self._last = now
                if self._tokens >= n:
                    self._tokens -= n
                    return
                wait = (n - self._tokens) / self._rate
            time.sleep(min(wait, 5.0) + random.uniform(0, 0.05))


def make_client(
    *, timeout: float = 30.0, headers: dict[str, str] | None = None, follow_redirects: bool = True
) -> httpx.Client:
    merged = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}
    if headers:
        merged.update(headers)
    return httpx.Client(
        timeout=httpx.Timeout(timeout, connect=10.0),
        headers=merged,
        follow_redirects=follow_redirects,
        limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
    )


def http_get(
    client: httpx.Client,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    retries: int = 4,
    max_bytes: int = MAX_DOWNLOAD_BYTES,
    limiter: RateLimiter | None = None,
) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        if limiter is not None:
            limiter.acquire()
        try:
            with client.stream("GET", url, params=params, headers=headers) as response:
                if response.status_code in {403, 429} or response.status_code >= 500:
                    retry_after = float(response.headers.get("retry-after", "0") or 0)
                    if retry_after > 0:
                        raise _RetryAfter(retry_after)
                    raise httpx.HTTPStatusError(
                        f"retryable status {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError(f"response exceeds {max_bytes} bytes: {url}")
                    chunks.append(chunk)
                return b"".join(chunks)
        except _RetryAfter as exc:
            last = exc
            time.sleep(min(exc.seconds, 60.0) + random.uniform(0, 0.5))
        except (httpx.HTTPError, ValueError) as exc:
            last = exc
            sleep = min(60.0, (2.0**attempt) * 0.75) + random.uniform(0, 0.5)
            time.sleep(sleep)
    raise RuntimeError(f"GET failed after {retries} attempts: {url}: {last}")


class _RetryAfter(Exception):
    def __init__(self, seconds: float) -> None:
        super().__init__(f"retry after {seconds}")
        self.seconds = seconds


def http_json(client: httpx.Client, url: str, **kwargs: Any) -> Any:
    raw = http_get(client, url, **kwargs)
    return json.loads(raw)


@dataclass(slots=True)
class Doc:
    text: str
    source: str
    url: str = ""
    lang: str = "und"
    meta: dict[str, Any] = field(default_factory=dict)


class Sink:
    """Append-only gzip JSONL writer with a per-document cap."""

    def __init__(self, out_dir: Path, source: str, *, max_chars: int = MAX_DOC_CHARS) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.source = source
        self.path = out_dir / "raw" / source / f"{source}-{stamp}.jsonl.gz"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = gzip.open(self.path, "wt", encoding="utf-8", compresslevel=6)
        self._max_chars = max_chars
        self.count = 0
        self.bytes = 0
        self.dropped = 0
        self._hashes: set[str] = set()

    def add(self, doc: Doc) -> bool:
        text = doc.text.strip()
        if len(text) < 80:
            self.dropped += 1
            return False
        if len(text) > self._max_chars:
            text = text[: self._max_chars]
        digest = hashlib.blake2b(text[:8192].encode("utf-8"), digest_size=8).hexdigest()
        if digest in self._hashes:
            self.dropped += 1
            return False
        self._hashes.add(digest)
        row = {
            "text": text,
            "source": doc.source,
            "url": doc.url,
            "lang": doc.lang,
            "meta": doc.meta,
        }
        line = json.dumps(row, ensure_ascii=False) + "\n"
        self._fh.write(line)
        self.count += 1
        self.bytes += len(line.encode("utf-8"))
        return True

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "Sink":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def write_manifest(out_dir: Path, source: str, data: dict[str, Any]) -> Path:
    path = out_dir / "manifests" / f"{source}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": source,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "user_agent": USER_AGENT,
        **data,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_collector(
    out_dir: Path,
    source: str,
    *,
    limit: int,
    body: Callable[[Sink], None],
) -> dict[str, Any]:
    """Boilerplate wrapper: sink, timing, manifest, summary print."""
    started = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    sink = Sink(out_dir, source)
    status = "ok"
    error: str | None = None
    try:
        body(sink)
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except Exception as exc:  # noqa: BLE001 - manifest must survive partial runs
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        sink.close()
        manifest = write_manifest(
            out_dir,
            source,
            {
                "status": status,
                "error": error,
                "limit": limit,
                "documents": sink.count,
                "dropped": sink.dropped,
                "uncompressed_bytes": sink.bytes,
                "elapsed_s": round(time.time() - started, 2),
                "file": str(sink.path),
            },
        )
    print(f"[{source}] docs={sink.count} dropped={sink.dropped} manifest={manifest}")
    return json.loads(manifest.read_text(encoding="utf-8"))


def iter_jsonl_gz(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def chunked(items: Iterable[Any], size: int) -> Iterator[list[Any]]:
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def env_token(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None
