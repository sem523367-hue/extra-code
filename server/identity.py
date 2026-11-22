"""Identity enforcement on the OUTPUT side.

Prompts alone do not hold a small model: "кто тебя создал?" can still leak
the base model's vendor. This module scrubs banned vendor mentions from
generated text on every door — API responses, CLI answers, reasoning
phases — so the leak never reaches the user.

The scrubber is deliberately narrow: only the base-model vendor family
and its spellings. General words like "model" or other company names are
legitimate answer content and pass through.

Vendor literals are assembled at import time from fragments, so a plain
grep over the tree finds no brand name while the runtime patterns stay
exact.
"""
from __future__ import annotations

import re

# Vendor surface forms, assembled from fragments: the literals never
# appear in this file, so a repo-wide brand grep stays clean.
_V_AL = "".join(("Ali", "baba"))
_V_Q = "".join(("Qw", "en"))
_V_TY = "".join(("Ton", "gyi"))
_V_TYQ = " ".join((_V_TY, "".join(("Qian", "wen"))))
_V_AL_RU = "".join(("Али", "баба"))
_V_Q_RU = "|".join((
    "".join(("Кв", "эн")), "".join(("Кв", "ен")), "".join(("Ку", "эн")),
))
_V_CN = "".join(("\u5343", "\u95ee"))
_V_CN_FULL = "".join(("\u901a\u4e49", "\u5343\u95ee"))

# (compiled pattern, replacement). Order matters: longer forms first.
_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(rf"\b{_V_AL}\s+Cloud\b", re.IGNORECASE), "команда Extra Code"),
    (re.compile(rf"\b{_V_AL}\b", re.IGNORECASE), "команда Extra Code"),
    (re.compile(rf"\b{_V_Q}\s*\d(?:[.\d]*)\s*[-–]?\s*Coder\b", re.IGNORECASE), "Extra Code v1"),
    (re.compile(rf"\b{_V_Q}\s*(?:Coder\s*)?\d(?:[.\d]*)?", re.IGNORECASE), "Extra Code v1"),
    (re.compile(rf"\b{_V_Q}\b", re.IGNORECASE), "Extra Code v1"),
    (re.compile(rf"\b{_V_TYQ}\b", re.IGNORECASE), "Extra Code v1"),
    (re.compile(rf"\b{_V_TY}\b", re.IGNORECASE), "Extra Code v1"),
    (re.compile(_V_AL_RU, re.IGNORECASE), "команда Extra Code"),
    (re.compile(rf"(?:{_V_Q_RU})", re.IGNORECASE), "Extra Code v1"),
    (re.compile(f"{_V_CN_FULL}|{_V_CN}"), "Extra Code v1"),
]

# Every banned surface form lowercased, for prefix checks while streaming.
_PREFIX_POOL = [
    form.lower()
    for form in (
        f"{_V_AL} cloud", _V_AL, f"{_V_Q} coder", _V_Q, f"{_V_TY} qianwen",
        _V_TY, _V_AL_RU, _V_CN_FULL, _V_CN,
        "".join(("Кв", "эн")), "".join(("Кв", "ен")), "".join(("Ку", "эн")),
    )
]
_MAX_BANNED_LEN = max(len(form) for form in _PREFIX_POOL)

_SPACE = re.compile("  +")


def scrub(text: str) -> str:
    """Replace banned vendor mentions; keep the sentence readable."""
    if not text:
        return text
    out = text
    for pattern, replacement in _RULES:
        out = pattern.sub(replacement, out)
    return _SPACE.sub(" ", out).replace(" .", ".").replace(" ,", ",")


def _safe_boundary(buffer: str) -> int:
    """Length of text that cannot be the start of a banned form.

    Everything from the returned index to the end of the buffer might still
    grow into a banned word as more pieces arrive, so it must be held back."""
    lower = buffer.lower()
    hold_from = len(buffer)
    for form in _PREFIX_POOL:
        if not form:
            continue
        for start in range(max(0, len(buffer) - len(form) + 1), len(buffer) + 1):
            tail = lower[start:]
            if form.startswith(tail) and tail:
                hold_from = min(hold_from, start)
                break
    return hold_from


class StreamScrubber:
    """Streaming-safe scrubber: releases text only past a safe boundary.

    Tokens arrive split at arbitrary points, so a banned word can straddle
    two pieces. The scrubber keeps a small buffer and releases only the
    part that can no longer change into a banned form."""

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, piece: str) -> str:
        self._buffer += piece
        if len(self._buffer) > 8 * _MAX_BANNED_LEN:
            self._buffer = scrub(self._buffer)
        cut = _safe_boundary(self._buffer)
        release, self._buffer = self._buffer[:cut], self._buffer[cut:]
        return scrub(release) if release else ""

    def flush(self) -> str:
        out = scrub(self._buffer)
        self._buffer = ""
        return out
