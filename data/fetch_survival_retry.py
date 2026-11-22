# -*- coding: utf-8 -*-
"""Retry pass for survival articles that failed on rate limiting (429).

Strategy: walk the same topic lists, skip files already on disk, fetch the
missing ones with exponential backoff (1s -> 2s -> 4s -> 8s) per title and a
1.0s base pause between requests. Wikipedia allows steady 1 rps with a
proper User-Agent; the first run burst too hard at 0.25s.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from fetch_survival import OUT, RU_TOPICS, TOPICS, fetch_extract, slug


def fetch_with_backoff(lang: str, title: str, attempts: int = 4) -> str | None:
    delay = 1.0
    for attempt in range(attempts):
        text = fetch_extract(lang, title)
        if text is not None:
            return text
        time.sleep(delay)
        delay = min(delay * 2, 8.0)
    return None


def main() -> None:
    ok = fail = 0
    for group, titles in TOPICS.items():
        for title in titles:
            t = title.strip()
            if not t or t.startswith("#"):
                continue
            out = OUT / f"en_{group}_{slug(t)}.txt"
            if out.exists() and out.stat().st_size > 1000:
                continue
            text = fetch_with_backoff("en", t)
            time.sleep(1.0)
            if text:
                out.write_text(text, encoding="utf-8")
                ok += 1
            else:
                fail += 1
                print(f"  MISS en:{t}")
        print(f"pass en/{group}: new={ok} miss={fail}", flush=True)
    for title in RU_TOPICS:
        t = title.strip()
        if not t:
            continue
        out = OUT / f"ru_{slug(t)}.txt"
        if out.exists() and out.stat().st_size > 1000:
            continue
        text = fetch_with_backoff("ru", t)
        time.sleep(1.0)
        if text:
            out.write_text(text, encoding="utf-8")
            ok += 1
        else:
            fail += 1
            print(f"  MISS ru:{t}")
    files = list(OUT.glob("*.txt"))
    print(json.dumps({"newly_fetched": ok, "still_missing": fail,
                      "files_on_disk": len(files),
                      "mb": round(sum(p.stat().st_size for p in files) / 1e6, 1)}))


if __name__ == "__main__":
    main()
