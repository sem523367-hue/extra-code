# -*- coding: utf-8 -*-
"""Fetch daily current-events pages: the real news feed, day by day.

The monthly Portal:Current events/August_2026 page is a thin 1 KB nav
shell that transcludes daily pages; the actual feed lives at
"Portal:Current events/2026 August 15" (space, not underscore, in the
title). ~660 pages for 2025 + 2026 through today.
"""
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(r"C:\Users\Sem52\Documents\Default Project\extra-code")
FACTS = ROOT / "data" / "corpus" / "facts_2026"
FACTS.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "ExtraCodeKB/1.0 (personal local knowledge base; no mirroring)"}
MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
# today = 2026-11-22: full 2025, 2026 through November 22
DAYS: list[tuple[int, int, int]] = []
for m in range(1, 13):
    for d in range(1, 32):
        DAYS.append((2025, m, d))
for m in range(1, 12):
    for d in range(1, 32):
        DAYS.append((2026, m, d))
for d in range(1, 23):
    DAYS.append((2026, 11, d))


def strip_wikitext(wt: str) -> str:
    wt = re.sub(r"\{\{Current events[^}]*\}\}", "", wt, count=1)
    wt = re.sub(r"<ref[^>]*/>", "", wt)
    wt = re.sub(r"<ref[^>]*>.*?</ref>", "", wt, flags=re.S)
    for _ in range(6):
        wt = re.sub(r"\{\{[^{}]*\}\}", "", wt)
    wt = re.sub(r"\{\|.*?\|\}", "", wt, flags=re.S)
    wt = re.sub(r"<![^>]*-->", "", wt)
    wt = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", wt)
    wt = re.sub(r"'{2,5}", "", wt)
    wt = re.sub(r"<[^>]+>", "", wt)
    wt = re.sub(r"\n{3,}", "\n\n", wt)
    return wt.strip()


def pull(title: str, out: Path) -> bool:
    if out.exists():
        return True
    url = (
        "https://en.wikipedia.org/w/index.php?title="
        + urllib.parse.quote(title.replace(" ", "_"), safe=":/")
        + "&action=raw"
    )
    for attempt in range(1, 6):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read().decode("utf-8", errors="ignore")
            time.sleep(1.0)
            text = strip_wikitext(body)
            if len(text) < 300:
                return False  # empty day or 404-ish stub
            out.write_text(text, encoding="utf-8")
            return True
        except Exception as exc:
            code = getattr(exc, "code", None)
            if code == 404:
                time.sleep(0.3)
                return False  # future/nonexistent day — quiet skip
            wait = 30 if code == 429 else 3 * attempt
            print(f"  fail {title}: {exc} -> wait {wait}s", flush=True)
            time.sleep(wait)
    return False


def main() -> int:
    t0 = time.time()
    ok = miss = 0
    for y, m, d in DAYS:
        title = f"Portal:Current events/{y} {MONTHS[m - 1]} {d}"
        out = FACTS / f"news_{y}-{m:02d}-{d:02d}.md"
        if pull(title, out):
            ok += 1
        else:
            miss += 1
        if (ok + miss) % 50 == 0:
            print(f"  {ok + miss}/{len(DAYS)} ok={ok}", flush=True)
    print(f"DONE ok={ok} miss={miss} elapsed={int(time.time() - t0)}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
