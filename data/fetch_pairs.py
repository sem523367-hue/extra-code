# -*- coding: utf-8 -*-
"""Fetch parallel-translation pairs ONLY, batched.

en_ru: resolve the ru title via langlinks (20 titles/request), then
batch extracts for both languages. zh_ru: titles are curated pairs,
just batch the extracts. Writes pair_<langs>_<slug>.txt with both
texts, gz-agnostic resume via _have. Run: python -m data.fetch_pairs
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from data.fetch_mega import (EN_RU_PAIRS, ZH_RU_PAIRS, OUT, _have,
                             api_get, fetch_extracts, slug)

EXTRACT_MIN = 800


def langlinks_ru(titles: list[str]) -> dict[str, str]:
    """en title -> ru title, 20 per request."""
    out: dict[str, str] = {}
    for i in range(0, len(titles), 20):
        batch = titles[i:i + 20]
        body = api_get("en", {
            "action": "query", "prop": "langlinks", "lllang": "ru",
            "lllimit": "max", "format": "json", "redirects": "1",
            "titles": "|".join(batch),
        })
        # redirects rewrite titles; map original -> resolved first
        redir: dict[str, str] = {}
        for r in body.get("query", {}).get("redirects", []):
            redir[r["to"]] = r["from"]
        pages = body.get("query", {}).get("pages", {})
        for page in pages.values():
            ru = None
            for ll in page.get("langlinks", []):
                if ll.get("lang") == "ru":
                    # with lllang set, the title lives under "*"
                    ru = ll.get("*") or ll.get("title")
                    break
            if not ru:
                continue
            src = page.get("title", "")
            out[src] = ru
            if src in redir:
                out[redir[src]] = ru
        time.sleep(1.0)
    return out


def make_pair(tag: str, src_title: str, src_lang: str,
              src_text: str, ru_title: str, ru_text: str) -> bool:
    out = OUT / f"pair_{tag}_{slug(ru_title)}.txt"
    if _have(out):
        return False
    mark = {"en_ru": "EN", "zh_ru": "ZH"}[tag]
    out.write_text(
        f"[{mark}] {src_title}\n\n{src_text}\n\n"
        f"=== ПЕРЕВОД (RU: {ru_title}) ===\n\n{ru_text}",
        encoding="utf-8")
    return True


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    # ---- en_ru ------------------------------------------------------
    todo_en = [t for t in EN_RU_PAIRS
               if not _have(OUT / f"pair_en_ru_{slug(t)}.txt")]
    # existing pairs were saved under the EN slug; also skip by content scan
    have_en = {p.name for p in OUT.glob("pair_en_ru_*")}
    print(f"en_ru: {len(todo_en)} to check", flush=True)
    if todo_en:
        links = langlinks_ru(todo_en)
        print(f"en_ru: langlinks resolved {len(links)}", flush=True)
        en_texts = fetch_extracts("en", todo_en)
        ru_titles = [links[t] for t in todo_en if t in links]
        ru_texts = fetch_extracts("ru", ru_titles) if ru_titles else {}
        made = 0
        for t in todo_en:
            ru_t = links.get(t)
            en_txt = en_texts.get(t)
            ru_txt = ru_texts.get(ru_t) if ru_t else None
            if en_txt and ru_txt:
                made += make_pair("en_ru", t, "en", en_txt, ru_t, ru_txt)
        print(f"en_ru: made {made} pairs", flush=True)

    # ---- zh_ru ------------------------------------------------------
    todo_zh = [(z, r) for z, r in ZH_RU_PAIRS
               if not _have(OUT / f"pair_zh_ru_{slug(r)}.txt")]
    print(f"zh_ru: {len(todo_zh)} to check", flush=True)
    if todo_zh:
        zh_titles = [z for z, _ in todo_zh]
        ru_titles = [r for _, r in todo_zh]
        zh_texts = fetch_extracts("zh", zh_titles)
        ru_texts = fetch_extracts("ru", ru_titles)
        made = 0
        for zh_t, ru_t in todo_zh:
            zh_txt = zh_texts.get(zh_t)
            ru_txt = ru_texts.get(ru_t)
            if zh_txt and ru_txt:
                made += make_pair("zh_ru", zh_t, "zh", zh_txt, ru_t, ru_txt)
        print(f"zh_ru: made {made} pairs", flush=True)

    n_en = len(list(OUT.glob("pair_en_ru_*")))
    n_zh = len(list(OUT.glob("pair_zh_ru_*")))
    print(json.dumps({"pair_en_ru": n_en, "pair_zh_ru": n_zh}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
