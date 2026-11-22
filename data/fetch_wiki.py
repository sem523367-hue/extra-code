# -*- coding: utf-8 -*-
"""Batched Wikipedia fetcher via raw wikitext (prop=revisions).

TextExtracts can't do full-text batching (exlimit>1 needs exintro, so
every "batch" quietly returned one article). Raw revisions DO batch:
50 titles per request. Wikitext is stripped to prose here.

Same PLAN/targets as fetch_mega; resume via _have (gz-aware).
Run from extra-code root: python -m data.fetch_wiki [--group name]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from data.fetch_mega import (EN_RU_PAIRS, PLAN, ZH_RU_PAIRS, OUT, _have,
                             api_get, category_members, slug)

BATCH = 50


# --- wikitext -> prose ------------------------------------------------

_RE_COMMENT = re.compile(r"<!--.*?-->", re.S)
_RE_REF = re.compile(r"<ref[^>]*?/>|<ref[^>]*?>.*?</ref>", re.S | re.I)
_RE_TAG = re.compile(r"<[^>]+>")
_RE_LINK_CAT = re.compile(r"\[\[(?:[a-z\-]{2,10}:)?[Cc]ategory:[^\]]*\]\]")
_RE_LINK = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")
_RE_EXT = re.compile(r"\[([^\] ]+)[^\]]*\]")
_RE_HEADING = re.compile(r"^(=+)\s*(.*?)\s*=+\s*$", re.M)
_RE_BOLD = re.compile(r"'{2,}")
_RE_ENTITY = re.compile(r"&([a-z]+|#[0-9]+);")
_ENTITIES = {"nbsp": " ", "mdash": "—", "ndash": "–", "hellip": "…",
             "laquo": "«", "raquo": "»", "nbsp;": " ", "amp": "&",
             "quot": '"', "lt": "<", "gt": ">"}


def _strip_nested(text: str, open_ch: str, close_ch: str) -> str:
    """Remove {{...}} / {|...|} blocks however deep they nest."""
    while True:
        out: list[str] = []
        depth = 0
        i = 0
        cut = False
        while i < len(text):
            if text.startswith(open_ch, i):
                if depth == 0:
                    cut = True
                depth += 1
                i += len(open_ch)
            elif text.startswith(close_ch, i):
                if depth > 0:
                    depth -= 1
                i += len(close_ch)
            elif depth == 0:
                out.append(text[i])
                i += 1
            else:
                i += 1
        text = "".join(out)
        if not cut:
            return text


def wikitext_to_prose(raw: str) -> str:
    text = _RE_COMMENT.sub(" ", raw)
    text = _strip_nested(text, "{{", "}}")   # templates incl. infoboxes
    text = _strip_nested(text, "{|", "|}")   # tables
    text = _RE_REF.sub(" ", text)
    text = _RE_LINK_CAT.sub(" ", text)
    text = _RE_LINK.sub(r"\1", text)
    text = _RE_EXT.sub(r"\1", text)
    text = _RE_TAG.sub(" ", text)
    text = _RE_HEADING.sub(r"\n\n\2\n", text)
    text = _RE_BOLD.sub("", text)
    text = _RE_ENTITY.sub(lambda m: _ENTITIES.get(m.group(1), " "), text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --- batched transport -------------------------------------------------

def fetch_wikitexts(lang: str, titles: list[str]) -> dict[str, str]:
    """{title: prose} for up to BATCH titles per API hit."""
    out: dict[str, str] = {}
    for i in range(0, len(titles), BATCH):
        batch = titles[i:i + BATCH]
        body = api_get(lang, {
            "action": "query", "prop": "revisions", "rvprop": "content",
            "rvslots": "main", "format": "json", "redirects": "1",
            "titles": "|".join(batch),
        })
        pages = body.get("query", {}).get("pages", {})
        for page in pages.values():
            if "revisions" not in page:
                continue
            slot = page["revisions"][0].get("slots", {}).get("main", {})
            raw = slot.get("*") or page["revisions"][0].get("*") or ""
            if not raw:
                continue
            if "#REDIRECT" in raw[:120].upper():
                continue
            prose = wikitext_to_prose(raw)
            if len(prose) > 900:
                out[page.get("title", "")] = prose
        time.sleep(1.2)
    return out


def _ci_map(d: dict[str, str]) -> dict[str, str]:
    return {k.casefold(): v for k, v in d.items()}


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default=None, help="only this group")
    ap.add_argument("--pairs", action="store_true", help="only pairs")
    args = ap.parse_args()

    if not args.pairs:
        for group, (target, en_cats, ru_cats) in PLAN.items():
            if args.group and group != args.group:
                continue
            queue: list[tuple[str, str]] = []
            for cat in en_cats:
                for t in category_members("en", cat):
                    queue.append(("en", t))
            for cat in ru_cats:
                for t in category_members("ru", cat):
                    queue.append(("ru", t))
            pending: dict[str, list[str]] = {"en": [], "ru": []}
            for lang, title in queue:
                out = OUT / f"{lang}_{group}_{slug(title)}.txt"
                if not _have(out):
                    pending[lang].append(title)
            have = len(list(OUT.glob(f"en_{group}_*"))) + \
                   len(list(OUT.glob(f"ru_{group}_*")))
            made = 0
            for lang, titles in pending.items():
                if have + made >= target + 50:
                    break
                if not titles:
                    continue
                titles = titles[: target - have + 100]
                texts = _ci_map(fetch_wikitexts(lang, titles))
                for t in titles:
                    txt = texts.get(t.casefold())
                    out = OUT / f"{lang}_{group}_{slug(t)}.txt"
                    if txt and not _have(out):
                        out.write_text(txt, encoding="utf-8")
                        made += 1
            print(f"{group}: have {have}, +{made}", flush=True)

    if args.pairs or args.group is None:
        # en_ru via langlinks (batched), zh_ru curated
        todo = [t for t in EN_RU_PAIRS
                if not _have(OUT / f"pair_en_ru_{slug(t)}.txt")]
        if todo:
            links: dict[str, str] = {}
            for i in range(0, len(todo), BATCH):
                body = api_get("en", {
                    "action": "query", "prop": "langlinks", "lllang": "ru",
                    "lllimit": "max", "format": "json", "redirects": "1",
                    "titles": "|".join(todo[i:i + BATCH]),
                })
                for p in body.get("query", {}).get("pages", {}).values():
                    ru = next((ll.get("*") or ll.get("title")
                               for ll in p.get("langlinks", [])
                               if ll.get("lang") == "ru"), None)
                    if ru:
                        links[p.get("title", "")] = ru
                time.sleep(1.2)
            en_texts = _ci_map(fetch_wikitexts("en", todo))
            ru_titles = [links[t] for t in todo if t in links]
            ru_texts = _ci_map(fetch_wikitexts("ru", ru_titles)) \
                if ru_titles else {}
            made = 0
            for t in todo:
                ru_t = links.get(t)
                en_txt = en_texts.get(t.casefold())
                ru_txt = ru_texts.get(ru_t.casefold()) if ru_t else None
                if en_txt and ru_txt:
                    out = OUT / f"pair_en_ru_{slug(ru_t)}.txt"
                    if not _have(out):
                        out.write_text(
                            f"[EN] {t}\n\n{en_txt}\n\n"
                            f"=== ПЕРЕВОД (RU: {ru_t}) ===\n\n{ru_txt}",
                            encoding="utf-8")
                        made += 1
            print(f"pair_en_ru: +{made}", flush=True)

        todo_zh = [(z, r) for z, r in ZH_RU_PAIRS
                   if not _have(OUT / f"pair_zh_ru_{slug(r)}.txt")]
        if todo_zh:
            zh_texts = _ci_map(fetch_wikitexts("zh", [z for z, _ in todo_zh]))
            ru_texts = _ci_map(fetch_wikitexts("ru", [r for _, r in todo_zh]))
            made = 0
            for zh_t, ru_t in todo_zh:
                zh_txt = zh_texts.get(zh_t.casefold())
                ru_txt = ru_texts.get(ru_t.casefold())
                if zh_txt and ru_txt:
                    out = OUT / f"pair_zh_ru_{slug(ru_t)}.txt"
                    if not _have(out):
                        out.write_text(
                            f"[ZH] {zh_t}\n\n{zh_txt}\n\n"
                            f"=== ПЕРЕВОД (RU: {ru_t}) ===\n\n{ru_txt}",
                            encoding="utf-8")
                        made += 1
            print(f"pair_zh_ru: +{made}", flush=True)

    n_en = len(list(OUT.glob("pair_en_ru_*")))
    n_zh = len(list(OUT.glob("pair_zh_ru_*")))
    total = len(list(OUT.glob("*.txt")))
    print(json.dumps({"pair_en_ru": n_en, "pair_zh_ru": n_zh,
                      "plain_txt_total": total}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
