"""Wikipedia collector: MediaWiki API (default) or a local pages-articles dump."""
from __future__ import annotations

import bz2
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

from .base import Doc, RateLimiter, Sink, http_json, make_client, run_collector

log = logging.getLogger(__name__)
API = "https://{lang}.wikipedia.org/w/api.php"


def _api_mode(sink: Sink, limit: int, langs: tuple[str, ...], api_rate: float) -> None:
    limiter = RateLimiter(api_rate, burst=2)
    with make_client() as client:
        failures = 0
        while sink.count < limit and failures < 5:
            progressed = False
            for lang in langs:
                if sink.count >= limit:
                    break
                params = {
                    "action": "query",
                    "generator": "random",
                    "grnnamespace": 0,
                    "prop": "extracts",
                    "explaintext": 1,
                    "exlimit": "max",
                    "grnlimit": 20,
                    "format": "json",
                    "formatversion": 2,
                }
                try:
                    payload = http_json(
                        client,
                        API.format(lang=lang),
                        params=params,
                        limiter=limiter,
                        retries=3,
                        max_bytes=8 * 1024 * 1024,
                    )
                except Exception as exc:  # noqa: BLE001 - retry counter below
                    failures += 1
                    log.warning("wikipedia %s request failed: %s", lang, exc)
                    continue
                failures = 0
                for page in payload.get("query", {}).get("pages", []):
                    text = (page.get("extract") or "").strip()
                    if len(text) < 500:
                        continue
                    title = page.get("title", "")
                    doc = Doc(
                        text=text,
                        source="wikipedia",
                        url=f"https://{lang}.wikipedia.org/?curid={page.get('pageid', '')}",
                        lang=lang,
                        meta={"title": title, "pageid": page.get("pageid")},
                    )
                    if sink.add(doc):
                        progressed = True
                        if sink.count % 100 == 0:
                            log.info("wikipedia: %d docs (%s)", sink.count, title)
                if sink.count >= limit:
                    break
            if not progressed and failures == 0:
                # Random generator can return only stubs; keep going anyway.
                continue
    if failures >= 5:
        raise RuntimeError("wikipedia API failed 5 consecutive times")


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _dump_mode(sink: Sink, dump: Path, limit: int, lang: str) -> None:
    log.info("wikipedia: streaming dump %s", dump)
    with bz2.open(dump, "rb") as fh:
        context = ET.iterparse(fh, events=("end",))
        for _, elem in context:
            if _local(elem.tag) != "page":
                continue
            title = ""
            text = ""
            redirect = False
            for node in elem.iter():
                name = _local(node.tag)
                if name == "title" and node.text:
                    title = node.text
                elif name == "redirect":
                    redirect = True
                elif name == "text" and node.text:
                    text = node.text
            elem.clear()
            if redirect or not text or len(text) < 500:
                continue
            doc = Doc(
                text=text,
                source="wikipedia",
                url=f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}",
                lang=lang,
                meta={"title": title, "format": "wikitext", "dump": dump.name},
            )
            if sink.add(doc):
                if sink.count % 500 == 0:
                    log.info("wikipedia dump: %d docs", sink.count)
            if sink.count >= limit:
                break


def _lang_from_name(path: Path) -> str:
    stem = path.name.split("wiki", 1)[0].split("-", 1)[0]
    return stem if stem in {"ru", "en", "de", "fr", "es", "zh"} else "und"


def collect(
    out_dir: Path,
    *,
    limit: int = 1000,
    langs: tuple[str, ...] = ("ru", "en"),
    dump: Path | None = None,
    api_rate: float = 2.0,
) -> dict[str, object]:
    def body(sink: Sink) -> None:
        if dump is not None:
            _dump_mode(sink, dump, limit, _lang_from_name(dump))
        else:
            _api_mode(sink, limit, langs, api_rate)

    return run_collector(out_dir, "wikipedia", limit=limit, body=body)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect Wikipedia text")
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--langs", default="ru,en")
    parser.add_argument("--dump", default=None)
    args = parser.parse_args()
    collect(
        Path(args.out),
        limit=args.limit,
        langs=tuple(args.langs.split(",")),
        dump=Path(args.dump) if args.dump else None,
    )
