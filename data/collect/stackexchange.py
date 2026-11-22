"""Stack Exchange corpus collector.

Pulls high-vote questions (and their top answers) from the Stack Exchange API
and flattens each thread into a single plain-text document.

    python data/collect/stackexchange.py --out data/raw --limit 2000

The API is quota-limited per IP (300 requests/day anonymous): one request per
question page plus one per page of answers, paced at 1 req/s, and any
``backoff`` the server asks for is honoured before the next call.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Sequence

import httpx

try:
    from .base import Doc, RateLimiter, Sink, chunked, http_json, make_client, run_collector
except ImportError:  # direct execution: python data/collect/stackexchange.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from data.collect.base import Doc, RateLimiter, Sink, chunked, http_json, make_client, run_collector

LOG = logging.getLogger(__name__)

API_ROOT = "https://api.stackexchange.com/2.3"
DEFAULT_SITES: tuple[str, ...] = (
    "stackoverflow",
    "ru.stackoverflow",
    "math.stackexchange",
    "superuser",
    "serverfault",
)

PAGE_SIZE = 100
MAX_IDS_PER_REQUEST = 100
MAX_ANSWERS_PER_QUESTION = 3
MIN_DOC_CHARS = 150
MAX_BACKOFF_S = 60.0


class HtmlToText(HTMLParser):
    """Minimal HTML to text: drops script/style, keeps <pre> layout readable."""

    SKIP_TAGS = frozenset({"script", "style", "head", "noscript", "iframe", "svg", "template"})
    BLOCK_TAGS = frozenset(
        {
            "address",
            "article",
            "aside",
            "blockquote",
            "dd",
            "div",
            "dl",
            "dt",
            "figcaption",
            "figure",
            "footer",
            "form",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "header",
            "hr",
            "li",
            "main",
            "nav",
            "ol",
            "p",
            "section",
            "table",
            "tbody",
            "td",
            "tfoot",
            "th",
            "thead",
            "tr",
            "ul",
        }
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0
        self._pre_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "pre":
            self._pre_depth += 1
            self._parts.append("\n")
        elif tag == "br":
            self._parts.append("\n")
        elif tag == "li":
            self._parts.append("\n- ")
        elif tag == "code" and not self._pre_depth:
            self._parts.append("`")
        elif tag in self.BLOCK_TAGS:
            self._parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP_TAGS:
            return  # self-closing skipped tag never opened a scope
        self.handle_starttag(tag, attrs)
        if tag == "pre":
            if self._pre_depth:
                self._pre_depth -= 1
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP_TAGS:
            if self._skip_depth:
                self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag == "pre":
            if self._pre_depth:
                self._pre_depth -= 1
            self._parts.append("\n")
        elif tag == "code" and not self._pre_depth:
            self._parts.append("`")
        elif tag in self.BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth or not data:
            return
        if self._pre_depth:
            self._parts.append(data)
        else:
            self._parts.append(re.sub(r"\s+", " ", data))

    def get_text(self) -> str:
        text = "".join(self._parts).replace("\xa0", " ")
        text = re.sub(r"[ \t]+\n", "\n", text)
        lines: list[str] = []
        blank_run = 0
        for line in text.split("\n"):
            line = line.rstrip()
            if not line:
                blank_run += 1
                if blank_run > 1:  # more than one blank line collapses to one
                    continue
                lines.append("")
                continue
            blank_run = 0
            lines.append(line if line.startswith((" ", "\t")) else re.sub(r"[ \t]{2,}", " ", line))
        return "\n".join(lines).strip()


def _fallback_strip(fragment: str) -> str:
    text = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", fragment)
    text = re.sub(r"(?s)<[^>]+>", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def html_to_text(fragment: str) -> str:
    """Convert an API-supplied HTML fragment to compact plain text."""
    if not fragment:
        return ""
    parser = HtmlToText()
    try:
        parser.feed(fragment)
        parser.close()
    except Exception:  # noqa: BLE001 - malformed markup must not kill a whole page
        return _fallback_strip(fragment)
    return parser.get_text()


def _api(
    client: httpx.Client,
    url: str,
    *,
    params: dict[str, Any],
    limiter: RateLimiter,
) -> dict[str, Any]:
    payload = http_json(client, url, params=params, limiter=limiter)
    if not isinstance(payload, dict):
        raise ValueError(f"unexpected Stack Exchange payload: {type(payload).__name__}")
    error_id = payload.get("error_id")
    if error_id is not None:
        raise RuntimeError(f"api error {error_id}: {payload.get('error_message')}")
    backoff = payload.get("backoff")
    if backoff:
        try:
            pause = min(float(backoff), MAX_BACKOFF_S)
        except (TypeError, ValueError):
            pause = 0.0
        if pause > 0:
            LOG.info("server requested %.1fs backoff", pause)
            time.sleep(pause)
    return payload


def _fetch_answers(
    client: httpx.Client,
    *,
    site: str,
    ids: list[int],
    limiter: RateLimiter,
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for batch in chunked(ids, MAX_IDS_PER_REQUEST):
        # The 2.3 API takes a semicolon-separated vector; a comma-joined path
        # returns 404 no_method, which silently cost every answer in an early run.
        joined = ";".join(str(qid) for qid in batch)
        payload = _api(
            client,
            f"{API_ROOT}/questions/{joined}/answers",
            params={
                "site": site,
                "filter": "withbody",
                "pagesize": PAGE_SIZE,
                "sort": "votes",
                "order": "desc",
            },
            limiter=limiter,
        )
        for item in payload.get("items") or []:
            if not isinstance(item, dict):
                continue
            try:
                question_id = int(item.get("question_id") or 0)
            except (TypeError, ValueError):
                continue
            if question_id > 0:
                grouped.setdefault(question_id, []).append(item)
    return grouped


def _build_doc(site: str, question: dict[str, Any], answers: list[dict[str, Any]]) -> Doc | None:
    title = html_to_text(str(question.get("title") or ""))
    body = html_to_text(str(question.get("body") or ""))
    if not title and not body:
        return None

    ranked = sorted(answers, key=lambda item: int(item.get("score") or 0), reverse=True)
    parts: list[str] = [f"Question: {title}", "", body]
    included = 0
    for answer in ranked[:MAX_ANSWERS_PER_QUESTION]:
        answer_body = html_to_text(str(answer.get("body") or ""))
        if not answer_body:
            continue
        score = int(answer.get("score") or 0)
        parts.extend(["", f"Answer (score {score}): {answer_body}"])
        included += 1

    text = "\n".join(parts).strip()
    if len(text) < MIN_DOC_CHARS:
        return None

    link = str(question.get("link") or "")
    site_lang = "ru" if site.startswith("ru.") else "en"
    try:
        question_id = int(question.get("question_id") or 0)
    except (TypeError, ValueError):
        question_id = 0
    try:
        score = int(question.get("score") or 0)
    except (TypeError, ValueError):
        score = 0
    tags = question.get("tags")
    return Doc(
        text=text,
        source="stackexchange",
        url=link,
        lang=site_lang,
        meta={
            "site": site,
            "question_id": question_id,
            "score": score,
            "tags": list(tags) if isinstance(tags, list) else [],
            "answers": included,
            "link": link,
        },
    )


def _run(
    sink: Sink,
    *,
    limit: int,
    sites: tuple[str, ...],
    max_pages: int,
    include_answers: bool,
) -> None:
    limiter = RateLimiter(1.0, burst=1)
    questions_seen = 0
    leftovers = 0

    with make_client(timeout=60.0) as client:
        for site in sites:
            page = 1
            while page <= max_pages:
                if sink.count >= limit:
                    break
                try:
                    payload = _api(
                        client,
                        f"{API_ROOT}/questions",
                        params={
                            "site": site,
                            "pagesize": PAGE_SIZE,
                            "sort": "votes",
                            "order": "desc",
                            "filter": "withbody",
                            "page": page,
                        },
                        limiter=limiter,
                    )
                except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                    LOG.warning("%s questions page %d failed: %s", site, page, exc)
                    break

                questions = [item for item in (payload.get("items") or []) if isinstance(item, dict)]
                if not questions:
                    break

                answers_by_question: dict[int, list[dict[str, Any]]] = {}
                if include_answers:
                    ids = []
                    for item in questions:
                        try:
                            qid = int(item.get("question_id") or 0)
                        except (TypeError, ValueError):
                            continue
                        if qid > 0:
                            ids.append(qid)
                    if ids:
                        try:
                            answers_by_question = _fetch_answers(
                                client, site=site, ids=ids, limiter=limiter
                            )
                        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                            LOG.warning("%s answers batch failed: %s", site, exc)

                for question in questions:
                    try:
                        qid = int(question.get("question_id") or 0)
                    except (TypeError, ValueError):
                        qid = 0
                    questions_seen += 1
                    doc = _build_doc(site, question, answers_by_question.get(qid, []))
                    if doc is None:
                        leftovers += 1
                        continue
                    if sink.add(doc):
                        if sink.count % 50 == 0:
                            LOG.info("stackexchange: %d docs written (site=%s)", sink.count, site)
                    else:
                        leftovers += 1
                    if sink.count >= limit:
                        break

                if sink.count >= limit:
                    break
                if payload.get("quota_remaining") == 0:
                    LOG.warning("api quota exhausted at %s page %d", site, page)
                    break
                if not payload.get("has_more"):
                    break
                page += 1

    LOG.info(
        "stackexchange done: docs=%d questions_seen=%d skipped_or_deduped=%d",
        sink.count,
        questions_seen,
        leftovers,
    )


def collect(
    out_dir: Path,
    *,
    limit: int = 2000,
    sites: tuple[str, ...] = DEFAULT_SITES,
    max_pages: int = 20,
    include_answers: bool = True,
) -> dict[str, object]:
    """Collect Stack Exchange question threads into ``out_dir/raw/stackexchange``."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    if max_pages <= 0:
        raise ValueError("max_pages must be positive")
    site_tuple = tuple(s.strip() for s in sites if s.strip())
    if not site_tuple:
        raise ValueError("at least one site is required")

    def body(sink: Sink) -> None:
        _run(sink, limit=limit, sites=site_tuple, max_pages=max_pages, include_answers=include_answers)

    return run_collector(out_dir, "stackexchange", limit=limit, body=body)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect Stack Exchange question threads")
    parser.add_argument("--out", type=Path, default=Path("data/raw"), help="output root directory")
    parser.add_argument("--limit", type=int, default=2000, help="maximum documents to write")
    parser.add_argument(
        "--sites",
        nargs="+",
        default=list(DEFAULT_SITES),
        metavar="SITE",
        help="Stack Exchange site slugs",
    )
    parser.add_argument("--no-answers", action="store_true", help="skip answer retrieval")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    manifest = collect(
        args.out,
        limit=args.limit,
        sites=tuple(args.sites),
        include_answers=not args.no_answers,
    )
    LOG.info("manifest: %s", manifest.get("file"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
