"""arXiv collector over the public Atom API.

Endpoint: http://export.arxiv.org/api/query

The API returns an Atom feed with one ``<entry>`` per paper. Requests are
serialised through a shared token bucket because arXiv explicitly asks for
three seconds between calls; the limiter sleeps, the caller does not.

Categories are walked round-robin so one slow or failing ``cs.*`` category
cannot starve ``math.OC`` or ``stat.ML``. A category is dropped after an empty
page; the run aborts only once every category has failed twice.

Public entry point::

    collect(out_dir, *, limit=5000, max_results=100,
            categories=("cs.CL", "cs.LG", "cs.AI", "cs.SE", "math.OC", "stat.ML"))
        -> dict[str, object]
"""
from __future__ import annotations

import argparse
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

from .base import Doc, RateLimiter, Sink, http_get, make_client, run_collector

logger = logging.getLogger(__name__)

ARXIV_API = "http://export.arxiv.org/api/query"
ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"
_NS = {"atom": ATOM_NS, "arxiv": ARXIV_NS}

# arXiv's published guidance: one request every three seconds.
REQUEST_RATE_PER_S = 1 / 3
ARXIV_MAX_PAGE = 100
MIN_ABSTRACT_CHARS = 200
MAX_PAGES = 20_000
LOG_EVERY_DOCS = 50
MAX_CONSECUTIVE_FAILURES = 2

_WS_RE = re.compile(r"\s+")


def _field(entry: ET.Element, path: str) -> str:
    """Return a whitespace-collapsed text field, or "" when absent/empty."""
    node = entry.find(path, _NS)
    if node is None or node.text is None:
        return ""
    return _WS_RE.sub(" ", node.text).strip()


def _fetch_page(
    client: httpx.Client,
    category: str,
    start: int,
    page_size: int,
    limiter: RateLimiter,
) -> list[ET.Element]:
    """Fetch one Atom page for ``category`` and return its ``entry`` elements.

    Raises whatever :func:`http_get` raises after its own bounded retries.
    """
    raw = http_get(
        client,
        ARXIV_API,
        params={
            "search_query": f"cat:{category}",
            "start": start,
            "max_results": page_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        },
        headers={"Accept": "application/atom+xml"},
        limiter=limiter,
    )
    root = ET.fromstring(raw)
    return list(root.findall("atom:entry", _NS))


def _arxiv_id_from_url(entry_id: str) -> str:
    """Extract ``2401.01234v1`` from ``http://arxiv.org/abs/2401.01234v1``."""
    marker = "/abs/"
    if marker in entry_id:
        return entry_id.rsplit(marker, 1)[-1]
    return entry_id.rsplit("/", 1)[-1] if entry_id else ""


def _entry_to_doc(entry: ET.Element) -> Doc | None:
    """Convert one Atom entry to a :class:`Doc`; ``None`` when it is too thin."""
    abstract = _field(entry, "atom:summary")
    if len(abstract) < MIN_ABSTRACT_CHARS:
        return None

    title = _field(entry, "atom:title")
    entry_id = _field(entry, "atom:id")
    published = _field(entry, "atom:published")
    updated = _field(entry, "atom:updated")

    authors = [
        name
        for name in (_field(author, "atom:name") for author in entry.findall("atom:author", _NS))
        if name
    ]
    categories = [
        str(category.get("term"))
        for category in entry.findall("atom:category", _NS)
        if category.get("term")
    ]
    primary_node = entry.find("arxiv:primary_category", _NS)
    primary = str(primary_node.get("term")) if primary_node is not None and primary_node.get("term") else ""

    text = "\n\n".join(
        part
        for part in (
            title,
            f"Authors: {', '.join(authors)}" if authors else "",
            f"Categories: {', '.join(categories)}" if categories else "",
            abstract,
        )
        if part
    )

    return Doc(
        text=text,
        source="arxiv",
        url=entry_id,
        lang="en",
        meta={
            "arxiv_id": _arxiv_id_from_url(entry_id),
            "published": published,
            "updated": updated,
            "categories": categories,
            "authors": authors,
            "primary_category": primary or (categories[0] if categories else ""),
        },
    )


def collect(
    out_dir: Path,
    *,
    limit: int = 5000,
    max_results: int = 100,
    categories: tuple[str, ...] = ("cs.CL", "cs.LG", "cs.AI", "cs.SE", "math.OC", "stat.ML"),
) -> dict[str, object]:
    """Collect up to ``limit`` arXiv papers into ``out_dir``.

    Args:
        out_dir: manifest/raw root handed to :func:`run_collector`.
        limit: maximum number of documents written.
        max_results: page size, clamped to arXiv's practical ceiling of 100.
        categories: arXiv category identifiers walked round-robin.

    Returns:
        The manifest dict produced by :func:`run_collector`.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")
    if max_results <= 0:
        raise ValueError("max_results must be positive")

    ordered = tuple(dict.fromkeys(category.strip() for category in categories if category.strip()))
    if not ordered:
        raise ValueError("categories must contain at least one arXiv category")
    page_size = max(1, min(max_results, ARXIV_MAX_PAGE))

    def body(sink: Sink) -> None:
        # burst=1: the first call goes out immediately, then one per three seconds.
        limiter = RateLimiter(REQUEST_RATE_PER_S, burst=1)
        cursors: dict[str, int] = {category: 0 for category in ordered}
        failures: dict[str, int] = {category: 0 for category in ordered}
        exhausted: set[str] = set()
        pages = 0
        skipped_short = 0

        with make_client() as client:
            while sink.count < limit and pages < MAX_PAGES:
                active = [category for category in ordered if category not in exhausted]
                if not active:
                    logger.info("all arXiv categories exhausted at %d docs", sink.count)
                    break

                for category in active:
                    if sink.count >= limit:
                        break
                    if pages >= MAX_PAGES:
                        break

                    try:
                        entries = _fetch_page(client, category, cursors[category], page_size, limiter)
                    except Exception as exc:  # noqa: BLE001 - one bad category must not kill the run
                        failures[category] += 1
                        logger.warning(
                            "arxiv %s start=%d failed (%d/%d): %s",
                            category,
                            cursors[category],
                            failures[category],
                            MAX_CONSECUTIVE_FAILURES,
                            exc,
                        )
                        if all(
                            other in exhausted or failures[other] >= MAX_CONSECUTIVE_FAILURES
                            for other in ordered
                        ):
                            raise RuntimeError(
                                f"every arXiv category failed at least {MAX_CONSECUTIVE_FAILURES} times; "
                                f"last failure: {category}: {exc}"
                            ) from exc
                        continue

                    failures[category] = 0
                    pages += 1

                    if not entries:
                        exhausted.add(category)
                        logger.info("arxiv %s exhausted after start=%d", category, cursors[category])
                        continue

                    cursors[category] += page_size
                    for entry in entries:
                        if sink.count >= limit:
                            break
                        doc = _entry_to_doc(entry)
                        if doc is None:
                            skipped_short += 1
                            continue
                        sink.add(doc)
                        if sink.count and sink.count % LOG_EVERY_DOCS == 0:
                            logger.info(
                                "arxiv: %d/%d docs across %d pages (%d categories active)",
                                sink.count,
                                limit,
                                pages,
                                len([c for c in ordered if c not in exhausted]),
                            )

        logger.info(
            "arxiv finished: docs=%d pages=%d short_skips=%d exhausted=%s",
            sink.count,
            pages,
            skipped_short,
            ",".join(sorted(exhausted)) or "none",
        )

    return run_collector(out_dir, "arxiv", limit=limit, body=body)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Collect arXiv papers into the corpus.")
    parser.add_argument("out_dir", type=Path, help="corpus output root")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--max-results", type=int, default=100)
    parser.add_argument(
        "--categories",
        default="cs.CL,cs.LG,cs.AI,cs.SE,math.OC,stat.ML",
        help="comma-separated arXiv category identifiers",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    collect(
        args.out_dir,
        limit=args.limit,
        max_results=args.max_results,
        categories=tuple(c.strip() for c in args.categories.split(",") if c.strip()),
    )


if __name__ == "__main__":
    main()
