"""Project Gutenberg collector, sourced through the Gutendex JSON API.

Gutendex indexes the Gutenberg catalogue and exposes it as paginated JSON:
https://gutendex.com/books

One document per book. A plain-text rendition is preferred over HTML/epub,
the Gutenberg header/footer boilerplate is stripped, and short ephemera
(poetry pamphlets, single-page tracts) are dropped because they dilute
pretraining signal.

Public entry point::

    collect(out_dir, *, limit=500, languages=("ru", "en"),
            max_book_bytes=5 * 1024 * 1024, subjects=()) -> dict[str, object]
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any

from .base import Doc, RateLimiter, Sink, http_get, http_json, make_client, run_collector

logger = logging.getLogger(__name__)

GUTENDEX_URL = "https://gutendex.com/books"
EBOOK_URL = "https://www.gutenberg.org/ebooks/{id}"

# Gutendex is a small volunteer-run index; half a request per second is polite
# and still moves ~1800 requests/hour.
REQUEST_RATE_PER_S = 0.5

# Below this a book is a poem, a tract, or a catalogue stub rather than prose.
MIN_BOOK_CHARS = 2_000

# Only bodies above this size get the blind head/tail trim used when the
# Gutenberg markers are missing; trimming a 3 kB essay would eat the essay.
SHORT_BODY_TRIM_THRESHOLD = 20_000
BODY_TRIM_FRACTION = 0.02

# Gutendex paginates ~32 books per page; this is a runaway guard only.
MAX_PAGES = 200

LOG_EVERY_DOCS = 50

# Ordered by fidelity: utf-8 first, then the ascii transcription, then anything
# textual. Keys are exactly the labels Gutendex emits under "formats".
FORMAT_PREFERENCE: tuple[str, ...] = (
    "text/plain; charset=utf-8",
    "text/plain; charset=us-ascii",
)

_START_RE = re.compile(
    r"^\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*$",
    re.MULTILINE | re.IGNORECASE,
)
_END_RE = re.compile(
    r"^\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*$",
    re.MULTILINE | re.IGNORECASE,
)


def strip_gutenberg_boilerplate(text: str) -> str:
    """Cut the licence header and footer from a Gutenberg plain-text file.

    Returns the body between the START and END markers. When either marker is
    missing the body is kept whole, minus a 2% head/tail trim for long files.
    """
    start = _START_RE.search(text)
    end = _END_RE.search(text)
    if start is not None and end is not None and end.start() > start.end():
        return text[start.end() : end.start()]
    if len(text) <= SHORT_BODY_TRIM_THRESHOLD:
        return text
    edge = int(len(text) * BODY_TRIM_FRACTION)
    if edge <= 0 or edge * 2 >= len(text):
        return text
    return text[edge : len(text) - edge]


def pick_text_url(formats: Any) -> str | None:
    """Choose a plain-text download URL out of a Gutendex ``formats`` map."""
    if not isinstance(formats, dict):
        return None
    for key in FORMAT_PREFERENCE:
        url = formats.get(key)
        if isinstance(url, str) and url:
            return url
    for key, url in formats.items():
        if isinstance(key, str) and key.startswith("text/plain") and isinstance(url, str) and url:
            return url
    return None


def subject_match(book_subjects: list[str], needles: tuple[str, ...]) -> bool:
    """Case-insensitive substring match against any requested subject."""
    if not needles:
        return True
    folded = [subject.casefold() for subject in book_subjects]
    return any(needle.casefold() in subject for needle in needles for subject in folded)


def collect(
    out_dir: Path,
    *,
    limit: int = 500,
    languages: tuple[str, ...] = ("ru", "en"),
    max_book_bytes: int = 5 * 1024 * 1024,
    subjects: tuple[str, ...] = (),
) -> dict[str, object]:
    """Collect up to ``limit`` Gutenberg books into ``out_dir``.

    Args:
        out_dir: manifest/raw root handed to :func:`run_collector`.
        limit: maximum number of documents written.
        languages: Gutendex language codes, comma-joined into the query.
        max_book_bytes: per-book download cap; longer renditions are skipped.
        subjects: optional case-insensitive substrings; empty keeps everything.

    Returns:
        The manifest dict produced by :func:`run_collector`.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")
    if max_book_bytes <= 0:
        raise ValueError("max_book_bytes must be positive")
    if not languages:
        raise ValueError("languages must contain at least one code")

    def body(sink: Sink) -> None:
        limiter = RateLimiter(REQUEST_RATE_PER_S, burst=1)
        params: dict[str, Any] | None = {
            "languages": ",".join(languages),
            "copyright": "false",
            "sort": "popular",
        }
        url: str | None = GUTENDEX_URL
        pages = 0
        scanned = 0
        skipped_subject = 0
        skipped_format = 0
        skipped_short = 0
        failed_downloads = 0

        with make_client() as client:
            while url and sink.count < limit and pages < MAX_PAGES:
                payload = http_json(client, url, params=params, limiter=limiter)
                params = None  # the `next` link already carries its query string
                pages += 1

                if not isinstance(payload, dict):
                    logger.warning("unexpected Gutendex payload on page %d: %s", pages, type(payload).__name__)
                    break

                results = payload.get("results")
                if not isinstance(results, list) or not results:
                    logger.info("Gutendex page %d returned no results, stopping", pages)
                    break

                for book in results:
                    if sink.count >= limit:
                        break
                    if not isinstance(book, dict):
                        continue
                    scanned += 1

                    book_subjects = [str(s) for s in (book.get("subjects") or []) if s]
                    if not subject_match(book_subjects, subjects):
                        skipped_subject += 1
                        continue

                    book_id = book.get("id")
                    if not isinstance(book_id, int):
                        logger.debug("skipping Gutendex record without an integer id: %r", book.get("title"))
                        continue

                    text_url = pick_text_url(book.get("formats"))
                    if text_url is None:
                        skipped_format += 1
                        continue

                    try:
                        raw = http_get(client, text_url, limiter=limiter, max_bytes=max_book_bytes)
                    except (RuntimeError, ValueError) as exc:
                        # RuntimeError covers exhausted retries; ValueError covers
                        # the byte ceiling. Neither is fatal to the run.
                        failed_downloads += 1
                        logger.warning("book download failed (id=%s): %s", book.get("id"), exc)
                        continue

                    text = strip_gutenberg_boilerplate(raw.decode("utf-8", errors="replace"))
                    if len(text) < MIN_BOOK_CHARS:
                        skipped_short += 1
                        continue

                    book_languages = [str(code) for code in (book.get("languages") or []) if code]
                    lang = book_languages[0] if book_languages else "und"
                    authors = [
                        str(author.get("name"))
                        for author in (book.get("authors") or [])
                        if isinstance(author, dict) and author.get("name")
                    ]

                    doc = Doc(
                        text=text,
                        source="books",
                        url=EBOOK_URL.format(id=book_id),
                        lang=lang,
                        meta={
                            "title": str(book.get("title") or ""),
                            "authors": authors,
                            "gutenberg_id": book_id,
                            "subjects": book_subjects,
                            "download_count": int(book.get("download_count") or 0),
                        },
                    )
                    sink.add(doc)
                    if sink.count and sink.count % LOG_EVERY_DOCS == 0:
                        logger.info(
                            "books: %d/%d docs after %d pages (scanned=%d skipped=%d)",
                            sink.count,
                            limit,
                            pages,
                            scanned,
                            skipped_subject + skipped_format + skipped_short + failed_downloads,
                        )

                url = payload.get("next") if isinstance(payload.get("next"), str) else None

        logger.info(
            "books finished: docs=%d pages=%d scanned=%d subject_skips=%d format_skips=%d short_skips=%d download_failures=%d",
            sink.count,
            pages,
            scanned,
            skipped_subject,
            skipped_format,
            skipped_short,
            failed_downloads,
        )

    return run_collector(out_dir, "books", limit=limit, body=body)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Collect Project Gutenberg books into the corpus.")
    parser.add_argument("out_dir", type=Path, help="corpus output root")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--languages", default="ru,en", help="comma-separated Gutendex language codes")
    parser.add_argument("--max-book-bytes", type=int, default=5 * 1024 * 1024)
    parser.add_argument("--subjects", default="", help="comma-separated subject substrings")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    collect(
        args.out_dir,
        limit=args.limit,
        languages=tuple(code.strip() for code in args.languages.split(",") if code.strip()),
        max_book_bytes=args.max_book_bytes,
        subjects=tuple(s.strip() for s in args.subjects.split(",") if s.strip()),
    )


if __name__ == "__main__":
    main()
