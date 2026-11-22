"""Web search + page fetch + snippet ranking for a small local model.

A 25M-parameter model cannot call tools: it has no room for the policy that
decides *when* to search, and no reliable way to emit a JSON tool call. So the
searching happens here, in Python, and the model only ever sees the answer-
shaped text that comes out. The retrieval does the reasoning; the model does the
phrasing.

    python -m server.web "кто изобрёл динамит"
    python -m server.web "GIL в Python" --pages 3 --max-chars 900 --json

Pieces:
    search(query)            -> list[Hit]   DuckDuckGo HTML + Wikipedia API
    fetch(url)               -> str         page text, tags stripped
    rank(query, chunks)      -> list[str]   term-overlap score, stable order
    build_context(query)     -> Context     search -> fetch -> rank -> join

Transport is urllib, not httpx, on purpose: DuckDuckGo and Wikimedia both serve
a 202/403 challenge to httpx's TLS fingerprint and a normal 200 to urllib from
the same machine. Verified on this box, not assumed.

Every network call is bounded: connect timeout, read timeout, byte ceiling,
and a hard cap on how many pages one query may open.
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import html
import io
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Iterable
from urllib.parse import parse_qs, unquote, urlparse

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)
DDG_HTML = "https://html.duckduckgo.com/html/"
WIKI_API = "https://{lang}.wikipedia.org/w/api.php"
MAX_PAGE_BYTES = 2_000_000
MAX_PAGE_CHARS = 200_000
TIMEOUT = 20.0

STOPWORDS = frozenset(    "а и но или что как это для при над под про чем кем где когда почему зачем "
    "the a an and or but of to in on for with is are was were be been what how "
    "why when where who which do does did can could should would".split()
)


@dataclass(slots=True)
class Hit:
    title: str
    url: str
    snippet: str = ""


@dataclass(slots=True)
class Context:
    query: str
    hits: list[Hit] = field(default_factory=list)
    text: str = ""
    sources: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "text": self.text,
            "sources": self.sources,
            "hits": [{"title": h.title, "url": h.url, "snippet": h.snippet} for h in self.hits],
        }


# ---------------------------------------------------------------------------
# transport
# ---------------------------------------------------------------------------

def _open(url: str, *, timeout: float = TIMEOUT, max_bytes: int = MAX_PAGE_BYTES) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept-Language": "ru,en;q=0.8",
            "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
            "Accept-Encoding": "gzip",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            return b""
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = response.read(65_536)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total >= max_bytes:
                break
    raw = b"".join(chunks)
    if raw[:2] == b"\x1f\x8b":
        try:
            raw = gzip.GzipFile(fileobj=io.BytesIO(raw)).read()
        except OSError:
            pass
    return raw


def _get_text(url: str, *, timeout: float = TIMEOUT) -> str:
    try:
        return _open(url, timeout=timeout).decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError, ValueError):
        return ""


# ---------------------------------------------------------------------------
# HTML -> text
# ---------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    """Strip tags, drop script/style/nav, keep block boundaries as newlines."""

    SKIP = {"script", "style", "noscript", "svg", "head", "nav", "footer", "form"}
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP:
            self._skip_depth += 1
        elif tag in self.BLOCK and self._skip_depth == 0:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag in self.BLOCK and self._skip_depth == 0:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        raw = html.unescape(raw)
        raw = re.sub(r"[ \t\r\f\v]+", " ", raw)
        raw = re.sub(r"\n\s*\n\s*\n+", "\n\n", raw)
        return raw.strip()


def html_to_text(markup: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(markup)
        parser.close()
    except Exception:  # malformed markup is normal on the open web
        return re.sub(r"<[^>]+>", " ", markup)[:MAX_PAGE_CHARS]
    return parser.text()[:MAX_PAGE_CHARS]


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def _unwrap_ddg(href: str) -> str:
    """DuckDuckGo wraps results as /l/?uddg=<encoded>. Unwrap to the real URL."""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg")
        if target:
            return unquote(target[0])
    return href


_RESULT_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
    r'(?:<a[^>]+class="result__snippet"[^>]*>(.*?)</a>)?',
    re.DOTALL | re.IGNORECASE,
)


def search_duckduckgo(query: str, *, limit: int = 5) -> list[Hit]:
    body = _get_text(DDG_HTML + "?" + urllib.parse.urlencode({"q": query, "kl": "ru-ru"}))
    if not body:
        return []
    hits: list[Hit] = []
    seen: set[str] = set()
    for match in _RESULT_RE.finditer(body):
        url = _unwrap_ddg(html.unescape(match.group(1)))
        if not url.startswith("http") or url in seen:
            continue
        seen.add(url)
        hits.append(Hit(
            title=html_to_text(match.group(2) or "").replace("\n", " ").strip()[:200],
            url=url,
            snippet=html_to_text(match.group(3) or "").replace("\n", " ").strip()[:400],
        ))
        if len(hits) >= limit:
            break
    return hits


def search_wikipedia(query: str, *, limit: int = 3, lang: str = "ru") -> list[Hit]:
    """Wikipedia's own API. Reliable, gives a real extract, no scraping."""
    url = WIKI_API.format(lang=lang) + "?" + urllib.parse.urlencode(
        {"action": "query", "list": "search", "srsearch": query, "format": "json",
         "srlimit": str(limit), "utf8": "1"}
    )
    body = _get_text(url)
    if not body:
        return []
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return []
    hits: list[Hit] = []
    for row in (payload.get("query") or {}).get("search") or []:
        title = str(row.get("title") or "")
        if not title:
            continue
        hits.append(Hit(
            title=title,
            url=f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
            snippet=html_to_text(str(row.get("snippet") or "")).replace("\n", " ").strip()[:400],
        ))
    return hits


def search(query: str, *, limit: int = 5) -> list[Hit]:
    """DuckDuckGo first, Wikipedia fills the tail. Never raises on network failure.

    DDG's HTML endpoint occasionally serves junk (bot-suspect pages about
    unrelated topics) — measured: "погоду в москве" came back with cartoon
    plot summaries. Hits whose title+url+snippet share no content term
    with the query are dropped before they can steer the answer.
    """
    if not query.strip():
        return []
    limit = max(1, min(limit, 10))
    terms = {t for t in _terms(query) if t not in STOPWORDS and len(t) > 2}
    hits: list[Hit] = []
    for hit in search_duckduckgo(query, limit=limit + 3):
        if terms:
            blob = f"{hit.title} {hit.url} {hit.snippet}".lower()
            if not any(t in blob for t in terms):
                continue
        hits.append(hit)
        if len(hits) >= limit:
            break
    if len(hits) < limit:
        have = {h.url for h in hits}
        for hit in search_wikipedia(query, limit=limit - len(hits)):
            if hit.url not in have:
                hits.append(hit)
    return hits[:limit]


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

def fetch(url: str, *, timeout: float = TIMEOUT, max_bytes: int = MAX_PAGE_BYTES) -> str:
    """Download one page and return readable text. Returns "" on anything odd."""
    if not url.startswith(("http://", "https://")):
        return ""
    try:
        raw = _open(url, timeout=timeout, max_bytes=max_bytes)
    except (urllib.error.URLError, OSError, ValueError):
        return ""
    if not raw:
        return ""
    return html_to_text(raw.decode("utf-8", errors="replace"))


# ---------------------------------------------------------------------------
# ranking
# ---------------------------------------------------------------------------

def _terms(text: str) -> set[str]:
    words = re.findall(r"[\w\u0400-\u04FF]+", text.lower())
    return {w for w in words if len(w) > 2 and w not in STOPWORDS}


def chunk(text: str, *, max_chars: int = 400) -> list[str]:
    """Sentence-aware chunks. Never splits a sentence it can keep whole."""
    sentences = re.split(r"(?<=[.!?…])\s+|\n{2,}", text)
    out: list[str] = []
    buffer = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            if buffer:
                out.append(buffer)
                buffer = ""
            out.extend(sentence[i:i + max_chars] for i in range(0, len(sentence), max_chars))
            continue
        if len(buffer) + len(sentence) + 1 > max_chars:
            out.append(buffer)
            buffer = sentence
        else:
            buffer = f"{buffer} {sentence}".strip()
    if buffer:
        out.append(buffer)
    return out


def rank(query: str, chunks: Iterable[str], *, top_k: int = 6) -> list[str]:
    """Term-overlap scoring with a length prior. Stable: ties keep input order."""
    wanted = _terms(query)
    material = list(chunks)
    if not wanted:
        return material[:top_k]
    scored: list[tuple[float, int, str]] = []
    for index, text in enumerate(material):
        if len(text) < 40:
            continue
        found = wanted & _terms(text)
        if not found:
            continue
        score = len(found) / len(wanted) + 0.15 * min(len(text), 400) / 400
        scored.append((-score, index, text))
    scored.sort()
    return [text for _, _, text in scored[:top_k]]


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

_YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")

# Leading imperatives people wrap questions in ("посмотри погоду в...").
# Sent as-is they steer DuckDuckGo toward command-word matches; stripped,
# the same question lands on the actual topic pages.
_COMMANDS = re.compile(
    r"^(?:посмотри|посмотреть|покажи|показать|скажи|сказать|узнай|узнать|"
    r"найди|найти|расскажи|рассказать|проверь|проверить|дай|дать|мне|"
    r"пожалуйста|please|show|tell|find|check|look\s+up)\b[\s,.!:;]*",
    re.IGNORECASE,
)


def search_query(question: str) -> str:
    """The question with leading command phrases removed, for search."""
    q = question.strip()
    for _ in range(4):
        stripped = _COMMANDS.sub("", q, count=1).lstrip(" \t,.!:;")
        if stripped == q:
            break
        q = stripped
    return q.strip() or question.strip()


def _fresh_enough(piece: str, min_year: int) -> bool:
    """False only when the fragment carries an explicit old year.

    Undated text is kept: no date is not proof of staleness, and dropping it
    would empty the answer for pages that simply do not print a year.
    """
    years = [int(y) for y in _YEAR.findall(piece)]
    if not years:
        return True
    return max(years) >= min_year


def build_context(
    query: str,
    *,
    results: int = 5,
    pages: int = 3,
    max_chars: int = 1200,
    per_page: int = 3,
    fresh: bool = False,
) -> Context:
    """Search, open the top pages, rank everything, join into one bounded block.

    `fresh=True` is for live-data questions: fragments whose newest explicit
    year is older than the current year are dropped instead of being served
    as today's numbers. Measured failure: a weather query answered "Сегодня
    ... 28 января 2022 года" from a stale page.
    """
    context = Context(query=query)
    context.hits = search(search_query(query), limit=results)
    if not context.hits:
        return context

    collected: list[str] = []
    for hit in context.hits[:pages]:
        text = fetch(hit.url)
        if not text:
            if hit.snippet:
                collected.append(hit.snippet)
            continue
        context.sources.append(hit.url)
        picked = rank(query, chunk(text), top_k=per_page)
        if picked:
            collected.extend(picked)
        elif hit.snippet:
            collected.append(hit.snippet)

    picked = rank(query, collected, top_k=8)
    if fresh:
        min_year = datetime.date.today().year
        kept = [p for p in picked if _fresh_enough(p, min_year)]
        # Every fragment undated or stale would leave nothing to answer
        # with; undated fragments are kept, so only a fully-stale set is
        # reported as empty.
        picked = kept
    body = ""
    for piece in picked:
        if len(body) + len(piece) + 2 > max_chars:
            break
        body = f"{body}\n\n{piece}".strip() if body else piece
    context.text = body[:max_chars]
    return context


def main(argv: list[str] | None = None) -> int:
    try:  # Windows console defaults to cp1251; page text is not cp1251.
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    parser = argparse.ArgumentParser(description="web search -> readable context")
    parser.add_argument("query")
    parser.add_argument("--results", type=int, default=5)
    parser.add_argument("--pages", type=int, default=3)
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    context = build_context(args.query, results=args.results, pages=args.pages,
                            max_chars=args.max_chars)
    if args.json:
        print(json.dumps(context.as_dict(), indent=2, ensure_ascii=False))
    else:
        for hit in context.hits:
            print(f"- {hit.title}\n  {hit.url}")
        print("\n--- context ---\n")
        print(context.text or "(nothing retrieved)")
    return 0 if (context.text or context.hits) else 1


if __name__ == "__main__":
    raise SystemExit(main())
