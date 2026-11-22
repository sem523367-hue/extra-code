# -*- coding: utf-8 -*-
"""Python stdlib docs straight from the CPython repo (raw .rst).

The docs.python.org text-archive index no longer exposes a plain link we
can regex, so pull the canonical module docs from cpython/main instead.
Saved as .md under data/corpus/docs_2026 so kb.py picks them up.
"""
import time
import urllib.request
from pathlib import Path

DOCS_DIR = Path(
    r"C:\Users\Sem52\Documents\Default Project\extra-code\data\corpus\docs_2026"
)
DOCS_DIR.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "curl/8.0"}
BASE = "https://raw.githubusercontent.com/python/cpython/main/Doc"

FILES = [
    "library/asyncio.rst", "library/asyncio-task.rst", "library/asyncio-stream.rst",
    "library/typing.rst", "library/dataclasses.rst", "library/pathlib.rst",
    "library/argparse.rst", "library/json.rst", "library/re.rst",
    "library/datetime.rst", "library/itertools.rst", "library/functools.rst",
    "library/logging.rst", "library/logging.handlers.rst",
    "library/sqlite3.rst", "library/collections.rst", "library/enum.rst",
    "library/socket.rst", "library/threading.rst", "library/multiprocessing.rst",
    "library/urllib.request.rst", "library/http.client.rst",
    "library/concurrent.futures.rst", "library/contextlib.rst",
    "library/abc.rst", "library/struct.rst", "library/hashlib.rst",
    "tutorial/introduction.rst", "tutorial/controlflow.rst",
    "tutorial/datastructures.rst", "tutorial/modules.rst",
    "tutorial/classes.rst", "tutorial/errors.rst", "tutorial/stdlib.rst",
    "tutorial/venv.rst",
]


def fetch(url: str, timeout: int = 60) -> bytes | None:
    for attempt in (1, 2):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as exc:
            print(f"  fail ({attempt}/2) {url.rsplit('/', 1)[-1]}: {exc}", flush=True)
            time.sleep(2 * attempt)
    return None


def main() -> int:
    saved = skipped = 0
    for rel in FILES:
        stem = rel.replace("/", "_").rsplit(".", 1)[0]
        out = DOCS_DIR / f"doc_cpy_{stem}.md"
        if out.exists():
            skipped += 1
            continue
        data = fetch(f"{BASE}/{rel}")
        if not data:
            continue
        text = data.decode("utf-8", errors="ignore")
        if len(text.strip()) < 300:
            continue
        # rst directives are noise for the term index; keep the prose/code
        out.write_text(text, encoding="utf-8")
        saved += 1
        print(f"[cpy] {stem}: {len(text)//1024} KB", flush=True)
    print(f"DONE saved={saved} skipped={skipped}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
