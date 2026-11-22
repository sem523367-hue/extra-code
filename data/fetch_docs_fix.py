# -*- coding: utf-8 -*-
"""Docs wave 3b: repair the 404s from fetch_code_langs.

- Python text docs: resolve the LIVE archive URL from the archives index
  instead of a hard-coded point release.
- MDN guide: pages moved to <slug>/index.md — try both shapes.
- Rust book: parse src/SUMMARY.md and fetch every real chapter file,
  replacing the stub chapter heads that landed last wave.
- dotnet docs: try folder/index.md shapes for the moved pages.
"""
import io
import re
import tarfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(r"C:\Users\Sem52\Documents\Default Project\extra-code")
DOCS_DIR = ROOT / "data" / "corpus" / "docs_2026"
DOCS_DIR.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "curl/8.0"}

PYDOC_KEEP = (
    "library/asyncio", "library/typing", "library/dataclasses", "library/pathlib",
    "library/enum", "library/argparse", "library/json", "library/re",
    "library/datetime", "library/itertools", "library/functools",
    "library/concurrent", "library/logging", "library/sqlite3",
    "library/urllib", "library/http", "library/socket", "library/threading",
    "library/multiprocessing", "library/collections", "tutorial",
)

MDN_TOPICS = [
    "introduction", "grammar_and_types", "control_flow_and_error_handling",
    "functions", "expressions_and_operators", "numbers_and_dates",
    "text_formatting", "regular_expressions", "indexed_collections",
    "keyed_collections", "working_with_objects", "details_of_the_object_model",
    "using_promises", "iterators_and_generators", "meta_programming", "modules",
]

DOTNET_CANDIDATES = [
    ("csharp_tour", [
        "https://raw.githubusercontent.com/dotnet/docs/main/docs/csharp/tour-of-csharp/index.md",
        "https://raw.githubusercontent.com/dotnet/docs/main/docs/csharp/tour-of-csharp/tour-of-csharp.md",
        "https://raw.githubusercontent.com/dotnet/docs/main/docs/csharp/tour-of-csharp/feature-tour.md",
    ]),
    ("csharp_pattern_matching", [
        "https://raw.githubusercontent.com/dotnet/docs/main/docs/csharp/fundamentals/pattern-matching/index.md",
        "https://raw.githubusercontent.com/dotnet/docs/main/docs/csharp/fundamentals/tutorials/pattern-matching.md",
    ]),
]


def fetch(url: str, timeout: int = 90) -> bytes | None:
    for attempt in (1, 2):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as exc:
            print(f"  fail ({attempt}/2) {url}: {exc}", flush=True)
            time.sleep(2 * attempt)
    return None


def save(name: str, data: bytes, min_len: int = 200) -> bool:
    text = data.decode("utf-8", errors="ignore")
    if len(text.strip()) < min_len:
        print(f"  skip tiny {name}: {len(text)}B", flush=True)
        return False
    (DOCS_DIR / f"doc_{name}.md").write_text(text, encoding="utf-8")
    print(f"[doc] {name}: {len(text)//1024} KB", flush=True)
    return True


def do_pydocs() -> None:
    if any(DOCS_DIR.glob("doc_py_tutorial*.txt")):
        print("[pydocs] already present", flush=True)
        return
    idx = fetch("https://docs.python.org/3/archives/", timeout=60)
    if not idx:
        print("[pydocs] index fetch failed", flush=True)
        return
    hits = re.findall(r"python-(3\.\d+\.\d+)-docs-text\.tar\.bz2", idx.decode("utf-8", "ignore"))
    if not hits:
        print("[pydocs] no archive link on index page", flush=True)
        return
    latest = sorted(hits)[-1]
    url = f"https://docs.python.org/3/archives/python-{latest}-docs-text.tar.bz2"
    print(f"[pydocs] resolving {url}", flush=True)
    data = fetch(url, timeout=600)
    if not data:
        return
    saved = 0
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:bz2") as tf:
        for member in tf:
            if not member.isfile() or not member.name.endswith(".txt"):
                continue
            low = member.name.lower()
            if not any(k in low for k in PYDOC_KEEP) or member.size > 400_000:
                continue
            fh = tf.extractfile(member)
            if fh is None:
                continue
            stem = Path(member.name).stem.replace("/", "_")
            (DOCS_DIR / f"doc_py_{stem}.txt").write_text(
                fh.read().decode("utf-8", errors="ignore"), encoding="utf-8"
            )
            saved += 1
    print(f"[pydocs] saved {saved} files (python {latest})", flush=True)


def do_mdn() -> None:
    for topic in MDN_TOPICS:
        out = DOCS_DIR / f"doc_js_mdn_{topic}.md"
        if out.exists():
            continue
        base = "https://raw.githubusercontent.com/mdn/content/main/files/en-us/web/javascript/guide"
        for cand in (f"{base}/{topic}.md", f"{base}/{topic}/index.md"):
            data = fetch(cand, timeout=60)
            if data and save(f"js_mdn_{topic}", data):
                break


def do_rust() -> None:
    summ = fetch("https://raw.githubusercontent.com/rust-lang/book/main/src/SUMMARY.md", timeout=60)
    if not summ:
        print("[rust] SUMMARY fetch failed", flush=True)
        return
    chapters = sorted(set(re.findall(r"\((ch[^)#]+\.md)\)", summ.decode("utf-8", "ignore"))))
    print(f"[rust] {len(chapters)} chapters in SUMMARY", flush=True)
    saved = 0
    for ch in chapters:
        stem = re.sub(r"[^a-z0-9]+", "_", ch.lower()).strip("_")
        out = DOCS_DIR / f"doc_rustbook_{stem}.md"
        if out.exists():
            saved += 1
            continue
        data = fetch(f"https://raw.githubusercontent.com/rust-lang/book/main/src/{ch}", timeout=60)
        if data and save(f"rustbook_{stem}", data, min_len=800):
            saved += 1
    print(f"[rust] saved {saved}/{len(chapters)}", flush=True)


def do_dotnet() -> None:
    for name, cands in DOTNET_CANDIDATES:
        out = DOCS_DIR / f"doc_{name}.md"
        if out.exists():
            continue
        for cand in cands:
            data = fetch(cand, timeout=60)
            if data and save(name, data):
                break


def main() -> None:
    t0 = time.time()
    do_pydocs()
    do_mdn()
    do_rust()
    do_dotnet()
    print(f"DONE elapsed={int(time.time()-t0)}s", flush=True)


if __name__ == "__main__":
    main()
