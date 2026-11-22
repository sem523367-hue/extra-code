# -*- coding: utf-8 -*-
"""Wave 3: code + modern docs for the remaining popular languages.

Sam asked for coding skill on ALL popular languages: how modern libraries
do things, their methods, their documentation. Wave 1 (fetch_x5) covered
theory; fetch_code_x5 covered py/cpp/rust/go/java source. This wave adds:

  - TheAlgorithms zips: Javascript, TypeScript, PHP, Ruby, C-Sharp, C,
    Swift  -> real idiomatic source per language
  - Full Python 3.13 text docs (asyncio, typing, dataclasses, pathlib...)
  - The Rust Book (all chapters, raw markdown)
  - React learn guides, Vue guide, MDN JavaScript guide chapters
  - C# tour + version history from the dotnet docs repo

Everything lands under data/corpus/ and is brand-scrubbed by kb.py at
index time. Files: ta_<lang>_*.txt (code), doc_<topic>.md/txt (docs).
"""
import io
import json
import os
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(r"C:\Users\Sem52\Documents\Default Project\extra-code")
CODE_DIR = ROOT / "data" / "corpus" / "real_code"
DOCS_DIR = ROOT / "data" / "corpus" / "docs_2026"
CODE_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "curl/8.0"}
MAX_PER_LANG = 450          # source files per language zip
MAX_FILE_BYTES = 120_000    # skip monster files
CODE_EXT = {".py", ".js", ".ts", ".tsx", ".jsx", ".php", ".rb", ".cs", ".c", ".h", ".swift"}

ZIPS = [
    ("javascript", "TheAlgorithms/Javascript"),
    ("typescript", "TheAlgorithms/TypeScript"),
    ("php", "TheAlgorithms/PHP"),
    ("ruby", "TheAlgorithms/Ruby"),
    ("csharp", "TheAlgorithms/C-Sharp"),
    ("c", "TheAlgorithms/C"),
    ("swift", "TheAlgorithms/Swift"),
]

# (out_name, url) — raw markdown from official doc repos, stable mains.
DOC_MD = [
    ("rust_ch00_foreword", "https://raw.githubusercontent.com/rust-lang/book/main/src/foreword.md"),
    ("rust_ch01", "https://raw.githubusercontent.com/rust-lang/book/main/src/ch01-00-getting-started.md"),
    ("rust_ch03", "https://raw.githubusercontent.com/rust-lang/book/main/src/ch03-00-common-programming-concepts.md"),
    ("rust_ch04", "https://raw.githubusercontent.com/rust-lang/book/main/src/ch04-00-understanding-ownership.md"),
    ("rust_ch06", "https://raw.githubusercontent.com/rust-lang/book/main/src/ch06-00-enums.md"),
    ("rust_ch10", "https://raw.githubusercontent.com/rust-lang/book/main/src/ch10-00-generics.md"),
    ("rust_ch13", "https://raw.githubusercontent.com/rust-lang/book/main/src/ch13-00-functional-features.md"),
    ("rust_ch15", "https://raw.githubusercontent.com/rust-lang/book/main/src/ch15-00-smart-pointers.md"),
    ("rust_ch16", "https://raw.githubusercontent.com/rust-lang/book/main/src/ch16-00-concurrency.md"),
    ("rust_ch17", "https://raw.githubusercontent.com/rust-lang/book/main/src/ch17-00-async-await.md"),
    ("rust_ch18", "https://raw.githubusercontent.com/rust-lang/book/main/src/ch18-00-oop.md"),
    ("rust_ch20", "https://raw.githubusercontent.com/rust-lang/book/main/src/ch20-00-final-project-a-web-server.md"),
    ("react_describing_the_ui", "https://raw.githubusercontent.com/reactjs/react.dev/main/src/content/learn/describing-the-ui.md"),
    ("react_managing_state", "https://raw.githubusercontent.com/reactjs/react.dev/main/src/content/learn/managing-state.md"),
    ("react_escape_hatches", "https://raw.githubusercontent.com/reactjs/react.dev/main/src/content/learn/escape-hatches.md"),
    ("react_think_in_react", "https://raw.githubusercontent.com/reactjs/react.dev/main/src/content/learn/thinking-in-react.md"),
    ("vue_introduction", "https://raw.githubusercontent.com/vuejs/docs/main/src/guide/introduction.md"),
    ("js_mdn_introduction", "https://raw.githubusercontent.com/mdn/content/main/files/en-us/web/javascript/guide/introduction.md"),
    ("js_mdn_grammar_types", "https://raw.githubusercontent.com/mdn/content/main/files/en-us/web/javascript/guide/grammar_and_types.md"),
    ("js_mdn_control_flow", "https://raw.githubusercontent.com/mdn/content/main/files/en-us/web/javascript/guide/control_flow_and_error_handling.md"),
    ("js_mdn_functions", "https://raw.githubusercontent.com/mdn/content/main/files/en-us/web/javascript/guide/functions.md"),
    ("js_mdn_expressions", "https://raw.githubusercontent.com/mdn/content/main/files/en-us/web/javascript/guide/expressions_and_operators.md"),
    ("js_mdn_numbers_dates", "https://raw.githubusercontent.com/mdn/content/main/files/en-us/web/javascript/guide/numbers_and_dates.md"),
    ("js_mdn_text_formatting", "https://raw.githubusercontent.com/mdn/content/main/files/en-us/web/javascript/guide/text_formatting.md"),
    ("js_mdn_regular_expressions", "https://raw.githubusercontent.com/mdn/content/main/files/en-us/web/javascript/guide/regular_expressions.md"),
    ("js_mdn_indexed_collections", "https://raw.githubusercontent.com/mdn/content/main/files/en-us/web/javascript/guide/indexed_collections.md"),
    ("js_mdn_keyed_collections", "https://raw.githubusercontent.com/mdn/content/main/files/en-us/web/javascript/guide/keyed_collections.md"),
    ("js_mdn_working_objects", "https://raw.githubusercontent.com/mdn/content/main/files/en-us/web/javascript/guide/working_with_objects.md"),
    ("js_mdn_details_model", "https://raw.githubusercontent.com/mdn/content/main/files/en-us/web/javascript/guide/details_of_the_object_model.md"),
    ("js_mdn_using_promises", "https://raw.githubusercontent.com/mdn/content/main/files/en-us/web/javascript/guide/using_promises.md"),
    ("js_mdn_iterators_generators", "https://raw.githubusercontent.com/mdn/content/main/files/en-us/web/javascript/guide/iterators_and_generators.md"),
    ("js_mdn_meta_programming", "https://raw.githubusercontent.com/mdn/content/main/files/en-us/web/javascript/guide/meta_programming.md"),
    ("js_mdn_modules", "https://raw.githubusercontent.com/mdn/content/main/files/en-us/web/javascript/guide/modules.md"),
    ("csharp_tour", "https://raw.githubusercontent.com/dotnet/docs/main/docs/csharp/tour-of-csharp/tour-of-csharp-features.md"),
    ("csharp_version_history", "https://raw.githubusercontent.com/dotnet/docs/main/docs/csharp/whats-new/csharp-version-history.md"),
    ("csharp_types_records", "https://raw.githubusercontent.com/dotnet/docs/main/docs/csharp/fundamentals/types/records.md"),
    ("csharp_pattern_matching", "https://raw.githubusercontent.com/dotnet/docs/main/docs/csharp/fundamentals/pattern-matching.md"),
    ("csharp_linq", "https://raw.githubusercontent.com/dotnet/docs/main/docs/csharp/linq/index.md"),
    ("csharp_async", "https://raw.githubusercontent.com/dotnet/docs/main/docs/csharp/asynchronous-programming/async-scenarios.md"),
]

PYDOCS_URL = "https://docs.python.org/3/archives/python-3.13.0-docs-text.tar.bz2"
PYDOC_KEEP = (
    "library/asyncio", "library/typing", "library/dataclasses", "library/pathlib",
    "library/enum", "library/argparse", "library/json", "library/re",
    "library/datetime", "library/itertools", "library/functools",
    "library/concurrent", "library/logging", "library/sqlite3",
    "library/urllib", "library/http", "library/socket", "library/threading",
    "library/multiprocessing", "library/collections", "tutorial",
)


def fetch(url: str, timeout: int = 90) -> bytes | None:
    for attempt in (1, 2):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as exc:
            print(f"  fetch fail ({attempt}/2) {url}: {exc}", flush=True)
            time.sleep(2 * attempt)
    return None


def save_code(lang: str, idx: int, text: str) -> None:
    out = CODE_DIR / f"ta_{lang}_{idx:04d}.txt"
    out.write_text(text, encoding="utf-8", errors="ignore")


def do_zip(lang: str, repo: str) -> int:
    url = f"https://codeload.github.com/{repo}/zip/refs/heads/master"
    data = fetch(url, timeout=300)
    if not data:
        url = f"https://codeload.github.com/{repo}/zip/refs/heads/main"
        data = fetch(url, timeout=300)
    if not data:
        print(f"[{lang}] ZIP FAILED", flush=True)
        return 0
    saved = 0
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
        names = [n for n in zf.namelist()
                 if not n.endswith("/") and Path(n).suffix.lower() in CODE_EXT
                 and ".github" not in n]
        # round-robin over top-level dirs for breadth, like wave code-x5
        by_dir: dict[str, list[str]] = {}
        for n in names:
            parts = n.split("/")
            key = parts[1] if len(parts) > 2 else "_root"
            by_dir.setdefault(key, []).append(n)
        ring = list(by_dir.keys())
        cursors = {k: 0 for k in ring}
        turned = True
        while saved < MAX_PER_LANG and turned:
            turned = False
            for k in ring:
                bucket = by_dir[k]
                c = cursors[k]
                if c >= len(bucket):
                    continue
                turned = True
                cursors[k] = c + 1
                n = bucket[c]
                info = zf.getinfo(n)
                if info.file_size > MAX_FILE_BYTES:
                    continue
                try:
                    text = zf.read(n).decode("utf-8", errors="ignore")
                except Exception:
                    continue
                if len(text.strip()) < 120:
                    continue
                saved += 1
                save_code(lang, saved, text)
    except (zipfile.BadZipFile, OSError) as exc:
        print(f"[{lang}] zip corrupt: {exc}", flush=True)
    print(f"[{lang}] saved {saved} files", flush=True)
    return saved


def do_docs() -> int:
    saved = 0
    for name, url in DOC_MD:
        if (DOCS_DIR / f"doc_{name}.md").exists():
            saved += 1
            continue
        data = fetch(url)
        if not data:
            continue
        text = data.decode("utf-8", errors="ignore")
        if len(text.strip()) < 200:
            continue
        (DOCS_DIR / f"doc_{name}.md").write_text(text, encoding="utf-8")
        saved += 1
        print(f"[doc] {name}: {len(text)//1024} KB", flush=True)
    return saved


def do_pydocs() -> int:
    out_dir = DOCS_DIR
    marker = out_dir / "doc_python_tutorial.txt"
    if marker.exists():
        return 0
    data = fetch(PYDOCS_URL, timeout=600)
    if not data:
        print("[pydocs] FAILED", flush=True)
        return 0
    saved = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:bz2") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                if not member.name.endswith(".txt"):
                    continue
                low = member.name.lower()
                if not any(k in low for k in PYDOC_KEEP):
                    continue
                if member.size > 400_000:
                    continue
                fh = tf.extractfile(member)
                if fh is None:
                    continue
                text = fh.read().decode("utf-8", errors="ignore")
                stem = Path(member.name).stem.replace("/", "_")
                (out_dir / f"doc_py_{stem}.txt").write_text(text, encoding="utf-8")
                saved += 1
    except (tarfile.TarError, OSError) as exc:
        print(f"[pydocs] tar corrupt: {exc}", flush=True)
    print(f"[pydocs] saved {saved} files", flush=True)
    return saved


def main() -> int:
    t0 = time.time()
    total = 0
    for lang, repo in ZIPS:
        total += do_zip(lang, repo)
    total += do_docs()
    total += do_pydocs()
    print(f"DONE files={total} elapsed={int(time.time()-t0)}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
