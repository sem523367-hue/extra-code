# -*- coding: utf-8 -*-
"""Wave 7: max out Python and C++ coding knowledge.

Python gold: pydantic, requests, flask, jinja, sqlalchemy, aiohttp, numpy.
C++ gold: Catch2, range-v3, abseil, entt, leveldb, bustub (cmu-db), benchmark.
Plus the isocpp FAQ (markdown Q&A straight from the source repo).

Output: data/corpus/real_code_py/, real_code_cpp/, docs_2026/ — kb.py
indexes all of them automatically.
"""
import io
import re
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(r"C:\Users\Sem52\Documents\Default Project\extra-code")
PY_DIR = ROOT / "data" / "corpus" / "real_code_py"
CPP_DIR = ROOT / "data" / "corpus" / "real_code_cpp"
DOCS = ROOT / "data" / "corpus" / "docs_2026"
for d in (PY_DIR, CPP_DIR, DOCS):
    d.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "ExtraCodeKB/1.0 (personal local knowledge base; no mirroring)"}
MAX_PER_REPO = 220
MAX_FILE_BYTES = 140_000

PY_REPOS = [
    ("pydantic", "pydantic/pydantic"),
    ("requests", "psf/requests"),
    ("flask", "pallets/flask"),
    ("jinja", "pallets/jinja"),
    ("sqlalchemy", "sqlalchemy/sqlalchemy"),
    ("aiohttp", "aio-libs/aiohttp"),
    ("numpy", "numpy/numpy"),
]

CPP_REPOS = [
    ("catch2", "catchorg/Catch2"),
    ("range3", "ericniebler/range-v3"),
    ("abseil", "abseil/abseil-cpp"),
    ("entt", "skypjack/entt"),
    ("leveldb", "google/leveldb"),
    ("bustub", "cmu-db/bustub"),
    ("gbench", "google/benchmark"),
]

PY_EXT = {".py", ".pyi"}
CPP_EXT = {".cpp", ".cc", ".h", ".hpp", ".hxx", ".cxx", ".inl"}


def fetch(url: str, timeout: int = 400) -> bytes | None:
    for attempt in range(1, 4):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as exc:
            code = getattr(exc, "code", None)
            if code == 404:
                return None
            wait = 20 if code == 429 else 3 * attempt
            print(f"  fail ({attempt}/3) {url.rsplit('/', 1)[-1][:44]}: "
                  f"{exc} -> wait {wait}s", flush=True)
            time.sleep(wait)
    return None


def repo_zip(repo: str) -> zipfile.ZipFile | None:
    for branch in ("master", "main"):
        data = fetch(
            f"https://codeload.github.com/{repo}/zip/refs/heads/{branch}"
        )
        if data:
            try:
                return zipfile.ZipFile(io.BytesIO(data))
            except zipfile.BadZipFile:
                print(f"[{repo}] zip corrupt on {branch}", flush=True)
    print(f"[{repo}] ZIP FAILED", flush=True)
    return None


def do_repo(tag: str, repo: str, out_dir: Path, exts: set) -> int:
    # skip repos already harvested (tag prefix on disk)
    if any(out_dir.glob(f"{tag}_*")):
        print(f"[{tag}] already present", flush=True)
        return 0
    zf = repo_zip(repo)
    if zf is None:
        return 0
    saved = 0
    try:
        names = [
            n for n in zf.namelist()
            if not n.endswith("/")
            and Path(n).suffix.lower() in exts
            and "/test" not in n.lower()
            and ".github" not in n
        ]
        for n in names:
            if saved >= MAX_PER_REPO:
                break
            info = zf.getinfo(n)
            if info.file_size > MAX_FILE_BYTES:
                continue
            try:
                text = zf.read(n).decode("utf-8", errors="ignore")
            except Exception:
                continue
            if len(text.strip()) < 150:
                continue
            saved += 1
            (out_dir / f"{tag}_{saved:04d}.txt").write_text(
                text, encoding="utf-8", errors="ignore"
            )
    except (zipfile.BadZipFile, OSError) as exc:
        print(f"[{tag}] zip read error: {exc}", flush=True)
    print(f"[{tag}] saved {saved}", flush=True)
    return saved


def do_cpp_faq() -> None:
    if any(DOCS.glob("cppfaq_*.md")):
        print("[faq] already present", flush=True)
        return
    zf = repo_zip("isocpp/faq")
    if zf is None:
        return
    saved = 0
    for n in zf.namelist():
        if not n.endswith(".md") or ".github" in n.lower():
            continue
        try:
            text = zf.read(n).decode("utf-8", errors="ignore")
        except Exception:
            continue
        if len(text.strip()) < 400:
            continue
        stem = re.sub(r"[^a-z0-9]+", "_", Path(n).stem.lower()).strip("_") or "f"
        out = DOCS / f"cppfaq_{stem}.md"
        if out.exists():
            continue
        out.write_text(text, encoding="utf-8")
        saved += 1
    print(f"[faq] saved {saved} md", flush=True)


def main() -> int:
    t0 = time.time()
    py_total = 0
    for tag, repo in PY_REPOS:
        py_total += do_repo(tag, repo, PY_DIR, PY_EXT)
    cpp_total = 0
    for tag, repo in CPP_REPOS:
        cpp_total += do_repo(tag, repo, CPP_DIR, CPP_EXT)
    do_cpp_faq()
    print(f"DONE py=+{py_total} cpp=+{cpp_total} elapsed={int(time.time()-t0)}s",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
