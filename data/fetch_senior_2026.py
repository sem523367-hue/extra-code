# -*- coding: utf-8 -*-
"""Wave 4: 2025-2026 world facts + senior-level Python and C++.

Sam wants the model to KNOW what happened in 2025-2026 (news, science,
AI, discoveries) and to code Python/C++ at senior level from real
projects. Three parts:

  1. Wikipedia year/science/AI/computing articles, EN+RU, action=raw,
     wikitext stripped -> data/corpus/facts_2026/*.md
  2. Gold-standard modern Python libs (FastAPI, httpx, rich, click,
     tqdm) source zips -> data/corpus/real_code_py/<lib>_N.txt
  3. Gold-standard modern C++ libs (fmt, nlohmann/json, spdlog, asio)
     source zips -> data/corpus/real_code_cpp/<lib>_N.txt
"""
import io
import re
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(r"C:\Users\Sem52\Documents\Default Project\extra-code")
FACTS = ROOT / "data" / "corpus" / "facts_2026"
FACTS.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "curl/8.0"}
MAX_PER_REPO = 220
MAX_FILE_BYTES = 140_000

WIKI = [
    ("en", "2025"), ("en", "2026"),
    ("en", "2025_in_science"), ("en", "2026_in_science"),
    ("en", "2025_in_computing"), ("en", "2026_in_computing"),
    ("en", "2025_in_artificial_intelligence"),
    ("en", "2026_in_artificial_intelligence"),
    ("en", "Timeline_of_computing_2020s"),
    ("en", "2025_Nobel_Prize_laureates"),
    ("ru", "2025_год"), ("ru", "2026_год"),
    ("ru", "2025_год_в_науке"),
]

PY_REPOS = [
    ("fastapi", "fastapi/fastapi"),
    ("httpx", "encode/httpx"),
    ("rich", "Textualize/rich"),
    ("click", "pallets/click"),
    ("tqdm", "tqdm/tqdm"),
    ("attrs", "python-attrs/attrs"),
]

CPP_REPOS = [
    ("fmt", "fmtlib/fmt"),
    ("nlohmann_json", "nlohmann/json"),
    ("spdlog", "gabime/spdlog"),
    ("asio", "chriskohlhoff/asio"),
    ("ctre", "hanickadot/compile-time-regular-expressions"),
]

PY_EXT = {".py", ".pyi"}
CPP_EXT = {".cpp", ".cc", ".h", ".hpp", ".hxx", ".cxx", ".inl"}


def fetch(url: str, timeout: int = 120) -> bytes | None:
    for attempt in (1, 2):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as exc:
            print(f"  fail ({attempt}/2) {url.split('/')[-1][:40]}: {exc}", flush=True)
            time.sleep(2 * attempt)
    return None


def strip_wikitext(wt: str) -> str:
    # keep headings, lists and prose; drop refs/templates/tables noise
    wt = re.sub(r"<ref[^>]*/>", "", wt)
    wt = re.sub(r"<ref[^>]*>.*?</ref>", "", wt, flags=re.S)
    wt = re.sub(r"\{\{[^{}]*\}\}", "", wt)            # innermost templates
    for _ in range(6):
        wt = re.sub(r"\{\{[^{}]*\}\}", "", wt)
    wt = re.sub(r"\{\|.*?\|\}", "", wt, flags=re.S)    # tables
    wt = re.sub(r"<![^>]*-->", "", wt)
    wt = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", wt)
    wt = re.sub(r"'''?", "", wt)
    wt = re.sub(r"<[^>]+>", "", wt)
    wt = re.sub(r"\n{3,}", "\n\n", wt)
    return wt.strip()


def do_wiki() -> int:
    saved = 0
    for lang, page in WIKI:
        out = FACTS / f"wiki_{lang}_{page}.md"
        if out.exists():
            saved += 1
            continue
        url = (
            f"https://{lang}.wikipedia.org/w/index.php?title="
            + urllib.parse.quote(page) + "&action=raw"
        )
        data = fetch(url)
        if not data:
            continue
        text = strip_wikitext(data.decode("utf-8", errors="ignore"))
        if len(text) < 500:
            continue
        out.write_text(text, encoding="utf-8")
        saved += 1
        print(f"[wiki] {lang}:{page}: {len(text)//1024} KB", flush=True)
    return saved


def do_repo(lang_dir: str, repos: list, exts: set) -> int:
    out_dir = ROOT / "data" / "corpus" / f"real_code_{lang_dir}"
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for tag, repo in repos:
        data = fetch(f"https://codeload.github.com/{repo}/zip/refs/heads/master", 400) \
            or fetch(f"https://codeload.github.com/{repo}/zip/refs/heads/main", 400)
        if not data:
            print(f"[{tag}] ZIP FAILED", flush=True)
            continue
        saved = 0
        try:
            zf = zipfile.ZipFile(io.BytesIO(data))
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
            print(f"[{tag}] zip corrupt: {exc}", flush=True)
        total += saved
        print(f"[{tag}] saved {saved}", flush=True)
    return total


def main() -> int:
    t0 = time.time()
    w = do_wiki()
    p = do_repo("py", PY_REPOS, PY_EXT)
    c = do_repo("cpp", CPP_REPOS, CPP_EXT)
    print(f"DONE wiki={w} py_files={p} cpp_files={c} elapsed={int(time.time()-t0)}s",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
