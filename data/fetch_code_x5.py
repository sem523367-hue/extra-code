# -*- coding: utf-8 -*-
"""Real runnable code x5: TheAlgorithms repositories as zip archives.

One download per language instead of one request per file: the GitHub
tree API answered 404 for guessed repo slugs and is capped at 60
anonymous calls per hour anyway. codeload serves the whole repository
as a zip, which is filtered locally.

Files land in data/corpus/real_code as .txt (the extension kb.py
indexes) with a two-line source header, capped at 40 KB, tests and
docs excluded. Selection is round-robin over top-level directories so
the cap does not fill with the alphabetically first topic only.
Resume-safe. Run from extra-code root: python -m data.fetch_code_x5
"""
from __future__ import annotations

import io
import re
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

OUT = Path("data/corpus/real_code")
OUT.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "extra-code-kb/1.0 (local research; contact: local)"}
MAX_BYTES = 40_000
MIN_BYTES = 200

# key -> (repo slug under TheAlgorithms, branch, file extension, file cap)
ARCHIVES = {
    "py": ("Python", "master", ".py", 1400),
    "cpp": ("C-Plus-Plus", "master", ".cpp", 700),
    "rust": ("Rust", "master", ".rs", 350),
    "go": ("Go", "master", ".go", 250),
    "java": ("Java", "master", ".java", 250),
}

_NOISE = re.compile(r"(^|/)(tests?|bench(mark)?s?|venv|\.github|docs|examples?)/", re.I)


def fetch_zip(repo: str, branch: str) -> bytes | None:
    """Whole-repository zip, trying the other default branch on a 404."""
    last: Exception | None = None
    for name in (branch, "main", "master"):
        url = f"https://codeload.github.com/TheAlgorithms/{repo}/zip/refs/heads/{name}"
        req = urllib.request.Request(url, headers=HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001
            last = exc
            if "404" in str(exc):
                continue
            wait = 10
            print(f"  {repo}: {exc}, retry in {wait}s", flush=True)
            time.sleep(wait)
    print(f"  {repo}: archive unavailable ({last})", flush=True)
    return None


def pick(entries: list[tuple[str, int]], ext: str, cap: int) -> list[str]:
    """Paths worth teaching: small enough to read, no tests or vendored code.

    One file per top-level directory per round, so breadth comes before
    depth and the cap is not exhausted by a single topic.
    """
    by_dir: dict[str, list[str]] = {}
    for path, size in sorted(entries):
        if not path.endswith(ext) or not (MIN_BYTES <= size <= MAX_BYTES):
            continue
        if _NOISE.search(path):
            continue
        by_dir.setdefault(path.split("/", 1)[0], []).append(path)
    out: list[str] = []
    while len(out) < cap:
        took = False
        for directory in sorted(by_dir):
            if by_dir[directory]:
                out.append(by_dir[directory].pop(0))
                took = True
                if len(out) >= cap:
                    break
        if not took:
            break
    return out


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    grand = 0
    for key, (repo, branch, ext, cap) in ARCHIVES.items():
        have = len(list(OUT.glob(f"ta_{key}_*")))
        if have >= cap:
            print(f"=== {repo}: {have} files already >= cap {cap}, skip", flush=True)
            continue
        print(f"=== {repo}: downloading archive", flush=True)
        blob = fetch_zip(repo, branch)
        if blob is None:
            continue
        new = 0
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            infos = [info for info in archive.infolist() if not info.is_dir()]
            root = infos[0].filename.split("/", 1)[0] if infos else ""
            entries = [(info.filename.split("/", 1)[1], info.file_size)
                       for info in infos if "/" in info.filename]
            paths = pick(entries, ext, cap)
            print(f"=== {repo}: {len(blob) / 1e6:.1f} MB archive, "
                  f"{len(entries)} files, {len(paths)} chosen", flush=True)
            for path in paths:
                out = OUT / f"ta_{key}_{path.replace('/', '__')}.txt"
                if out.exists() and out.stat().st_size > 100:
                    continue
                try:
                    data = archive.read(f"{root}/{path}")
                except KeyError:
                    continue
                if not (MIN_BYTES <= len(data) <= MAX_BYTES):
                    continue
                header = (f"source: TheAlgorithms/{repo} :: {path}\n"
                          "license: MIT\n\n").encode()
                out.write_bytes(header + data)
                new += 1
        grand += new
        print(f"=== {repo} done: +{new} (total {have + new})", flush=True)
        time.sleep(1)

    total = len(list(OUT.glob("ta_*")))
    print(f"TOTAL ta_* files: {total}, +{grand} this run", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
