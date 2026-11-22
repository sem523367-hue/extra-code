# -*- coding: utf-8 -*-
"""Wave 7b: Catch2 + isocpp/faq — repos whose default branch is neither
master nor main (Catch2 lives on 'develop', the FAQ repo has its own
branch layout). Resolve via the API, then codeload zip / tarball."""
import io
import json
import re
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(r"C:\Users\Sem52\Documents\Default Project\extra-code")
CPP_DIR = ROOT / "data" / "corpus" / "real_code_cpp"
DOCS = ROOT / "data" / "corpus" / "docs_2026"
UA = {"User-Agent": "ExtraCodeKB/1.0 (personal local knowledge base; no mirroring)"}
MAX_FILES = 220
MAX_BYTES = 140_000


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
            print(f"  fail ({attempt}/3) {exc}", flush=True)
            time.sleep(3 * attempt)
    return None


def default_branch(repo: str) -> str | None:
    meta = fetch(f"https://api.github.com/repos/{repo}", timeout=60)
    if not meta:
        return None
    try:
        return json.loads(meta.decode("utf-8")).get("default_branch")
    except json.JSONDecodeError:
        return None


def harvest(repo: str, tag: str, out_dir: Path, exts: set, md_only: bool = False) -> None:
    if any(out_dir.glob(f"{tag}_*")):
        print(f"[{tag}] already present", flush=True)
        return
    branch = default_branch(repo)
    print(f"[{tag}] default branch: {branch}", flush=True)
    data = None
    for b in filter(None, (branch, "master", "main", "develop")):
        data = fetch(f"https://codeload.github.com/{repo}/zip/refs/heads/{b}")
        if data:
            try:
                zf = zipfile.ZipFile(io.BytesIO(data))
                print(f"[{tag}] zip ok on {b}", flush=True)
                break
            except zipfile.BadZipFile:
                data = None
                continue
    if data is None:
        print(f"[{tag}] FAILED all branches", flush=True)
        return
    saved = 0
    try:
        for n in zf.namelist():
            if saved >= MAX_FILES:
                break
            if n.endswith("/") or ".github" in n.lower():
                continue
            if md_only:
                if not n.endswith(".md"):
                    continue
            elif Path(n).suffix.lower() not in exts or "/test" in n.lower():
                continue
            info = zf.getinfo(n)
            if info.file_size > MAX_BYTES:
                continue
            try:
                text = zf.read(n).decode("utf-8", errors="ignore")
            except Exception:
                continue
            if len(text.strip()) < (400 if md_only else 150):
                continue
            stem = re.sub(r"[^a-z0-9]+", "_", Path(n).stem.lower()).strip("_") or "f"
            out = out_dir / f"{tag}_{saved + 1:04d}_{stem}.txt"
            if out.exists():
                continue
            out.write_text(text, encoding="utf-8", errors="ignore")
            saved += 1
    except (zipfile.BadZipFile, OSError) as exc:
        print(f"[{tag}] read error: {exc}", flush=True)
        return
    print(f"[{tag}] saved {saved}", flush=True)


def main() -> int:
    harvest("catchorg/Catch2", "catch2", CPP_DIR,
            {".cpp", ".cc", ".h", ".hpp", ".hxx", ".cxx", ".inl"})
    harvest("isocpp/faq", "cppfaq", DOCS, set(), md_only=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
