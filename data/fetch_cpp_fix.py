# -*- coding: utf-8 -*-
"""Wave 6b: patch the two wave-6 gaps.

1. CppCoreGuidelines.md — the isocpp repo default branch is 'master',
   the wave-6 fetch used 'main' and got a silent 404. Try master first.
2. cpp_best_practices — codeload zip failed on both branches mid-wave;
   retry zip, then fall back to the api.github.com tarball redirect.
"""
import io
import re
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(r"C:\Users\Sem52\Documents\Default Project\extra-code")
DOCS = ROOT / "data" / "corpus" / "docs_2026"
DOCS.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "ExtraCodeKB/1.0 (personal local knowledge base; no mirroring)"}


def fetch(url: str, timeout: int = 300) -> bytes | None:
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


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower() or "doc"


def do_core_guidelines() -> None:
    out = DOCS / "cpp_core_guidelines.md"
    if out.exists() and out.stat().st_size > 100_000:
        print("[cpp] core guidelines already present", flush=True)
        return
    for branch in ("master", "main"):
        data = fetch(
            "https://raw.githubusercontent.com/isocpp/CppCoreGuidelines/"
            f"{branch}/CppCoreGuidelines.md",
            timeout=600,
        )
        if data and len(data) > 200_000:
            out.write_text(data.decode("utf-8", errors="ignore"),
                           encoding="utf-8")
            print(f"[cpp] core guidelines ({branch}): {len(data)//1024} KB",
                  flush=True)
            return
    print("[cpp] core guidelines FAILED on both branches", flush=True)


def do_cppbp() -> None:
    if list(DOCS.glob("cppbp_*.md")):
        print("[cppbp] already present", flush=True)
        return
    zf = None
    for branch in ("master", "main"):
        data = fetch(
            f"https://codeload.github.com/cpp-best-practices/cpp_best_practices/zip/refs/heads/{branch}",
            timeout=600,
        )
        if data:
            try:
                zf = zipfile.ZipFile(io.BytesIO(data))
                print(f"[cppbp] zip ok ({branch})", flush=True)
                break
            except zipfile.BadZipFile:
                print(f"[cppbp] zip corrupt ({branch})", flush=True)
                zf = None
    if zf is None:
        # tarball redirect from the API — different CDN path than codeload
        data = fetch(
            "https://api.github.com/repos/cpp-best-practices/cpp_best_practices/tarball",
            timeout=600,
        )
        print("[cppbp] tarball fallback not a zip — skip "
              "(would need tarfile)", flush=True)
        return
    saved = 0
    for n in zf.namelist():
        if not n.endswith(".md"):
            continue
        if ".github" in n.lower():
            continue
        try:
            text = zf.read(n).decode("utf-8", errors="ignore")
        except Exception:
            continue
        if len(text.strip()) < 300:
            continue
        stem = slug(Path(n).stem)
        out = DOCS / f"cppbp_{stem}.md"
        if out.exists():
            continue
        out.write_text(text, encoding="utf-8")
        saved += 1
    print(f"[cppbp] saved {saved} md", flush=True)


def main() -> None:
    t0 = time.time()
    do_core_guidelines()
    do_cppbp()
    print(f"DONE elapsed={int(time.time()-t0)}s", flush=True)


if __name__ == "__main__":
    main()
