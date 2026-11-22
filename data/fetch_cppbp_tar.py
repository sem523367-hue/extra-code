# -*- coding: utf-8 -*-
"""Wave 6c: cpp_best_practices via the API tarball.

Codeload 404s on both master/main — the default branch has another name
(or the repo moved). The api.github.com tarball redirect always resolves
the default branch, no guessing needed. gzip tar in, md files out.
"""
import io
import json
import re
import tarfile
import time
import urllib.request
from pathlib import Path

DOCS = Path(
    r"C:\Users\Sem52\Documents\Default Project\extra-code\data\corpus\docs_2026"
)
UA = {"User-Agent": "ExtraCodeKB/1.0 (personal local knowledge base; no mirroring)"}
REPO = "cpp-best-practices/cpp_best_practices"


def fetch(url: str, timeout: int = 300) -> bytes | None:
    for attempt in range(1, 4):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as exc:
            code = getattr(exc, "code", None)
            if code == 404:
                print(f"  404 {url[:90]}", flush=True)
                return None
            print(f"  fail ({attempt}/3) {exc}", flush=True)
            time.sleep(3 * attempt)
    return None


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower() or "doc"


def main() -> int:
    if list(DOCS.glob("cppbp_*.md")):
        print("[cppbp] already present", flush=True)
        return 0

    # resolve the real default branch first (diagnosis + right codeload url)
    meta = fetch(f"https://api.github.com/repos/{REPO}", timeout=60)
    branch = None
    if meta:
        try:
            branch = json.loads(meta.decode("utf-8")).get("default_branch")
        except json.JSONDecodeError:
            pass
    print(f"[cppbp] default branch: {branch}", flush=True)

    data = None
    if branch:
        data = fetch(
            f"https://codeload.github.com/{REPO}/tar.gz/refs/heads/{branch}",
            timeout=600,
        )
    if not data:
        data = fetch(f"https://api.github.com/repos/{REPO}/tarball", timeout=600)
    if not data:
        print("[cppbp] FAILED — no tarball either", flush=True)
        return 1

    saved = 0
    try:
        tf = tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")
        for member in tf.getmembers():
            if not member.isfile() or not member.name.endswith(".md"):
                continue
            if ".github" in member.name.lower():
                continue
            fh = tf.extractfile(member)
            if fh is None:
                continue
            text = fh.read().decode("utf-8", errors="ignore")
            if len(text.strip()) < 300:
                continue
            stem = slug(Path(member.name).stem)
            out = DOCS / f"cppbp_{stem}.md"
            if out.exists():
                continue
            out.write_text(text, encoding="utf-8")
            saved += 1
    except (tarfile.TarError, OSError) as exc:
        print(f"[cppbp] tar corrupt: {exc}", flush=True)
        return 1
    print(f"[cppbp] saved {saved} md (total raw {len(data)//1024} KB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
