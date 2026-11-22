# -*- coding: utf-8 -*-
"""Wave 6d: find cpp_best_practices (repo moved/renamed) via search API."""
import io
import json
import re
import tarfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

DOCS = Path(
    r"C:\Users\Sem52\Documents\Default Project\extra-code\data\corpus\docs_2026"
)
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

    # candidates: the classic name, jason turner's account, and search
    candidates = [
        "cpp-best-practices/cpp_best_practices",
        "lefticus/cpp_best_practices",
        "lefticus/cpp-best-practices",
    ]
    q = urllib.parse.quote("cpp best practices book in:name,description")
    data = fetch(f"https://api.github.com/search/repositories?q={q}&per_page=5",
                 timeout=60)
    if data:
        try:
            for item in json.loads(data.decode("utf-8")).get("items", []):
                candidates.append(item["full_name"])
        except json.JSONDecodeError:
            pass
    print("[cppbp] candidates:", candidates, flush=True)

    branch = None
    repo = None
    for cand in candidates:
        meta = fetch(f"https://api.github.com/repos/{cand}", timeout=60)
        if not meta:
            continue
        try:
            j = json.loads(meta.decode("utf-8"))
        except json.JSONDecodeError:
            continue
        if int(j.get("size", 0)) < 1000:  # kb; the book repo is several MB
            print(f"  skip tiny {cand} ({j.get('size')} KB)", flush=True)
            continue
        repo, branch = j["full_name"], j.get("default_branch")
        break
    if not repo:
        print("[cppbp] no candidate repo found", flush=True)
        return 1
    print(f"[cppbp] repo: {repo} branch: {branch}", flush=True)

    blob = fetch(
        f"https://codeload.github.com/{repo}/tar.gz/refs/heads/{branch}",
        timeout=600,
    ) or fetch(f"https://api.github.com/repos/{repo}/tarball", timeout=600)
    if not blob:
        print("[cppbp] tarball FAILED", flush=True)
        return 1

    saved = 0
    try:
        tf = tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz")
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
    print(f"[cppbp] saved {saved} md from {repo}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
