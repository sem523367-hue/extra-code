"""GitHub code collector: search repos, download tarballs, emit source files.

Only repositories with a recognised permissive/copyleft license are ingested,
and the license is recorded per document. Set GITHUB_TOKEN to lift the search
rate limit from 10 req/min to 30 req/min.
"""
from __future__ import annotations

import io
import logging
import tarfile
from pathlib import Path

from .base import Doc, RateLimiter, Sink, env_token, http_get, http_json, make_client, run_collector

log = logging.getLogger(__name__)

SEARCH = "https://api.github.com/search/repositories"
LICENSES = {
    "mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc", "mpl-2.0",
    "unlicense", "0bsd", "lgpl-3.0", "gpl-3.0", "agpl-3.0", "cc0-1.0",
}
QUERIES = (
    "language:python stars:>1000",
    "language:rust stars:>800",
    "language:cpp stars:>800",
    "language:typescript stars:>1000",
    "language:go stars:>800",
    "language:java stars:>1000",
    "language:c stars:>500",
    "language:javascript stars:>1500",
)
SKIP_PARTS = {
    "node_modules", "vendor", "dist", "build", "third_party", "third-party",
    "site-packages", "__pycache__", ".git", "target", "assets", "static",
}
LANG_BY_EXT = {
    ".py": "python", ".pyi": "python", ".rs": "rust", ".c": "c", ".h": "c",
    ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp",
    ".cu": "cuda", ".go": "go", ".java": "java", ".kt": "kotlin", ".ts": "typescript",
    ".tsx": "typescript", ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
    ".cs": "csharp", ".rb": "ruby", ".php": "php", ".swift": "swift", ".lua": "lua",
    ".sh": "bash", ".ps1": "powershell", ".sql": "sql", ".r": "r", ".scala": "scala",
    ".hs": "haskell", ".ex": "elixir", ".exs": "elixir", ".zig": "zig", ".nim": "nim",
    ".jl": "julia", ".dart": "dart", ".md": "markdown", ".rst": "rst", ".toml": "toml",
    ".yaml": "yaml", ".yml": "yaml", ".json": "json",
}


def _decode(raw: bytes) -> str | None:
    if b"\x00" in raw[:4096]:
        return None
    for encoding in ("utf-8", "utf-8-sig", "cp1251"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def _keep(path: str, size: int, max_file_bytes: int) -> bool:
    parts = path.lower().split("/")
    if any(part in SKIP_PARTS for part in parts):
        return False
    if ".min." in path or path.endswith(".min.js"):
        return False
    suffix = Path(path).suffix.lower()
    return suffix in LANG_BY_EXT and size <= max_file_bytes and size > 0


def collect(
    out_dir: Path,
    *,
    limit: int = 5000,
    queries: tuple[str, ...] = QUERIES,
    max_repos: int = 120,
    max_files_per_repo: int = 200,
    max_file_bytes: int = 262_144,
) -> dict[str, object]:
    token = env_token("GITHUB_TOKEN", "GH_TOKEN")
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    search_limiter = RateLimiter(0.12 if not token else 0.4, burst=1)
    repos_seen = 0

    def body(sink: Sink) -> None:
        nonlocal repos_seen
        with make_client(timeout=60.0, headers=headers) as client:
            for query in queries:
                for page in range(1, 4):
                    if repos_seen >= max_repos or sink.count >= limit:
                        return
                    try:
                        payload = http_json(
                            client,
                            SEARCH,
                            params={"q": query, "sort": "stars", "order": "desc", "per_page": 100, "page": page},
                            limiter=search_limiter,
                            retries=3,
                            max_bytes=8 * 1024 * 1024,
                        )
                    except Exception as exc:  # noqa: BLE001
                        log.warning("github search failed (%s page %d): %s", query, page, exc)
                        break
                    items = payload.get("items") or []
                    if not items:
                        break
                    for repo in items:
                        if repos_seen >= max_repos or sink.count >= limit:
                            return
                        license_info = (repo.get("license") or {}).get("spdx_id") or ""
                        if license_info.lower() not in LICENSES:
                            continue
                        repos_seen += 1
                        full_name = repo.get("full_name")
                        branch = repo.get("default_branch") or "main"
                        if not full_name:
                            continue
                        log.info("github: %s (%s, %s stars)", full_name, license_info, repo.get("stargazers_count"))
                        try:
                            blob = http_get(
                                client,
                                f"https://codeload.github.com/{full_name}/tar.gz/refs/heads/{branch}",
                                retries=2,
                                max_bytes=64 * 1024 * 1024,
                            )
                        except Exception as exc:  # noqa: BLE001
                            log.warning("github tarball failed %s: %s", full_name, exc)
                            continue
                        files = 0
                        total_docs = 0
                        try:
                            with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
                                for member in tar:
                                    if files >= max_files_per_repo or sink.count >= limit:
                                        break
                                    if not member.isfile() or not _keep(member.name, member.size, max_file_bytes):
                                        continue
                                    handle = tar.extractfile(member)
                                    if handle is None:
                                        continue
                                    text = _decode(handle.read(max_file_bytes + 1))
                                    if text is None or len(text) < 200:
                                        continue
                                    rel = "/".join(member.name.split("/")[1:])
                                    files += 1
                                    doc = Doc(
                                        text=text,
                                        source="github",
                                        url=f"https://github.com/{full_name}/blob/{branch}/{rel}",
                                        lang=LANG_BY_EXT.get(Path(rel).suffix.lower(), "und"),
                                        meta={
                                            "repo": full_name,
                                            "path": rel,
                                            "license": license_info,
                                            "stars": repo.get("stargazers_count"),
                                            "branch": branch,
                                        },
                                    )
                                    if sink.add(doc):
                                        total_docs += 1
                        except tarfile.TarError as exc:
                            log.warning("github tar error %s: %s", full_name, exc)
                        log.info("github: %s -> %d docs", full_name, total_docs)

    return run_collector(out_dir, "github", limit=limit, body=body)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect permissively licensed GitHub code")
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--max-repos", type=int, default=120)
    args = parser.parse_args()
    collect(Path(args.out), limit=args.limit, max_repos=args.max_repos)
