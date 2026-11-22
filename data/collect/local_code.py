"""Local source-code collector.

Walks the projects already on this machine and emits one document per source
file, with the path in the metadata so the model can learn file structure.
Secrets are excluded by name and by a lightweight content check.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Iterable

from .base import Doc, Sink, env_token, run_collector

log = logging.getLogger(__name__)

SKIP_DIRS = {
    ".git", ".hg", ".svn", ".idea", ".vscode", ".vs", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "node_modules", "bower_components", "venv", ".venv",
    "env", ".env", "dist", "build", "target", "out", "bin", "obj", ".next", ".nuxt",
    ".turbo", ".cache", "site-packages", "vendor", "third_party", "Pods", ".gradle",
    ".dart_tool", "coverage", "htmlcov", ".terraform", "cmake-build-debug",
}

SKIP_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "Cargo.lock",
    "go.sum", "composer.lock", "Gemfile.lock",
}

SECRET_NAME_RE = re.compile(
    r"(^\.env|\.env$|\.pem$|\.key$|\.pfx$|\.p12$|id_rsa|id_ed25519|credentials|"
    r"secrets?\.(json|ya?ml|toml|ini)$|\.npmrc$|\.pypirc$|token)", re.IGNORECASE
)
SECRET_CONTENT_RE = re.compile(
    r"(BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|"
    r"AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,})"
)

LANG_BY_EXT = {
    ".py": "python", ".pyi": "python", ".rs": "rust", ".c": "c", ".h": "c",
    ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp",
    ".cu": "cuda", ".go": "go", ".java": "java", ".kt": "kotlin", ".kts": "kotlin",
    ".ts": "typescript", ".tsx": "typescript", ".js": "javascript", ".jsx": "javascript",
    ".mjs": "javascript", ".cjs": "javascript", ".cs": "csharp", ".rb": "ruby",
    ".php": "php", ".swift": "swift", ".m": "objective-c", ".mm": "objective-c++",
    ".lua": "lua", ".sh": "bash", ".ps1": "powershell", ".psm1": "powershell",
    ".sql": "sql", ".r": "r", ".scala": "scala", ".hs": "haskell", ".ex": "elixir",
    ".exs": "elixir", ".zig": "zig", ".nim": "nim", ".jl": "julia", ".dart": "dart",
    ".vue": "vue", ".svelte": "svelte", ".html": "html", ".css": "css", ".scss": "scss",
    ".md": "markdown", ".rst": "rst", ".toml": "toml", ".yaml": "yaml", ".yml": "yaml",
    ".json": "json", ".ini": "ini", ".cfg": "ini", ".cmake": "cmake", ".gradle": "groovy",
}

DEFAULT_ROOTS = (
    Path.home() / "Documents",
    Path.home() / "Desktop",
    Path.home() / "source",
    Path.home() / "projects",
    Path.home() / "dev",
    Path.home() / "code",
)


def _decode(raw: bytes) -> str | None:
    if b"\x00" in raw[:4096]:
        return None
    for encoding in ("utf-8", "utf-8-sig", "cp1251"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def _looks_like_code(text: str) -> bool:
    """Reject prose dumps and minified one-liners without syntax."""
    lines = text.splitlines()
    if len(lines) < 3:
        return False
    longest = max((len(line) for line in lines), default=0)
    if longest > 5000:
        return False
    alnum = sum(ch.isalnum() for ch in text[:20_000])
    symbols = sum(ch in "{}[]();=<>+-*/#:.,\"'" for ch in text[:20_000])
    return alnum > 0 and (symbols / max(1, alnum + symbols)) > 0.05


def _iter_files(roots: Iterable[Path], extensions: set[str], max_bytes: int):
    seen_roots: set[Path] = set()
    for root in roots:
        try:
            resolved = root.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if resolved in seen_roots or not resolved.is_dir():
            continue
        seen_roots.add(resolved)
        for dirpath, dirnames, filenames in os.walk(resolved, followlinks=False):
            dirnames[:] = [
                d for d in dirnames
                if d.lower() not in SKIP_DIRS and not d.startswith(".") and d != "node_modules"
            ]
            for name in filenames:
                path = Path(dirpath) / name
                if name in SKIP_FILES or SECRET_NAME_RE.search(name):
                    continue
                suffix = path.suffix.lower()
                if suffix not in extensions:
                    continue
                try:
                    if path.stat().st_size > max_bytes:
                        continue
                except OSError:
                    continue
                yield path


def collect(
    out_dir: Path,
    *,
    limit: int = 20_000,
    roots: tuple[Path, ...] | None = None,
    extensions: tuple[str, ...] = tuple(LANG_BY_EXT),
    max_file_bytes: int = 512 * 1024,
) -> dict[str, object]:
    configured = env_token("LOCAL_CODE_ROOTS")
    if roots is None and configured:
        roots = tuple(Path(p) for p in configured.split(os.pathsep) if p.strip())
    search_roots = roots or DEFAULT_ROOTS
    allowed = {e if e.startswith(".") else f".{e}" for e in extensions}
    stats: dict[str, int] = {}

    def body(sink: Sink) -> None:
        for path in _iter_files(search_roots, allowed, max_file_bytes):
            if sink.count >= limit:
                return
            try:
                raw = path.read_bytes()
            except OSError:
                continue
            text = _decode(raw)
            if text is None or not _looks_like_code(text):
                continue
            if SECRET_CONTENT_RE.search(text):
                continue
            lang = LANG_BY_EXT.get(path.suffix.lower(), "und")
            doc = Doc(
                text=text,
                source="local_code",
                url=path.as_uri(),
                lang=lang,
                meta={"path": str(path), "bytes": len(raw)},
            )
            if sink.add(doc):
                stats[lang] = stats.get(lang, 0) + 1
                if sink.count % 100 == 0:
                    log.info("local_code: %d files (%s)", sink.count, path)

    manifest = run_collector(out_dir, "local_code", limit=limit, body=body)
    manifest["by_language"] = stats
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect source code from local projects")
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--limit", type=int, default=20_000)
    parser.add_argument("--roots", nargs="*", default=None)
    args = parser.parse_args()
    collect(
        Path(args.out),
        limit=args.limit,
        roots=tuple(Path(p) for p in args.roots) if args.roots else None,
    )
