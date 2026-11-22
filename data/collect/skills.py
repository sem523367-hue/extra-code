"""Collect agent skills, command definitions and instruction files.

These are high-value tiny documents: they teach instruction-following shape,
tool-use schemas and Russian/English mixed technical prose. Sources are read
from the local machine only, nothing is uploaded.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from .base import Doc, Sink, env_token, run_collector

log = logging.getLogger(__name__)

TEXT_SUFFIXES = {".md", ".mdx", ".txt", ".yaml", ".yml", ".json", ".toml"}
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".cache"}
FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
NAME_RE = re.compile(r"^name:\s*(.+)$", re.MULTILINE)
DESC_RE = re.compile(r"^description:\s*(.+)$", re.MULTILINE)


def _decode(raw: bytes) -> str | None:
    if b"\x00" in raw[:2048]:
        return None
    for encoding in ("utf-8", "utf-8-sig", "cp1251"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def _default_roots() -> list[Path]:
    home = Path.home()
    roots: list[Path] = [
        home / ".claude" / "skills",
        home / ".claude" / "agents",
        home / ".claude" / "commands",
        home / ".claude" / "plugins",
        home / ".config" / "opencode",
        home / ".opencode",
        home / ".codex",
    ]
    for name in ("CLAUDE.md", "AGENTS.md", "claude.md"):
        roots.append(home / name)
    configured = env_token("SKILL_ROOTS")
    if configured:
        roots.extend(Path(p) for p in configured.split(os.pathsep) if p.strip())
    return roots


def _walk(root: Path, max_depth: int = 6):
    if root.is_file():
        yield root
        return
    if not root.is_dir():
        return
    base_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        if len(current.parts) - base_depth >= max_depth:
            dirnames[:] = []
        dirnames[:] = [d for d in dirnames if d.lower() not in SKIP_DIRS and not d.startswith(".git")]
        for name in filenames:
            path = current / name
            if path.suffix.lower() in TEXT_SUFFIXES and not name.startswith(".env"):
                yield path


def _kind(path: Path) -> str:
    lower = path.as_posix().lower()
    if "skill" in lower:
        return "skill"
    if "agent" in lower:
        return "agent"
    if "command" in lower:
        return "command"
    if path.name.lower() in {"agents.md", "claude.md"}:
        return "instructions"
    return "config"


def collect(
    out_dir: Path,
    *,
    limit: int = 5000,
    roots: tuple[Path, ...] | None = None,
    max_file_bytes: int = 256 * 1024,
) -> dict[str, object]:
    search_roots = roots or tuple(_default_roots())
    seen: set[Path] = set()

    def body(sink: Sink) -> None:
        for root in search_roots:
            for path in _walk(root):
                if sink.count >= limit:
                    return
                try:
                    resolved = path.resolve()
                    if resolved in seen:
                        continue
                    seen.add(resolved)
                    if resolved.stat().st_size > max_file_bytes:
                        continue
                    raw = resolved.read_bytes()
                except (OSError, RuntimeError):
                    continue
                text = _decode(raw)
                if text is None:
                    continue
                text = text.strip()
                if len(text) < 120:
                    continue
                front = FRONTMATTER_RE.match(text)
                name = NAME_RE.search(front.group(1)).group(1).strip() if front and NAME_RE.search(front.group(1)) else resolved.stem
                desc = DESC_RE.search(front.group(1)).group(1).strip() if front and DESC_RE.search(front.group(1)) else ""
                body_text = f"# {name}\n\n{text}" if not text.startswith("#") else text
                doc = Doc(
                    text=body_text,
                    source="skills",
                    url=resolved.as_uri(),
                    lang="ru" if _cyrillic_ratio(text) > 0.15 else "en",
                    meta={"path": str(resolved), "kind": _kind(resolved), "name": name, "description": desc},
                )
                if sink.add(doc) and sink.count % 200 == 0:
                    log.info("skills: %d files", sink.count)

    return run_collector(out_dir, "skills", limit=limit, body=body)


def _cyrillic_ratio(text: str) -> float:
    letters = [ch for ch in text[:20_000] if ch.isalpha()]
    if not letters:
        return 0.0
    return sum("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in letters) / len(letters)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect local agent skills and instruction files")
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--roots", nargs="*", default=None)
    args = parser.parse_args()
    collect(Path(args.out), limit=args.limit, roots=tuple(Path(p) for p in args.roots) if args.roots else None)
