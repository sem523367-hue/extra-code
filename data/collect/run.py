"""One entry point for every collector.

    python -m data.collect.run --out data/raw --limit 2000 --sources all

Each source writes its own gzip-JSONL shard and manifest under data/raw, so a
crashed run never destroys earlier work.
"""
from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path

log = logging.getLogger("extra_code.collect")

SOURCES = {
    "wikipedia": "data.collect.wikipedia",
    "github": "data.collect.github",
    "stackexchange": "data.collect.stackexchange",
    "books": "data.collect.books",
    "arxiv": "data.collect.arxiv",
    "local_code": "data.collect.local_code",
    "skills": "data.collect.skills",
}


def run_source(name: str, out_dir: Path, limit: int, **kwargs: object) -> bool:
    module_name = SOURCES.get(name)
    if module_name is None:
        log.error("unknown source %r (known: %s)", name, ", ".join(sorted(SOURCES)))
        return False
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        log.error("collector %s is unavailable: %s", name, exc)
        return False
    try:
        manifest = module.collect(out_dir, limit=limit, **kwargs)
    except KeyboardInterrupt:
        raise
    except Exception:
        log.exception("collector %s failed", name)
        return False
    log.info(
        "%s: %s documents, file=%s",
        name,
        manifest.get("documents"),
        manifest.get("file"),
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect training corpora")
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--limit", type=int, default=1000, help="documents per source")
    parser.add_argument(
        "--sources",
        default="all",
        help="comma-separated: " + ", ".join(SOURCES) + ", or all",
    )
    parser.add_argument("--langs", default="ru,en", help="wikipedia languages")
    parser.add_argument("--roots", nargs="*", default=None, help="local_code roots")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    requested = list(SOURCES) if args.sources == "all" else [s.strip() for s in args.sources.split(",") if s.strip()]
    out_dir = Path(args.out)
    failed: list[str] = []
    for name in requested:
        extra: dict[str, object] = {}
        if name == "wikipedia":
            extra["langs"] = tuple(args.langs.split(","))
        elif name == "local_code" and args.roots:
            extra["roots"] = tuple(Path(p) for p in args.roots)
        if not run_source(name, out_dir, args.limit, **extra):
            failed.append(name)
    if failed:
        log.warning("sources failed: %s", ", ".join(failed))
        sys.exit(1 if len(failed) == len(requested) else 0)


if __name__ == "__main__":
    main()
