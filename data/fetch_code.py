# -*- coding: utf-8 -*-
"""Fetch REAL production code from top open-source projects.

Not synthetic examples: actual files that ship in redis, CPython, Go,
musl, llama.cpp. The KB indexes them so the model can quote real-world
idioms (sds strings, CPython decorators, Go stdlib patterns) instead of
inventing style.

Files are capped at ~120 KB, saved as .txt with a source header.
Resume-safe. Run from extra-code root: python -m data.fetch_code
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

OUT = Path("data/corpus/real_code")
OUT.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "extra-code-kb/1.0 (local research; contact: local)"}
MAX_BYTES = 120_000
PAUSE_S = 0.4

# (repo label, raw url, out name)
FILES = [
    # redis — the C style reference: small, dense, battle-tested
    ("redis", "https://raw.githubusercontent.com/redis/redis/unstable/src/sds.c", "redis_sds.c"),
    ("redis", "https://raw.githubusercontent.com/redis/redis/unstable/src/dict.c", "redis_dict.c"),
    ("redis", "https://raw.githubusercontent.com/redis/redis/unstable/src/zmalloc.c", "redis_zmalloc.c"),
    ("redis", "https://raw.githubusercontent.com/redis/redis/unstable/src/intset.c", "redis_intset.c"),
    ("redis", "https://raw.githubusercontent.com/redis/redis/unstable/src/listpack.c", "redis_listpack.c"),
    ("redis", "https://raw.githubusercontent.com/redis/redis/unstable/src/ae.c", "redis_ae_eventloop.c"),
    ("redis", "https://raw.githubusercontent.com/redis/redis/unstable/src/anet.c", "redis_anet_sockets.c"),
    ("redis", "https://raw.githubusercontent.com/redis/redis/unstable/src/t_string.c", "redis_t_string_commands.c"),
    ("redis", "https://raw.githubusercontent.com/redis/redis/unstable/src/sha1.c", "redis_sha1.c"),
    ("redis", "https://raw.githubusercontent.com/redis/redis/unstable/src/crc16.c", "redis_crc16.c"),
    # musl libc — minimal, textbook C
    ("musl", "https://raw.githubusercontent.com/bminor/musl/master/src/string/strlen.c", "musl_strlen.c"),
    ("musl", "https://raw.githubusercontent.com/bminor/musl/master/src/string/strcmp.c", "musl_strcmp.c"),
    ("musl", "https://raw.githubusercontent.com/bminor/musl/master/src/string/memcpy.c", "musl_memcpy.c"),
    ("musl", "https://raw.githubusercontent.com/bminor/musl/master/src/string/strstr.c", "musl_strstr.c"),
    ("musl", "https://raw.githubusercontent.com/bminor/musl/master/src/stdlib/qsort.c", "musl_qsort.c"),
    ("musl", "https://raw.githubusercontent.com/bminor/musl/master/src/stdlib/strtod.c", "musl_strtod.c"),
    ("musl", "https://raw.githubusercontent.com/bminor/musl/master/src/math/sqrt.c", "musl_sqrt.c"),
    # CPython — python written by the masters
    ("cpython", "https://raw.githubusercontent.com/python/cpython/3.12/Lib/functools.py", "cpython_functools.py"),
    ("cpython", "https://raw.githubusercontent.com/python/cpython/3.12/Lib/contextlib.py", "cpython_contextlib.py"),
    ("cpython", "https://raw.githubusercontent.com/python/cpython/3.12/Lib/queue.py", "cpython_queue.py"),
    ("cpython", "https://raw.githubusercontent.com/python/cpython/3.12/Lib/heapq.py", "cpython_heapq.py"),
    ("cpython", "https://raw.githubusercontent.com/python/cpython/3.12/Lib/bisect.py", "cpython_bisect.py"),
    ("cpython", "https://raw.githubusercontent.com/python/cpython/3.12/Lib/statistics.py", "cpython_statistics.py"),
    ("cpython", "https://raw.githubusercontent.com/python/cpython/3.12/Lib/collections/__init__.py", "cpython_collections.py"),
    ("cpython", "https://raw.githubusercontent.com/python/cpython/3.12/Lib/dataclasses.py", "cpython_dataclasses.py"),
    ("cpython", "https://raw.githubusercontent.com/python/cpython/3.12/Lib/gzip.py", "cpython_gzip.py"),
    ("cpython", "https://raw.githubusercontent.com/python/cpython/3.12/Lib/csv.py", "cpython_csv.py"),
    ("cpython", "https://raw.githubusercontent.com/python/cpython/3.12/Lib/json/encoder.py", "cpython_json_encoder.py"),
    ("cpython", "https://raw.githubusercontent.com/python/cpython/3.12/Lib/json/decoder.py", "cpython_json_decoder.py"),
    ("cpython", "https://raw.githubusercontent.com/python/cpython/3.12/Lib/asyncio/queues.py", "cpython_asyncio_queues.py"),
    # Go stdlib — the readability school
    ("go", "https://raw.githubusercontent.com/golang/go/master/src/strings/strings.go", "go_strings.go"),
    ("go", "https://raw.githubusercontent.com/golang/go/master/src/bytes/bytes.go", "go_bytes.go"),
    ("go", "https://raw.githubusercontent.com/golang/go/master/src/sort/sort.go", "go_sort.go"),
    ("go", "https://raw.githubusercontent.com/golang/go/master/src/errors/errors.go", "go_errors.go"),
    ("go", "https://raw.githubusercontent.com/golang/go/master/src/net/url/url.go", "go_net_url.go"),
    ("go", "https://raw.githubusercontent.com/golang/go/master/src/strconv/atoi.go", "go_strconv_atoi.go"),
    ("go", "https://raw.githubusercontent.com/golang/go/master/src/encoding/json/encode.go", "go_json_encode.go"),
    ("go", "https://raw.githubusercontent.com/golang/go/master/src/fmt/scan.go", "go_fmt_scan.go"),
    ("go", "https://raw.githubusercontent.com/golang/go/master/src/time/format.go", "go_time_format.go"),
    ("go", "https://raw.githubusercontent.com/golang/go/master/src/io/io.go", "go_io.go"),
    # llama.cpp — GGML inference engine (the stack this project runs on)
    ("llama.cpp", "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/examples/simple/simple.cpp", "llamacpp_simple.cpp"),
    ("llama.cpp", "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/tool-calls/tool-calls.cpp", "llamacpp_tool_calls.cpp"),
    # git — strbuf, the growable-buffer idiom
    ("git", "https://raw.githubusercontent.com/git/git/master/strbuf.c", "git_strbuf.c"),
]


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read(MAX_BYTES)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    ok = fail = 0
    for repo, url, name in FILES:
        out = OUT / f"{name}.txt"
        if out.exists() and out.stat().st_size > 500:
            ok += 1
            continue
        try:
            raw = fetch(url)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERR {repo}/{name}: {exc}", flush=True)
            fail += 1
            time.sleep(PAUSE_S)
            continue
        if not raw:
            fail += 1
            continue
        text = raw.decode("utf-8", errors="replace")
        header = (
            f"# Реальный продакшн-код: {repo}\n"
            f"# Источник: {url}\n"
            f"# Это работающий файл из релиза проекта, не учебный пример.\n\n"
        )
        out.write_text(header + text, encoding="utf-8")
        ok += 1
        print(f"  + {repo}/{name} ({len(raw)//1024} KB)", flush=True)
        time.sleep(PAUSE_S)
    total = sum(1 for _ in OUT.glob("*.txt"))
    mb = sum(p.stat().st_size for p in OUT.glob("*.txt")) / 1e6
    print(json.dumps({"ok": ok, "failed": fail,
                      "files": total, "mb": round(mb, 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
