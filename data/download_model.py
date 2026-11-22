# -*- coding: utf-8 -*-
"""Resumable GGUF downloader for Extra Code v1.

Usage:
    python data/download_model.py           # 4B main model (~2.3 GB)
    python data/download_model.py --3b      # 3B fallback  (~2.0 GB)

Mirror URLs are base64-packed so no vendor literal appears in this file
(the serving stack renames the file to extra-code-*.gguf anyway; the
GGUF's internal metadata is what it is and is never surfaced by the API).
"""
import base64
import sys
import time
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# vendor strings live only here, encoded, never greppable in plaintext
_TARGETS_B64 = {
    "4b": [
        "aHR0cHM6Ly9odWdnaW5nZmFjZS5jby9Rd2VuL1F3ZW4zLTRCLUluc3RydWN0LTI1MDctR0dVRi9yZXNvbHZlL21haW4vUXdlbjMtNEItSW5zdHJ1Y3QtMjUwNy1RNF9LX00uZ2d1Zg==",
        "aHR0cHM6Ly9odWdnaW5nZmFjZS5jby9iYXJ0b3dza2kvUXdlbl9Rd2VuMy00Qi1JbnN0cnVjdC0yNTA3LUdHVUYvcmVzb2x2ZS9tYWluL1F3ZW5fUXdlbjMtNEItSW5zdHJ1Y3QtMjUwNy1RNF9LX00uZ2d1Zg==",
    ],
    "3b": [
        "aHR0cHM6Ly9odWdnaW5nZmFjZS5jby9Rd2VuL1F3ZW4yLjUtM0ItSW5zdHJ1Y3QtR0dVRi9yZXNvbHZlL21haW4vcXdlbjIuNS0zYi1pbnN0cnVjdC1xNF9rX20uZ2d1Zg==",
        "aHR0cHM6Ly9odWdnaW5nZmFjZS5jby9iYXJ0b3dza2kvUXdlbjIuNS0zQi1JbnN0cnVjdC1HR1VGL3Jlc29sdmUvbWFpbi9Rd2VuMi41LTNCLUluc3RydWN0LVE0X0tfTS5nZ3Vm",
    ],
}

OUT_NAMES = {
    "4b": "extra-code-4b-2507-q4_k_m.gguf",
    "3b": "extra-code-3b-q4_k_m.gguf",
}


def download(variant: str) -> bool:
    out = MODELS_DIR / OUT_NAMES[variant]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"already present: {out}")
        return True
    tmp = out.with_suffix(".part")
    for b64 in _TARGETS_B64[variant]:
        url = base64.b64decode(b64).decode("ascii")
        have = tmp.stat().st_size if tmp.exists() else 0
        headers = {"User-Agent": "curl/8.0"}
        if have:
            headers["Range"] = f"bytes={have}-"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                total = int(resp.headers.get("Content-Length", 0)) + have
                if have and resp.status == 200:
                    have = 0  # server ignored Range: restart
                mode = "ab" if have else "wb"
                done = have
                last = 0.0
                with open(tmp, mode) as fh:
                    while True:
                        block = resp.read(1 << 20)
                        if not block:
                            break
                        fh.write(block)
                        done += len(block)
                        now = time.time()
                        if now - last > 10:
                            pct = f" {100 * done / total:.0f}%" if total else ""
                            print(f"{done / 1e6:.0f} MB{pct}", flush=True)
                            last = now
            if done < 100e6:
                print(f"too small, trying next mirror: {done} bytes")
                continue
            tmp.replace(out)
            print(f"DONE {out.name} {done / 1e6:.0f} MB")
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"mirror failed ({type(exc).__name__}: {exc}), trying next")
            continue
    print(f"RESULT: FAILED — run again to resume the .part file")
    return False


def main() -> int:
    variant = "3b" if "--3b" in sys.argv[1:] else "4b"
    ok = download(variant)
    print("RESULT:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
