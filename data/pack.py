"""Pack tokenized documents into flat uint16 files.

Layout: [doc tokens..., </s>, doc tokens..., </s>, ...] for train and val.
Two files on purpose: training memmaps train.bin and never touches val.bin.
uint16 caps the vocabulary at 65536; the tokenizer enforces that limit.
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

log = logging.getLogger("extra_code.pack")

FLUSH_TOKENS = 1_000_000


def iter_docs(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("text")
            if isinstance(text, str) and text:
                yield text


def pack_split(tokenizer: Tokenizer, source: Path, target: Path, eos_id: int) -> tuple[int, int]:
    buffer: list[int] = []
    docs = tokens = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as out:
        for text in iter_docs(source):
            ids = tokenizer.encode(text).ids
            if not ids:
                continue
            buffer.extend(ids)
            buffer.append(eos_id)
            docs += 1
            if len(buffer) >= FLUSH_TOKENS:
                np.asarray(buffer, dtype=np.uint16).tofile(out)
                tokens += len(buffer)
                buffer.clear()
            if docs % 10_000 == 0:
                log.info("%s: %d docs, %d tokens", target.name, docs, tokens + len(buffer))
        if buffer:
            np.asarray(buffer, dtype=np.uint16).tofile(out)
            tokens += len(buffer)
    log.info("%s done: %d docs, %d tokens, %.1f MB", target.name, docs, tokens, target.stat().st_size / 1e6)
    return docs, tokens


def validate(path: Path, tokenizer: Tokenizer, expected_vocab: int) -> dict[str, object]:
    data = np.memmap(path, dtype=np.uint16, mode="r")
    if data.size < 16:
        raise ValueError(f"{path} is too small to validate")
    head = tokenizer.decode([int(x) for x in data[:64]], skip_special_tokens=False)
    tail = tokenizer.decode([int(x) for x in data[-64:]], skip_special_tokens=False)
    max_id = int(data.max())
    if max_id >= expected_vocab:
        raise ValueError(f"{path}: token id {max_id} exceeds vocab {expected_vocab}")
    return {
        "tokens": int(data.size),
        "bytes": path.stat().st_size,
        "max_token_id": max_id,
        "head": head[:200],
        "tail": tail[-200:],
    }


def pack(
    clean_dir: Path,
    tokenizer_path: Path,
    out_dir: Path,
) -> dict[str, object]:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab = tokenizer.get_vocab_size()
    if vocab > 65_536:
        raise ValueError(f"vocab {vocab} does not fit uint16")
    eos_id = tokenizer.token_to_id("</s>")
    if eos_id is None:
        raise ValueError("tokenizer has no </s> token; refusing to pack without EOS")

    train_docs, train_tokens = pack_split(
        tokenizer, clean_dir / "train.jsonl.gz", out_dir / "train.bin", eos_id
    )
    val_docs, val_tokens = pack_split(tokenizer, clean_dir / "val.jsonl.gz", out_dir / "val.bin", eos_id)
    if train_tokens < 1000:
        raise ValueError("training set is too small; collect more data first")

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "vocab_size": vocab,
        "eos_id": eos_id,
        "dtype": "uint16",
        "train": {"docs": train_docs, "tokens": train_tokens, "file": str(out_dir / "train.bin")},
        "val": {"docs": val_docs, "tokens": val_tokens, "file": str(out_dir / "val.bin")},
        "train_check": validate(out_dir / "train.bin", tokenizer, vocab),
        "val_check": validate(out_dir / "val.bin", tokenizer, vocab) if val_tokens else None,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("packed: train=%d tokens val=%d tokens", train_tokens, val_tokens)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack the clean corpus into uint16 memmap files")
    parser.add_argument("--clean", default="data/clean")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--out", default="data")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    meta = pack(Path(args.clean), Path(args.tokenizer), Path(args.out))
    print(json.dumps({k: meta[k] for k in ("vocab_size", "eos_id")}, indent=2))


if __name__ == "__main__":
    main()
