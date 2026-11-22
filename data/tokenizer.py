"""Train the 32k byte-level BPE tokenizer.

Byte-level BPE keeps every byte representable, so Russian, code, emoji and
broken UTF-8 all round-trip without <unk>. The special tokens include the
ChatML markers used by the server prompt builder.
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

log = logging.getLogger("extra_code.tokenizer")

SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<pad>", "<|im_start|>", "<|im_end|>"]


def iter_texts(path: Path, limit: int | None = None) -> Iterator[str]:
    count = 0
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("text")
            if isinstance(text, str) and text.strip():
                yield text
                count += 1
                if limit is not None and count >= limit:
                    return


def train(
    train_path: Path,
    out_path: Path,
    *,
    vocab_size: int = 32_000,
    limit_docs: int | None = None,
    min_frequency: int = 2,
) -> dict[str, object]:
    if not 256 <= vocab_size <= 65_536:
        raise ValueError("vocab_size must be between 256 and 65536 for uint16 packing")
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    log.info("training BPE: vocab=%d from %s", vocab_size, train_path)
    tokenizer.train_from_iterator(iter_texts(train_path, limit_docs), trainer=trainer)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))
    real_vocab = tokenizer.get_vocab_size()

    samples = [
        "Привет, мир. Это проверка русского текста.",
        "def add(a: int, b: int) -> int:\n    return a + b\n",
        "emoji 🥔 and symbols <|im_end|> stay representable byte-wise",
    ]
    checks = []
    for sample in samples:
        encoded = tokenizer.encode(sample)
        decoded = tokenizer.decode(encoded.ids, skip_special_tokens=False)
        checks.append({"sample": sample[:40], "tokens": len(encoded.ids), "roundtrip": decoded == sample})

    meta = {
        "path": str(out_path),
        "vocab_size": real_vocab,
        "requested_vocab": vocab_size,
        "special_tokens": SPECIAL_TOKENS,
        "model": "byte-level-bpe",
        "roundtrip": all(c["roundtrip"] for c in checks),
        "checks": checks,
    }
    (out_path.parent / "special_tokens.json").write_text(
        json.dumps({"special_tokens": SPECIAL_TOKENS, "eos_id": tokenizer.token_to_id("</s>")}, indent=2),
        encoding="utf-8",
    )
    (out_path.parent / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("tokenizer saved: vocab=%d roundtrip=%s", real_vocab, meta["roundtrip"])
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the BPE tokenizer on the clean corpus")
    parser.add_argument("--train", default="data/clean/train.jsonl.gz")
    parser.add_argument("--out", default="tokenizer/tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--limit-docs", type=int, default=None)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train(
        Path(args.train),
        Path(args.out),
        vocab_size=args.vocab_size,
        limit_docs=args.limit_docs,
        min_frequency=args.min_frequency,
    )


if __name__ == "__main__":
    main()
