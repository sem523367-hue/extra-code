"""End-to-end smoke: synthetic corpus -> clean -> tokenizer -> pack -> pretrain -> SFT.

Not a pytest test (it shells out and takes ~1 minute). Run manually:
    python tests/smoke_e2e.py
"""
from __future__ import annotations

import gzip
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".smoke"
PY = sys.executable
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_corpus() -> None:
    rng = random.Random(7)
    words = (
        "token model vector matrix tensor gradient layer attention context batch loss "
        "training corpus tokenizer server request response stream cache memory kernel "
        "функция модель вектор матрица градиент слой внимание контекст обучение сервер "
        "запрос ответ поток память ядро данные текст язык программа отчёт система "
        "decode encode sample temperature window sequence output input weight bias "
        "compile deploy docker service proxy latency throughput benchmark dataset"
    ).split()
    raw = WORK / "raw" / "synthetic"
    raw.mkdir(parents=True, exist_ok=True)
    with gzip.open(raw / "synthetic-1.jsonl.gz", "wt", encoding="utf-8") as fh:
        for i in range(140):
            sentences = [
                " ".join(rng.choice(words) for _ in range(rng.randint(12, 30))) + "."
                for _ in range(14)
            ]
            text = "\n".join(sentences)
            if i % 7 == 0:
                text += "\ndef add(a: int, b: int) -> int:\n    return a + b\n"
            row = {"text": text, "source": "synthetic", "url": "", "lang": "ru", "meta": {"i": i}}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_sft() -> None:
    path = WORK / "sft" / "train.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(11)
    with path.open("w", encoding="utf-8") as fh:
        for i in range(40):
            fh.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "system", "content": "Ты компактная локальная модель."},
                            {"role": "user", "content": f"Объясни тему номер {i} коротко."},
                            {
                                "role": "assistant",
                                "content": f"Тема {i} — это протокол обмена данными и проверку гипотез.",
                            },
                        ]
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    (WORK / "config.json").write_text(
        json.dumps(
            {
                "vocab_size": 512,
                "d_model": 64,
                "n_layers": 2,
                "n_heads": 4,
                "n_kv_heads": 2,
                "d_ff": 128,
                "max_seq_len": 96,
                "rope_theta": 10000.0,
                "rms_norm_eps": 1e-05,
                "tie_embeddings": True,
                "dropout": 0.0,
            }
        ),
        encoding="utf-8",
    )


def run(args: list[str]) -> None:
    print("$", " ".join(args))
    result = subprocess.run(args, cwd=ROOT, text=True)
    if result.returncode != 0:
        raise SystemExit(f"command failed: {args}")


def main() -> None:
    for stale in ("raw", "clean", "data", "tokenizer", "runs"):
        target = WORK / stale
        if target.exists():
            import shutil

            shutil.rmtree(target)
    build_corpus()
    build_sft()

    from data.prepare import prepare

    report = prepare(WORK / "raw", WORK / "clean", threshold=0.8)
    assert report["kept"] > 20, report
    print("prepare:", report)

    run(
        [
            PY, "-m", "data.tokenizer",
            "--train", str(WORK / "clean" / "train.jsonl.gz"),
            "--out", str(WORK / "tokenizer" / "tokenizer.json"),
            "--vocab-size", "512",
        ]
    )
    run(
        [
            PY, "-m", "data.pack",
            "--clean", str(WORK / "clean"),
            "--tokenizer", str(WORK / "tokenizer" / "tokenizer.json"),
            "--out", str(WORK / "data"),
        ]
    )
    run(
        [
            PY, "-m", "train.pretrain",
            "--config", str(WORK / "config.json"),
            "--data-dir", str(WORK / "data"),
            "--out", str(WORK / "runs" / "base"),
            "--steps", "8",
            "--batch-size", "2",
            "--grad-accum", "2",
            "--block-size", "64",
            "--eval-interval", "4",
            "--eval-batches", "2",
            "--ckpt-interval", "4",
            "--log-interval", "1",
            "--dtype", "fp32",
            "--device", "cpu",
        ]
    )
    run(
        [
            PY, "-m", "train.sft",
            "--model", str(WORK / "runs" / "base" / "final" / "model.pt"),
            "--data", str(WORK / "sft" / "train.jsonl"),
            "--tokenizer", str(WORK / "tokenizer" / "tokenizer.json"),
            "--out", str(WORK / "runs" / "sft"),
            "--epochs", "1",
            "--batch-size", "2",
            "--grad-accum", "2",
            "--max-len", "96",
            "--dtype", "fp32",
            "--device", "cpu",
        ]
    )
    assert (WORK / "runs" / "sft" / "final" / "model.pt").exists()
    summary = json.loads((WORK / "runs" / "sft" / "summary.json").read_text(encoding="utf-8"))
    assert summary["best_val"] == summary["best_val"], summary
    print("SMOKE OK", summary)


if __name__ == "__main__":
    main()
