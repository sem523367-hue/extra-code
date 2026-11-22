"""Perplexity for the base model and for every exported GGUF, one command each.

Two backends, because the two things being measured live in different places:

  torch  - the fp32/bf16 checkpoint, scored exactly, window by window, with the
           model's own logits. This is the baseline the quantized files are
           compared against, and it is also the number that must not move when
           a lossless acceleration flag is switched on.
  gguf   - drives `llama-perplexity` from llama.cpp over the same text file, so
           Q8_0 / Q4_K_M are measured by the engine that will serve them.

Scoring is GPT-2 style: a sliding window of `--block-size` tokens with
`--stride` (default half the block). Every token after the first window is
scored exactly once, so the estimate is comparable across block sizes and
across backends that use the same window.

    python -m eval.perplexity torch --checkpoint runs/sft/final/model.pt \
        --text eval/val.txt --block-size 512 --stride 256 --max-windows 64

    python -m eval.perplexity gguf --model export/gguf/extra-code-q4_k_m.gguf \
        --text eval/val.txt --ctx 512 --chunks 32

    python -m eval.perplexity compare --baseline base.json --candidate q4.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

DEFAULT_TEXT = "data/clean/val.jsonl.gz"
PPL_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)\s*(?:\+/?-\s*([0-9.]+))?")


def read_eval_text(path: Path) -> str:
    """Accept a plain text file, or a .jsonl/.jsonl.gz shard of {"text": ...}.

    The default is the validation split produced by data/prepare.py, so the same
    file feeds the PyTorch baseline and llama-perplexity without a conversion
    step that could accidentally differ between the two.
    """
    if not path.exists():
        raise FileNotFoundError(f"evaluation text not found: {path}")
    if path.suffix == ".gz" or path.name.endswith(".jsonl"):
        import gzip

        opener = gzip.open if path.suffix == ".gz" else open
        parts: list[str] = []
        with opener(path, "rt", encoding="utf-8") as fh:  # type: ignore[operator]
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if path.name.endswith(".jsonl") or path.suffix == ".gz":
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        parts.append(line)
                        continue
                    text = row.get("text") if isinstance(row, dict) else None
                    if isinstance(text, str) and text.strip():
                        parts.append(text)
                else:
                    parts.append(line)
        return "\n\n".join(parts)
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# torch backend
# ---------------------------------------------------------------------------

def _load_checkpoint(path: Path):
    import torch

    from model.model import MiniLLM, ModelConfig

    payload = torch.load(path, map_location="cpu", weights_only=False)
    config_dict = payload.get("config")
    if not config_dict:
        sidecar = path.parent / "config.json"
        if not sidecar.exists():
            raise FileNotFoundError(
                f"{path} has no embedded config and {sidecar} is missing; "
                "pass a checkpoint written by train/pretrain.py or train/sft.py"
            )
        config_dict = json.loads(sidecar.read_text(encoding="utf-8"))
    config = ModelConfig.from_dict(config_dict)
    model = MiniLLM(config)
    state = payload.get("model") or payload.get("state_dict") or payload
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model, config


def _resolve_dtype(name: str, device: str) -> Any:
    import torch

    if name == "auto":
        name = "bf16" if device == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
    if name == "bf16" and device == "cpu":
        name = "fp32"  # CPU bf16 matmul is slower and not more accurate
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def perplexity_torch(
    checkpoint: Path,
    text_path: Path,
    tokenizer_path: Path,
    *,
    block_size: int = 512,
    stride: int | None = None,
    max_windows: int = 64,
    device: str = "cpu",
    dtype: str = "auto",
) -> dict[str, Any]:
    import torch
    from tokenizers import Tokenizer

    text = read_eval_text(text_path)
    if len(text.strip()) < 100:
        raise ValueError(f"{text_path} is too short to evaluate ({len(text)} chars)")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    if len(ids) < block_size:
        raise ValueError(
            f"text tokenizes to {len(ids)} tokens, shorter than --block-size {block_size}"
        )

    stride = stride or block_size // 2
    if not 1 <= stride <= block_size:
        raise ValueError("--stride must be between 1 and --block-size")

    model, config = _load_checkpoint(checkpoint)
    if block_size > config.max_seq_len:
        raise ValueError(
            f"--block-size {block_size} exceeds the model's max_seq_len {config.max_seq_len}"
        )
    torch_dtype = _resolve_dtype(dtype, device)
    model.to(device=device, dtype=torch_dtype)
    model.eval()

    nll_sum = 0.0
    scored = 0
    windows = 0
    last_target = 0
    started = time.perf_counter()

    with torch.inference_mode():
        for start in range(0, len(ids) - 1, stride):
            if windows >= max_windows:
                break
            end = min(start + block_size, len(ids))
            if end - start < 2:
                break
            chunk = torch.tensor(ids[start:end], dtype=torch.long, device=device).unsqueeze(0)
            logits = model(chunk).logits[0, :-1].float()
            targets = chunk[0, 1:]

            first = 0 if start == 0 else max(1, last_target + 1 - start)
            if first >= targets.numel():
                continue
            window_logits = logits[first:]
            window_targets = targets[first:]
            nll = torch.nn.functional.cross_entropy(
                window_logits, window_targets, reduction="sum"
            )
            nll_sum += float(nll)
            scored += int(window_targets.numel())
            last_target = end - 1
            windows += 1

    if scored == 0:
        raise RuntimeError("no tokens were scored; increase --block-size or --max-windows")
    elapsed = time.perf_counter() - started
    mean_nll = nll_sum / scored
    return {
        "backend": "torch",
        "model": str(checkpoint),
        "tokenizer": str(tokenizer_path),
        "text": str(text_path),
        "device": device,
        "dtype": {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}[torch_dtype],
        "block_size": block_size,
        "stride": stride,
        "max_windows": max_windows,
        "windows": windows,
        "tokens_scored": scored,
        "nll": round(mean_nll, 6),
        "perplexity": round(math.exp(min(mean_nll, 50.0)), 4),
        "seconds": round(elapsed, 2),
        "tokens_per_s": round(scored / elapsed, 1) if elapsed > 0 else None,
    }


# ---------------------------------------------------------------------------
# gguf backend
# ---------------------------------------------------------------------------

def _find_perplexity_binary(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"--perplexity-binary not found: {path}")
        return path
    candidates = [
        Path("llama.cpp/build/bin/llama-perplexity"),
        Path("llama.cpp/build/bin/llama-perplexity.exe"),
        Path("llama.cpp/build/bin/Release/llama-perplexity.exe"),
        Path("../llama.cpp/build/bin/llama-perplexity"),
        Path("../llama.cpp/build/bin/llama-perplexity.exe"),
        Path("../llama.cpp/build/bin/Release/llama-perplexity.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    found = shutil.which("llama-perplexity")
    if found:
        return Path(found)
    raise FileNotFoundError(
        "llama-perplexity not found. Build llama.cpp "
        "(cmake -B build -DGGML_NATIVE=ON && cmake --build build --config Release -j) "
        "and pass --perplexity-binary, or put it on PATH."
    )


def perplexity_gguf(
    model_path: Path,
    text_path: Path,
    *,
    ctx: int = 512,
    chunks: int = 32,
    batch: int | None = None,
    threads: int | None = None,
    binary: str | None = None,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"GGUF not found: {model_path}")
    if not text_path.exists():
        raise FileNotFoundError(f"evaluation text not found: {text_path}")
    if model_path.suffix.lower() != ".gguf":
        raise ValueError(f"{model_path} does not look like a GGUF file")
    with model_path.open("rb") as fh:
        if fh.read(4) != b"GGUF":
            raise ValueError(f"{model_path} is not a GGUF file (bad magic)")

    exe = _find_perplexity_binary(binary)
    # llama-perplexity reads plain text only; the validation split is gzip JSONL,
    # so materialise exactly the same characters it would get from a text file.
    text = read_eval_text(text_path)
    if len(text.strip()) < 100:
        raise ValueError(f"{text_path} is too short to evaluate ({len(text)} chars)")
    tmp = tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False)
    try:
        tmp.write(text)
        tmp.close()
        cmd = [
            str(exe),
            "-m", str(model_path),
            "-f", tmp.name,
            "-c", str(ctx),
            "-b", str(batch or ctx),
            "--chunks", str(chunks),
        ]
        if threads:
            cmd += ["-t", str(threads)]
        if extra_args:
            cmd += extra_args

        started = time.perf_counter()
        print("$", " ".join(cmd), file=sys.stderr)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.perf_counter() - started
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    output = f"{proc.stdout}\n{proc.stderr}"
    match = None
    for match in PPL_RE.finditer(output):
        pass
    if proc.returncode != 0 or match is None:
        tail = "\n".join(output.strip().splitlines()[-20:])
        raise RuntimeError(
            f"llama-perplexity failed (exit {proc.returncode}) and no 'Final estimate' line "
            f"was found.\n{tail}"
        )
    return {
        "backend": "gguf",
        "model": str(model_path),
        "model_mb": round(model_path.stat().st_size / 1e6, 2),
        "text": str(text_path),
        "ctx": ctx,
        "batch": batch or ctx,
        "chunks": chunks,
        "perplexity": round(float(match.group(1)), 4),
        "std": round(float(match.group(2)), 4) if match.group(2) else None,
        "seconds": round(elapsed, 2),
        "command": " ".join(cmd),
    }


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------

def compare(
    baseline_path: Path,
    candidate_path: Path,
    *,
    max_abs: float = 0.01,
    max_rel: float = 0.05,
) -> dict[str, Any]:
    base = json.loads(baseline_path.read_text(encoding="utf-8"))
    cand = json.loads(candidate_path.read_text(encoding="utf-8"))
    base_ppl = float(base["perplexity"])
    cand_ppl = float(cand["perplexity"])
    if base_ppl <= 0 or cand_ppl <= 0:
        raise ValueError("perplexity must be positive")
    delta = cand_ppl - base_ppl
    rel = delta / base_ppl
    ok_abs = abs(delta) <= max_abs
    ok_rel = rel <= max_rel
    return {
        "baseline": {"model": base.get("model"), "perplexity": base_ppl},
        "candidate": {"model": cand.get("model"), "perplexity": cand_ppl},
        "delta": round(delta, 6),
        "relative": round(rel, 6),
        "max_abs": max_abs,
        "max_rel": max_rel,
        "within_abs": ok_abs,
        "within_rel": ok_rel,
        "lossless": ok_abs,          # acceleration acceptance: +-0.01 absolute
        "acceptable_quant": ok_rel,  # quant acceptance: <= 5% relative
        "pass": bool(ok_abs or ok_rel),
    }


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------

def _write(path: str | None, payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwritten: {out}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Perplexity for checkpoints and GGUF files")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_torch = sub.add_parser("torch", help="score a .pt checkpoint with the PyTorch model")
    p_torch.add_argument("--checkpoint", default="runs/sft/final/model.pt")
    p_torch.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    p_torch.add_argument("--text", default=DEFAULT_TEXT)
    p_torch.add_argument("--block-size", type=int, default=512)
    p_torch.add_argument("--stride", type=int, default=None)
    p_torch.add_argument("--max-windows", type=int, default=64)
    p_torch.add_argument("--device", default="cpu")
    p_torch.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    p_torch.add_argument("--out", default=None)

    p_gguf = sub.add_parser("gguf", help="score a .gguf file with llama-perplexity")
    p_gguf.add_argument("--model", required=True)
    p_gguf.add_argument("--text", default=DEFAULT_TEXT)
    p_gguf.add_argument("--ctx", type=int, default=512)
    p_gguf.add_argument("--chunks", type=int, default=32)
    p_gguf.add_argument("--batch", type=int, default=None)
    p_gguf.add_argument("--threads", type=int, default=None)
    p_gguf.add_argument("--perplexity-binary", default=None)
    p_gguf.add_argument("--out", default=None)

    p_cmp = sub.add_parser("compare", help="diff two perplexity JSON files")
    p_cmp.add_argument("--baseline", required=True)
    p_cmp.add_argument("--candidate", required=True)
    p_cmp.add_argument("--max-abs", type=float, default=0.01)
    p_cmp.add_argument("--max-rel", type=float, default=0.05)
    p_cmp.add_argument("--out", default=None)

    args = parser.parse_args(argv)
    try:
        if args.cmd == "torch":
            payload = perplexity_torch(
                Path(args.checkpoint),
                Path(args.text),
                Path(args.tokenizer),
                block_size=args.block_size,
                stride=args.stride,
                max_windows=args.max_windows,
                device=args.device,
                dtype=args.dtype,
            )
            _write(args.out, payload)
            return 0
        if args.cmd == "gguf":
            payload = perplexity_gguf(
                Path(args.model),
                Path(args.text),
                ctx=args.ctx,
                chunks=args.chunks,
                batch=args.batch,
                threads=args.threads,
                binary=args.perplexity_binary,
            )
            _write(args.out, payload)
            return 0
        payload = compare(
            Path(args.baseline),
            Path(args.candidate),
            max_abs=args.max_abs,
            max_rel=args.max_rel,
        )
        _write(args.out, payload)
        return 0 if payload["pass"] else 1
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
