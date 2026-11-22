"""Run the code question set through a trained checkpoint and score the output.

    python -m eval.code_check --model runs/sft/final/model.pt
    python -m eval.code_check --limit 10 --temperature 0.2

Writes eval/code-check.json with every answer, token counts and timing, and
prints a short preview. This is a sanity report, not a benchmark against hosted
models: it answers "does the model produce the right shape and language".
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

from model.chat import DEFAULT_SYSTEM, build_prompt, load, pick_model, pick_tokenizer, sample


def read_prompts(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@torch.no_grad()
def generate(model, tokenizer, prompt_ids: list[int], *, device: torch.device,
             max_new_tokens: int, temperature: float, top_k: int, top_p: float,
             repetition_penalty: float) -> tuple[str, int, float, float]:
    eos_ids = {tokenizer.token_to_id("</s>"), tokenizer.token_to_id("<|im_end|>")}
    eos_ids.discard(None)
    window = max(1, model.config.max_seq_len - max_new_tokens)
    ids = prompt_ids[-window:]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    past = None
    generated: list[int] = []
    started = time.perf_counter()
    first: float | None = None
    for _ in range(max_new_tokens):
        out = model(input_ids if past is None else input_ids[:, -1:], use_cache=True, past_key_values=past)
        past = out.past_key_values
        next_id = sample(out.logits[0, -1, :], temperature=temperature, top_k=top_k, top_p=top_p,
                         recent=(ids + generated)[-64:], repetition_penalty=repetition_penalty)
        if next_id in eos_ids:
            break
        generated.append(next_id)
        input_ids = torch.tensor([[next_id]], dtype=torch.long, device=device)
        if first is None:
            first = time.perf_counter() - started
    elapsed = time.perf_counter() - started
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text, len(generated), (first or elapsed) * 1000, len(generated) / max(elapsed, 1e-9)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the code question set through a checkpoint")
    parser.add_argument("--model", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--prompts", default="eval/code_prompts.jsonl")
    parser.add_argument("--out", default="eval/code-check.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model_path = pick_model(args.model)
    tokenizer_path = pick_tokenizer(args.tokenizer, model_path)
    model, tokenizer, device = load(model_path, tokenizer_path, device=args.device, dtype="auto", ctx=None)
    prompts = read_prompts(Path(args.prompts))
    if args.limit:
        prompts = prompts[: args.limit]

    results = []
    for index, row in enumerate(prompts, 1):
        messages = [("system", DEFAULT_SYSTEM)]
        messages += [(m["role"], m["content"]) for m in row["messages"]]
        prompt_ids = build_prompt(tokenizer, messages)
        text, tokens, ttft, tok_s = generate(
            model, tokenizer, prompt_ids, device=device,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p, repetition_penalty=args.repetition_penalty,
        )
        results.append({
            "id": row.get("id", str(index)),
            "prompt": row["messages"][-1]["content"],
            "answer": text,
            "tokens": tokens,
            "ttft_ms": round(ttft, 1),
            "tok_s": round(tok_s, 1),
        })
        preview = text.replace("\n", " ⏎ ")[:160]
        print(f"[{index}/{len(prompts)}] {row.get('id', index)}: {preview}", flush=True)

    speeds = [r["tok_s"] for r in results if r["tokens"] > 0]
    summary = {
        "model": str(model_path),
        "params": sum(p.numel() for p in model.parameters()),
        "device": device.type,
        "questions": len(results),
        "avg_tok_s": round(statistics.mean(speeds), 1) if speeds else 0.0,
        "p50_tok_s": round(statistics.median(speeds), 1) if speeds else 0.0,
        "avg_ttft_ms": round(statistics.mean([r["ttft_ms"] for r in results]), 1) if results else 0.0,
        "empty_answers": sum(1 for r in results if not r["answer"]),
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
