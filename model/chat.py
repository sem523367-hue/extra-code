"""Interactive chat REPL for a trained extra-code checkpoint.

Loads a PyTorch checkpoint + tokenizer, renders the same ChatML the SFT loop
was trained on, and streams tokens from the KV cache.

    python -m model.chat
    python -m model.chat --model runs/sft/final/model.pt --temperature 0.7
    echo "что такое GIL?" | python -m model.chat --once

Commands inside the REPL: /reset, /system <text>, /temp <x>, /help, /exit.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model.model import MiniLLM, ModelConfig

DEFAULT_SYSTEM = "Ты Extra code — компактная локальная модель. Отвечай кратко и по делу."
STOP_MARKERS = ("<|im_end|>", "<|im_start|>", "</s>")


def pick_model(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise SystemExit(f"checkpoint not found: {path}")
        return path
    for candidate in (
        Path("runs/sft/final/model.pt"),
        Path("runs/base/final/model.pt"),
        Path("runs/base/latest.pt"),
        Path("runs/sft/best.pt"),
    ):
        if candidate.exists():
            return candidate
    raise SystemExit(
        "no checkpoint found. Train first:\n"
        "  python -m train.pretrain --steps 600 --ckpt-interval 100\n"
        "  python -m train.sft --model runs/base/final/model.pt --epochs 1"
    )


def pick_tokenizer(explicit: str | None, model_path: Path) -> Path:
    if explicit:
        return Path(explicit)
    for candidate in (model_path.parent / "tokenizer.json", Path("tokenizer/tokenizer.json")):
        if candidate.exists():
            return candidate
    raise SystemExit("tokenizer.json not found")


def load(model_path: Path, tokenizer_path: Path, *, device: str, dtype: str, ctx: int | None) -> tuple[MiniLLM, Tokenizer, torch.device]:
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    state = payload.get("model") or payload.get("state_dict")
    if state is None:
        raise SystemExit(f"{model_path} has no model weights")
    config = ModelConfig.from_dict(payload.get("config") or {})
    if ctx:
        config.max_seq_len = max(config.max_seq_len, ctx)
    model = MiniLLM(config)
    model.load_state_dict(state, strict=True)

    torch_device = torch.device(device)
    torch_dtype = torch.float32
    if dtype == "auto":
        torch_dtype = torch.float32
    elif dtype == "bf16":
        torch_dtype = torch.bfloat16 if torch_device.type == "cuda" else torch.float32
    elif dtype == "fp16":
        torch_dtype = torch.float16 if torch_device.type == "cuda" else torch.float32
    model = model.to(device=torch_device, dtype=torch_dtype).eval()

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return model, tokenizer, torch_device


def build_prompt(tokenizer: Tokenizer, messages: list[tuple[str, str]]) -> list[int]:
    """Same layout as train/sft.py: <|im_start|>role\\ncontent<|im_end|> ..."""
    start_id = tokenizer.token_to_id("<|im_start|>")
    end_id = tokenizer.token_to_id("<|im_end|>")
    if start_id is None or end_id is None:
        raise SystemExit("tokenizer is missing ChatML special tokens")
    ids: list[int] = []
    for role, content in messages:
        ids.append(start_id)
        ids.extend(tokenizer.encode(f"{role}\n", add_special_tokens=False).ids)
        ids.extend(tokenizer.encode(content, add_special_tokens=False).ids)
        ids.append(end_id)
    ids.append(start_id)
    ids.extend(tokenizer.encode("assistant\n", add_special_tokens=False).ids)
    return ids


def sample(logits: torch.Tensor, *, temperature: float, top_k: int, top_p: float,
           recent: list[int], repetition_penalty: float) -> int:
    logits = logits.float()
    if repetition_penalty != 1.0 and recent:
        for token_id in set(recent):
            value = logits[token_id]
            logits[token_id] = value / repetition_penalty if value > 0 else value * repetition_penalty
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    logits = logits / max(temperature, 1e-5)
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, k)
        logits[logits < values[-1]] = -float("inf")
    probs = F.softmax(logits, dim=-1)
    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative - sorted_probs > top_p
        sorted_probs[remove] = 0.0
        probs = torch.zeros_like(probs).scatter_(0, sorted_idx, sorted_probs)
        probs = probs / probs.sum().clamp_min(1e-9)
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def stream_reply(
    model: MiniLLM,
    tokenizer: Tokenizer,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    device: torch.device,
) -> tuple[str, int, float]:
    """Yield tokens through a callback while keeping the KV cache warm."""
    eos_ids = {tokenizer.token_to_id("</s>"), tokenizer.token_to_id("<|im_end|>")}
    eos_ids.discard(None)
    window = max(1, model.config.max_seq_len - max_new_tokens)
    ids = prompt_ids[-window:]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    past = None
    generated: list[int] = []
    printed = 0
    started = time.perf_counter()
    first_token_at: float | None = None

    for step in range(max_new_tokens):
        out = model(input_ids if past is None else input_ids[:, -1:], use_cache=True, past_key_values=past)
        past = out.past_key_values
        next_id = sample(out.logits[0, -1, :], temperature=temperature, top_k=top_k, top_p=top_p,
                         recent=(ids + generated)[-64:], repetition_penalty=repetition_penalty)
        if next_id in eos_ids:
            break
        generated.append(next_id)
        input_ids = torch.tensor([[next_id]], dtype=torch.long, device=device)
        if first_token_at is None:
            first_token_at = time.perf_counter() - started
        text = tokenizer.decode(generated, skip_special_tokens=True)
        if len(text) > printed:
            sys.stdout.write(text[printed:])
            sys.stdout.flush()
            printed = len(text)
        if "\n<|im" in text:
            break

    elapsed = time.perf_counter() - started
    text = tokenizer.decode(generated, skip_special_tokens=True)
    for marker in STOP_MARKERS:
        if marker in text:
            text = text.split(marker, 1)[0]
    return text, len(generated), (elapsed, first_token_at or elapsed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with a trained extra-code checkpoint")
    parser.add_argument("--model", default=None, help="path to model.pt (auto-detects runs/)")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--ctx", type=int, default=None, help="override max context (RoPE table)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="auto", choices=["auto", "fp32", "bf16", "fp16"])
    parser.add_argument("--once", action="store_true", help="read one prompt from stdin, print reply, exit")
    args = parser.parse_args()

    model_path = pick_model(args.model)
    tokenizer_path = pick_tokenizer(args.tokenizer, model_path)
    model, tokenizer, device = load(model_path, tokenizer_path, device=args.device, dtype=args.dtype, ctx=args.ctx)

    params = sum(p.numel() for p in model.parameters())
    print(f"extra-code | {params / 1e6:.1f}M params | {device.type} | {model_path}", file=sys.stderr)
    print(f"ctx={model.config.max_seq_len} vocab={model.config.vocab_size} | /help for commands", file=sys.stderr)

    system = args.system
    history: list[tuple[str, str]] = []
    temperature = args.temperature

    def answer(user_text: str) -> None:
        history.append(("user", user_text))
        messages = [("system", system), *history]
        prompt_ids = build_prompt(tokenizer, messages)
        reserve = min(args.max_new_tokens, max(32, model.config.max_seq_len // 2))
        if len(prompt_ids) + reserve > model.config.max_seq_len:
            messages = [("system", system), *history[-6:]]
            prompt_ids = build_prompt(tokenizer, messages)
        text, tokens, timing = stream_reply(
            model,
            tokenizer,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=device,
        )
        elapsed, ttft = timing
        print()
        print(f"[{tokens} tok, TTFT {ttft * 1000:.0f} ms, {tokens / max(elapsed, 1e-9):.1f} tok/s]", file=sys.stderr)
        history.append(("assistant", text))

    if args.once:
        user_text = sys.stdin.read().strip()
        if user_text:
            answer(user_text)
        print()
        return

    while True:
        try:
            user_text = input("\nuser> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user_text:
            continue
        if user_text in {"/exit", "/quit", ":q"}:
            return
        if user_text == "/help":
            print("/reset  clear history | /system <text> replace system prompt")
            print("/temp <x> set temperature | /exit quit")
            continue
        if user_text == "/reset":
            history.clear()
            print("history cleared")
            continue
        if user_text.startswith("/system "):
            system = user_text[len("/system "):].strip()
            print("system prompt updated")
            continue
        if user_text.startswith("/temp "):
            try:
                temperature = float(user_text.split(maxsplit=1)[1])
            except ValueError:
                print("usage: /temp 0.7")
                continue
            print(f"temperature = {temperature}")
            continue
        if user_text.startswith("/"):
            print("unknown command, /help")
            continue

        print("assistant> ", end="", flush=True)
        answer(user_text)


if __name__ == "__main__":
    main()
