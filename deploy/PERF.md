# Acceleration and caching — what is on, what it buys, how it is proven

This file maps the "МЕТОДЫ УСКОРЕНИЯ И КЕШИРОВАНИЯ LLM" spec onto this repo:
the exact command for each method, the metric that moves, and the acceptance
command that proves it. Anything that is *not* lossless is called out as such
instead of being sold as free.

Numbers below are targets from the spec. Measured numbers come from your
hardware and are produced by `bench/benchmark.py` + `eval/perplexity.py`;
they are deliberately not invented here.

---

## Two serving shapes

| shape | process | what it is good for | acceleration available |
|---|---|---|---|
| **shim** | `server/main.py` (FastAPI + `llama-cpp-python`) | auth, per-key RPM/TPM, queueing, `/v1/tokenize`, the Telegram bot | model load once, streaming, bounded queue. No cross-request prefix cache, single generation at a time |
| **engine** | `llama-server` (llama.cpp) | raw throughput, prefix cache, FlashAttention, speculative decoding, continuous batching | the full spec |

The shim exists because an OpenAI-compatible API on its own does not give you
key management, per-key token budgets, request IDs, graceful shutdown or a
tokenizer endpoint. `llama-server` exists because it is the only one of the two
that actually implements the acceleration layer.

Run the engine for speed and put the shim in front only if you need auth. Nginx
already exposes both (`/v1/` → shim, `/llama/` → engine) so the comparison is
one config edit, not a rewrite.

```bash
# engine, all flags on
deploy/systemd/llama-server.service      # systemd path
docker compose --profile accelerated up  # container path
```

---

## Is it lossless?

| method | output identical? | why |
|---|---|---|
| Prefix cache / host-memory cache | **yes** | KV of a prefix is a pure function of that prefix. Reusing it is memoisation |
| Continuous batching | **yes** | each sequence keeps its own KV and mask; scheduling order does not enter the math |
| PagedAttention / RadixAttention | **yes** | block tables change *where* KV lives, not what it contains |
| FlashAttention | **yes** | exact softmax, only the loop order over blocks changes |
| Speculative decoding | **yes** | a token is accepted only with the target model's own probability ratio; rejected candidates are resampled from the target distribution |
| **KV cache quantization** (`-ctk q8_0`, `-ctv q8_0`) | **no** | it quantizes the cache itself. Faster and smaller, but it is a quality trade, not a free win |
| **Weight quantization** (Q8_0, Q4_K_M) | **no** | gated by perplexity instead (≤5% relative for Q4_K_M) |

Two practical notes on "identical":

* Greedy comparison needs `temperature: 0` and a fixed seed. `eval/equivalence.py`
  does exactly that.
* FlashAttention changes the summation order inside attention, so the last
  mantissa bit can differ. On a near-tie that can flip an argmax. Gate
  FlashAttention with perplexity (`±0.01`), not with character equality; gate
  speculative decoding with character equality, since it runs the *same*
  attention kernel with and without the draft.

---

## 1. FlashAttention

```bash
llama-server -m models/extra-code-q4_k_m.gguf -fa on --ctx-size 16384 --parallel 4
```

* `-fa` alone is accepted by older builds; `-fa on|off|auto` by newer ones.
  Check with `llama-server --help | grep -i flash`.
* Meaningful on CUDA and Metal. On a pure CPU build it is a no-op — do not
  expect the 1.5–2× there.
* Verify it is active: the startup log mentions the flash-attention kernel, or
  `llama-bench -m model.gguf -fa 1` versus `-fa 0` gives the delta directly.

```bash
llama-bench -m models/extra-code-q4_k_m.gguf -fa 0,1 -p 512 -n 128 -r 3
```

## 2. Prefix caching

`llama-server` keeps the KV of each slot and, on the next request, reuses the
longest common prefix instead of re-prefilling it. That is the mechanism behind
the TTFT drop; `--prompt-cache` persists a slot state to disk so it also
survives a restart.

```bash
llama-server -m models/extra-code-q4_k_m.gguf \
  --ctx-size 16384 --parallel 4 \
  --prompt-cache /var/cache/extra-code/prompt.cache \
  --prompt-cache-ro \
  -fa on
```

* `--ctx-size` is the **total** context; each of the `--parallel` slots gets
  `ctx-size / parallel`. With `--ctx-size 16384 --parallel 4` every request sees
  4096 tokens. Size it from your real prompt length, not from habit.
* `--prompt-cache-ro` marks the file read-only, which is right for a shared
  system prompt baked at deploy time and wrong if you expect it to be updated.
* The cache is per slot. A request that lands on a different slot pays full
  prefill again — which is why the benchmark repeats the same prompt and
  reports the median, not the best case.

Measure:

```bash
python -m bench.benchmark --base-url http://127.0.0.1:8080 \
  --api-key "$LLAMA_API_KEY" --prefix-repeat 6 --skip batch,concurrency \
  --min-prefix-speedup 5
```

The report's `prefix.speedup` is `cold_ttft / median(warm_ttft)`. Exit code 1
when it is below `--min-prefix-speedup`, so it can gate a deploy.

## 3. Host-memory prompt cache

Keeps prompt KV in system RAM and copies it back into the slot when a matching
prefix arrives. Buys VRAM headroom on small cards and keeps TTFT low for a
rotating set of prefixes.

```bash
llama-server -m models/extra-code-q4_k_m.gguf \
  --ctx-size 16384 --parallel 4 \
  --cache-ram 4096 \
  -fa on
```

`--cache-ram` is MiB of host memory for cached prompt state. It needs a recent
llama.cpp — if the flag is rejected, update the checkout; there is no
equivalent in older builds. Same measurement as prefix caching: the warm TTFT
should stay flat while the number of distinct prefixes grows past what fits in
VRAM.

## 4. PagedAttention (vLLM) / RadixAttention (SGLang)

llama.cpp does not implement either — its KV is a contiguous per-slot buffer.
If the spec's throughput target (×2–2.5, >64 concurrent sequences) is the goal,
switch engine:

```bash
# vLLM: PagedAttention is the core allocator, prefix caching is the flag
vllm serve ./export/hf \
  --enable-prefix-caching \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.9 \
  --served-model-name extra-code-30m
```

```bash
# SGLang: RadixAttention (longest-common-prefix reuse) is on by default
python -m sglang.launch_server --model-path ./export/hf \
  --port 30000 --mem-fraction-static 0.85
```

`export/to_hf.py` writes the HF Llama layout both of them load. Point
`--base-url` of the benchmark at them; nothing else changes.

## 5. Speculative decoding

Needs a second, smaller GGUF that shares the tokenizer and the vocabulary. Train
one by distilling the target (or take an existing small model with the same
32k BPE — a different tokenizer silently degrades acceptance, it does not
error).

```bash
llama-server -m models/extra-code-q4_k_m.gguf \
  -md models/extra-code-draft-q4_k_m.gguf \
  --draft-max 8 --draft-min 4 --draft-p-min 0.8 \
  --ctx-size 16384 --parallel 4 -fa on
```

The drop-in that swaps these flags in is
`deploy/systemd/llama-server.service.d/speculative.conf`.

Correctness gate — same prompts, greedy, two servers (one with `-md`, one
without), character-for-character:

```bash
python -m eval.equivalence \
  --a http://127.0.0.1:8080 --b http://127.0.0.1:8081 \
  --a-key "$KEY_NO_DRAFT" --b-key "$KEY_DRAFT" \
  --prompts eval/prompts.jsonl --out eval/equivalence-spec.json
```

Exit 0 means identical output on the whole suite.

## 6. Continuous batching

```bash
# llama.cpp
llama-server -m models/extra-code-q4_k_m.gguf --parallel 4 --cont-batching -fa on

# vLLM
vllm serve ./export/hf --max-num-seqs 64 --max-num-batched-tokens 8192
```

```bash
python -m bench.benchmark --base-url http://127.0.0.1:8080 \
  --api-key "$LLAMA_API_KEY" --concurrency 4 --skip prefix,batch
```

Compare `concurrency.aggregate_tokens_per_s` at concurrency 1, 4 and 8. A single
request does not get faster; the aggregate should. If per-request tok/s collapses
while aggregate stays flat, the slots are contending for the same compute — drop
`--parallel` or move to a bigger GPU.

---

## Acceptance checklist

| criterion | command | pass |
|---|---|---|
| Prefix cache works | `bench/benchmark.py --prefix-repeat 6 --min-prefix-speedup 5` | exit 0, `prefix.speedup ≥ 5` |
| FlashAttention active | startup log, or `llama-bench -fa 0,1` | `-fa on` shows the FA kernel and a lower time |
| Spec decoding identical | `eval/equivalence.py --a ... --b ...` | `identical: true` |
| Continuous batching holds N | `bench/benchmark.py --concurrency 4` | 0 errors, aggregate tok/s > single-stream tok/s |
| Quality unchanged (±0.01) | `eval/perplexity.py gguf` before/after + `compare` | `lossless: true` |
| No approximation in attention | code review of the flags | no `-ctk q8_0`, no `-ctv q8_0`, no `--no-kv-offload` tricks |

Perplexity, baseline versus accelerated and versus Q4_K_M:

```bash
python -m eval.perplexity gguf --model models/extra-code-f16.gguf \
  --ctx 512 --chunks 32 --out eval/ppl-f16.json
python -m eval.perplexity gguf --model models/extra-code-q8_0.gguf \
  --ctx 512 --chunks 32 --out eval/ppl-q8.json
python -m eval.perplexity gguf --model models/extra-code-q4_k_m.gguf \
  --ctx 512 --chunks 32 --out eval/ppl-q4.json

python -m eval.perplexity compare --baseline eval/ppl-f16.json --candidate eval/ppl-q8.json
python -m eval.perplexity compare --baseline eval/ppl-f16.json --candidate eval/ppl-q4.json
```

`compare` prints `lossless` (±0.01 absolute, the acceleration gate) and
`acceptable_quant` (≤5% relative, the quantization gate), and exits 1 when both
fail.

---

## What this repo does **not** claim

* **No cross-request prefix cache inside `server/main.py`.** `llama-cpp-python`
  keeps one context and resets it per request, so the shim cannot memoise a
  prefix across requests. Use `llama-server` behind the same nginx and the same
  auth for that. This is a real architectural limit, not a configuration
  oversight.
* **No PagedAttention in llama.cpp.** The ×2–2.5 throughput number requires
  vLLM or SGLang; the export path to HF format is here so switching is a server
  flag change, not a re-export.
* **No draft model ships in this repo.** Speculative decoding needs one that
  shares the tokenizer; without it the flags are inert.
* **Q4_K_M is not lossless.** ≤5% perplexity is the gate, and on a 25M model
  that budget is tight enough that Q8_0 is the safer default for quality work.
