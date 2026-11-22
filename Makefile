# Extra code — compact local LLM, end to end.
#
# Windows note: these recipes are POSIX-flavoured. On PowerShell run the same
# `python -m ...` commands directly; `make` is only a shortcut.

PY      ?= python
DATA    ?= data
TOK     ?= tokenizer
RUNS    ?= runs
CONFIG  ?= model/config.json

.DEFAULT_GOAL := help
.PHONY: help install install-train data prepare tokenize sft pack pretrain sft-train \
        export api bot test smoke bench ppl equivalence clean clean-data distclean

help:
	@echo "install      pip install -e .[dev]"
	@echo "install-train  pip install -e .[train,export]"
	@echo "data         collect the corpus into data/raw"
	@echo "prepare      clean, dedup, split into data/clean"
	@echo "tokenize     train the 32k BPE tokenizer"
	@echo "sft          build data/sft/{train,val}.jsonl"
	@echo "pack         tokenize the corpus into data/{train,val}.bin"
	@echo "pretrain     base model -> runs/base/final/model.pt"
	@echo "sft-train    instruction tuning -> runs/sft/final/model.pt"
	@echo "export       checkpoint -> export/hf (then export/to_gguf.sh for GGUF)"
	@echo "api          uvicorn on 127.0.0.1:8000"
	@echo "bot          aiogram bot against API_BASE_URL"
	@echo "test         pytest"
	@echo "smoke        full synthetic pipeline (CPU, ~1 min)"
	@echo "bench        latency/throughput/prefix-cache benchmark"
	@echo "ppl          perplexity of runs/sft/final/model.pt"
	@echo "equivalence  greedy output diff between two endpoints"

install:
	$(PY) -m pip install -e ".[dev]"

install-train:
	$(PY) -m pip install -e ".[train,export]"

data:
	$(PY) -m data.collect.run --out $(DATA)/raw --limit 1000 --sources all

prepare:
	$(PY) -m data.prepare --raw $(DATA)/raw --out $(DATA)/clean --threshold 0.8

tokenize:
	$(PY) -m data.tokenizer --train $(DATA)/clean/train.jsonl.gz \
		--out $(TOK)/tokenizer.json --vocab-size 32000

sft:
	$(PY) -m data.make_sft --clean $(DATA)/clean --out $(DATA)/sft --per-task 4000

pack:
	$(PY) -m data.pack --clean $(DATA)/clean \
		--tokenizer $(TOK)/tokenizer.json --out $(DATA)

pretrain:
	$(PY) -m train.pretrain --config $(CONFIG) --data-dir $(DATA) \
		--out $(RUNS)/base --steps 50000 --batch-size 16 --grad-accum 8 \
		--block-size 512 --wandb extra-code

sft-train:
	$(PY) -m train.sft --model $(RUNS)/base/final/model.pt --config $(CONFIG) \
		--data $(DATA)/sft/train.jsonl --tokenizer $(TOK)/tokenizer.json \
		--out $(RUNS)/sft --epochs 3 --batch-size 4 --grad-accum 8 --max-len 512

export:
	$(PY) -m export.to_hf --checkpoint $(RUNS)/sft/final/model.pt \
		--tokenizer $(TOK) --out export/hf --dtype fp16

api:
	$(PY) -m uvicorn server.main:app --host 127.0.0.1 --port 8000 --workers 1

bot:
	$(PY) -m bot.main

test:
	$(PY) -m pytest tests -q

smoke:
	$(PY) tests/smoke_e2e.py

bench:
	$(PY) -m bench.benchmark --base-url http://127.0.0.1:8000 \
		--api-key "$$API_KEY" --prefix-repeat 5 --concurrency 4

ppl:
	$(PY) -m eval.perplexity torch --checkpoint $(RUNS)/sft/final/model.pt \
		--block-size 512 --stride 256 --max-windows 64 \
		--out eval/ppl-torch.json

equivalence:
	$(PY) -m eval.equivalence --a http://127.0.0.1:8000 --b http://127.0.0.1:8080 \
		--prompts eval/prompts.jsonl

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ runs .smoke

clean-data:
	rm -rf $(DATA)/raw $(DATA)/clean $(DATA)/*.bin $(DATA)/meta.json $(TOK)

distclean: clean clean-data
	rm -rf export/hf export/gguf models
