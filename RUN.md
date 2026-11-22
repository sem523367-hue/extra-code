# Как запустить и протестировать Extra code

Всё ниже — команды для PowerShell в папке проекта:

```powershell
cd "C:\Users\Sem52\Documents\Default Project\extra-code"
```

Четыре пути: **(0)** поднять всё одной командой (движок + API), **(1)** проверить
API на mock прямо сейчас, **(2)** поговорить с PyTorch-чекпоинтом, **(3)** поднять
GGUF в `llama-server` и гонять по нему сервер, бота и бенчмарки. Модель одна и та
же, меняется только движок.

---

## 0. Одна команда: движок + API (основной путь)

Поднимает `llama-server` с Extra Code (порт 8080) и extra-code API (порт 8000),
дожидается здоровья обоих, переиспользует уже живые порты:

```powershell
.\start.ps1      # или из cmd: start.bat
```

Готово:

- `.\ask.bat "вопрос"` — спросить (кириллица работает, интернет-поиск включён,
  локальная база знаний включена; `--no-web` / `--no-local` — отключить)
- OpenAI API: `http://127.0.0.1:8000/v1` (ключ `local-dev-key`)
- Swagger: <http://127.0.0.1:8000/docs>
- Движок напрямую (свой веб-интерфейс в браузере): <http://127.0.0.1:8080>
- Остановить всё: `.\stop.ps1` (или `stop.bat`)

Локальная база знаний: вся история, датасеты и пособия лежат в `data/kb`.
Пересобрать индекс после добавления материалов:

```powershell
python -m server.kb --build
python -m server.kb --query "формула серной кислоты"
```

---

## 1. Проверить сервер и протокол прямо сейчас (mock)

Mock-бэкенд не требует весов: он проверяет все эндпоинты, SSE, авторизацию,
очередь и лимиты.

```powershell
$env:BACKEND = "mock"
$env:API_KEY = "dev-key"
python -m uvicorn server.main:app --host 127.0.0.1 --port 8000
```

Во втором окне:

```powershell
$env:API_KEY = "dev-key"
Invoke-RestMethod http://127.0.0.1:8000/health

$body = @{
  model = "extra-code-30m"
  messages = @(@{ role = "user"; content = "привет" })
  max_tokens = 32
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/chat/completions `
  -Headers @{ Authorization = "Bearer $env:API_KEY" } `
  -ContentType "application/json" -Body $body
```

Swagger: <http://127.0.0.1:8000/docs>. Тесты протокола:

```powershell
python -m pytest tests -q
```

Без ключа сервер отвечает `401`, при переполнении очереди — `429` с
`Retry-After`, поток идёт как `data: {...}` и заканчивается `data: [DONE]`.

---

## 2. Поговорить с моделью (PyTorch, без GGUF)

Нужен чекпоинт. Он появляется после претрейна:

```powershell
$env:TRAIN_NUM_THREADS = "4"
$env:OMP_NUM_THREADS = "4"
python -m train.pretrain --config model/config.json --data-dir data `
  --out runs/base --steps 700 --batch-size 8 --grad-accum 2 --block-size 256 `
  --lr 3e-4 --warmup 100 --ckpt-interval 100 --eval-interval 200 `
  --dtype fp32 --device cpu
```

`runs/base/latest.pt` обновляется каждые 100 шагов, `runs/base/final/model.pt`
появляется в конце. Затем SFT на коде:

```powershell
python -m data.make_code_sft --out data/sft
python -m data.merge_sft --out data/sft/all.jsonl --extractive 2500

python -m train.sft --model runs/base/final/model.pt --config model/config.json `
  --data data/sft/all.jsonl --tokenizer tokenizer/tokenizer.json `
  --out runs/sft --epochs 1 --batch-size 4 --grad-accum 8 --max-len 256 `
  --lr 1e-5 --dtype fp32 --device cpu
```

Чат:

```powershell
python -m model.chat
# или короче, один вопрос:
"что такое GIL?" | python -m model.chat --once
```

Команды в REPL: `/reset`, `/system <текст>`, `/temp 0.5`, `/help`, `/exit`.
Параметры: `--temperature 0.7 --top-k 40 --top-p 0.9 --max-new-tokens 160`.
`--model` можно указать явно: `python -m model.chat --model runs/sft/final/model.pt`.

Проверка на наборе вопросов по коду (30 штук: Python, C++, Java, C, C#,
ассемблер, реверс, алгоритмы):

```powershell
python -m eval.code_check --model runs/sft/final/model.pt --temperature 0.2
```

Отчёт: `eval/code-check.json` — ответы, токены, TTFT, tok/s.

---

## 3. GGUF и llama.cpp (быстрый инференс)

Экспорт в HF-раскладку, которую понимает конвертер llama.cpp:

```powershell
python -m export.to_hf --checkpoint runs/sft/final/model.pt `
  --tokenizer tokenizer --out export/hf --dtype fp16
```

Конвертер берётся из репозитория llama.cpp (не пишем GGUF руками):

```powershell
python -m pip install gguf
Invoke-WebRequest https://raw.githubusercontent.com/ggml-org/llama.cpp/master/convert_hf_to_gguf.py `
  -OutFile tools/llama.cpp/convert_hf_to_gguf.py

python tools/llama.cpp/convert_hf_to_gguf.py export/hf `
  --outfile export/gguf/extra-code-f16.gguf --outtype f16

tools/llama.cpp/llama-quantize.exe export/gguf/extra-code-f16.gguf `
  export/gguf/extra-code-q8_0.gguf Q8_0
tools/llama.cpp/llama-quantize.exe export/gguf/extra-code-f16.gguf `
  export/gguf/extra-code-q4_k_m.gguf Q4_K_M
```

`export/to_gguf.py` делает то же одной командой:

```powershell
python -m export.to_gguf --hf export/hf --out export/gguf --llama-cpp tools/llama.cpp
```

Запуск движка с ускорениями из второго ТЗ:

```powershell
tools/llama.cpp-vulkan/llama-server.exe `
  -m export/gguf/extra-code-q4_k_m.gguf `
  --host 127.0.0.1 --port 8080 `
  -c 1024 --parallel 4 --cont-batching -fa --cache-ram 4096
```

Проверка, что FlashAttention реально включён: в логе загрузки строка
`flash_attn = 1`. `--cache-ram` требует свежую сборку; если флага нет —
обнови бинарники.

Сервер Extra code поверх GGUF:

```powershell
python -m pip install llama-cpp-python
$env:BACKEND = "llama_cpp"
$env:MODEL_PATH = "export/gguf/extra-code-q4_k_m.gguf"
$env:API_KEY = "dev-key"
python -m uvicorn server.main:app --host 127.0.0.1 --port 8000
```

`MODEL_PATH` по умолчанию — `model/extra-code-30m-q4_k_m.gguf`; проще
скопировать туда файл или задать переменную.

---

## 4. Бот

```powershell
$env:BOT_TOKEN = "123:ABC"
$env:API_BASE_URL = "http://127.0.0.1:8000"
$env:API_KEY = "dev-key"
python -m bot.main
```

Команды: `/start`, `/help`, `/reset`, `/model`, `/stats`. В группе бот
отвечает на упоминание или реплай (`GROUP_MODE=mention`). История — SQLite в
`bot/data/chats.db`, лимит `12` запросов в минуту на пользователя.

---

## 5. Бенчмарк и критерии приёмки второго ТЗ

```powershell
$env:BENCH_BASE_URL = "http://127.0.0.1:8000"
$env:BENCH_API_KEY = "dev-key"
python -m bench.benchmark --prefix-repeat 5 --concurrency 4 --out bench/results.json
```

Что смотреть:

| Критерий | Где | Ожидание |
|---|---|---|
| Prefix caching | `bench/results.json` → `prefix` | тёплый TTFT ≥5× ниже холодного |
| FlashAttention | лог `llama-server` | `flash_attn = 1` |
| Continuous batching | бенч при `--concurrency 4` | 0 ошибок, запросы не сериализуются |
| Perplexity не изменилась | `eval/perplexity.py compare` | разница ≤0.01 |
| Speculative decoding | `llama-server -md draft.gguf` | выход совпадает с базовым |

Perplexity:

```powershell
python -m eval.perplexity torch --checkpoint runs/sft/final/model.pt `
  --block-size 512 --stride 256 --max-windows 64 --out eval/ppl-torch.json
```

Для сравнения двух GGUF:

```powershell
python -m eval.perplexity compare --a export/gguf/extra-code-q4_k_m.gguf `
  --b export/gguf/extra-code-q8_0.gguf --data data/val.bin
```

---

## Что честно работает, а что нет

- **Работает и проверено:** сервер на mock (16 тестов), корпус 4752 документа /
  8.53M токенов, токенизатор 32k с roundtrip, претрейн на CPU, SFT-пайплайн,
  REPL, экспорт в HF, сборка llama.cpp (CPU и Vulkan).
- **Не проверено на этой машине:** CUDA-путь (карта AMD, CUDA нет),
  `llama-cpp-python` (нет MSVC для сборки), GGUF-конвертация до появления
  чекпоинта. Это не «должно работать» — это значит «не запускалось».
- **Скорость:** CPU-претрейн ~250 ток/с, 700 шагов ≈ 3 часа. Инференс на
  PyTorch CPU для 25M модели — единицы токенов в секунду; GGUF на Vulkan будет
  быстрее, но 4 ГБ VRAM и старая архитектура — не 40 ток/с, которые в ТЗ
  записаны для GPU с CUDA.
- **Качество:** 25M параметров и 0.3 эпохи претрейна — это короткие ответы по
  шаблону, а не рассуждения. SFT на коде даёт правильную форму ответа
  (объяснение → код → деталь), но факты модель будет путать. Это ограничение
  размера и бюджета, а не пайплайна.
