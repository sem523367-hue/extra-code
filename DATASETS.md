# Датасеты для extra-code

Всё, что помечено **сконвертировано**, проверено `python -m tools.verify_sft`
на реальном токенизаторе проекта. Размеры сняты с HF API 21.11.2026.

Конвертер: `python data/from_hf.py --parquet <file> --preset alpaca|pair --out data/sft/<name>.jsonl`
Корпус:    `python -m data.corpus_from_parquet <file> --column text --out data/corpus/<name>.txt`
Проверка:  `python -m tools.verify_sft data/sft/<name>.jsonl`

## Что уже на диске и готово к тренировке

| Файл | Источник | Строк | Токенов | Supervised |
|---|---|---|---|---|
| `data/sft/hf_python.jsonl` | python_code_instructions_18k | 6 000 | 1 012 098 | 709 966 |
| `data/sft/hf_math.jsonl` | MathInstruct | 4 000 | 766 431 | 414 920 |
| `data/sft/hf_sciq.jsonl` | sciq | 3 000 | 157 867 | 17 717 |
| `data/sft/hf_chem.jsonl` | ChemistryQA | 1 981 | 153 396 | 23 036 |
| `data/sft/hf_chem_camel.jsonl` | camel-ai/chemistry | 792 | 185 863 | 165 880 |
| **итого SFT** | | **15 773** | **2 275 655** | **1 331 519** |
| `data/corpus/history.txt` | k-12-world-history-100k | 20 000 док. | ~35M | претрейн |

Для сравнения: весь прежний SFT-набор проекта (`data/sft/train.jsonl`) — 7 771 строка.
Набор выше — 1.33M supervised-токенов живых знаний по пяти предметам.

## Математика

| Датасет | Размер | Строк | Схема | Статус |
|---|---|---|---|---|
| `TIGER-Lab/MathInstruct` | 94.5 МБ | 262 039 | `instruction`, `output`, `source` | **на диске + SFT 4 000 строк** |
| `meta-math/MetaMathQA` | 377 МБ | 395 000 | `query`, `response` | — |
| `AI-MO/NuminaMath-CoT` | 1.18 ГБ | 860 000 | `problem`, `solution` | — |
| `EleutherAI/hendrycks_math` | 4.7 МБ | 12 500 | `problem`, `solution`, `level` | — |
| `HuggingFaceTB/cosmopedia / data/auto_math_text` | 4.25 ГБ | — | сырой текст (претрейн) | — |

Готово к заливке:
```powershell
python data\from_hf.py --parquet data\raw\math\TIGER-Lab__MathInstruct_0.parquet `
  --instruction instruction --output output --out data\sft\hf_math.jsonl --limit 8000
```

## Программирование

| Датасет | Размер | Строк | Схема | Статус |
|---|---|---|---|---|
| `iamtarun/python_code_instructions_18k_alpaca` | 10.8 МБ | 18 612 | `instruction`, `input`, `output` | **на диске**, SFT: 6 000 строк / 710k supervised токенов |
| `bigcode/self-oss-instruct-sc2-exec-filter-50k` | 86 МБ | 50 000 | `messages`-подобная | — |
| `ise-uiuc/Magicoder-OSS-Instruct-75K` | 194 МБ | 75 000 | chat | — |
| `HuggingFaceTB/smollm-corpus / python-edu` | 614 МБ | — | сырой Python (претрейн) | — |
| `codeparrot/codeparrot-clean` | 12.2 ГБ | — | сырой код (претрейн) | — |
| `nvidia/OpenCodeReasoning` | 9.28 ГБ | 548k | reasoning-трейсы | избыточен для <100M |

## Русский язык

| Датасет | Размер | Строк | Схема | Статус |
|---|---|---|---|---|
| `Den4ikAI/russian_instructions_2` | 222 МБ | 200 000 | alpaca | — |
| `Den4ikAI/russian_dialogues` | 1.36 ГБ | 1 100 000 | диалоги | — |
| `deepvk/ru-HNP` | 85 МБ | — | новости (претрейн) | — |
| `wikimedia/wikipedia / 20231101.ru` | 4.65 ГБ | 1 920 000 | сырой текст | — |
| `HuggingFaceFW/fineweb-2 / rus_Cyrl` | 1.9 ТБ (441 parquet) | ~100M | сырой веб | качать только шардами |

Русская Википедия — лучший претрейн-корпус под 25M–500M русскую модель:
```powershell
# список шардов: https://huggingface.co/api/datasets/wikimedia/wikipedia/tree/main/20231101.ru
$u = "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.ru/train-00000-of-00021.parquet"
Invoke-WebRequest -Uri $u -OutFile data\raw\ru\wiki_00000.parquet
```

## История

| Датасет | Размер | Строк | Схема | Статус |
|---|---|---|---|---|
| `sujitpandey/k-12-world-history-100k` | 212 МБ | 99 986 | `text`, `subject`, `grade` | **на диске + корпус 20 000 док. / 133 МБ** |
| `alenaa/russian_history_book` | 2.2 МБ | — | сырой текст | — |
| `cais/mmlu` (subset `high_school_world_history`) | 257 МБ (весь) | 238 в сабсете | тест | только eval |

k-12 history — сырой текст → в претрейн-корпус, не в SFT:
```powershell
python -m data.corpus_from_parquet data\raw\history\sujitpandey__k-12-world-history-100k_0.parquet --column text --out data\corpus\history.txt
```

## Химия

| Датасет | Размер | Строк | Схема | Статус |
|---|---|---|---|---|
| `avaliev/ChemistryQA` | 9.8 МБ | 3 430 | `question`, `answer` | **на диске + SFT 1 981 строка** |
| `camel-ai/chemistry` | 20.7 МБ (zip, 2 891 json) | 2 891 | `message_1`, `message_2` | **на диске + SFT 792 строки** |

```powershell
python data\from_hf.py --parquet data\raw\chem\avaliev__ChemistryQA_0.parquet `
  --prompt-field question --response-field answer --out data\sft\hf_chem.jsonl --limit 3000
```

## Общие знания

| Датасет | Размер | Строк | Схема | Статус |
|---|---|---|---|---|
| `allenai/sciq` | 3.8 МБ | 11 679 | `question`, `correct_answer`, `support` | **на диске + SFT 3 000 строк** |
| `TIGER-Lab/MMLU-Pro` | 4.0 МБ | 12 032 | `question`, `options`, `answer` | **на диске** (test) |
| `cais/mmlu` | 257 МБ | 14 042 | тест 57 предметов | — |
| `allenai/ai2_arc` | 1.2 МБ | 3 370 | тест | — |
| `roneneldan/TinyStories` | 7.3 ГБ | 2 100 000 | сырой текст | для <30M моделей |
| `HuggingFaceTB/cosmopedia` | 87.9 ГБ | 157M | синтетика: khanacademy 47 МБ, openstax 331 МБ, stanford 3.1 ГБ, wikihow 479 МБ | по subdir'ам |

sciq → SFT:
```powershell
python data\from_hf.py --parquet data\raw\general\allenai__sciq_0.parquet `
  --prompt-field question --response-field correct_answer --out data\sft\hf_sciq.jsonl --limit 5000
```

## Gated (нужно принять условия на странице + HF токен)

- `nampdn-ai/tiny-codes` — 1.6M строк, сделан специально под крошечные модели.
- `bigcode/the-stack-dedup` — 3.1 ТБ кода, вручную отфильтрованный.
- `Den4ikAI/ru_alpaca`, `Den4ikAI/ru_turbo_alpaca`, `bond005/ru_squad` и часть
  `ai-forever/*` — отдают 401 без аутентификации: `huggingface-cli login` снимает.

## Сколько нужно токенов (замер на этой машине, CPU, 4 потока)

Замерено 21.11.2026, AdamW, forward+backward, `bench.json`:

| Конфиг | Параметры | Скорость |
|---|---|---|
| текущий (384d/6L) | 25.27M | **171.9 ток/с** |
| 1024d/16L/4kv | 276M | **16.7 ток/с** |
| ~1280d/22L (экстрап.) | ~500M | ~9 ток/с |
| ~1536d/24L (экстрап.) | ~1B | ~4.5 ток/с |

Время претрейна по Чинчилле (20 токенов на параметр), чистая арифметика:

| Параметры | Токенов нужно | Время на этом CPU |
|---|---|---|
| 25M | 500M | **33.7 дня** (реально с повторами: 13 дней) |
| 276M | 5.5B | 10.5 лет |
| 500M | 10B | 35 лет |
| 1B | 20B | 141 год |

**Вывод:** дотянуть СВОЮ модель до 500M–1B обучением с нуля на этой машине
нельзя — не «сложно», а арифметически. RX 580 4 ГБ тоже не вариант: для
обучения 500M нужно ~8 ГБ VRAM под веса+градиенты+Adam.

Рабочие пути:

1. **25M дотренировать** — 13–34 дня CPU, данные уже на диске.
2. **Инференс 1.5B** — Extra Code 2.5-Coder-1.5B уже скачан и работает с интернетом.
3. **Свои знания в чужую модель** — LoRA-финтюн готовой модели через
   `llama-cpp-python`/llama.cpp train: часы на CPU, не годы.
4. **С нуля до 500M–1B** — только облако: ~3e19 FLOPs для 500M×10B токенов,
   одна RTX 4090 справится за ~10 суток (~$200–250 аренды), 8×A100 за ~1.5 дня.

Сейчас в `data/train.bin` — **8.53M** токенов. На диске уже лежит ~320 МБ
сырых данных ≈ 80M токенов после токенизации.

