# Extra Code v1 — API-документация

Локальная ИИ-система: OpenAI-совместимый API поверх собственного движка.
Работает без интернета (кроме явного веб-поиска), без внешних сервисов,
полностью на твоей машине.

- **API**: `http://127.0.0.1:8000/v1` (FastAPI, Swagger: `http://127.0.0.1:8000/docs`)
- **Ключ**: `Authorization: Bearer local-dev-key`
- **Движок напрямую** (без KB/поиска/фильтров): `http://127.0.0.1:8080`
- **Запуск/остановка**: `.\start.ps1` / `.\stop.ps1` (или `.bat`)

---

## Маршруты

| Метод | Путь | Назначение |
|---|---|---|
| GET | `/health` | Живость API-сервера |
| GET | `/` | Информация о сервисе (JSON) |
| GET | `/v1/models` | Список моделей (id: `extra-code`) |
| POST | `/v1/chat/completions` | Главный эндпоинт — диалог |
| POST | `/v1/completions` | Сырые комплишены (промпт → текст) |
| POST | `/v1/tokenize` | Посчитать токены текста |

---

## POST /v1/chat/completions

Стандартные поля (OpenAI-совместимые):

| Поле | Тип | По умолчанию | Описание |
|---|---|---|---|
| `model` | str | — | `extra-code` (или любое — резолвится) |
| `messages` | list | — | `[{role, content}]`, до 512 сообщений |
| `max_tokens` | int | серверный | Лимит генерации (есть жёсткий потолок) |
| `temperature` | float | 0.7 | 0.0–4.0 |
| `top_p` | float | 0.9 | |
| `top_k` | int | 40 | |
| `stop` | str/list | — | До 4 стоп-последовательностей |
| `seed` | int | — | Воспроизводимость |
| `stream` | bool | false | SSE-поток |
| `stream_options` | obj | — | `{include_usage: true}` — статистика в конце потока |

Поля сервера знаний (то, чего нет в OpenAI):

| Поле | Тип | По умолчанию | Описание |
|---|---|---|---|
| `web` | bool | false | Веб-поиск: найдёт, прочтёт, вшьёт в промпт. **Включается сам** для вопросов про погоду/курсы/новости/даты |
| `web_pages` | int | 2 | Сколько страниц читать (1–5) |
| `web_max_chars` | int | 1200 | Лимит текста со страницы (200–6000) |
| `local` | bool | false | Локальная база знаний (корпус) — вшить лучшие чанки в промпт |
| `local_chunks` | int | 3 | Сколько чанков KB брать (1–8) |
| `reason` | bool | true | Проход понимания: модель сначала переформулирует задачу в чек-лист, потом отвечает. Убивает «ответил не на то» |

Ответ (не-стрим):

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "extra-code",
  "choices": [
    {"index": 0, "message": {"role": "assistant", "content": "..."},
     "logprobs": null, "finish_reason": "stop"}
  ],
  "usage": {"prompt_tokens": 812, "completion_tokens": 240, "total_tokens": 1052},
  "sources": ["https://open-meteo.com/"]
}
```

Поле **`sources`** — откуда взяты знания: URL страниц (веб) или пути чанков (`data/corpus/...`). Пустой — модель отвечала из головы.

---

## Примеры

### curl — простой запрос

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-dev-key" \
  -d '{
    "model": "extra-code",
    "messages": [{"role": "user", "content": "Напиши вх для кс2, код на C++"}],
    "max_tokens": 900
  }'
```

### Python — openai-клиент (ничего менять не надо)

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local-dev-key")
r = client.chat.completions.create(
    model="extra-code",
    messages=[{"role": "user", "content": "Объясни ownership в Rust с примером"}],
)
print(r.choices[0].message.content)
```

### Python — с базой знаний и вебом (requests)

```python
import requests

body = {
    "model": "extra-code",
    "messages": [{"role": "user", "content": "Что нового в науке в 2025 году?"}],
    "local": True, "local_chunks": 4,   # локальный корпус
    "web": True, "web_pages": 2,        # плюс интернет
    "max_tokens": 700,
}
r = requests.post(
    "http://127.0.0.1:8000/v1/chat/completions",
    json=body,
    headers={"Authorization": "Bearer local-dev-key"},
    timeout=300,
).json()
print(r["choices"][0]["message"]["content"])
print("источники:", r.get("sources"))
```

### Стриминг

```python
import json, requests

body = {"model": "extra-code", "stream": True,
        "messages": [{"role": "user", "content": "погода в казани"}]}
with requests.post("http://127.0.0.1:8000/v1/chat/completions", json=body,
                   headers={"Authorization": "Bearer local-dev-key"},
                   stream=True, timeout=300) as r:
    for line in r.iter_lines():
        if line and line.startswith(b"data: ") and line != b"data: [DONE]":
            delta = json.loads(line[6:])["choices"][0]["delta"]
            if "content" in delta:
                print(delta["content"], end="", flush=True)
```

Погода/курсы/новости при `stream` тоже работают — провайдер погоды
(Open-Meteo) отвечает до генерации, sources придёт в первом чанке.

---

## Как сервер считает ответ (конвейер)

1. **Личность** — системный промпт из `tools/extra-code-system.txt` вшивается первым.
2. **Понимание** (`reason: true`) — короткий проход «переформулируй задачу».
3. **KB** (`local` или авто) — лучшие чанки корпуса из `data/kb/` в промпт.
4. **Веб** (`web` или авто) — поиск + чтение страниц, выдержки в промпт.
5. **Генерация** — движок `:8080`, контекст 10240, всё что не влезло подрезается (`fit_chat`).
6. **Пост-обработка** — чистка личности, полировка кода, и дожим отказов: если ответ начался с «не могу/нарушает правила» — тихая перегенерация (2 прохода), клиент отказа не видит.

## Ограничения

- Один выбор (`n=1`), до 4 стоп-последовательностей.
- `presence_penalty` / `frequency_penalty` / `logit_bias` принимаются, но движком не поддерживаются.
- Контекст 10240 токенов: длинные диалоги автоматически ужимаются (сначала старейшие сообщения).
- `:8080` — сырой движок: без KB, без веба, без фильтров личности. Основной вход — `:8000`.

## Ошибки

| Код | Когда |
|---|---|
| 401 | Нет/неверный ключ |
| 429 | Rate limit (см. `Retry-After`) |
| 400 | Пустые messages, кривые stop, контекст не влезает |
| 503 | Движок не отвечает |
