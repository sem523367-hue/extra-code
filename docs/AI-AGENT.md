# AI-AGENT.md — самозапуск Extra Code v1 для ИИ-агентов

Инструкция для любого ИИ-агента (OpenCode, Claude Code, Codex, автономный
скрипт), которому дали машину с Windows и сказали «поставь и подними».
Выполнять по порядку, каждый шаг проверять, прежде чем идти дальше.

## Целевое состояние

- Движок llama.cpp на `http://127.0.0.1:8080` (health: `GET /health` → 200)
- API-сервер на `http://127.0.0.1:8000` (health: `GET /health` → 200)
- Ответ на тестовый вопрос через `/v1/chat/completions` приходит осмысленный
- Модель называет себя «Extra Code v1» на вопрос «кто ты»

## Шаги

1. **Проверь окружение:**
   - Windows 10/11, PowerShell доступен
   - `python --version` → 3.11+ (если нет — поставить с python.org)
   - ~8 ГБ свободного места на диске
   - GPU с 4 ГБ VRAM — плюс, не обязательство (CPU-режим работает)

2. **Склонируй/распакуй репозиторий** в любую папку без пробелов/кириллицы
   в пути (например `C:\extra-code`). Дальше все команды — из корня.

3. **Зависимости:**
   ```powershell
   pip install fastapi "uvicorn[standard]" python-dotenv
   ```

4. **Модель** (~2.3 ГБ; скрипт докачивает при обрыве, зеркала переключает сам):
   ```powershell
   python data\download_model.py
   ```
   Проверка: `models\extra-code-4b-2507-q4_k_m.gguf` существует, размер ~2.3 ГБ.

5. **Движок llama.cpp.** Варианты по приоритету:
   - есть сетевой доступ: скачать Windows-релиз с
     `https://github.com/ggml-org/llama.cpp/releases`
     (vulkan-x64 — если есть GPU, else обычный win-x64), распаковать так,
     чтобы `tools\llama.cpp-vulkan\llama-server.exe` (или `tools\llama.cpp\llama-server.exe`) существовал;
   - движка нет и скачать нельзя: стоп, сообщить — без него стек не поднимется.

6. **Запуск:**
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\start.ps1
   ```
   Скрипт ждёт health обоих портов сам (до 180 с движок, до 90 с API).
   Логи при сбое: `engine-err.log`, `server-err.log` в корне.

7. **Верификация (все пункты обязательны):**
   ```powershell
   # health
   curl.exe -s http://127.0.0.1:8000/health
   # ответ модели
   curl.exe -s http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer local-dev-key" -d "{\"model\":\"extra-code\",\"messages\":[{\"role\":\"user\",\"content\":\"кто ты?\"}],\"max_tokens\":100}"
   ```
   Ожидание: в ответе есть «Extra Code v1». Погода живьём:
   вопрос `"погода в москве"` → в поле `sources` — `open-meteo.com`.

8. **База знаний (опционально, после успешного шага 7):**
   ```powershell
   python -m server.kb --build
   ```
   Фетчеры корпуса (`data\fetch_*.py`) запускать только по прямому указанию
   владельца машины — они качают гигабайты из интернета.

## Правила для агента

- Не редактировать `tools/extra-code-system.txt` и шаблон
  `tools/extra-code-chat-template.jinja` — это идентичность системы.
- Не переименовывать файлы в `models\` — `start.ps1` ищет точные имена.
- Всё, что не получилось, фиксировать честно: какой шаг, какая ошибка,
  что уже проверено. Не докручивать несуществующие статусы.
- Остановка стека: `.\stop.ps1`. Перезапуск безопасен в любой момент.
