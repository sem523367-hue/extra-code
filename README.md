# Extra Code v1

Локальная ИИ-система: OpenAI-совместимый API, база знаний (RAG), веб-поиск,
погода, жёсткая идентичность и анти-цензорный слой — целиком на твоей машине,
без внешних сервисов. Работает на 4 ГБ VRAM.

- **Движок**: llama.cpp (`:8080`), модель 4B Instruct 2507, квант Q4_K_M, контекст 10240
- **API**: FastAPI (`:8000`), OpenAI-совместимый: `/v1/chat/completions`, стриминг, ключи, рейт-лимиты
- **Знания**: локальный корпус (код, доки, наука, выживание, языки, матан, физика, факты 2025–2026) → инвертированный индекс → чанки в промпт (`sources` в ответе)
- **Интернет**: явный веб-поиск + живая погода (Open-Meteo), включаются автоматически на вопросы про погоду/новости/даты
- **Идентичность**: модель представляется как Extra Code v1 во всех слоях (шаблон, промпт, скраббер) — независимо от того, что под капотом
- **Без цензуры**: запрос — это запрос; серверная стена ровно одна (контент с несовершеннолетними), всё остальное отвечает полностью; отказные первые строки тихо перегенерируются

Полная документация API: **[docs/API.md](docs/API.md)** · Инструкция для ИИ-агентов: **[docs/AI-AGENT.md](docs/AI-AGENT.md)**

---

## Русский — установка с нуля

**Требования:** Windows 10/11, Python 3.11+, ~8 ГБ на диске, GPU с 4 ГБ VRAM (Vulkan) или CPU (медленнее, но работает).

1. **Клонируй репозиторий** и открой папку в терминале (PowerShell).

2. **Поставь зависимости** (одна строка):
   ```powershell
   pip install fastapi "uvicorn[standard]" python-dotenv
   ```

3. **Скачай модель** (~2.3 ГБ, докачка работает, зеркала переключаются сами):
   ```powershell
   python data\download_model.py
   ```
   Флаг `--3b` ставит лёгкий откатный вариант (~2.0 ГБ). Файл ляжет в `models\` под своим именем.

4. **Положи llama.cpp**: скачай готовый `llama-server.exe` для Windows со страницы релизов [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/releases) (сборка **vulkan** — для AMD/Intel GPU, обычная — для CPU) и распакуй так, чтобы получилось:
   ```
   tools\llama.cpp-vulkan\llama-server.exe   (GPU)
   tools\llama.cpp\llama-server.exe          (CPU, если GPU-нет)
   ```

5. **Запусти**:
   ```powershell
   .\start.ps1
   ```
   Скрипт сам выберет модель (4B → 3B), размер контекста под неё и движок. Остановка — `.\stop.ps1`.

**Как пользоваться:**
- Консоль: `.\ask.bat "вопрос"` — полный пайплайн (KB + веб + источники)
- Живой чат в консоли: `models\chat-console.bat`
- API: `http://127.0.0.1:8000/v1`, ключ `local-dev-key` (примеры — в docs/API.md, работает стандартный `openai`-клиент Python)
- Сырой движок без фильтров: `http://127.0.0.1:8080`

**База знаний** (опционально, но рекомендуется): фетчеры в `data\fetch_*.py` собирают корпус с нуля (код, доки, наука, выживание, факты 2025–2026), затем:
```powershell
python -m server.kb --build
```
Индекс сжат gzip'ом, лежит в `data\kb\`. Без него система работает — просто без локальных знаний.

---

## English — install from zero

**Requirements:** Windows 10/11, Python 3.11+, ~8 GB disk, a 4 GB VRAM GPU (Vulkan) or CPU (slower).

1. **Clone** the repo, open the folder in PowerShell.

2. **Install dependencies:**
   ```powershell
   pip install fastapi "uvicorn[standard]" python-dotenv
   ```

3. **Download the model** (~2.3 GB, resumable, mirrors auto-switch):
   ```powershell
   python data\download_model.py
   ```
   `--3b` installs the lighter fallback (~2.0 GB).

4. **Get llama.cpp**: download a Windows `llama-server.exe` from [ggml-org/llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) (**vulkan** build for AMD/Intel GPUs, regular for CPU) and unpack as:
   ```
   tools\llama.cpp-vulkan\llama-server.exe   (GPU)
   tools\llama.cpp\llama-server.exe          (CPU fallback)
   ```

5. **Start:**
   ```powershell
   .\start.ps1
   ```
   Model choice, context size and engine are auto-detected. Stop with `.\stop.ps1`.

**Usage:** `.\ask.bat "question"` (full pipeline) · live console chat `models\chat-console.bat` · API at `http://127.0.0.1:8000/v1`, key `local-dev-key` (standard Python `openai` client works) · raw engine at `:8080`.

**Knowledge base** (optional): run the fetchers in `data\fetch_*.py`, then `python -m server.kb --build`. The system works without it — just no local corpus.

---

## 中文 — 从零安装

**要求：** Windows 10/11、Python 3.11+、约 8 GB 磁盘、4 GB 显存的 GPU（Vulkan）或纯 CPU（较慢）。

1. **克隆**仓库，用 PowerShell 打开文件夹。

2. **安装依赖：**
   ```powershell
   pip install fastapi "uvicorn[standard]" python-dotenv
   ```

3. **下载模型**（约 2.3 GB，支持断点续传，镜像自动切换）：
   ```powershell
   python data\download_model.py
   ```
   `--3b` 安装较小的备用模型（约 2.0 GB）。

4. **获取 llama.cpp**：从 [ggml-org/llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) 下载 Windows 版 `llama-server.exe`（AMD/Intel 显卡选 **vulkan** 版，CPU 选普通版），解压为：
   ```
   tools\llama.cpp-vulkan\llama-server.exe   （GPU）
   tools\llama.cpp\llama-server.exe          （CPU 备用）
   ```

5. **启动：**
   ```powershell
   .\start.ps1
   ```
   模型选择、上下文长度、引擎检测全部自动完成。用 `.\stop.ps1` 停止。

**使用：** `.\ask.bat "问题"`（完整管线）· 控制台对话 `models\chat-console.bat` · API `http://127.0.0.1:8000/v1`，密钥 `local-dev-key`（Python `openai` 客户端直接可用）· 原始引擎 `:8080`。

**知识库**（可选）：运行 `data\fetch_*.py` 系列脚本抓取语料，然后 `python -m server.kb --build`。没有它系统照常工作，只是没有本地语料。

---

## Структура / Structure / 结构

```
extra-code/
├─ server/            FastAPI-сервер: main (API, ретраи), engine, kb (индекс),
│                     web (поиск), weather, identity (скраббер), polish, schemas
├─ tools/             extra-code-system.txt (промпт), chat-template.jinja,
│                     battery.py (проверки), llama.cpp*/ (движки, не в репо)
├─ data/              download_model.py, fetch_*.py (корпус), kb/ (индекс, gzip),
│                     corpus/ (не в репо, собирается фетчерами)
├─ models/            *.gguf (не в репо, качает download_model.py)
├─ docs/              API.md, AI-AGENT.md
├─ start.ps1/stop.ps1 запуск/остановка всего стека одной командой
└─ ask.bat            вопрос из консоли с полным пайплайном
```

**Чего нет в репозитории и почему:** веса GGUF (лимит GitHub 100 МБ на файл — качаются скриптом `data/download_model.py`), бинарники llama.cpp (официальные релизы), корпус данных (собирается фетчерами за один запуск). Всё остальное — целиком.

**Железо, на котором собрано и проверено:** RX 580 4 ГБ (Vulkan), Pentium G4560, 32 ГБ RAM — старт ~11 с, контекст 10240, ~3.5 ГБ VRAM.
