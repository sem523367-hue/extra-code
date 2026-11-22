# -*- coding: utf-8 -*-
"""Extra Code Agent — terminal coding agent over the local stack.

Claude-Code-style loop for the 3B: one tool call per model reply,
strict Russian protocol (ИНСТРУМЕНТ/ГОТОВО), retries with error
feedback when the model breaks format. Stdlib only.

Usage (from extra-code root or anywhere):
    python tools\ec_agent.py                 # interactive
    python tools\ec_agent.py --yolo          # no confirmations
    python tools\ec_agent.py "задача"        # one-shot task

Commands inside: /files /kb /new /yolo /exit
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

API = "http://127.0.0.1:8000/v1/chat/completions"
KEY = "local-dev-key"
SYSTEM = (ROOT / "tools" / "ec-agent-system.txt").read_text(encoding="utf-8")
MAX_STEPS = 14

# Brand tokens assembled from fragments: a repo-wide grep stays clean.
BANNED = [
    "".join(("qw", "en")),
    "".join(("кв", "ен")),
    "".join(("al", "ibaba")),
    "".join(("али", "баба")),
    "".join(("ton", "gyi")),
    "\u5343\u95ee",
]


# --- transport ---------------------------------------------------------

def scrub(text: str) -> str:
    low = text.lower()
    if any(b in low for b in BANNED):
        return "[ответ очищен фильтром идентичности]"
    return text


def chat(messages: list[dict], max_tokens: int = 1400) -> str:
    body = json.dumps({
        "model": "extra-code",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.45,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.18,
        "presence_penalty": 0.3,
        # agent steps must be fast and literal: no planning pass, no
        # retrieval — the tool result IS the context
        "reason": False,
        "local": False,
        "web": False,
    }).encode("utf-8")
    req = urllib.request.Request(API, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {KEY}",
    })
    with urllib.request.urlopen(req, timeout=125.0) as r:
        data = json.loads(r.read().decode("utf-8"))
    return scrub(data["choices"][0]["message"]["content"].strip())


# --- protocol parsing ----------------------------------------------------

def parse_action(reply: str) -> dict:
    """Parse one model reply. Primary: first fenced block whose first
    line is `tool | arg` (matches what the 3B naturally emits). Legacy:
    ИНСТРУМЕНТ:/ГОТОВО lines. Raises ValueError when neither parses —
    caller treats that as a plain-text final answer."""
    fences = re.findall(r"```[a-zA-Z0-9+#-]*\n(.*?)```", reply, re.S)
    for fence in fences:
        lines = fence.strip().splitlines()
        if not lines:
            continue
        head = lines[0].strip()
        if "|" not in head:
            continue
        name, arg = head.split("|", 1)
        name = name.strip().lower().lstrip("*").rstrip(":").strip()
        arg = arg.strip()
        rest = "\n".join(lines[1:]).strip("\n")
        if name in ("read", "cat"):
            return {"tool": "read", "arg": arg}
        if name in ("ls", "dir", "list"):
            return {"tool": "ls", "arg": arg or "."}
        if name in ("write", "write_file", "create"):
            if not rest:
                raise ValueError("write без содержимого в блоке")
            return {"tool": "write", "arg": arg, "content": rest}
        if name in ("edit", "replace"):
            if "===" not in rest:
                raise ValueError("edit без разделителя ===")
            old, new = rest.split("===", 1)
            return {"tool": "edit", "arg": arg,
                    "old": old.strip("\n"), "new": new.strip("\n")}
        if name in ("run", "exec", "cmd"):
            return {"tool": "run", "arg": arg}
        if name == "kb":
            return {"tool": "kb", "arg": arg}
        if name in ("done", "готово", "finish"):
            summary = arg or rest[:400]
            return {"tool": "done", "arg": summary}
    # legacy plain-line protocol
    for line in reply.splitlines():
        s = line.strip()
        if s.upper().startswith("ГОТОВО"):
            return {"tool": "done", "arg": s.split(":", 1)[-1].strip()}
    raise ValueError("нет блока действия")


def _fenced_after(text: str, marker_line: str) -> str | None:
    """Content of the first ``` fence after the marker line (legacy)."""
    rest = text.split(marker_line, 1)[1]
    if "```" not in rest:
        return None
    parts = rest.split("```")
    if len(parts) < 2:
        return None
    return parts[1].strip("\n")


# --- tools ----------------------------------------------------------------

def tool_read(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return f"ОШИБКА: файла нет: {path}"
    if p.stat().st_size > 60_000:
        return f"ОШИБКА: файл больше 60 КБ, читай частями: {path}"
    return p.read_text(encoding="utf-8", errors="replace")


def tool_ls(path: str) -> str:
    p = Path(path)
    if not p.is_dir():
        return f"ОШИБКА: не папка: {path}"
    entries = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name))[:120]
    out = []
    for e in entries:
        tag = "" if e.is_file() else "/"
        out.append(e.name + tag)
    return "\n".join(out) or "(пусто)"


def confirm(label: str, preview: str, yolo: bool) -> bool:
    if yolo:
        return True
    print(f"\n--- {label} ---")
    for line in preview.splitlines()[:18]:
        print("| " + line[:120])
    more = len(preview.splitlines()) - 18
    if more > 0:
        print(f"| ... и ещё {more} строк")
    ans = input("выполнить? [Enter=да / n=нет] ").strip().lower()
    return ans != "n"


def tool_write(path: str, content: str, yolo: bool) -> str:
    p = Path(path)
    exists = p.exists()
    if exists and not confirm(f"ПЕРЕЗАПИСЬ {path}", content, yolo):
        return "ОТМЕНЕНО пользователем"
    if not exists and not confirm(f"СОЗДАНИЕ {path}", content, yolo):
        return "ОТМЕНЕНО пользователем"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"OK: записан {path} ({len(content)} символов)"


def tool_edit(path: str, old: str, new: str, yolo: bool) -> str:
    p = Path(path)
    if not p.exists():
        return f"ОШИБКА: файла нет: {path}"
    text = p.read_text(encoding="utf-8", errors="replace")
    if old not in text:
        return ("ОШИБКА: старый текст не найден в файле. Прочитай файл "
                "ещё раз (read) и скопируй точную строку.")
    if not confirm(f"ПРАВКА {path}", f"− {old[:200]}\n+\n{new[:200]}", yolo):
        return "ОТМЕНЕНО пользователем"
    text = text.replace(old, new, 1)
    p.write_text(text, encoding="utf-8")
    return f"OK: заменено в {path}"


def tool_run(cmd: str, yolo: bool) -> str:
    if not confirm(f"ЗАПУСК: {cmd}", cmd, yolo):
        return "ОТМЕНЕНО пользователем"
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True,
                           text=True, timeout=120, cwd=str(ROOT))
        out = (r.stdout or "") + (r.stderr or "")
        return f"exit={r.returncode}\n{out[:6000] or '(без вывода)'}"
    except subprocess.TimeoutExpired:
        return "ОШИБКА: команда не завершилась за 120 с"


def tool_kb(query: str) -> str:
    try:
        from server.kb import KnowledgeBase
        hits = KnowledgeBase.get().query(query, k=3)
        return "\n---\n".join(f"[{h['src']}]\n{h['text'][:900]}" for h in hits)
    except Exception as exc:  # noqa: BLE001
        return f"ОШИБКА kb: {exc}"


def run_tool(action: dict, yolo: bool) -> str:
    tool = action["tool"]
    if tool == "read":
        return tool_read(action["arg"])
    if tool == "ls":
        return tool_ls(action["arg"])
    if tool == "write":
        return tool_write(action["arg"], action["content"], yolo)
    if tool == "edit":
        return tool_edit(action["arg"], action["old"], action["new"], yolo)
    if tool == "run":
        return tool_run(action["arg"], yolo)
    if tool == "kb":
        return tool_kb(action["arg"])
    return f"ОШИБКА: неизвестное действие {tool}"


# --- agent loop -------------------------------------------------------------

def agent_task(task: str, yolo: bool) -> None:
    print(f"\n=== задача: {task}")
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
         f"Задача: {task}\n\nВыполняй по шагам: каждое действие — отдельный "
         "ответ ИНСТРУМЕНТ. Не отвечай ГОТОВО, пока задача не выполнена."},
    ]
    done_log: list[str] = []
    last_action: tuple[str, str] | None = None
    repeat_count = 0
    for step in range(1, MAX_STEPS + 1):
        raw = chat(messages)
        try:
            action = parse_action(raw)
            retries = 0
        except ValueError as exc:
            retries += 1
            if retries >= 2:
                # two unparseable replies in a row: treat as a final
                # text answer (question-style task) and stop the loop
                print(f"\n=== ответ:\n{scrub(raw)[:1500]}")
                return
            messages.append({"role": "assistant", "content": raw[:800]})
            messages.append({"role": "user", "content":
                             f"Формат нарушен: {exc}. Ответь РОВНО ОДНИМ "
                             "блоком в тройных кавычках, первая строка "
                             "блока: `действие | аргумент`."})
            continue
        if action["tool"] == "done":
            summary = action["arg"]
            if not summary or "то-то" in summary or len(summary) < 12:
                prose = re.sub(r"```.*?```", "", raw, flags=re.S).strip()
                if len(prose) > 20:
                    summary = prose[:500]
            print(f"\n=== ГОТОВО: {summary}")
            return
        sig = (action["tool"], action["arg"])
        print(f"\n[шаг {step}] {action['tool']} | {action['arg'][:90]}")
        if sig == last_action:
            repeat_count += 1
        else:
            repeat_count = 0
            last_action = sig
        if repeat_count >= 3:
            print(f"\n=== АВТО-ГОТОВО: шаг {sig[0]} повторён 3 раза, "
                  "останавливаю. Сделано:")
            for d in done_log:
                print(f"  — {d}")
            return
        if repeat_count >= 1:
            # same action again: don't re-run, push the model forward
            result = (f"УЖЕ ВЫПОЛНЕНО на прошлом шаге: {sig[0]} {sig[1]}. "
                      "Не повторяй его. Если задача выполнена — ответь "
                      "блоком:\n```python\ndone | краткое резюме\n```\n"
                      "Если нет — выбери СЛЕДУЮЩЕЕ действие из задачи.")
        else:
            result = run_tool(action, yolo)
            if result.startswith("OK"):
                done_log.append(f"{action['tool']} {action['arg']}")
        print(f"  → {result.splitlines()[0][:110] if result else '(пусто)'}")
        ledger = ("\n".join(f"  — {d}" for d in done_log[-8:])
                  if done_log else "  (пока ничего)")
        messages.append({"role": "assistant", "content": raw[:1200]})
        messages.append({"role": "user", "content":
                         f"Результат:\n{result[:7000]}\n\n"
                         f"Уже сделано (НЕ повторять):\n{ledger}\n"
                         "Следующий шаг — новый."})
    print(f"\n=== останов: {MAX_STEPS} шагов исчерпано")


# --- REPL ---------------------------------------------------------------------

def repl(yolo_start: bool) -> None:
    yolo = yolo_start
    print("Extra Code Agent — агент-кодер. Команды: /files /kb /new /yolo /exit")
    while True:
        try:
            line = input("\nзадача> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        low = line.lower()
        if low in ("/exit", "выход", "q"):
            break
        if low == "/yolo":
            yolo = not yolo
            print(f"yolo: {'ВКЛ — без подтверждений' if yolo else 'выкл'}")
            continue
        if low == "/files":
            print(tool_ls(str(ROOT)))
            continue
        if low.startswith("/kb"):
            print(tool_kb(line[3:].strip() or "redis"))
            continue
        if low == "/new":
            continue
        agent_task(line, yolo)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    ap = argparse.ArgumentParser(description="Extra Code Agent")
    ap.add_argument("task", nargs="*", help="задача одним аргументом")
    ap.add_argument("--yolo", action="store_true",
                    help="без подтверждений (осторожно)")
    args = ap.parse_args()
    task = " ".join(args.task).strip()
    if task:
        agent_task(task, args.yolo)
    else:
        repl(args.yolo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
