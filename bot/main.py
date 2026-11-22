"""Aiogram 3 bot: a thin client of the OpenAI-compatible API.

    python -m bot.main

The bot never loads the model. It keeps per-chat history in SQLite, streams
answers by editing one Telegram message, and rate-limits each user in memory.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from pathlib import Path

import httpx
from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ChatAction, ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from bot.api_client import ApiClient, ApiError
from bot.config import BotConfig
from bot.storage import ChatStore

log = logging.getLogger("extra_code.bot")

TELEGRAM_LIMIT = 4000
EDIT_INTERVAL_S = 1.2


class UserLimiter:
    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._events: dict[int, deque[float]] = defaultdict(deque)

    def check(self, user_id: int) -> float:
        """Returns 0 when allowed, otherwise seconds to wait."""
        now = time.monotonic()
        events = self._events[user_id]
        while events and now - events[0] >= 60.0:
            events.popleft()
        if len(events) >= self.per_minute:
            return max(0.5, 60.0 - (now - events[0]))
        events.append(now)
        return 0.0


def split_message(text: str, limit: int = TELEGRAM_LIMIT) -> list[str]:
    text = text.strip()
    if not text:
        return ["(пустой ответ)"]
    chunks: list[str] = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(text[:cut])
        text = text[cut:].lstrip()
    chunks.append(text)
    return chunks


def build_dispatcher(config: BotConfig) -> tuple[Bot, Dispatcher, ApiClient, ChatStore]:
    bot = Bot(token=config.token, default=DefaultBotProperties(parse_mode=None))
    dp = Dispatcher()
    router = Router()
    api = ApiClient(config.api_base_url, config.api_key, timeout_s=config.request_timeout_s)
    store = ChatStore(config.db_path, max_messages=config.max_context_messages)
    limiter = UserLimiter(config.rate_limit_per_min)

    async def safe_edit(placeholder: Message, text: str) -> None:
        body = text[:TELEGRAM_LIMIT]
        if not body:
            return
        try:
            await placeholder.edit_text(body)
        except TelegramBadRequest as exc:
            if "message is not modified" not in str(exc).lower():
                log.debug("edit failed: %s", exc)

    @router.message(CommandStart())
    async def on_start(message: Message) -> None:
        await message.answer(
            f"Extra code на линии. Модель: {config.default_model}.\n"
            "Просто напиши сообщение. /reset — очистить контекст, /stats — лимиты."
        )

    @router.message(Command("help"))
    async def on_help(message: Message) -> None:
        await message.answer(
            "/start — приветствие\n"
            "/reset — сбросить историю чата\n"
            "/model — список моделей на сервере\n"
            "/stats — состояние сервера и лимиты\n"
            "Обычный текст — запрос к модели."
        )

    @router.message(Command("reset"))
    async def on_reset(message: Message) -> None:
        store.clear(message.chat.id)
        await message.answer("Контекст очищен.")

    @router.message(Command("model"))
    async def on_model(message: Message) -> None:
        try:
            models = await api.models()
        except (ApiError, httpx.HTTPError) as exc:
            log.warning("model list failed: %s", exc)
            await message.answer("Сервер не отдал список моделей. Попробуй позже.")
            return
        await message.answer("Модели: " + ", ".join(models))

    @router.message(Command("stats"))
    async def on_stats(message: Message) -> None:
        lines = [
            f"модель: {config.default_model}",
            f"контекст: {config.max_context_messages} сообщений",
            f"лимит: {config.rate_limit_per_min} запросов/мин",
            f"max_tokens: {config.max_tokens}",
        ]
        try:
            health = await api.health()
            lines.append(f"сервер: {health.get('status')} / {health.get('engine', {}).get('backend')}")
            lines.append(f"uptime: {health.get('uptime_s')}s, inflight: {health.get('inflight')}")
        except (ApiError, httpx.HTTPError) as exc:
            lines.append(f"сервер недоступен: {type(exc).__name__}")
        await message.answer("\n".join(lines))

    def addressed(message: Message, bot: Bot) -> bool:
        if message.chat.type == "private":
            return True
        if config.group_mode == "off":
            return False
        if config.group_mode == "all":
            return True
        if message.reply_to_message and message.reply_to_message.from_user:
            return message.reply_to_message.from_user.id == bot.id
        username = bot.username
        return bool(username and f"@{username.lower()}" in (message.text or "").lower())

    @router.message(F.text)
    async def on_text(message: Message, bot: Bot) -> None:
        if message.from_user is None or message.from_user.is_bot:
            return
        if not addressed(message, bot):
            return
        wait = limiter.check(message.from_user.id)
        if wait > 0:
            await message.answer(f"Слишком часто. Подожди {int(wait)} с.")
            return

        text = (message.text or "").strip()
        if bot.username:
            text = text.replace(f"@{bot.username}", "").strip()
        if not text:
            return

        history = store.history(message.chat.id)
        messages = [{"role": "system", "content": config.system_prompt}, *history, {"role": "user", "content": text}]
        await bot.send_chat_action(message.chat.id, ChatAction.TYPING)

        placeholder: Message | None = None
        buffer: list[str] = []
        last_edit = 0.0

        async def on_delta(piece: str) -> None:
            nonlocal last_edit, placeholder
            buffer.append(piece)
            if not config.stream_edits:
                return
            now = time.monotonic()
            if now - last_edit < EDIT_INTERVAL_S:
                return
            last_edit = now
            if placeholder is None:
                placeholder = await message.answer("…")
            await safe_edit(placeholder, "".join(buffer))

        try:
            result = await api.chat(
                messages,
                model=config.default_model,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                stream=config.stream_edits,
                on_delta=on_delta if config.stream_edits else None,
            )
        except ApiError as exc:
            log.warning("api error status=%s code=%s: %s", exc.status, exc.code, exc)
            await message.answer("Модель сейчас недоступна. Детали записаны в лог сервера.")
            return
        except (httpx.HTTPError, asyncio.TimeoutError) as exc:
            log.warning("transport error: %s", exc)
            await message.answer("Не дозвонился до API. Проверь, что сервер запущен.")
            return

        answer = result.text or "".join(buffer)
        chunks = split_message(answer)
        if placeholder is not None:
            await safe_edit(placeholder, chunks[0])
            for chunk in chunks[1:]:
                await message.answer(chunk)
            # If the streamed draft was truncated, replace with the full first chunk.
            await safe_edit(placeholder, chunks[0])
        else:
            for chunk in chunks:
                await message.answer(chunk)

        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": answer})
        store.save(message.chat.id, history)

    dp.include_router(router)
    return bot, dp, api, store


async def run(config: BotConfig) -> None:
    bot, dp, api, _store = build_dispatcher(config)
    try:
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        await api.close()
        await bot.session.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    config = BotConfig.from_env()
    asyncio.run(run(config))


if __name__ == "__main__":
    main()
