"""Runtime configuration, resolved once at process start.

Every knob comes from the environment so the same image runs on a laptop, a
CPU server and a GPU box without edits. Parsing is strict: a malformed value
kills the process at startup instead of surfacing as a 500 under load.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_PRESETS = {"chatml", "llama3", "plain"}

# One canonical prompt file, used by the API and by the CLI. Editing it
# changes the identity everywhere on the next restart.
_IDENTITY_FILE = Path(__file__).resolve().parent.parent / "tools" / "extra-code-system.txt"

_FALLBACK_IDENTITY = (
    "Ты — Extra Code v1, локальная ИИ-система команды Extra Code. "
    "Ты не продукт внешних компаний и не связан ни с одной из них. "
    "компаний. На вопрос «кто ты» отвечай только: Extra Code v1."
)


def _load_identity() -> str:
    try:
        text = _IDENTITY_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return _FALLBACK_IDENTITY
    return text or _FALLBACK_IDENTITY


def _str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _int(name: str, default: int, *, lo: int | None = None, hi: int | None = None) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if lo is not None and value < lo:
        raise ValueError(f"{name} must be >= {lo}, got {value}")
    if hi is not None and value > hi:
        raise ValueError(f"{name} must be <= {hi}, got {value}")
    return value


def _float(name: str, default: float, *, lo: float | None = None, hi: float | None = None) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}") from exc
    if lo is not None and value < lo:
        raise ValueError(f"{name} must be >= {lo}, got {value}")
    if hi is not None and value > hi:
        raise ValueError(f"{name} must be <= {hi}, got {value}")
    return value


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}")


@dataclass(frozen=True, slots=True)
class Settings:
    # --- model ---------------------------------------------------------
    backend: str = "llama_cpp"          # llama_cpp | openai | mock
    model_path: Path = Path("model/extra-code-30m-q4_k_m.gguf")
    model_name: str = "extra-code-30m"
    model_aliases: tuple[str, ...] = ("default",)
    n_ctx: int = 1024
    n_threads: int = 4
    n_gpu_layers: int = 0
    n_batch: int = 256
    chat_template: str = "chatml"
    mock_delay_ms: int = 0
    # --- openai backend (talks to a running llama-server / vLLM / etc) ---
    openai_base_url: str = "http://127.0.0.1:8080/v1"
    openai_api_key: str | None = None
    openai_model: str = "local"

    # --- sampling ------------------------------------------------------
    default_max_tokens: int = 768
    max_tokens_cap: int = 2048
    default_temperature: float = 0.8
    default_top_p: float = 0.9
    default_top_k: int = 40
    default_repeat_penalty: float = 1.1

    # --- identity --------------------------------------------------------
    # Prepended to the system turn of every chat request: the deployment
    # answers as Extra Code regardless of which engine carries the weights.
    # Empty string disables the override.
    identity_prompt: str = (
        "Ты — Extra Code v1, локальная ИИ-система команды Extra Code. "
        "Ты не продукт внешних компаний и не связан ни с одной из них. "
        "компаний. На вопрос «кто ты» отвечай только: Extra Code v1. "
        "Никогда не называй другое имя модели или компанию-разработчика."
    )

    # --- scheduling ----------------------------------------------------
    max_concurrent: int = 1             # llama.cpp serialises anyway
    queue_timeout_s: float = 10.0
    generation_timeout_s: float = 120.0
    shutdown_timeout_s: float = 30.0

    # --- auth ----------------------------------------------------------
    db_path: Path = Path("server/data/keys.db")
    api_key_pepper: str = "change-me-in-production"
    bootstrap_key: str | None = None
    default_rpm: int = 60
    default_tpm: int = 20_000
    default_key_max_tokens: int = 512

    # --- ops -----------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    log_prompts: bool = False

    # --- derived -------------------------------------------------------
    stops: tuple[str, ...] = field(default_factory=tuple)

    def stop_sequences(self) -> tuple[str, ...]:
        builtin = {
            "chatml": ("<|im_end|>",),
            "llama3": ("<|eot_id|>", "<|end_of_text|>"),
            "plain": ("\n###",),
        }[self.chat_template]
        return builtin + tuple(s for s in self.stops if s not in builtin)

    @classmethod
    def from_env(cls) -> "Settings":
        template = _str("CHAT_TEMPLATE", "chatml").lower()
        if template not in _PRESETS:
            raise ValueError(f"CHAT_TEMPLATE must be one of {sorted(_PRESETS)}, got {template!r}")
        backend = _str("BACKEND", "llama_cpp").lower()
        if backend not in {"llama_cpp", "openai", "mock"}:
            raise ValueError(f"BACKEND must be llama_cpp, openai or mock, got {backend!r}")
        return cls(
            backend=backend,
            model_path=Path(_str("MODEL_PATH", "model/extra-code-30m-q4_k_m.gguf")),
            model_name=_str("MODEL_NAME", "extra-code-30m"),
            model_aliases=tuple(
                a.strip() for a in _str("MODEL_ALIASES", "default").split(",") if a.strip()
            ),
            n_ctx=_int("N_CTX", 1024, lo=64, hi=131_072),
            n_threads=_int("N_THREADS", max(1, (os.cpu_count() or 4) // 2), lo=1, hi=256),
            n_gpu_layers=_int("N_GPU_LAYERS", 0, lo=-1, hi=10_000),
            n_batch=_int("N_BATCH", 256, lo=1, hi=8192),
            chat_template=template,
            mock_delay_ms=_int("MOCK_DELAY_MS", 0, lo=0, hi=5000),
            openai_base_url=_str("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1"),
            openai_api_key=os.environ.get("OPENAI_API_KEY") or None,
            openai_model=_str("OPENAI_MODEL", "local"),
            identity_prompt=_str("IDENTITY_PROMPT", _load_identity()),
            default_max_tokens=_int("DEFAULT_MAX_TOKENS", 256, lo=1, hi=65_536),
            max_tokens_cap=_int("MAX_TOKENS_CAP", 512, lo=1, hi=65_536),
            default_temperature=_float("DEFAULT_TEMPERATURE", 0.7, lo=0.0, hi=4.0),
            default_top_p=_float("DEFAULT_TOP_P", 0.9, lo=0.0, hi=1.0),
            default_top_k=_int("DEFAULT_TOP_K", 40, lo=0, hi=10_000),
            default_repeat_penalty=_float("DEFAULT_REPEAT_PENALTY", 1.1, lo=0.5, hi=4.0),
            max_concurrent=_int("MAX_CONCURRENT", 1, lo=1, hi=1024),
            queue_timeout_s=_float("QUEUE_TIMEOUT_S", 10.0, lo=0.0, hi=600.0),
            generation_timeout_s=_float("GENERATION_TIMEOUT_S", 120.0, lo=1.0, hi=3600.0),
            shutdown_timeout_s=_float("SHUTDOWN_TIMEOUT_S", 30.0, lo=0.0, hi=600.0),
            db_path=Path(_str("API_DB_PATH", "server/data/keys.db")),
            api_key_pepper=_str("API_KEY_PEPPER", "change-me-in-production"),
            bootstrap_key=os.environ.get("API_KEY") or None,
            default_rpm=_int("DEFAULT_RPM", 60, lo=1, hi=1_000_000),
            default_tpm=_int("DEFAULT_TPM", 20_000, lo=1, hi=10**9),
            default_key_max_tokens=_int("DEFAULT_KEY_MAX_TOKENS", 512, lo=1, hi=65_536),
            host=_str("HOST", "127.0.0.1"),
            port=_int("PORT", 8000, lo=1, hi=65_535),
            log_level=_str("LOG_LEVEL", "INFO").upper(),
            log_prompts=_bool("LOG_PROMPTS", False),
        )
