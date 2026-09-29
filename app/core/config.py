import logging
import warnings
from pydantic_settings import BaseSettings, SettingsConfigDict

# Default value used at dev time. Production MUST override this via environment variable.
_WEAK_SECRET_KEY = "change_this_in_production_secret_key"

# ------------------------------------------------------------------ #
# Known good Groq models that support reasoning_effort="none".         #
# qwen/qwen3.6-27b was DEPRECATED by Groq and returns 404/400 errors. #
# qwen/qwen3.8-27b is the current supported successor.                 #
# ------------------------------------------------------------------ #
_RECOMMENDED_GROQ_MODEL = "qwen/qwen3.8-27b"


class Settings(BaseSettings):
    GROQ_API_KEY: str
    HF_TOKEN: str | None = None
    WEBRAG_USER_AGENT: str = "WebRAG/1.0 (public webpage research application)"

    DATABASE_URL: str = "sqlite:///./webrag.db"
    SECRET_KEY: str = _WEAK_SECRET_KEY
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    CORS_ORIGINS: str = "http://localhost:5173"
    DATA_PATH: str = "data"
    # Render uses an ephemeral filesystem — /tmp survives a request but not
    # a dyno restart.  FAISS indexes are rebuilt from PostgreSQL chunks on
    # first access after a restart via the recovery path in RetrievalService.
    # Set FAISS_INDEX_DIR=/tmp/webrag/data/index on Render.
    FAISS_INDEX_DIR: str = "/tmp/webrag/data/index"

    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    # IMPORTANT: qwen/qwen3.6-27b was deprecated on Groq. If you see
    # "AI model is currently unavailable or misconfigured", check that
    # GROQ_MODEL on Render is set to a live model (e.g. qwen/qwen3.8-27b).
    GROQ_MODEL: str = "qwen/qwen3.8-27b"

    TOP_K: int = 5
    RETRIEVAL_MULTIPLIER: int = 3
    RRF_K: int = 60

    MMR_LAMBDA: float = 0.7

    METADATA_BOOST_MAX: float = 0.15

    LOG_LEVEL: str = "INFO"

    # ------------------------------------------------------------------ #
    # Rate Limiting                                                        #
    # Note: process-local only — not distributed across multiple workers   #
    # ------------------------------------------------------------------ #
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW_SECONDS: int = 60

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
    )


settings = Settings()

# ------------------------------------------------------------------ #
# Production safety check — warn loudly if weak SECRET_KEY is in use  #
# ------------------------------------------------------------------ #
if settings.SECRET_KEY == _WEAK_SECRET_KEY:
    warnings.warn(
        "SECRET_KEY is set to the default development value. "
        "This is INSECURE in production. "
        "Generate a strong key with: python -c \"import secrets; print(secrets.token_hex(32))\" "
        "and set it as the SECRET_KEY environment variable.",
        stacklevel=1,
    )

# ------------------------------------------------------------------ #
# Production safety check — surface a missing HF_TOKEN at startup,    #
# not only when a user first triggers document ingestion. Note that   #
# this value is read once per process at import time: if HF_TOKEN is  #
# added or rotated later (in .env or on Render), the process must be  #
# restarted/redeployed to pick up the new value — it is not hot-reloaded. #
# ------------------------------------------------------------------ #
def _check_hf_token(hf_token: str | None) -> None:
    if not hf_token:
        warnings.warn(
            "HF_TOKEN is not set. Document ingestion (embedding generation) will "
            "fail until a valid Hugging Face access token is configured via the "
            "HF_TOKEN environment variable.",
            stacklevel=1,
        )


_check_hf_token(settings.HF_TOKEN)


# ------------------------------------------------------------------ #
# Startup configuration summary — logged (not printed) so it appears  #
# in Render logs. NEVER log the values of secrets.                     #
# ------------------------------------------------------------------ #
def log_startup_config() -> None:
    """Emit a structured summary of critical config at process startup.
    Logs presence/absence of required variables — NEVER their values."""
    _log = logging.getLogger("app.startup")
    _log.info("[STARTUP] WebRAG configuration summary:")
    _log.info("[STARTUP]   GROQ_API_KEY   : %s", "set" if settings.GROQ_API_KEY else "MISSING")
    _log.info("[STARTUP]   HF_TOKEN       : %s", "set" if settings.HF_TOKEN else "MISSING (embeddings will fail)")
    _log.info("[STARTUP]   GROQ_MODEL     : %s", settings.GROQ_MODEL)
    _log.info("[STARTUP]   EMBEDDING_MODEL: %s", settings.EMBEDDING_MODEL)
    _log.info("[STARTUP]   FAISS_INDEX_DIR: %s", settings.FAISS_INDEX_DIR)
    _log.info("[STARTUP]   DATABASE_URL   : %s", "set (sqlite)" if "sqlite" in settings.DATABASE_URL else "set (postgresql)")
    _log.info("[STARTUP]   SECRET_KEY     : %s", "weak default — CHANGE IN PRODUCTION" if settings.SECRET_KEY == _WEAK_SECRET_KEY else "set")
    if settings.GROQ_MODEL == "qwen/qwen3.6-27b":
        _log.warning(
            "[STARTUP] WARNING: GROQ_MODEL=qwen/qwen3.6-27b is DEPRECATED on Groq. "
            "This will cause APIStatusError on every request. "
            "Set GROQ_MODEL=qwen/qwen3.8-27b in your Render environment variables."
        )