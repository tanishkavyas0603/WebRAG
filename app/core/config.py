import warnings
from pydantic_settings import BaseSettings, SettingsConfigDict

# Default value used at dev time. Production MUST override this via environment variable.
_WEAK_SECRET_KEY = "change_this_in_production_secret_key"


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
    FAISS_INDEX_DIR: str = "./data/index"

    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    GROQ_MODEL: str = "qwen/qwen3.6-27b"

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