from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    GROQ_API_KEY: str
    HF_TOKEN: str | None = None
    WEBRAG_USER_AGENT: str = "WebRAG/1.0 (public webpage research application)"

    DATABASE_URL: str = "sqlite:///./webrag.db"
    SECRET_KEY: str = "change_this_in_production_secret_key"
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
    # Rate Limiting                                                      #
    # ------------------------------------------------------------------ #
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW_SECONDS: int = 60

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
    )


settings = Settings()