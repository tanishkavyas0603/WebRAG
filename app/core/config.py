from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ------------------------------------------------------------------ #
    # External API                                                         #
    # ------------------------------------------------------------------ #
    GROQ_API_KEY: str
    HF_TOKEN: str | None = None
    WEBRAG_USER_AGENT: str = "WebRAG/1.0 (public webpage research application)"

    # ------------------------------------------------------------------ #
    # Database & Security                                                  #
    # ------------------------------------------------------------------ #
    DATABASE_URL: str = "sqlite:///./webrag.db"
    SECRET_KEY: str = "change_this_in_production_secret_key"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    CORS_ORIGINS: str = "http://localhost:5173"
    DATA_PATH: str = "data"
    FAISS_INDEX_DIR: str = "./data/index"

    # ------------------------------------------------------------------ #
    # Models                                                               #
    # ------------------------------------------------------------------ #
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    GROQ_MODEL: str = "llama-3.1-8b-instant"

    # ------------------------------------------------------------------ #
    # Retrieval — dense                                                    #
    # ------------------------------------------------------------------ #
    TOP_K: int = 5
    # Over-retrieve by this factor before fusion; e.g. TOP_K=5 → fetch 15
    # from each retriever, fuse, then trim.
    RETRIEVAL_MULTIPLIER: int = 3

    # ------------------------------------------------------------------ #
    # Retrieval — hybrid fusion                                            #
    # ------------------------------------------------------------------ #
    # Smoothing constant for Reciprocal Rank Fusion.  k=60 is the value
    # from the original Cormack et al. 2009 paper and is the default used
    # by Elasticsearch and Microsoft ARES.
    RRF_K: int = 60

    # ------------------------------------------------------------------ #
    # Retrieval — diversity (MMR)                                          #
    # ------------------------------------------------------------------ #
    # λ=1.0 → pure relevance (no diversity).  λ=0.0 → pure diversity.
    # 0.7 balances relevance with novelty; empirically good for short-doc
    # corpora where chunks share vocabulary.
    MMR_LAMBDA: float = 0.7

    # ------------------------------------------------------------------ #
    # Metadata boosting                                                    #
    # ------------------------------------------------------------------ #
    # Maximum fractional boost applied to RRF scores when a chunk's
    # section/title matches the detected query intent.
    # 0.15 → a matching chunk can be lifted by up to 15% of its score.
    METADATA_BOOST_MAX: float = 0.15

    # ------------------------------------------------------------------ #
    # Logging                                                              #
    # ------------------------------------------------------------------ #
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
    )


settings = Settings()