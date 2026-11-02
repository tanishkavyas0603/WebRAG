from pydantic import BaseModel, Field


class RetrievalResult(BaseModel):

    chunk_id: int
    title: str
    section: str
    preview: str
    content: str

    # Mutable ranking score — starts as cosine, becomes RRF after fusion,
    # then gets boosted by the metadata ranker.
    similarity_score: float

    # The original FAISS cosine similarity. NEVER overwritten after
    # _dense_search(). Passed to ConfidenceService.
    cosine_score: float = Field(default=0.0)

    source: str

    # Fractional boost applied by MetadataRanker. 0.0 if not boosted.
    metadata_boost: float = Field(default=0.0)


class Source(BaseModel):
    """Public-facing source shown in the API response and UI."""

    title: str
    section: str
    preview: str

    # Relevance score shown in UI. This is the cosine similarity (0–1),
    # NOT the RRF score. RRF scores (~0.016) are meaningless to users.
    relevance_score: float

    source: str

    # True if the metadata ranker applied a boost to this chunk.
    boosted: bool = Field(default=False)


class QueryResponse(BaseModel):
    """Complete API response for a /query request."""

    answer: str

    # ------------------------------------------------------------------
    # Retrieval diagnostics (engineer-facing)
    # ------------------------------------------------------------------
    query_expanded: str = Field(
        default="",
        description="Expanded query used for retrieval (abbreviations resolved)",
    )
    retrieval_strategy: str = Field(
        default="hybrid_rrf",
        description="Retrieval strategy used",
    )

    # ------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------
    confidence: float          # 0–100
    confidence_label: str      # "High" | "Medium" | "Low"

    # ------------------------------------------------------------------
    # Sources
    # ------------------------------------------------------------------
    sources: list[Source]

    # ------------------------------------------------------------------
    # Performance
    # ------------------------------------------------------------------
    response_time_ms: float