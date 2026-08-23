"""
Structured logging for the Health RAG system.

Design decisions:
- One-time root logger configuration via configure_logging().
  Calling basicConfig() inside get_logger() is an anti-pattern: it no-ops
  after the first call, making log-level changes unpredictable.
- A RAGRequestLogger helper wraps per-request structured events.
  This mirrors the pattern used in production observability stacks
  (Datadog, Google Cloud Logging) where every log line carries a
  trace/request ID so you can reconstruct the full journey of a query.
"""

import logging
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.core.config import settings

_CONFIGURED = False


def configure_logging() -> None:
    """Configure the root logger once at application startup."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
    root.handlers.clear()
    root.addHandler(handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


# --------------------------------------------------------------------------- #
# Per-request structured logger                                                #
# --------------------------------------------------------------------------- #

@dataclass
class RAGRequestLogger:
    """
    Accumulates structured metrics for a single RAG request.

    Usage:
        req = RAGRequestLogger(query="What is PM-JAY?")
        req.log_expansion(expanded="Pradhan Mantri Jan Arogya Yojana")
        req.log_retrieval(dense_count=15, sparse_count=15, final_count=5)
        req.log_confidence(score=0.81, label="High")
        req.log_latency(expansion_ms=2.1, retrieval_ms=45.3, llm_ms=920.0)
        req.emit()
    """

    query: str
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    _logger: logging.Logger = field(
        default_factory=lambda: get_logger("rag.request"),
        repr=False,
        compare=False,
    )
    _data: dict[str, Any] = field(default_factory=dict, repr=False)

    def log_expansion(self, expanded: str, strategy: str = "abbreviation+synonym") -> None:
        self._data["query_expanded"] = expanded
        self._data["expansion_strategy"] = strategy
        self._logger.debug(
            "[%s] Query expanded | original=%r expanded=%r",
            self.request_id,
            self.query,
            expanded,
        )

    def log_retrieval(
        self,
        dense_count: int,
        sparse_count: int,
        after_fusion: int,
        final_count: int,
        strategy: str = "hybrid_rrf",
    ) -> None:
        self._data.update(
            {
                "retrieval_strategy": strategy,
                "dense_candidates": dense_count,
                "sparse_candidates": sparse_count,
                "after_fusion": after_fusion,
                "final_chunks": final_count,
            }
        )
        self._logger.info(
            "[%s] Retrieval | strategy=%s dense=%d sparse=%d fused=%d final=%d",
            self.request_id,
            strategy,
            dense_count,
            sparse_count,
            after_fusion,
            final_count,
        )

    def log_confidence(self, score: float, label: str, top_score: float, score_gap: float) -> None:
        self._data.update(
            {
                "confidence": score,
                "confidence_label": label,
                "top_score": top_score,
                "score_gap": score_gap,
            }
        )
        self._logger.info(
            "[%s] Confidence | score=%.3f label=%s top_score=%.3f gap=%.3f",
            self.request_id,
            score,
            label,
            top_score,
            score_gap,
        )

    def log_latency(
        self,
        expansion_ms: float,
        retrieval_ms: float,
        llm_ms: float,
    ) -> None:
        total = expansion_ms + retrieval_ms + llm_ms
        self._data.update(
            {
                "latency_expansion_ms": round(expansion_ms, 2),
                "latency_retrieval_ms": round(retrieval_ms, 2),
                "latency_llm_ms": round(llm_ms, 2),
                "latency_total_ms": round(total, 2),
            }
        )
        self._logger.info(
            "[%s] Latency | expand=%.1fms retrieval=%.1fms llm=%.1fms total=%.1fms",
            self.request_id,
            expansion_ms,
            retrieval_ms,
            llm_ms,
            total,
        )

    def emit(self) -> None:
        """Emit final structured summary log for this request."""
        self._logger.info(
            "[%s] Request complete | query=%r %s",
            self.request_id,
            self.query,
            " ".join(f"{k}={v}" for k, v in self._data.items()),
        )