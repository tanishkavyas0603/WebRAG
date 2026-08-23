"""
Metadata-Aware Ranker
=====================
Applies intent-based score boosts to RRF-fused retrieval results.

Design decisions:
- Intent detection is deterministic keyword matching, NOT an LLM call.
  This adds < 1ms latency and is 100% auditable.
- Boosts are fractional (capped at METADATA_BOOST_MAX = 0.15) so they
  improve ranking without overriding retrieval signal. A chunk with
  score 0.40 and a title match becomes 0.46 at most — not 0.90.
- Boost logic mirrors Elasticsearch's function_score query pattern and
  Azure AI Search's semantic field weights.
- Each RetrievedChunk gets a `metadata_boost` field so the API response
  is transparent about why a chunk ranked where it did.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.core.config import settings
from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.models.response import RetrievalResult

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Intent definitions
# ---------------------------------------------------------------------------
# Each intent has:
#   - query_patterns:   regex patterns to detect intent in the user query
#   - title_patterns:   patterns boosted in chunk titles
#   - section_patterns: patterns boosted in chunk section field
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _IntentRule:
    name: str
    query_patterns: list[str]
    title_patterns: list[str]
    section_patterns: list[str]


_INTENT_RULES: list[_IntentRule] = [
    _IntentRule(
        name="insurance_scheme",
        query_patterns=[
            r"\binsurance\b", r"\bscheme\b", r"\byojana\b",
            r"\bpm-?jay\b", r"\bayushman\b", r"\bcoverage\b",
            r"\bbeneficiar\b", r"\bpremium\b",
        ],
        title_patterns=[
            r"\binsurance\b", r"\byojana\b", r"\bscheme\b",
            r"\bayushman\b", r"\bpm-?jay\b",
        ],
        section_patterns=[r"health insurance", r"insurance"],
    ),
    _IntentRule(
        name="digital_health",
        query_patterns=[
            r"\bdigital\b", r"\babdm\b", r"\babha\b",
            r"\bhealth id\b", r"\btelemedicine\b", r"\bhealth record\b",
            r"\be-?health\b", r"\bonline\b",
        ],
        title_patterns=[
            r"\bdigital\b", r"\babdm\b", r"\babha\b",
            r"\btelemedicine\b",
        ],
        section_patterns=[r"digital health", r"digital"],
    ),
    _IntentRule(
        name="infrastructure",
        query_patterns=[
            r"\binfrastructure\b", r"\bhospital\b", r"\baiims\b",
            r"\bmedical college\b", r"\bpm-?abhim\b", r"\bcritical care\b",
            r"\bbed\b", r"\bicu\b", r"\bfacility\b", r"\bblock\b",
        ],
        title_patterns=[
            r"\binfrastructure\b", r"\bhospital\b", r"\baiims\b",
            r"\bpm-?abhim\b",
        ],
        section_patterns=[r"infrastructure"],
    ),
    _IntentRule(
        name="primary_care",
        query_patterns=[
            r"\bprimary\b", r"\bwellness\b", r"\bhwc\b",
            r"\bab-?hwc\b", r"\bhealth.*centre\b", r"\bphc\b",
            r"\bsub.?centre\b", r"\bcommunity health\b",
        ],
        title_patterns=[
            r"\bwellness\b", r"\bhwc\b", r"\bprimary\b",
        ],
        section_patterns=[r"primary", r"wellness"],
    ),
    _IntentRule(
        name="maternal_child_health",
        query_patterns=[
            r"\bmaternal\b", r"\bmother\b", r"\bnewborn\b",
            r"\bchild\b", r"\bjssk\b", r"\bjsy\b",
            r"\bdelivery\b", r"\bpregnant\b",
        ],
        title_patterns=[
            r"\bmaternal\b", r"\bjssk\b", r"\bjsy\b", r"\bnewborn\b",
        ],
        section_patterns=[r"maternal", r"child"],
    ),
    _IntentRule(
        name="mission",
        query_patterns=[
            r"\bmission\b", r"\bprogramme\b", r"\binitiative\b",
            r"\bnhm\b", r"\bnhp\b",
        ],
        title_patterns=[r"\bmission\b", r"\bprogramme\b"],
        section_patterns=[r"mission"],
    ),
]


def _matches_any(text: str, patterns: list[str]) -> bool:
    """Return True if any pattern matches the text (case-insensitive)."""
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in patterns)


class MetadataRanker:
    """
    Re-ranks a list of RetrievalResult objects using intent-based
    metadata boosts.

    The boost is computed as:
        boost_score = base_rrf_score × (1 + total_boost)

    where total_boost ∈ [0, METADATA_BOOST_MAX] accumulates fractional
    contributions from title and section matches.
    """

    def __init__(self) -> None:
        self._max_boost = settings.METADATA_BOOST_MAX

    def _detect_intents(self, query: str) -> list[_IntentRule]:
        """Return all intent rules that match the query."""
        return [
            rule
            for rule in _INTENT_RULES
            if _matches_any(query, rule.query_patterns)
        ]

    def _compute_chunk_boost(
        self,
        chunk: "RetrievalResult",
        intents: list[_IntentRule],
    ) -> float:
        """
        Compute the total fractional boost for a chunk.

        Each matching intent contributes a partial boost:
          - Title match   → 60% of the per-intent budget
          - Section match → 40% of the per-intent budget

        Total is capped at METADATA_BOOST_MAX.
        """
        if not intents:
            return 0.0

        per_intent_budget = self._max_boost / len(intents)
        total = 0.0

        for rule in intents:
            contribution = 0.0
            if _matches_any(chunk.title, rule.title_patterns):
                contribution += per_intent_budget * 0.60
            if _matches_any(chunk.section, rule.section_patterns):
                contribution += per_intent_budget * 0.40
            total += contribution

        return min(total, self._max_boost)

    def rerank(
        self,
        query: str,
        chunks: list["RetrievalResult"],
    ) -> list["RetrievalResult"]:
        """
        Apply metadata boosts and return chunks sorted by boosted score.

        The original similarity_score is used as the base. The
        metadata_boost field is set on each chunk for API transparency.
        """
        intents = self._detect_intents(query)

        if not intents:
            logger.debug("No intents detected for query — metadata boost skipped.")
            return chunks

        intent_names = [r.name for r in intents]
        logger.info("Detected intents: %s", intent_names)

        boosted: list[tuple[float, "RetrievalResult"]] = []

        for chunk in chunks:
            boost_fraction = self._compute_chunk_boost(chunk, intents)
            boosted_score = chunk.similarity_score * (1.0 + boost_fraction)

            # similarity_score is updated with the boosted value for ranking.
            # cosine_score is intentionally NOT touched — it is the preserved
            # original FAISS score used by ConfidenceService.
            chunk_copy = chunk.model_copy(
                update={
                    "similarity_score": round(boosted_score, 5),
                    "metadata_boost": round(boost_fraction, 4),
                }
            )
            boosted.append((boosted_score, chunk_copy))

        boosted.sort(key=lambda x: x[0], reverse=True)

        reranked = [chunk for _, chunk in boosted]

        logger.info(
            "Metadata rerank applied | intents=%s | chunks=%d",
            intent_names,
            len(reranked),
        )

        return reranked
