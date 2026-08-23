"""
Confidence Estimation Service
==============================
Computes answer-grounded confidence from cosine similarity signals only.

WHY THE PREVIOUS VERSION WAS BROKEN:
--------------------------------------
The previous implementation passed RRF scores into a function calibrated
for cosine similarity. RRF scores are tiny (typically 0.008–0.033,
depending on rank position) because they are computed as 1/(k + rank).
At k=60, the maximum possible RRF score for a top-1 result is 1/61 ≈ 0.016.
Dividing that by the soft cap (0.85) produced norm_top ≈ 0.02 — essentially
zero — making every answer appear "Low" confidence regardless of quality.

CORRECT APPROACH:
------------------
Confidence must be computed on the ORIGINAL cosine similarity scores
from the dense retriever, NOT on RRF scores. We now store the dense
cosine score separately and pass it here.

Three signals remain correct in concept; only the input has changed:

  Signal 1 — Top cosine similarity (weight 0.50)
    The dense embedding similarity of the best-matching chunk.
    MiniLM cosine similarities for in-domain queries typically sit
    in 0.35–0.80. Capped at 0.85 to prevent the score saturating at
    100% on near-identical text.

  Signal 2 — Score gap / separation (weight 0.30)
    (top1 - top2) / top1.
    A large gap means one clear best answer. A small gap means
    many similar chunks → the answer may be scattered or ambiguous.
    Weight raised from 0.25 to 0.30 because gap is a stronger signal
    than agreement for small corpora.

  Signal 3 — Coverage (weight 0.20)
    Fraction of final chunks whose cosine similarity exceeds a minimum
    relevance threshold (0.30). If 4 of 5 chunks are above threshold,
    the answer is well-supported from multiple angles.
    Replaces the previous "agreement_ratio" which was computed
    incorrectly (intersecting final chunks with the full dense candidate
    set, producing near-100% agreement always).

Label thresholds (calibrated for MiniLM on health policy queries):
    High   >= 60  → Answer is well-grounded
    Medium >= 35  → Partial or uncertain grounding
    Low     < 35  → Likely off-topic or outside corpus
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.logging import get_logger

logger = get_logger(__name__)

# MiniLM cosine similarity rarely exceeds 0.85 for in-domain queries.
_SCORE_SOFT_CAP = 0.85

# A chunk must have at least this cosine similarity to count as "relevant".
_RELEVANCE_THRESHOLD = 0.30

_THRESHOLD_HIGH = 60.0
_THRESHOLD_MEDIUM = 35.0


@dataclass(frozen=True)
class ConfidenceResult:
    score: float          # 0–100, shown in UI
    label: str            # "High" | "Medium" | "Low"
    top_cosine: float     # raw cosine of best chunk (diagnostic)
    score_gap: float      # top1_cosine - top2_cosine (diagnostic)
    coverage: float       # fraction of chunks above relevance threshold


class ConfidenceService:
    """
    Computes a calibrated confidence score from cosine similarity signals.

    IMPORTANT: Pass cosine_scores from the dense retriever directly.
    Do NOT pass RRF scores — they are on a completely different scale.
    """

    @staticmethod
    def compute(
        cosine_scores: list[float],
    ) -> ConfidenceResult:
        """
        Args:
            cosine_scores: Cosine similarity scores of the final Top-K chunks,
                           in descending order. These must be the original FAISS
                           inner-product scores, not RRF scores.

        Returns:
            ConfidenceResult with score, label, and component breakdown.
        """
        if not cosine_scores:
            return ConfidenceResult(
                score=0.0,
                label="Low",
                top_cosine=0.0,
                score_gap=0.0,
                coverage=0.0,
            )

        top_cosine = cosine_scores[0]

        # ----------------------------------------------------------------
        # Signal 1: Normalised top cosine similarity
        # ----------------------------------------------------------------
        norm_top = min(top_cosine / _SCORE_SOFT_CAP, 1.0)

        # ----------------------------------------------------------------
        # Signal 2: Score gap (separation between top-1 and top-2)
        # ----------------------------------------------------------------
        if len(cosine_scores) >= 2:
            gap = top_cosine - cosine_scores[1]
            norm_gap = min(gap / max(top_cosine, 1e-9), 1.0)
        else:
            norm_gap = 1.0  # single result — no ambiguity

        # ----------------------------------------------------------------
        # Signal 3: Coverage — how many chunks are genuinely relevant?
        # ----------------------------------------------------------------
        relevant_count = sum(
            1 for s in cosine_scores if s >= _RELEVANCE_THRESHOLD
        )
        coverage = relevant_count / len(cosine_scores)

        # ----------------------------------------------------------------
        # Weighted combination
        # ----------------------------------------------------------------
        raw_score = (
            0.50 * norm_top
            + 0.30 * norm_gap
            + 0.20 * coverage
        )

        confidence = round(raw_score * 100, 1)
        label = _label(confidence)

        logger.debug(
            "Confidence | top_cosine=%.3f gap=%.3f coverage=%.2f -> %.1f (%s)",
            top_cosine,
            gap if len(cosine_scores) >= 2 else 0.0,
            coverage,
            confidence,
            label,
        )

        return ConfidenceResult(
            score=confidence,
            label=label,
            top_cosine=round(top_cosine, 4),
            score_gap=round(
                top_cosine - (cosine_scores[1] if len(cosine_scores) > 1 else 0.0), 4
            ),
            coverage=round(coverage, 3),
        )


def _label(score: float) -> str:
    if score >= _THRESHOLD_HIGH:
        return "High"
    if score >= _THRESHOLD_MEDIUM:
        return "Medium"
    return "Low"
