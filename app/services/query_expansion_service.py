"""
Query Expansion Service
=======================
Expands user queries before embedding and BM25 search to bridge the
vocabulary gap between user shorthand (abbreviations, acronyms) and the
document corpus.

Design decisions:
- Deterministic dictionary-based expansion: zero latency, zero API calls,
  100% auditable. Production systems (Azure AI Search, Elasticsearch) call
  this a "synonym map" and configure it at index time; we apply it at
  query time because our index is pre-built.
- Returns both the original query AND an expanded string so the caller can
  pass both to the dense encoder (for semantic similarity on expanded text)
  and to BM25 (for lexical match on the full form).
- Synonym injection appends related terms, not replaces — this prevents
  over-specificity when the abbreviation resolves to multiple concepts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Domain knowledge: Indian health policy abbreviations
# ---------------------------------------------------------------------------
# Each key is the canonical abbreviation (case-insensitive match).
# Each value is a tuple of (full_form, [related_synonyms]).
# The full_form is always injected; synonyms are injected only when the
# query does NOT already contain them.
# ---------------------------------------------------------------------------

_ABBREVIATION_MAP: dict[str, tuple[str, list[str]]] = {
    # --- Flagship insurance scheme ---
    "pm-jay": (
        "Pradhan Mantri Jan Arogya Yojana",
        ["Ayushman Bharat", "health insurance", "coverage", "beneficiaries"],
    ),
    "pmjay": (
        "Pradhan Mantri Jan Arogya Yojana",
        ["Ayushman Bharat", "health insurance", "coverage"],
    ),
    # --- Digital health mission ---
    "abdm": (
        "Ayushman Bharat Digital Mission",
        ["digital health", "ABHA", "health ID", "health records"],
    ),
    "abha": (
        "Ayushman Bharat Health Account",
        ["health ID", "digital health", "ABDM"],
    ),
    # --- Primary care network ---
    "ab-hwc": (
        "Ayushman Bharat Health and Wellness Centres",
        ["health wellness centre", "primary care", "HWC"],
    ),
    "hwc": (
        "Health and Wellness Centres",
        ["primary care", "Ayushman Bharat", "AB-HWC"],
    ),
    # --- Health infrastructure mission ---
    "pm-abhim": (
        "Pradhan Mantri Ayushman Bharat Health Infrastructure Mission",
        ["health infrastructure", "hospitals", "critical care"],
    ),
    "pmabhim": (
        "Pradhan Mantri Ayushman Bharat Health Infrastructure Mission",
        ["health infrastructure", "hospitals"],
    ),
    # --- Maternal & child health ---
    "jssk": (
        "Janani Shishu Suraksha Karyakram",
        ["maternal health", "newborn care", "free delivery"],
    ),
    "jsy": (
        "Janani Suraksha Yojana",
        ["maternal health", "institutional delivery", "cash incentive"],
    ),
    # --- Emergency transport ---
    "pmssma": (
        "Pradhan Mantri Swasthya Suraksha Mission",
        ["AIIMS", "medical colleges", "health infrastructure"],
    ),
    # --- Mental health ---
    "nmhp": (
        "National Mental Health Programme",
        ["mental health", "psychiatry", "district mental health"],
    ),
    "dmhp": (
        "District Mental Health Programme",
        ["mental health", "psychiatry"],
    ),
    # --- Tuberculosis ---
    "nikshay": (
        "Nikshay patient support programme",
        ["tuberculosis", "TB", "patient support"],
    ),
    "rntcp": (
        "Revised National Tuberculosis Control Programme",
        ["tuberculosis", "TB elimination"],
    ),
    # --- National health authority ---
    "nha": (
        "National Health Authority",
        ["health authority", "PM-JAY implementation", "Ayushman Bharat"],
    ),
    # --- Emergency ambulance ---
    "emri": (
        "Emergency Management and Research Institute",
        ["ambulance", "108", "emergency services"],
    ),
}

# ---------------------------------------------------------------------------
# Synonym groups for concept-level expansion (applied to any query)
# These handle cases where a user uses one term but documents use another.
# ---------------------------------------------------------------------------
_CONCEPT_SYNONYMS: list[tuple[str, list[str]]] = [
    # If user says "scheme", also search for yojana/mission/programme
    (r"\bscheme\b", ["yojana", "mission", "programme", "initiative"]),
    (r"\byojana\b", ["scheme", "mission", "programme"]),
    (r"\bmission\b", ["yojana", "scheme", "programme"]),
    # Hospital types
    (r"\bhospital\b", ["medical college", "health centre", "facility"]),
    # Coverage / beneficiaries
    (r"\bbeneficiar(?:y|ies)\b", ["coverage", "enrolled", "insured"]),
    (r"\bcoverage\b", ["beneficiaries", "enrolled", "insured families"]),
    # Digital
    (r"\bdigital health\b", ["ABDM", "ABHA", "health records", "telemedicine"]),
]


@dataclass(frozen=True)
class ExpandedQuery:
    """Result of query expansion.

    Attributes:
        original:    The raw user query, unchanged.
        expanded:    Full-form query for dense embedding (longest).
        bm25_terms:  Extra terms to append to BM25 token list.
        was_expanded: True if any expansion actually fired.
    """

    original: str
    expanded: str
    bm25_terms: list[str]
    was_expanded: bool


class QueryExpansionService:
    """
    Expands abbreviations and injects synonyms into health-policy queries.

    The expansion pipeline runs in two passes:
      1. Abbreviation pass  — replaces known acronyms with their full forms
                              and appends related synonyms.
      2. Concept pass       — appends synonym variants for generic health terms.

    The result is used as follows by the retrieval layer:
      - expanded  → encoded by SentenceTransformer (richer semantic context)
      - bm25_terms → appended to BM25 token list (better term matching)
    """

    def expand(self, query: str) -> ExpandedQuery:
        query_lower = query.lower().strip()
        expanded_parts: list[str] = [query]
        bm25_terms: list[str] = []
        was_expanded = False

        # ----------------------------------------------------------------
        # Pass 1: Abbreviation expansion
        # ----------------------------------------------------------------
        for abbrev, (full_form, synonyms) in _ABBREVIATION_MAP.items():
            pattern = re.compile(
                r"\b" + re.escape(abbrev) + r"\b",
                re.IGNORECASE,
            )
            if pattern.search(query_lower):
                was_expanded = True
                # Add full form to the expanded query text
                if full_form.lower() not in query_lower:
                    expanded_parts.append(full_form)

                # Add synonyms as BM25 extra terms (not part of the
                # dense query to avoid over-diluting the embedding)
                for syn in synonyms:
                    if syn.lower() not in query_lower:
                        bm25_terms.append(syn)

                logger.debug(
                    "Abbreviation expansion: %r → %r (synonyms: %s)",
                    abbrev,
                    full_form,
                    synonyms,
                )

        # ----------------------------------------------------------------
        # Pass 2: Concept synonym expansion (only for BM25 terms)
        # ----------------------------------------------------------------
        for pattern_str, synonyms in _CONCEPT_SYNONYMS:
            if re.search(pattern_str, query_lower, re.IGNORECASE):
                for syn in synonyms:
                    if syn.lower() not in query_lower and syn not in bm25_terms:
                        bm25_terms.append(syn)

        # Build the final expanded string for the dense encoder.
        # Keep it concise: original + full forms only (not all synonyms).
        expanded = " ".join(expanded_parts)

        if was_expanded:
            logger.info(
                "Query expanded | original=%r -> expanded=%r | bm25_extras=%d terms",
                query,
                expanded,
                len(bm25_terms),
            )

        return ExpandedQuery(
            original=query,
            expanded=expanded,
            bm25_terms=bm25_terms,
            was_expanded=was_expanded,
        )
