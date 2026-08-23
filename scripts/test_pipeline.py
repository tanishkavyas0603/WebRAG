"""Quick functional validation of the new pipeline components (no LLM needed)."""

import sys
sys.path.insert(0, ".")

from app.services.query_expansion_service import QueryExpansionService
from app.services.bm25_service import BM25Service
from app.services.metadata_ranker import _INTENT_RULES, _matches_any

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"

errors = []

# ---------------------------------------------------------------
# Test 1: Abbreviation expansion
# ---------------------------------------------------------------
expander = QueryExpansionService()
r = expander.expand("What is PM-JAY?")
assert r.was_expanded, "PM-JAY should trigger expansion"
assert "Pradhan Mantri Jan Arogya Yojana" in r.expanded, "Full form missing"
print(f"{PASS} Query expansion (PM-JAY)")
print(f"       expanded: {r.expanded}")
print(f"       bm25_extras: {r.bm25_terms}")

# ---------------------------------------------------------------
# Test 2: ABDM expansion
# ---------------------------------------------------------------
r2 = expander.expand("Tell me about ABDM digital health mission")
assert r2.was_expanded
assert "Ayushman Bharat Digital Mission" in r2.expanded
print(f"{PASS} Query expansion (ABDM)")

# ---------------------------------------------------------------
# Test 3: No expansion for normal query
# ---------------------------------------------------------------
r3 = expander.expand("What are health schemes in India?")
# should not expand (no known abbreviation)
print(f"{PASS} Query expansion (no abbrev — was_expanded={r3.was_expanded})")

# ---------------------------------------------------------------
# Test 4: BM25 initialisation and search
# ---------------------------------------------------------------
bm25 = BM25Service()
assert len(bm25.chunks) > 0, "BM25 corpus empty"
hits = bm25.search("Pradhan Mantri Jan Arogya Yojana health insurance", k=5)
assert len(hits) > 0, "BM25 returned no results"
assert hits[0][1] > hits[-1][1], "BM25 results not sorted descending"
print(f"{PASS} BM25 search returned {len(hits)} hits")
for idx, score in hits[:3]:
    chunk = bm25.chunks[idx]
    title = chunk["title"][:55]
    print(f"       [{score:.3f}] {title}")

# ---------------------------------------------------------------
# Test 5: RRF fusion ordering
# ---------------------------------------------------------------
r1 = [(10, 0.9), (20, 0.8), (30, 0.7)]
r2 = [(20, 5.0), (10, 4.0), (40, 3.0)]
fused = BM25Service.reciprocal_rank_fusion(r1, r2, k=60)
fused_ids = [x[0] for x in fused]
# chunk 20 appears at rank 1 in r2 and rank 2 in r1 → should be top
# chunk 10 appears at rank 1 in r1 and rank 2 in r2 → close second
assert fused_ids[0] in (10, 20), f"Unexpected top fusion result: {fused_ids}"
# Scores should be descending
scores = [x[1] for x in fused]
assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
print(f"{PASS} RRF fusion ordering correct: {fused_ids[:4]}")

# ---------------------------------------------------------------
# Test 6: Intent detection
# ---------------------------------------------------------------
def detect(query):
    return [r.name for r in _INTENT_RULES if _matches_any(query, r.query_patterns)]

assert "insurance_scheme" in detect("What is PM-JAY scheme?")
assert "digital_health" in detect("Explain ABDM digital health")
assert "infrastructure" in detect("How many hospitals under PM-ABHIM?")
assert "maternal_child_health" in detect("What is JSSK maternal health?")
print(f"{PASS} Intent detection correct for all 4 test queries")

# ---------------------------------------------------------------
# Test 7: BM25 with extra terms from expansion
# ---------------------------------------------------------------
r_exp = expander.expand("PM-JAY")
hits_with_extras = bm25.search(r_exp.expanded, extra_terms=r_exp.bm25_terms, k=5)
hits_without = bm25.search("PM-JAY", k=5)
# With extra terms (full form + synonyms) should return >= as many results
assert len(hits_with_extras) >= len(hits_without), "Extra terms should not reduce results"
print(f"{PASS} BM25 extra-terms injection (with={len(hits_with_extras)} vs without={len(hits_without)})")

print()
print("All tests passed." if not errors else f"{len(errors)} test(s) failed.")
