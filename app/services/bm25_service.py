import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi
from app.core.constants import INDEX_DIR
from app.core.logging import get_logger

logger = get_logger(__name__)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)

def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())

class BM25Service:
    def __init__(self, document_id: int):
        self.document_id = document_id
        self.metadata_path = Path(INDEX_DIR) / f"doc_{document_id}_metadata.pkl"
        self.chunks = []
        self.corpus = []
        self.bm25 = None

    def build(self, chunks: list[dict]):
        self.chunks = chunks
        self.corpus = [_tokenize(chunk["content"]) for chunk in self.chunks]
        self.bm25 = BM25Okapi(self.corpus)
        logger.info(f"BM25 built for document {self.document_id} | chunks={len(self.chunks)}")
        
    def load(self):
        if not self.metadata_path.exists():
            logger.warning(f"Metadata not found for BM25 document {self.document_id}")
            return False
            
        with open(self.metadata_path, "rb") as f:
            self.chunks = pickle.load(f)
            
        self.corpus = [_tokenize(chunk["content"]) for chunk in self.chunks]
        self.bm25 = BM25Okapi(self.corpus)
        logger.info(f"BM25 loaded for document {self.document_id} | chunks={len(self.chunks)}")
        return True

    def search(self, query: str, extra_terms: list[str] | None = None, k: int = 10) -> list[tuple[int, float]]:
        if not self.bm25:
            return []
            
        tokens = _tokenize(query)
        if extra_terms:
            tokens += _tokenize(" ".join(extra_terms))

        scores = self.bm25.get_scores(tokens)
        ranked = sorted(
            ((idx, float(score)) for idx, score in enumerate(scores) if score != 0.0),
            key=lambda x: x[1],
            reverse=True,
        )[:k]

        return ranked

    @staticmethod
    def reciprocal_rank_fusion(*rankings: list[tuple[int, float]], k: int = 60) -> list[tuple[int, float]]:
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, (idx, _score) in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)