import numpy as np
from app.services.embedding_service import EmbeddingService

from app.core.config import settings
from app.core.logging import get_logger
from app.models.response import RetrievalResult
from app.services.bm25_service import BM25Service
from app.services.metadata_ranker import MetadataRanker
from app.services.query_expansion_service import ExpandedQuery, QueryExpansionService
from app.vectorstore.faiss_store import FAISSVectorStore

logger = get_logger(__name__)

_MIN_COSINE_THRESHOLD = 0.15

class RetrievalService:
    def __init__(self, document_id: int):
        self.document_id = document_id
        self.embedding_service = EmbeddingService()

        self.store = FAISSVectorStore(document_id)
        self.index, self.metadata = self.store.load()
        if self.index is None:
            raise RuntimeError(f"FAISS index not found for document {document_id}")

        self.bm25 = BM25Service(document_id)
        self.bm25.load()
        
        self.expander = QueryExpansionService()
        self.metadata_ranker = MetadataRanker()

        self._fetch_k = settings.TOP_K * settings.RETRIEVAL_MULTIPLIER

    def search(self, question: str) -> tuple[list[RetrievalResult], ExpandedQuery, list[float]]:
        expanded = self.expander.expand(question)
        query_embedding: np.ndarray = self._embed(expanded.expanded)
        
        dense_results = self._dense_search(query_embedding, k=self._fetch_k)
        sparse_results = self._sparse_search(expanded.expanded, extra_terms=expanded.bm25_terms, k=self._fetch_k)
        
        fused = self._fuse(dense_results, sparse_results)
        
        logger.info(f"Retrieval | dense={len(dense_results)} sparse={len(sparse_results)} fused={len(fused)}")

        fused = [c for c in fused if c.cosine_score >= _MIN_COSINE_THRESHOLD]
        reranked = self.metadata_ranker.rerank(question, fused)
        
        diverse = self._mmr_filter(query_embedding=query_embedding, chunks=reranked, top_k=settings.TOP_K, lam=settings.MMR_LAMBDA)
        cosine_scores = [c.cosine_score for c in diverse]

        return diverse, expanded, cosine_scores

    def _embed(self, text: str) -> np.ndarray:
        vec = self.embedding_service.generate_embeddings([text])
        return vec[0]

    def _dense_search(self, query_embedding: np.ndarray, k: int) -> list[RetrievalResult]:
        query_vec = query_embedding.reshape(1, -1).astype("float32")
        scores, indices = self.index.search(query_vec, k)

        results: list[RetrievalResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
            chunk = self.metadata[idx]
            cosine = float(score)
            results.append(
                RetrievalResult(
                    chunk_id=chunk["id"],
                    title=chunk["title"],
                    section=chunk.get("section", ""),
                    preview=chunk.get("preview", ""),
                    content=chunk["content"],
                    similarity_score=cosine,
                    cosine_score=cosine,
                    source=chunk.get("source", "Webpage Content"),
                    metadata_boost=0.0,
                )
            )
        return results

    def _sparse_search(self, query_text: str, extra_terms: list[str], k: int) -> list[RetrievalResult]:
        ranked = self.bm25.search(query_text, extra_terms=extra_terms, k=k)
        results: list[RetrievalResult] = []
        for idx, _bm25_score in ranked:
            if idx >= len(self.bm25.chunks):
                continue
            chunk = self.bm25.chunks[idx]
            results.append(
                RetrievalResult(
                    chunk_id=chunk["id"],
                    title=chunk["title"],
                    section=chunk.get("section", ""),
                    preview=chunk.get("preview", ""),
                    content=chunk["content"],
                    similarity_score=0.0,
                    cosine_score=0.0,
                    source=chunk.get("source", "Webpage Content"),
                    metadata_boost=0.0,
                )
            )
        return results

    def _fuse(self, dense: list[RetrievalResult], sparse: list[RetrievalResult]) -> list[RetrievalResult]:
        dense_ranking = [(r.chunk_id, r.similarity_score) for r in dense]
        sparse_ranking = [(r.chunk_id, r.similarity_score) for r in sparse]
        fused_ids = BM25Service.reciprocal_rank_fusion(dense_ranking, sparse_ranking, k=settings.RRF_K)

        id_to_chunk = {r.chunk_id: r for r in sparse}
        id_to_chunk.update({r.chunk_id: r for r in dense})

        fused_results = []
        for chunk_id, rrf_score in fused_ids:
            chunk = id_to_chunk.get(chunk_id)
            if chunk:
                fused_results.append(chunk.model_copy(update={"similarity_score": round(rrf_score, 6)}))
        return fused_results

    def _mmr_filter(self, query_embedding: np.ndarray, chunks: list[RetrievalResult], top_k: int, lam: float) -> list[RetrievalResult]:
        if len(chunks) <= top_k:
            return chunks
        texts = [c.content for c in chunks]
        chunk_embeddings = self.embedding_service.generate_embeddings(texts)
        relevance = chunk_embeddings @ query_embedding

        selected_indices = []
        remaining = list(range(len(chunks)))

        for _ in range(min(top_k, len(chunks))):
            if not remaining:
                break
            if not selected_indices:
                best = max(remaining, key=lambda i: relevance[i])
            else:
                selected_embs = chunk_embeddings[selected_indices]
                sim_to_selected = chunk_embeddings[remaining] @ selected_embs.T
                if sim_to_selected.ndim == 1:
                    sim_to_selected = sim_to_selected.reshape(-1, 1)
                max_sim = sim_to_selected.max(axis=1)
                rel_scores = relevance[remaining]
                mmr_scores = lam * rel_scores - (1.0 - lam) * max_sim
                best_local = int(np.argmax(mmr_scores))
                best = remaining[best_local]
            selected_indices.append(best)
            remaining.remove(best)

        return [chunks[i] for i in selected_indices]