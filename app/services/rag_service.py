import re
import time
from dataclasses import dataclass
from groq import Groq, APIStatusError, APIConnectionError, GroqError

class LLMError(Exception):
    pass

from app.core.config import settings
from app.core.logging import RAGRequestLogger, get_logger
from app.models.response import QueryResponse, RetrievalResult, Source
from app.services.confidence_service import ConfidenceService
from app.services.prompt_service import PromptService
from app.services.retrieval_service import RetrievalService

logger = get_logger(__name__)

_MAX_CHUNK_CHARS = 1_500

@dataclass
class _PipelineResult:
    chunks: list[RetrievalResult]
    query_expanded: str
    cosine_scores: list[float]

def _build_sources(chunks: list[RetrievalResult]) -> list[Source]:
    seen: dict[str, Source] = {}
    for chunk in chunks:
        source = Source(
            title=chunk.title,
            section=chunk.section,
            preview=chunk.preview,
            relevance_score=round(chunk.cosine_score, 3),
            source=chunk.source,
            boosted=chunk.metadata_boost > 0.0,
        )
        existing = seen.get(chunk.title)
        if existing is None or source.relevance_score > existing.relevance_score:
            seen[chunk.title] = source
    return sorted(seen.values(), key=lambda s: s.relevance_score, reverse=True)

def _empty_response(question: str) -> QueryResponse:
    return QueryResponse(
        answer=(
            "The information was not found in the webpage."
        ),
        query_expanded=question,
        retrieval_strategy="hybrid_rrf",
        sources=[],
        confidence=0.0,
        confidence_label="Low",
        response_time_ms=0.0,
    )


class RAGService:
    def __init__(self, document_id: int):
        if not settings.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set in environment.")
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.retriever = RetrievalService(document_id)

    def answer(self, question: str, chat_history: list[dict[str, str]] = None) -> QueryResponse:
        req_log = RAGRequestLogger(query=question)
        t_total_start = time.perf_counter()

        if chat_history is None:
            chat_history = []

        standalone_question = self._rewrite_query(question, chat_history)

        t_retrieval_start = time.perf_counter()
        pipeline = self._retrieve(standalone_question, req_log)
        retrieval_ms = (time.perf_counter() - t_retrieval_start) * 1_000

        if not pipeline.chunks:
            return _empty_response(standalone_question)

        confidence = ConfidenceService.compute(cosine_scores=pipeline.cosine_scores)
        req_log.log_confidence(
            score=confidence.score, label=confidence.label,
            top_score=confidence.top_cosine, score_gap=confidence.score_gap
        )

        truncated_chunks = [
            chunk.model_copy(update={"content": chunk.content[:_MAX_CHUNK_CHARS]})
            for chunk in pipeline.chunks
        ]
        messages = PromptService.build_messages(question, truncated_chunks, chat_history)

        context_char_count = sum(len(c.content) for c in truncated_chunks)
        logger.info(f"[DIAGNOSTICS] final chunk count: {len(truncated_chunks)}")
        logger.info(f"[DIAGNOSTICS] context character count: {context_char_count}")
        logger.info(f"[DIAGNOSTICS] context passed to LLM: True")

        t_llm_start = time.perf_counter()
        llm_answer = self._call_llm(messages)
        llm_ms = (time.perf_counter() - t_llm_start) * 1_000
        
        logger.info(f"[DIAGNOSTICS] LLM response content length: {len(llm_answer)}")

        total_ms = round((time.perf_counter() - t_total_start) * 1_000, 2)

        req_log.log_latency(expansion_ms=0.0, retrieval_ms=retrieval_ms, llm_ms=llm_ms)
        req_log.emit()

        sources = _build_sources(pipeline.chunks)

        return QueryResponse(
            answer=llm_answer,
            query_expanded=pipeline.query_expanded,
            retrieval_strategy="hybrid_rrf",
            sources=sources,
            confidence=confidence.score,
            confidence_label=confidence.label,
            response_time_ms=total_ms,
        )

    def _rewrite_query(self, question: str, chat_history: list[dict[str, str]]) -> str:
        if not chat_history:
            return question

        prompt = (
            "Given the following chat history and a follow-up question, "
            "rephrase the follow-up question to be a standalone question, "
            "in its original language, without changing its meaning.\n"
            "Chat History:\n"
        )
        for msg in chat_history[-3:]: # only use last 3 messages for rewriting context
            prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
        prompt += f"Follow-up Question: {question}\nStandalone Question:"

        try:
            response = self.client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=128,
            )
            raw_response = response.choices[0].message.content.strip()
            
            # Strip <think>...</think> blocks using regex
            cleaned = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()
            
            # Strip common prefixes that models might hallucinate
            prefixes_to_strip = [
                r'(?i)^draft:', r'(?i)^answer:', r'(?i)^rewritten query:', 
                r'(?i)^standalone question:', r'(?i)^here is.*?:', r'(?i)^rephrased:'
            ]
            for prefix in prefixes_to_strip:
                cleaned = re.sub(prefix, '', cleaned).strip()
                
            # Remove any surrounding quotes
            cleaned = cleaned.strip('"\'`')
            
            if not cleaned or len(cleaned) > 200:
                cleaned = question
                
            logger.info(f"[QUERY_REWRITE] original='{question}'")
            logger.info(f"[QUERY_REWRITE] raw_response='{raw_response}'")
            logger.info(f"[QUERY_REWRITE] cleaned='{cleaned}'")
            
            return cleaned
        except APIStatusError as e:
            logger.error(f"Groq APIStatusError: {e}")
            raise LLMError("AI model is currently unavailable or misconfigured. Please try again later.")
        except GroqError as e:
            logger.error(f"Groq GroqError: {e}")
            raise LLMError("AI model connection failed. Please try again later.")

    def _retrieve(self, question: str, req_log: RAGRequestLogger) -> _PipelineResult:
        chunks, expanded, cosine_scores = self.retriever.search(question)
        req_log.log_expansion(
            expanded=expanded.expanded,
            strategy="abbreviation+synonym" if expanded.was_expanded else "none",
        )
        req_log.log_retrieval(
            dense_count=settings.TOP_K * settings.RETRIEVAL_MULTIPLIER,
            sparse_count=settings.TOP_K * settings.RETRIEVAL_MULTIPLIER,
            after_fusion=len(chunks), final_count=len(chunks), strategy="hybrid_rrf",
        )
        return _PipelineResult(chunks=chunks, query_expanded=expanded.expanded, cosine_scores=cosine_scores)

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        try:
            response = self.client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=messages,
                temperature=0,
                max_tokens=1_024,
            )
            content = response.choices[0].message.content
            # Strip <think>...</think> blocks
            content = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL).strip()
            return content
        except APIStatusError as e:
            logger.error(f"Groq APIStatusError in _call_llm: {e}")
            raise LLMError("AI model is currently unavailable or misconfigured. Please try again later.")
        except GroqError as e:
            logger.error(f"Groq GroqError in _call_llm: {e}")
            raise LLMError("AI model connection failed. Please try again later.")
