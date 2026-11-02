import re
import time
from dataclasses import dataclass
from groq import Groq, APIStatusError, APIConnectionError, GroqError, RateLimitError

# ------------------------------------------------------------------ #
# Application-specific exceptions                                      #
# These are raised by service layers and translated to safe HTTP       #
# responses at the API layer. They carry user-safe messages only.      #
# ------------------------------------------------------------------ #

class LLMError(Exception):
    """Raised when Groq/LLM generation fails. Message is safe to show users."""
    pass


class EmbeddingServiceError(Exception):
    """Raised when HF embedding API fails. Message is safe to show users."""
    pass


class RetrievalError(Exception):
    """Raised when retrieval pipeline fails critically. Message is safe to show users."""
    pass


from app.core.config import settings
from app.core.logging import RAGRequestLogger, get_logger
from app.models.response import QueryResponse, RetrievalResult, Source
from app.services.confidence_service import ConfidenceService
from app.services.prompt_service import PromptService
from app.services.retrieval_service import RetrievalService

logger = get_logger(__name__)

_MAX_CHUNK_CHARS = 1_500

# ------------------------------------------------------------------ #
# Model-specific parameter guards                                      #
# reasoning_effort is ONLY supported by Qwen reasoning models on Groq. #
# Sending it to non-Qwen models (e.g. openai/gpt-oss-120b, llama-*)   #
# causes HTTP 400, which surfaces to the user as a 503.               #
# ------------------------------------------------------------------ #
_REASONING_EFFORT_MODELS = ("qwen",)  # lowercase prefix match


def _supports_reasoning_effort(model_id: str) -> bool:
    """Return True only for models that accept reasoning_effort on Groq."""
    return any(model_id.lower().startswith(p) for p in _REASONING_EFFORT_MODELS)


def _build_groq_kwargs(model_id: str, extra: dict | None = None) -> dict:
    """Build Groq chat.completions.create kwargs, injecting reasoning_effort
    only for models that actually support it.  Never mutates the caller's dict."""
    kwargs: dict = extra.copy() if extra else {}
    if _supports_reasoning_effort(model_id):
        kwargs["reasoning_effort"] = "none"
        logger.debug(f"[GROQ_KWARGS] reasoning_effort=none injected for model={model_id}")
    else:
        logger.debug(f"[GROQ_KWARGS] reasoning_effort omitted for model={model_id}")
    return kwargs

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

        logger.info(f"[REQUEST_START] question={repr(question[:80])} model={settings.GROQ_MODEL}")

        standalone_question = self._rewrite_query(question, chat_history)

        t_retrieval_start = time.perf_counter()
        pipeline = self._retrieve(standalone_question, req_log)
        retrieval_ms = (time.perf_counter() - t_retrieval_start) * 1_000

        if not pipeline.chunks:
            logger.info("[REQUEST_SUCCESS] No relevant chunks found \u2014 returning empty response")
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

        logger.info(f"[LLM_START] model={settings.GROQ_MODEL} chunks={len(truncated_chunks)} context_chars={context_char_count}")
        t_llm_start = time.perf_counter()
        llm_answer = self._call_llm(messages)
        llm_ms = (time.perf_counter() - t_llm_start) * 1_000
        logger.info(f"[LLM_SUCCESS] elapsed_ms={llm_ms:.1f} response_chars={len(llm_answer)}")

        logger.info(f"[DIAGNOSTICS] LLM response content length: {len(llm_answer)}")
        logger.info(f"[DIAGNOSTICS] llm_answer exact value: {repr(llm_answer)}")

        total_ms = round((time.perf_counter() - t_total_start) * 1_000, 2)

        req_log.log_latency(expansion_ms=0.0, retrieval_ms=retrieval_ms, llm_ms=llm_ms)
        req_log.emit()

        logger.info(f"[REQUEST_SUCCESS] total_ms={total_ms}")

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
            call_kwargs = _build_groq_kwargs(settings.GROQ_MODEL, {
                "model": settings.GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 128,
            })
            logger.info(
                f"[LLM_START] stage=query_rewrite model={settings.GROQ_MODEL} "
                f"reasoning_effort={'none' if _supports_reasoning_effort(settings.GROQ_MODEL) else 'omitted'}"
            )
            response = self.client.chat.completions.create(**call_kwargs)
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
            cleaned = cleaned.strip("\"'`")

            if not cleaned or len(cleaned) > 200:
                cleaned = question

            logger.info(f"[QUERY_REWRITE] original='{question}'")
            logger.info(f"[QUERY_REWRITE] raw_response='{raw_response}'")
            logger.info(f"[QUERY_REWRITE] cleaned='{cleaned}'")

            return cleaned
        except RateLimitError as e:
            logger.warning(f"[LLM_FAILURE] stage=query_rewrite reason=rate_limit model={settings.GROQ_MODEL}")
            raise LLMError("The AI service is currently receiving too many requests. Please wait a moment and try again.")
        except APIStatusError as e:
            status_code = e.status_code if hasattr(e, "status_code") else 0
            if status_code in (400, 404):
                logger.error(
                    f"[LLM_FAILURE] stage=query_rewrite reason=api_error status={status_code} "
                    f"model={settings.GROQ_MODEL} "
                    f"hint=Check_GROQ_MODEL_env_var_and_reasoning_effort_param_compatibility "
                    f"reasoning_effort_sent={_supports_reasoning_effort(settings.GROQ_MODEL)}"
                )
            else:
                logger.error(
                    f"[LLM_FAILURE] stage=query_rewrite reason=api_error status={status_code} "
                    f"model={settings.GROQ_MODEL}"
                )
            raise LLMError("AI model is currently unavailable or misconfigured. Please try again later.")
        except GroqError as e:
            logger.error(f"[LLM_FAILURE] stage=query_rewrite reason=groq_error type={type(e).__name__} model={settings.GROQ_MODEL}")
            raise LLMError("AI model connection failed. Please try again later.")
        except Exception:
            logger.exception(f"[LLM_FAILURE] stage=query_rewrite reason=unexpected model={settings.GROQ_MODEL}")
            raise LLMError("AI model is currently unavailable or misconfigured. Please try again later.")

    def _retrieve(self, question: str, req_log: RAGRequestLogger) -> _PipelineResult:
        logger.info(f"[RETRIEVAL_START] question={repr(question[:80])}")
        try:
            chunks, expanded, cosine_scores = self.retriever.search(question)
        except Exception:
            # Covers embedding/FAISS/BM25 failures surfaced during search().
            # EmbeddingService already logs its own [EMBEDDING_FAILURE] with
            # specifics; this is the retrieval-pipeline-level failure log.
            logger.exception(f"[RETRIEVAL_FAILURE] question={repr(question[:80])}")
            raise RetrievalError("Document search is currently unavailable. Please try again later.")

        req_log.log_expansion(
            expanded=expanded.expanded,
            strategy="abbreviation+synonym" if expanded.was_expanded else "none",
        )
        req_log.log_retrieval(
            dense_count=settings.TOP_K * settings.RETRIEVAL_MULTIPLIER,
            sparse_count=settings.TOP_K * settings.RETRIEVAL_MULTIPLIER,
            after_fusion=len(chunks), final_count=len(chunks), strategy="hybrid_rrf",
        )
        logger.info(f"[RETRIEVAL_SUCCESS] chunks={len(chunks)}")
        return _PipelineResult(chunks=chunks, query_expanded=expanded.expanded, cosine_scores=cosine_scores)

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        logger.info(
            f"[LLM_START] stage=call_llm model={settings.GROQ_MODEL} "
            f"reasoning_effort={'none' if _supports_reasoning_effort(settings.GROQ_MODEL) else 'omitted'}"
        )
        try:
            call_kwargs = _build_groq_kwargs(settings.GROQ_MODEL, {
                "model": settings.GROQ_MODEL,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 768,
            })
            response = self.client.chat.completions.create(**call_kwargs)
            content = response.choices[0].message.content
            logger.info(f"[DIAGNOSTICS] raw Groq content exact value: {repr(content)}")

            if content is None:
                logger.error(f"[LLM_FAILURE] stage=call_llm reason=none_content model={settings.GROQ_MODEL}")
                raise LLMError("The AI service returned an empty response. Please try again.")

            # Strip <think>...</think> blocks (robust approach)
            # Split on closing tag and take content after, handles unclosed/unopened tags
            if '</think>' in content:
                content = content.split('</think>')[-1]
            content = content.strip()
            logger.info(f"[DIAGNOSTICS] content after stripping think blocks: {repr(content)}")

            # BUG-02 FIX: Guard against empty response after stripping.
            # Without this, an empty string reaches the frontend as a blank message bubble.
            if not content:
                logger.warning(f"[LLM_FAILURE] stage=call_llm reason=empty_after_strip model={settings.GROQ_MODEL}")
                return "I was unable to generate a response. Please try rephrasing your question."

            logger.info(f"[LLM_SUCCESS] stage=call_llm response_chars={len(content)}")
            return content
        except LLMError:
            raise  # re-raise without wrapping
        except RateLimitError as e:
            logger.warning(f"[LLM_FAILURE] stage=call_llm reason=rate_limit model={settings.GROQ_MODEL}")
            raise LLMError("The AI service is currently receiving too many requests. Please wait a moment and try again.")
        except APIStatusError as e:
            status_code = e.status_code if hasattr(e, "status_code") else 0
            if status_code in (400, 404):
                logger.error(
                    f"[LLM_FAILURE] stage=call_llm reason=api_error status={status_code} "
                    f"model={settings.GROQ_MODEL} "
                    f"hint=Check_GROQ_MODEL_env_var_and_reasoning_effort_param_compatibility "
                    f"reasoning_effort_sent={_supports_reasoning_effort(settings.GROQ_MODEL)}"
                )
            else:
                logger.error(
                    f"[LLM_FAILURE] stage=call_llm reason=api_error status={status_code} "
                    f"model={settings.GROQ_MODEL}"
                )
            raise LLMError("AI model is currently unavailable or misconfigured. Please try again later.")
        except GroqError as e:
            logger.error(f"[LLM_FAILURE] stage=call_llm reason=groq_error type={type(e).__name__} model={settings.GROQ_MODEL}")
            raise LLMError("AI model connection failed. Please try again later.")
        except Exception:
            logger.exception(f"[LLM_FAILURE] stage=call_llm reason=unexpected model={settings.GROQ_MODEL}")
            raise LLMError("AI model is currently unavailable or misconfigured. Please try again later.")
