"""
Regression test for the "Failed to generate answer" root cause.

GROQ_MODEL (qwen/qwen3.6-27b) is a reasoning model that spends completion
tokens on hidden/visible <think> reasoning before writing the final answer.
Without reasoning_effort="none", it reliably burns the entire max_tokens
budget on reasoning for any non-trivial context and returns truncated or
empty content (finish_reason="length"), which surfaces to users as
"Failed to generate answer". See app/services/rag_service.py.
"""
import pytest
from unittest.mock import patch, MagicMock

from app.services.rag_service import RAGService
from app.models.response import RetrievalResult


def _make_chunk() -> RetrievalResult:
    return RetrievalResult(
        chunk_id=1,
        content="HTTP is a protocol.",
        title="HTTP",
        section="",
        preview="",
        similarity_score=0.9,
        cosine_score=0.9,
        metadata_boost=0.0,
        source="test",
    )


@pytest.fixture
def rag_service():
    with patch('app.services.rag_service.Groq'), \
         patch('app.services.rag_service.RetrievalService'):
        service = RAGService(document_id=1)
        service.client.chat.completions.create = MagicMock()
        service.retriever.search = MagicMock()
        return service


def _mock_response(content: str):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


def test_call_llm_disables_reasoning(rag_service):
    rag_service.client.chat.completions.create.return_value = _mock_response("The answer.")
    rag_service._call_llm([{"role": "user", "content": "hi"}])

    _, kwargs = rag_service.client.chat.completions.create.call_args
    assert kwargs.get("reasoning_effort") == "none"


def test_rewrite_query_disables_reasoning(rag_service):
    rag_service.client.chat.completions.create.return_value = _mock_response("Standalone question?")
    rag_service._rewrite_query("follow up", [{"role": "user", "content": "hello"}])

    _, kwargs = rag_service.client.chat.completions.create.call_args
    assert kwargs.get("reasoning_effort") == "none"


def test_answer_end_to_end_disables_reasoning(rag_service):
    dummy_chunk = _make_chunk()
    rag_service.retriever.search.return_value = ([dummy_chunk], MagicMock(expanded="what is HTTP", was_expanded=False), [0.9])
    rag_service.client.chat.completions.create.return_value = _mock_response("HTTP is a protocol.")

    response = rag_service.answer("what is HTTP")

    assert response.answer == "HTTP is a protocol."
    _, kwargs = rag_service.client.chat.completions.create.call_args
    assert kwargs.get("reasoning_effort") == "none"
