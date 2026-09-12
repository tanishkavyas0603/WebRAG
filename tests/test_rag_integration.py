import pytest
from unittest.mock import patch, MagicMock
from app.services.rag_service import RAGService, _empty_response
from app.models.response import RetrievalResult, QueryResponse


def _make_chunk(chunk_id: int = 1, content: str = "HTTP is a protocol.", title: str = "HTTP",
                similarity_score: float = 0.9) -> RetrievalResult:
    """
    BUG-06 FIX: Helper builds RetrievalResult with correct types.
    Previous tests used chunk_id="1" (string) and final_score=1.0 (nonexistent field).
    Pydantic v2 coerced the string to int silently, masking the type mismatch.
    """
    return RetrievalResult(
        chunk_id=chunk_id,            # int — correct type
        content=content,
        title=title,
        section="",
        preview="",
        similarity_score=similarity_score,
        cosine_score=similarity_score,  # cosine_score must also be set
        metadata_boost=0.0,
        source="test",
    )


@pytest.fixture
def rag_service():
    with patch('app.services.rag_service.Groq'), \
         patch('app.services.rag_service.RetrievalService'):
        service = RAGService(document_id=1)
        # Setup mocks
        service.client.chat.completions.create = MagicMock()
        service.retriever.search = MagicMock()
        return service


def test_query_rewriting_never_passes_think_into_retrieval(rag_service):
    history = [{"role": "user", "content": "hello"}]
    
    # Mock LLM returning think block
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "<think>reasoning...</think>What is HTTP?"
    
    # Setup LLM calls (first for rewrite, second for answer)
    rag_service.client.chat.completions.create.side_effect = [mock_response, mock_response]
    
    # BUG-06 FIX: Use correct types and fields for RetrievalResult
    dummy_chunk = _make_chunk(chunk_id=1, content="HTTP is...", title="HTTP", similarity_score=1.0)
    
    rag_service.retriever.search.return_value = (
        [dummy_chunk],
        MagicMock(expanded="What is HTTP?", was_expanded=False, bm25_terms=[]),
        [1.0]
    )
    
    rag_service.answer("what is HTTP?", history)
    
    # Verify retrieval was called with cleaned query (no <think> block)
    rag_service.retriever.search.assert_called_with("What is HTTP?")


def test_retrieval_final_zero_produces_fallback(rag_service):
    """When retrieval returns no chunks, answer must be the fallback string — never empty."""
    rag_service.retriever.search.return_value = (
        [],
        MagicMock(expanded="xyz", was_expanded=False, bm25_terms=[]),
        []
    )
    
    response = rag_service.answer("what is HTTP?")
    
    assert response.answer == "The information was not found in the webpage."
    assert len(response.sources) == 0
    
    # LLM should NOT be called when there are no chunks to ground on
    rag_service.client.chat.completions.create.assert_not_called()


@patch('app.services.rag_service.PromptService.build_messages')
def test_retrieval_final_gt_zero_sends_context_to_llm(mock_build, rag_service):
    # Mock LLM response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "HTTP is a protocol."
    rag_service.client.chat.completions.create.return_value = mock_response
    
    # BUG-06 FIX: Use correct types for RetrievalResult
    dummy_chunk = _make_chunk(chunk_id=1, content="HTTP is a protocol used for web.", title="HTTP", similarity_score=0.9)
    rag_service.retriever.search.return_value = (
        [dummy_chunk],
        MagicMock(expanded="what is HTTP", was_expanded=False, bm25_terms=[]),
        [0.9]
    )
    
    # Mock build_messages to return a minimal prompt
    mock_build.return_value = [{"role": "system", "content": "dummy"}]
    
    response = rag_service.answer("what is HTTP?")
    
    # Verify build_messages was called with our chunk
    assert mock_build.called
    chunks_passed = mock_build.call_args[0][1]
    assert len(chunks_passed) == 1
    assert chunks_passed[0].content == "HTTP is a protocol used for web."
    
    # Verify LLM was called to generate the final answer
    rag_service.client.chat.completions.create.assert_called()
    assert response.answer == "HTTP is a protocol."


def test_answer_never_returns_empty_string(rag_service):
    """
    BUG-02 regression test: If LLM returns only a <think> block that gets stripped away,
    the answer must NOT be an empty string. Should be the fallback message.
    """
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    # LLM returns ONLY a think block — after stripping, content is empty
    mock_response.choices[0].message.content = "<think>I have no answer</think>"
    rag_service.client.chat.completions.create.return_value = mock_response

    dummy_chunk = _make_chunk(chunk_id=1, content="Some webpage content.", similarity_score=0.8)
    rag_service.retriever.search.return_value = (
        [dummy_chunk],
        MagicMock(expanded="what?", was_expanded=False, bm25_terms=[]),
        [0.8]
    )

    response = rag_service.answer("what?")

    # The answer must NOT be empty — BUG-02 would have returned ""
    assert response.answer != ""
    assert len(response.answer) > 0


def test_answer_never_returns_none(rag_service):
    """
    BUG-02 regression test: If LLM returns None content, the answer must be an
    appropriate error message, not None or a blank message.
    """
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = None
    rag_service.client.chat.completions.create.return_value = mock_response

    from groq import APIStatusError
    # When content is None, _call_llm raises LLMError which propagates to answer()
    # which should re-raise it (caught by conversations.py as 503)
    dummy_chunk = _make_chunk(chunk_id=1, content="Some content.", similarity_score=0.8)
    rag_service.retriever.search.return_value = (
        [dummy_chunk],
        MagicMock(expanded="test", was_expanded=False, bm25_terms=[]),
        [0.8]
    )

    from app.services.rag_service import LLMError
    with pytest.raises(LLMError):
        rag_service.answer("test?")
