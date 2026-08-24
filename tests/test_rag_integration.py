import pytest
from unittest.mock import patch, MagicMock
from app.services.rag_service import RAGService, _empty_response
from app.models.response import RetrievalResult, QueryResponse

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
    
    # Mock retrieval to return dummy results
    dummy_chunk = RetrievalResult(chunk_id="1", content="HTTP is...", title="HTTP", section="", preview="", similarity_score=1.0, metadata_boost=0.0, source="test", final_score=1.0)
    
    rag_service.retriever.search.return_value = ([dummy_chunk], MagicMock(expanded="What is HTTP?", was_expanded=False), [1.0])
    
    rag_service.answer("what is HTTP?", history)
    
    # Verify retrieval was called with cleaned query
    rag_service.retriever.search.assert_called_with("What is HTTP?")

def test_retrieval_final_zero_produces_fallback(rag_service):
    # Mock retrieval returning nothing
    rag_service.retriever.search.return_value = ([], MagicMock(expanded="xyz", was_expanded=False), [])
    
    response = rag_service.answer("what is HTTP?")
    
    assert response.answer == "The information was not found in the webpage."
    assert len(response.sources) == 0
    
    # LLM should not be called to generate an answer
    rag_service.client.chat.completions.create.assert_not_called()

@patch('app.services.rag_service.PromptService.build_messages')
def test_retrieval_final_gt_zero_sends_context_to_llm(mock_build, rag_service):
    # Mock LLM response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "HTTP is a protocol."
    rag_service.client.chat.completions.create.return_value = mock_response
    
    # Mock retrieval returning results
    dummy_chunk = RetrievalResult(chunk_id="1", content="HTTP is a protocol used for web.", title="HTTP", section="", preview="", similarity_score=0.9, metadata_boost=0.0, source="test", final_score=0.9)
    rag_service.retriever.search.return_value = ([dummy_chunk], MagicMock(expanded="what is HTTP", was_expanded=False), [0.9])
    
    # Mock build_messages to just return a dummy message
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
