import pytest
from unittest.mock import patch, MagicMock

from app.services.rag_service import RAGService
from groq.types.chat import ChatCompletion, ChatCompletionMessage
from groq.types.chat.chat_completion import Choice

@pytest.fixture
def rag_service():
    with patch('app.services.rag_service.Groq'), \
         patch('app.services.rag_service.RetrievalService'):
        service = RAGService(document_id=1)
        # Mock the completions creation to be easily overrideable
        service.client.chat.completions.create = MagicMock()
        return service

def _mock_groq_response(content: str):
    message = ChatCompletionMessage(
        content=content,
        role="assistant"
    )
    choice = Choice(
        finish_reason="stop",
        index=0,
        message=message
    )
    # Using mock instead of proper model instantiation to bypass validation if any
    response = MagicMock(spec=ChatCompletion)
    response.choices = [choice]
    return response

def test_rewrite_no_history_returns_original(rag_service):
    # If no history, it should return exactly the same string without calling Groq
    question = "what is HTTP"
    result = rag_service._rewrite_query(question, [])
    assert result == question
    rag_service.client.chat.completions.create.assert_not_called()

def test_rewrite_normal_clean(rag_service):
    history = [{"role": "user", "content": "hello"}]
    rag_service.client.chat.completions.create.return_value = _mock_groq_response("What is HTTP?")
    
    result = rag_service._rewrite_query("what is HTTP", history)
    assert result == "What is HTTP?"

def test_rewrite_strips_think_block(rag_service):
    history = [{"role": "user", "content": "hello"}]
    mock_content = "<think>\nThinking about this...\n</think>\nWhat are the components of HTTP?"
    rag_service.client.chat.completions.create.return_value = _mock_groq_response(mock_content)
    
    result = rag_service._rewrite_query("components of HTTP", history)
    assert result == "What are the components of HTTP?"

def test_rewrite_strips_prefixes(rag_service):
    history = [{"role": "user", "content": "hello"}]
    mock_content = "Draft: What is HTTP?"
    rag_service.client.chat.completions.create.return_value = _mock_groq_response(mock_content)
    
    result = rag_service._rewrite_query("what is HTTP", history)
    assert result == "What is HTTP?"
    
    mock_content2 = "Rewritten Query: What is HTTP?"
    rag_service.client.chat.completions.create.return_value = _mock_groq_response(mock_content2)
    assert rag_service._rewrite_query("what is HTTP", history) == "What is HTTP?"

def test_rewrite_empty_fallback(rag_service):
    history = [{"role": "user", "content": "hello"}]
    mock_content = "<think> I don't know what to do </think>"
    rag_service.client.chat.completions.create.return_value = _mock_groq_response(mock_content)
    
    result = rag_service._rewrite_query("what is HTTP", history)
    assert result == "what is HTTP" # Falls back to original query

def test_rewrite_excessively_long_fallback(rag_service):
    history = [{"role": "user", "content": "hello"}]
    mock_content = "A" * 250
    rag_service.client.chat.completions.create.return_value = _mock_groq_response(mock_content)
    
    result = rag_service._rewrite_query("what is HTTP", history)
    assert result == "what is HTTP" # Falls back to original query

def test_rewrite_complex_followup(rag_service):
    history = [
        {"role": "user", "content": "tell me about TCP"},
        {"role": "assistant", "content": "TCP is a protocol"}
    ]
    mock_content = "<think>The user wants to know how it compares to UDP</think>\nHow does TCP compare to UDP?"
    rag_service.client.chat.completions.create.return_value = _mock_groq_response(mock_content)
    
    result = rag_service._rewrite_query("what about UDP?", history)
    assert result == "How does TCP compare to UDP?"
