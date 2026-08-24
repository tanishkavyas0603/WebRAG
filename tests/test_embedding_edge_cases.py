import pytest
import httpx
from unittest.mock import MagicMock
from app.services.embedding_service import EmbeddingService
from app.core.config import settings

@pytest.fixture(autouse=True)
def set_hf_token(monkeypatch, request):
    if "test_missing_hf_token" not in request.node.name:
        monkeypatch.setattr(settings, "HF_TOKEN", "dummy_token")

def test_missing_hf_token(monkeypatch):
    monkeypatch.setattr(settings, "HF_TOKEN", None)
    with pytest.raises(ValueError, match="HF_TOKEN environment variable is missing"):
        EmbeddingService()

def test_hf_401_403(monkeypatch):
    service = EmbeddingService()
    
    def mock_post(*args, **kwargs):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        return mock_resp
        
    monkeypatch.setattr(httpx.Client, "post", mock_post)
    with pytest.raises(RuntimeError, match="Hugging Face authentication failed"):
        service.generate_embeddings(["test chunk"])

def test_hf_429(monkeypatch):
    service = EmbeddingService()
    
    def mock_post(*args, **kwargs):
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        return mock_resp
        
    monkeypatch.setattr(httpx.Client, "post", mock_post)
    with pytest.raises(RuntimeError, match="Hugging Face API rate limit exceeded"):
        service.generate_embeddings(["test chunk"])

def test_hf_503_retry_and_fail(monkeypatch):
    service = EmbeddingService()
    # Mock time.sleep to run fast
    monkeypatch.setattr("time.sleep", lambda x: None)
    
    call_count = 0
    def mock_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        return mock_resp
        
    monkeypatch.setattr(httpx.Client, "post", mock_post)
    with pytest.raises(RuntimeError, match="Hugging Face API is unavailable"):
        service.generate_embeddings(["test chunk"])
        
    assert call_count == 3  # Should retry 3 times

def test_hf_timeout(monkeypatch):
    service = EmbeddingService()
    monkeypatch.setattr("time.sleep", lambda x: None)
    
    call_count = 0
    def mock_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise httpx.TimeoutException("Timeout")
        
    monkeypatch.setattr(httpx.Client, "post", mock_post)
    with pytest.raises(RuntimeError, match="Hugging Face API request timed out"):
        service.generate_embeddings(["test chunk"])
        
    assert call_count == 3

def test_successful_hf_response(monkeypatch):
    # Testing that batching and basic response works correctly
    service = EmbeddingService()
    
    call_count = 0
    def mock_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        # Return 1 dummy embedding vector of size 384 for each input
        inputs = kwargs.get("json", {}).get("inputs", [])
        mock_resp.json.return_value = [[0.5] * 384 for _ in inputs]
        return mock_resp
        
    monkeypatch.setattr(httpx.Client, "post", mock_post)
    
    # 35 chunks should trigger 2 batches (batch size 32)
    chunks = ["chunk"] * 35
    result = service.generate_embeddings(chunks)
    
    assert call_count == 2
    assert result.shape == (35, 384)
