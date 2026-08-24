import pytest
import httpx
from unittest.mock import MagicMock
import numpy as np

@pytest.fixture(autouse=True)
def mock_hf_inference(monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "HF_TOKEN", "dummy_token_for_tests")
    original_client_post = httpx.Client.post

    def mocked_post(self, url, *args, **kwargs):
        if "api-inference.huggingface.co" in url or "router.huggingface.co" in url:
            # Mock the response
            payload = kwargs.get("json", {})
            inputs = payload.get("inputs", [])
            # Return dummy embeddings of size 384 for each input.
            # Must be non-zero to pass cosine similarity threshold > 0.15 in retrieval_service.py
            dummy_embeddings = np.ones((len(inputs), 384)).tolist()
            
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = dummy_embeddings
            mock_response.raise_for_status.return_value = None
            return mock_response
        
        return original_client_post(self, url, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "post", mocked_post)
