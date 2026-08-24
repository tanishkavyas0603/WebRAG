import pytest
from unittest.mock import patch

from app.services.rag_service import RAGService
from app.core.config import settings

def test_missing_groq_api_key_raises_error():
    """Verify that a missing GROQ_API_KEY is handled clearly upon RAGService initialization."""
    # Temporarily remove GROQ_API_KEY
    original_key = settings.GROQ_API_KEY
    settings.GROQ_API_KEY = ""
    
    try:
        with pytest.raises(RuntimeError, match="GROQ_API_KEY is not set in environment."):
            RAGService(document_id=1)
    finally:
        # Restore the key
        settings.GROQ_API_KEY = original_key
