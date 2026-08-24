import os
import shutil
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.core.database import SessionLocal, Base, engine
from app.models.db import User, Document, Chunk, Conversation
from app.api.deps import get_current_user
import app.core.constants as constants

# Use a test database or clear existing
Base.metadata.create_all(bind=engine)

def override_get_current_user():
    db = SessionLocal()
    user = db.query(User).filter_by(email="test_faiss_recovery@example.com").first()
    if not user:
        user = User(email="test_faiss_recovery@example.com", password_hash="dummy")
        db.add(user)
        db.commit()
        db.refresh(user)
    db.close()
    return user

app.dependency_overrides[get_current_user] = override_get_current_user

def test_production_faiss_recovery_scenario():
    db = SessionLocal()
    
    # 1. Setup: Create user, document, chunks, conversation
    user = override_get_current_user()
    
    doc = Document(user_id=user.id, url="http://test-recovery.com", content_hash="hash_rec_1", status="ready", content="HTTP is a protocol.")
    db.add(doc)
    db.commit()
    db.refresh(doc)
    
    chunk1 = Chunk(document_id=doc.id, chunk_index=1, content="HTTP stands for HyperText Transfer Protocol.", metadata_={"title": "HTTP", "section": "", "preview": "HTTP stands..."})
    db.add(chunk1)
    
    conv = Conversation(user_id=user.id, document_id=doc.id, title="Test Chat")
    db.add(conv)
    db.commit()
    db.refresh(conv)
    
    # 2. Assert FAISS file does NOT exist (we never built it for this dummy doc)
    index_dir = Path(constants.INDEX_DIR)
    faiss_file = index_dir / f"doc_{doc.id}.faiss"
    if faiss_file.exists():
        faiss_file.unlink()
        
    assert not faiss_file.exists(), "FAISS file should not exist, simulating ephemeral storage loss."
    
    # 3. Simulate user sending a message to this conversation
    # We mock Groq since we don't want to actually call the LLM in this test
    # We mock HuggingFace so we don't need real token in this test
    client = TestClient(app)
    
    with patch('app.services.rag_service.Groq') as MockGroq, \
         patch('app.services.embedding_service.httpx.Client') as MockHTTPXClient:
         
        # Mock HF Embeddings
        mock_hf_response = MagicMock()
        mock_hf_response.status_code = 200
        mock_hf_response.json.return_value = [[0.1] * 384] # 384-dimensional fake embedding
        mock_httpx_instance = MockHTTPXClient.return_value.__enter__.return_value
        mock_httpx_instance.post.return_value = mock_hf_response
        
        # Mock Groq Answer
        mock_llm_response = MagicMock()
        mock_llm_response.choices = [MagicMock()]
        mock_llm_response.choices[0].message.content = "HTTP is a protocol."
        
        mock_groq_instance = MockGroq.return_value
        mock_groq_instance.chat.completions.create.return_value = mock_llm_response
        
        response = client.post(f"/api/conversations/{conv.id}/messages", json={
            "message": "What is HTTP?"
        })
        
        assert response.status_code == 200, f"Request failed: {response.text}"
        data = response.json()
        assert "HTTP is a protocol" in data["content"]
        
    # 4. Verify that FAISS file was created during the request!
    assert faiss_file.exists(), "FAISS file was not automatically rebuilt during the request!"
