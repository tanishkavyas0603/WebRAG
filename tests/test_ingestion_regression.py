import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.database import SessionLocal
from app.models.db import Document, User
import asyncio
import time

def test_regression_ingestion_background_task_scheduled(monkeypatch):
    """
    Proves that POST /api/documents/ingest schedules the task
    and the document eventually transitions status.
    """
    client = TestClient(app)
    db = SessionLocal()
    
    # 1. Setup a test user and get a token
    test_email = f"test_regression_{time.time()}@example.com"
    client.post("/api/auth/register", json={"email": test_email, "password": "password123"})
    login_resp = client.post("/api/auth/login", data={"username": test_email, "password": "password123"})
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # 2. Mock the actual ingestion to avoid external network calls in test
    # We will simulate a successful fetch that finishes instantly
    class MockIngestionService:
        def __init__(self, db, user_id, url):
            pass
            
        async def fetch_html(self):
            return "<html><title>Test Page</title><body><p>This is test content for regression.</p></body></html>"
            
        def extract_text(self, html):
            return "This is test content for regression.", "Test Page"
            
    monkeypatch.setattr("app.api.documents.DocumentIngestionService", MockIngestionService)
    
    # Mock FAISS and BM25 to avoid heavy computation
    monkeypatch.setattr("app.api.documents.BM25Service", lambda doc_id: type("MockBM25", (), {"build": lambda self, x: None})())
    monkeypatch.setattr("app.api.documents.FAISSVectorStore", lambda doc_id: type("MockFAISS", (), {"build": lambda self, x: None, "save_metadata": lambda self, x: None})())
    
    # Mock EmbeddingService
    monkeypatch.setattr("app.api.documents.EmbeddingService", lambda: type("MockEmbed", (), {"generate_embeddings": lambda self, x: []})())
    
    # 3. Trigger ingestion
    test_url = f"https://example.com/regression/{time.time()}"
    resp = client.post("/api/documents/ingest", json={"url": test_url}, headers=headers)
    
    assert resp.status_code == 200
    doc_id = resp.json()["id"]
    
    # Fast API TestClient executes BackgroundTasks synchronously immediately AFTER returning the response!
    # So by the time we check the DB, the task has ALREADY RUN!
    
    # 4. Verify the document transitioned
    db.expire_all()
    doc = db.query(Document).filter(Document.id == doc_id).first()
    
    # Status should be 'ready' or 'failed', definitely not 'pending' anymore!
    assert doc.status in ["ready", "failed"], f"Document is stuck in {doc.status}"
    
    db.close()
