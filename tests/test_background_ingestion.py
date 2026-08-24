import pytest
from app.models.db import Document
from app.api.documents import process_document_background
from app.core.database import SessionLocal

def test_background_ingestion_exception_marks_failed(monkeypatch):
    db = SessionLocal()
    # Create a dummy document
    doc = Document(user_id=1, url="http://example.com", content_hash="", status="pending")
    db.add(doc)
    db.commit()
    doc_id = doc.id
    
    # Mock DocumentIngestionService to raise an Exception
    class MockIngestionService:
        def __init__(self, *args, **kwargs):
            pass
        async def fetch_html(self):
            raise Exception("Simulated fatal ingestion error")
            
    monkeypatch.setattr("app.api.documents.DocumentIngestionService", MockIngestionService)
    
    # Run the background task
    process_document_background(doc_id, "http://example.com", 1)
    
    # Check the database
    db.expire_all()
    updated_doc = db.query(Document).filter(Document.id == doc_id).first()
    assert updated_doc.status == "failed"
    assert "Simulated fatal ingestion error" in updated_doc.error_message
    
    db.close()
