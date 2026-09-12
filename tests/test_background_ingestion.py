"""
Background ingestion task tests.

NOTE: test was updated during the production audit (BUG-08 fix).
The original test asserted that the raw exception message appeared in
document.error_message, but that was the *old insecure behavior*.

BUG-08 fix: Internal error messages (e.g. stack traces, HF_TOKEN mentions)
must NOT reach document.error_message. The new behavior produces a
sanitized, user-friendly string instead.

The updated test now correctly asserts:
  1. document.status == "failed"  (unchanged)
  2. document.error_message is a non-empty, user-facing string
  3. Internal technical details are NOT present in error_message
"""
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

    # ── Status must be 'failed' ────────────────────────────────────────────────
    assert updated_doc.status == "failed"

    # ── BUG-08 FIX: error_message must be sanitized ───────────────────────────
    # The old assertion was:
    #   assert "Simulated fatal ingestion error" in updated_doc.error_message
    # That was verifying the *insecure* behavior where raw internal exception text
    # reached the frontend. The correct behavior is a sanitized user-facing message.

    assert updated_doc.error_message is not None
    assert len(updated_doc.error_message) > 0, "error_message must not be blank"

    # Raw internal details must NOT leak to users
    assert "Simulated fatal ingestion error" not in updated_doc.error_message, (
        "BUG-08: Raw internal exception text must not appear in error_message. "
        "It would be shown directly to users in the Dashboard UI."
    )

    db.close()
