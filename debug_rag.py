import os
import sys
from app.services.rag_service import RAGService
from app.core.database import SessionLocal
from app.models.db import Document

def main():
    db = SessionLocal()
    doc = db.query(Document).first()
    if doc:
        print('Found doc:', doc.id)
        rag = RAGService(document_id=doc.id)
        resp = rag.answer('What is the main topic of this document?')
        print('Response answer length:', len(resp.answer))
        print('Response answer:', repr(resp.answer))
    else:
        print('No doc found')

if __name__ == "__main__":
    main()
