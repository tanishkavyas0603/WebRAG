from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
import traceback

from app.core.database import get_db, SessionLocal
from app.api.deps import get_current_user
from app.models.db import User, Document, Chunk
from app.models.schemas import DocumentIngestRequest, DocumentResponse
from app.services.ingestion_service import DocumentIngestionService
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.vectorstore.faiss_store import FAISSVectorStore
from app.services.bm25_service import BM25Service
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()

async def process_document_background(document_id: int, url: str, user_id: int):
    # Use a new DB session for background task
    db = SessionLocal()
    document = db.query(Document).filter(Document.id == document_id).first()
    
    try:
        # Ingestion
        ingestion_service = DocumentIngestionService(db, user_id, url)
        
        try:
            html = await ingestion_service.fetch_html()
            clean_text, title = ingestion_service.extract_text(html)
        except Exception as e:
            logger.error(f"Ingestion failed for {url}: {e}")
            document.status = "failed"
            document.error_message = str(e)
            db.commit()
            db.close()
            return

        document.content = clean_text
        document.title = title[:255]
        
        # Chunking
        chunking_service = ChunkingService(db, document)
        chunks = chunking_service.run()
        
        if not chunks:
            document.status = "failed"
            document.error_message = "No text content found to chunk."
            db.commit()
            db.close()
            return

        # Prepare for indexing
        chunk_dicts = [
            {
                "id": c.id,
                "content": c.content,
                "title": c.metadata_.get("title", ""),
                "section": c.metadata_.get("section", ""),
                "preview": c.metadata_.get("preview", "")
            }
            for c in chunks
        ]

        # BM25 Index
        bm25_service = BM25Service(document_id)
        bm25_service.build(chunk_dicts)

        # FAISS Index
        embed_service = EmbeddingService()
        embeddings = embed_service.generate_embeddings(chunks)

        faiss_store = FAISSVectorStore(document_id)
        faiss_store.build(embeddings)
        faiss_store.save_metadata(chunk_dicts)
        
        # Mark as ready
        document.status = "ready"
        db.commit()
        logger.info(f"Document {document_id} processed successfully.")
    except Exception as e:
        logger.error(f"Unexpected error processing document {document_id}: {traceback.format_exc()}")
        document.status = "failed"
        document.error_message = "An unexpected error occurred during processing."
        db.commit()
    finally:
        db.close()


@router.post("/ingest", response_model=DocumentResponse)
async def ingest_document(
    request: DocumentIngestRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Quick check for existing processing/ready document with this URL
    existing_doc = db.query(Document).filter(
        Document.user_id == current_user.id,
        Document.url == request.url,
        Document.status.in_(["pending", "processing", "ready"])
    ).first()
    
    if existing_doc:
        return existing_doc
        
    new_doc = Document(
        user_id=current_user.id,
        url=request.url,
        content_hash="", # temporary, updated in background
        status="pending"
    )
    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)
    
    background_tasks.add_task(process_document_background, new_doc.id, request.url, current_user.id)
    
    return new_doc


@router.get("/{document_id}/status", response_model=DocumentResponse)
def get_document_status(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    document = db.query(Document).filter(Document.id == document_id, Document.user_id == current_user.id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document
