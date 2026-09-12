import asyncio
import hashlib
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
import traceback

from app.core.database import get_db, SessionLocal
from app.api.deps import get_current_user, get_rate_limiter
from app.models.db import User, Document, Chunk
from app.models.schemas import DocumentIngestRequest, DocumentResponse
from app.services.ingestion_service import DocumentIngestionService, IngestionError, SSRFProtectionError
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.vectorstore.faiss_store import FAISSVectorStore
from app.services.bm25_service import BM25Service
from app.core.logging import get_logger
from urllib.parse import urlparse

logger = get_logger(__name__)

router = APIRouter()


def _user_friendly_error(exc: Exception) -> str:
    """
    BUG-08 FIX: Map internal exception messages to user-safe strings.

    Internal errors like "Hugging Face authentication failed (HTTP 401). Please check your HF_TOKEN."
    or "Embedding API network request failed: ..." must not be shown directly to users.
    Log the full exception internally; return a sanitised message for the DB/UI.
    """
    msg = str(exc)

    # Errors from our own IngestionError class (already user-friendly from ingestion_service.py)
    if isinstance(exc, SSRFProtectionError):
        return "This URL is not allowed for security reasons."
    if isinstance(exc, IngestionError):
        return msg  # IngestionError messages are already user-facing

    # Embedding / HF API errors
    if "HF_TOKEN" in msg or "Hugging Face authentication" in msg or "huggingface" in msg.lower():
        return "The embedding service is currently unavailable. Please try again later."
    if "Embedding API" in msg or "embedding" in msg.lower():
        return "Failed to generate document embeddings. Please try again later."

    # Generic catch-all — do not expose internal details
    logger.warning(f"[INGESTION] Unmapped internal error being sanitised: {msg}")
    return "An unexpected error occurred during document processing. Please try again."


def process_document_background(document_id: int, url: str, user_id: int):
    logger.info(f"[INGESTION] BACKGROUND TASK STARTED DOCUMENT ID={document_id}")
    
    # Use a new DB session for background task
    try:
        db = SessionLocal()
    except Exception as e:
        logger.error(f"[INGESTION] BACKGROUND TASK FAILED creating DB session: {e}\n{traceback.format_exc()}")
        return
        
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            logger.error(f"[INGESTION] Document {document_id} not found in DB.")
            db.close()
            return

        logger.info(f"[INGESTION] Starting document {document_id}")
        
        # Ingestion
        ingestion_service = DocumentIngestionService(db, user_id, url)
        
        logger.info("[INGESTION] Fetching webpage")
        
        # Safe async wrapper since background tasks in FastAPI run in threads if defined as 'def'
        try:
            html = asyncio.run(ingestion_service.fetch_html())
            logger.info("[INGESTION] Webpage fetched")
            clean_text, title = ingestion_service.extract_text(html)
            logger.info("[INGESTION] Text extracted")
        except Exception as e:
            logger.error(f"[INGESTION] BACKGROUND TASK FAILED (fetching/extracting): {e}\n{traceback.format_exc()}")
            document.status = "failed"
            # BUG-08 FIX: Map internal errors to user-friendly messages.
            # Raw exception messages (e.g. "HF_TOKEN missing") must not reach users.
            document.error_message = _user_friendly_error(e)
            db.commit()
            return
            
        logger.info(f"[INGESTION] Extracted content: {len(clean_text)} characters")

        document.content = clean_text
        document.title = title[:255]
        document.content_hash = hashlib.sha256(clean_text.encode('utf-8')).hexdigest()
        
        # Chunking
        chunking_service = ChunkingService(db, document)
        chunks = chunking_service.run()
        
        if not chunks:
            error_msg = "No text content found to chunk."
            logger.error(f"[INGESTION] BACKGROUND TASK FAILED: {error_msg}")
            document.status = "failed"
            document.error_message = error_msg
            db.commit()
            return

        logger.info(f"[INGESTION] Chunks created: {len(chunks)}")

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
        logger.info("[INGESTION] Building BM25 index")
        bm25_service = BM25Service(document_id)
        bm25_service.build(chunk_dicts)

        # FAISS Index
        logger.info("[EMBEDDING] Starting embedding generation")
        embed_service = EmbeddingService()
        embeddings = embed_service.generate_embeddings(chunks)
        logger.info(f"[EMBEDDING] Embeddings generated: {len(embeddings)}")

        faiss_store = FAISSVectorStore(document_id)
        faiss_store.build(embeddings)
        faiss_store.save_metadata(chunk_dicts)
        
        logger.info("[INGESTION] FAISS index built")
        
        # Mark as ready
        document.status = "ready"
        db.commit()
        logger.info("[INGESTION] Document marked completed")
        
    except Exception as e:
        logger.error(f"[INGESTION] BACKGROUND TASK FAILED: {str(e)}\n{traceback.format_exc()}")
        db.rollback() # Rollback any pending uncommitted changes
        
        try:
            # Refresh document inside this session to mark it as failed
            document = db.query(Document).filter(Document.id == document_id).first()
            if document:
                document.status = "failed"
                # BUG-08 FIX: sanitise error message before storing — never expose internals to users
                document.error_message = _user_friendly_error(e)
                db.commit()
        except Exception as inner_e:
            logger.error(f"[INGESTION] Critical failure updating document status: {str(inner_e)}")
    finally:
        db.close()


@router.post("/ingest", response_model=DocumentResponse)
async def ingest_document(
    request: DocumentIngestRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_rate_limiter())
):
    logger.info("[INGESTION] POST /documents/ingest RECEIVED")
    
    parsed = urlparse(request.url)
    if not parsed.scheme or not parsed.hostname:
        raise HTTPException(status_code=422, detail="Invalid URL format")
    
    # Quick check for existing ready document with this URL
    existing_doc = db.query(Document).filter(
        Document.user_id == current_user.id,
        Document.url == request.url
    ).first()
    
    if existing_doc:
        if existing_doc.status == "ready":
            logger.info(f"[INGESTION] Returning existing ready document ID={existing_doc.id}")
            return existing_doc
        else:
            # The document is stuck in pending/processing (zombie task from restart) or failed.
            # We delete it so we can start fresh.
            logger.info(f"[INGESTION] Deleting stuck/failed document ID={existing_doc.id} to start fresh")
            db.delete(existing_doc)
            db.commit()
        
    new_doc = Document(
        user_id=current_user.id,
        url=request.url,
        content_hash="", # temporary, updated in background
        status="pending"
    )
    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)
    
    logger.info(f"[INGESTION] CREATED DOCUMENT ID={new_doc.id}")
    
    logger.info(f"[INGESTION] CALLING add_task FOR DOCUMENT ID={new_doc.id}")
    background_tasks.add_task(process_document_background, new_doc.id, request.url, current_user.id)
    logger.info(f"[INGESTION] add_task COMPLETED FOR DOCUMENT ID={new_doc.id}")
    
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
