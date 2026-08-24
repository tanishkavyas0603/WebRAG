import asyncio
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

def process_document_background(document_id: int, url: str, user_id: int):
    logger.info(f"[INGESTION] BACKGROUND TASK STARTED for document {document_id}")
    
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
            document.error_message = str(e)
            db.commit()
            return
            
        logger.info(f"[INGESTION] Extracted content: {len(clean_text)} characters")

        document.content = clean_text
        document.title = title[:255]
        
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
                document.error_message = str(e)
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
    
    logger.info(f"[INGESTION] Scheduling background ingestion for document {new_doc.id}")
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
