from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List

from app.core.database import get_db
from app.api.deps import get_current_user
from app.models.db import User, Conversation, Document, Message
from app.models.schemas import ConversationCreate, ConversationResponse, MessageCreate, MessageResponse, Source
from app.services.conversation_service import ConversationService
from app.services.rag_service import RAGService
from urllib.parse import urlparse
from app.core.logging import get_logger

logger = get_logger(__name__)

def clean_document_title(doc_title: str, doc_url: str) -> str:
    if not doc_title:
        try:
            hostname = urlparse(doc_url).hostname
            return hostname if hostname else "Webpage Chat"
        except:
            return "Webpage Chat"

    if "Tutorial - User Guide - FastAPI" in doc_title:
        return "FastAPI — Tutorial"
    if "Overview of HTTP - HTTP | MDN" in doc_title:
        return "HTTP Overview — MDN"
    if "Python For Beginners" in doc_title:
        return "Python for Beginners"

    parts = [p.strip() for p in doc_title.replace("|", "-").split("-")]
    if len(parts) > 1:
        return f"{parts[0]} — {parts[-1]}"
    return parts[0]

router = APIRouter()

@router.get("", response_model=List[ConversationResponse])
def list_conversations(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    conversations = db.query(Conversation).filter(Conversation.user_id == current_user.id).order_by(Conversation.updated_at.desc()).all()
    for conv in conversations:
        conv.document_url = conv.document.url if conv.document else None
        
        # Clean bad existing titles on the fly for display
        if conv.title in ["New Chat", "Untitled"] or "<think>" in conv.title or "Untitled" in conv.title:
            doc_title = conv.document.title if conv.document else None
            conv.title = clean_document_title(doc_title, conv.document_url)
            
    return conversations

@router.post("", response_model=ConversationResponse)
def create_conversation(req: ConversationCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    doc = db.query(Document).filter(Document.id == req.document_id, Document.user_id == current_user.id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found or access denied")
    
    if doc.status != "ready":
        raise HTTPException(status_code=400, detail="Document is not ready for chatting")
        
    conv = Conversation(
        user_id=current_user.id, 
        document_id=req.document_id,
        title=clean_document_title(doc.title, doc.url)
    )
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv

@router.get("/{conversation_id}", response_model=ConversationResponse)
def get_conversation(conversation_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    conv = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
        
    conv.document_url = conv.document.url if conv.document else None
    if conv.title in ["New Chat", "Untitled"] or "<think>" in conv.title or "Untitled" in conv.title:
        doc_title = conv.document.title if conv.document else None
        conv.title = clean_document_title(doc_title, conv.document_url)
        
    return conv

@router.delete("/{conversation_id}")
def delete_conversation(conversation_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    conv = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    db.delete(conv)
    db.commit()
    return {"status": "deleted"}

@router.get("/{conversation_id}/messages", response_model=List[MessageResponse])
def get_messages(conversation_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    conv = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
        
    return conv.messages



@router.post("/{conversation_id}/messages", response_model=MessageResponse)
def send_message(
    conversation_id: int, 
    req: MessageCreate, 
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    conv = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
        
    conv_svc = ConversationService(db)
    
    # Save user message
    user_msg = conv_svc.add_message(conversation_id, "user", req.message)
    
    # Title generation is now handled synchronously at conversation creation,
    # so we don't need a background LLM task for it anymore.
    
    # Fetch recent history
    chat_history = conv_svc.get_recent_messages(conversation_id, limit=6)
    # The last message is the one we just added, so we don't need it in the 'history' passed to rewrite
    history_for_rag = chat_history[:-1]
    
    # Run RAG
    try:
        rag = RAGService(document_id=conv.document_id)
        rag_response = rag.answer(req.message, chat_history=history_for_rag)
        
        # Serialize citations
        citations = [s.model_dump() for s in rag_response.sources]
        
        # Save assistant message
        assistant_msg = conv_svc.add_message(
            conversation_id, 
            "assistant", 
            rag_response.answer, 
            citations=citations
        )
        return assistant_msg
        
    except Exception as e:
        # Check if the exception class name is LLMError to avoid circular imports if needed, 
        # or we can just import LLMError at the top. Let's import it locally or check class name.
        if type(e).__name__ == "LLMError":
            raise HTTPException(status_code=503, detail=str(e))
        logger.error(f"RAG failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate answer")
