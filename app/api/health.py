from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.core.database import get_db

router = APIRouter()

@router.get("")
def health_check(db: Session = Depends(get_db)):
    status = {
        "status": "healthy",
        "database": "disconnected",
        "rag": "ready",
        "llm": "available" # We assume Groq is available if key is set
    }
    
    try:
        db.execute(text("SELECT 1"))
        status["database"] = "connected"
    except Exception:
        status["status"] = "unhealthy"
        
    return status
