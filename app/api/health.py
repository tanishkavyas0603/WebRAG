from fastapi import APIRouter, Response
from sqlalchemy import text
from app.core.database import SessionLocal
from app.core.config import settings
from app.core.logging import get_logger
from pathlib import Path

router = APIRouter()
logger = get_logger(__name__)


@router.get("")
def health_check():
    """
    Liveness check — answers: is the FastAPI process running?
    Render uses this to decide if the dyno should be restarted.
    Must remain fast and never block on external services.
    """
    return {"status": "ok", "service": "webrag"}


@router.get("/ready")
def readiness_check(response: Response):
    """
    Readiness check — answers: is the service ready to serve requests?
    Validates:
      - Database connectivity
      - Required configuration is present (HF_TOKEN, GROQ_API_KEY, GROQ_MODEL)
      - FAISS index directory is writable

    Never makes external LLM or embedding API calls.
    Never exposes secret values \u2014 only presence/absence.
    """
    checks: dict[str, str] = {}
    healthy = True

    # 1. Database connectivity
    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as e:
        logger.error(f"[READINESS] Database check failed: {type(e).__name__}")
        checks["database"] = "error"
        healthy = False

    # 2. Required configuration \u2014 log presence only, never values
    if not settings.GROQ_API_KEY:
        logger.error("[READINESS] GROQ_API_KEY is not configured")
        checks["groq_api_key"] = "missing"
        healthy = False
    else:
        checks["groq_api_key"] = "set"

    if not settings.HF_TOKEN:
        logger.warning("[READINESS] HF_TOKEN is not configured \u2014 embeddings will fail")
        checks["hf_token"] = "missing"
        # Not marking unhealthy: the app can serve cached FAISS if index exists
    else:
        checks["hf_token"] = "set"

    checks["groq_model"] = settings.GROQ_MODEL

    # Warn if the configured model is the known-deprecated one
    if settings.GROQ_MODEL == "qwen/qwen3.6-27b":
        logger.warning(
            "[READINESS] GROQ_MODEL=qwen/qwen3.6-27b is DEPRECATED on Groq. "
            "Update GROQ_MODEL to qwen/qwen3.8-27b in Render environment variables."
        )
        checks["groq_model_status"] = "deprecated"
        healthy = False
    else:
        checks["groq_model_status"] = "ok"

    # 3. FAISS index directory \u2014 check it exists and is writable
    try:
        index_dir = Path(settings.FAISS_INDEX_DIR)
        index_dir.mkdir(parents=True, exist_ok=True)
        test_file = index_dir / ".write_test"
        test_file.write_text("ok")
        test_file.unlink()
        checks["faiss_index_dir"] = "writable"
    except Exception as e:
        logger.error(f"[READINESS] FAISS index dir not writable: {settings.FAISS_INDEX_DIR} \u2014 {type(e).__name__}")
        checks["faiss_index_dir"] = "error"
        healthy = False

    if not healthy:
        response.status_code = 503

    return {
        "status": "ready" if healthy else "not_ready",
        "checks": checks,
    }
