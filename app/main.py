from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth import router as auth_router
from app.api.documents import router as documents_router
from app.api.conversations import router as conversations_router
from app.api.health import router as health_router

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_ALEMBIC_INI = Path(__file__).resolve().parent.parent / "alembic.ini"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Apply any pending schema migrations on startup. This is what makes a
    # fresh deployment (e.g. a new Render Postgres database) work without a
    # manual "alembic upgrade head" step; it is a no-op when already current.
    # Failure here is logged, not fatal — the app still starts so /api/health
    # remains reachable for diagnosis rather than crash-looping.
    try:
        from alembic import command
        from alembic.config import Config

        alembic_cfg = Config(str(_ALEMBIC_INI))
        command.upgrade(alembic_cfg, "head")
        logger.info("Database migrations are up to date.")
    except Exception as e:
        logger.error(f"Failed to apply database migrations at startup: {e}")
    yield


app = FastAPI(
    title="WebRAG",
    version="1.0.0",
    description="Chat with any webpage. Ask questions. Get grounded answers.",
    lifespan=lifespan,
)

# Parse comma-separated origins
cors_origins = [origin.strip() for origin in settings.CORS_ORIGINS.split(",") if origin.strip()]

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(auth_router, prefix="/api/auth", tags=["auth"])
app.include_router(documents_router, prefix="/api/documents", tags=["documents"])
app.include_router(conversations_router, prefix="/api/conversations", tags=["conversations"])
app.include_router(health_router, prefix="/api/health", tags=["health"])