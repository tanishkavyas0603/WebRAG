from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth import router as auth_router
from app.api.documents import router as documents_router
from app.api.conversations import router as conversations_router
from app.api.health import router as health_router

from app.core.config import settings

app = FastAPI(
    title="WebRAG",
    version="1.0.0",
    description="Chat with any webpage. Ask questions. Get grounded answers."
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