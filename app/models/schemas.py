from pydantic import BaseModel, EmailStr, ConfigDict
from typing import List, Optional
from datetime import datetime


class UserCreate(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    email: str
    created_at: datetime


class Token(BaseModel):
    access_token: str
    token_type: str


class DocumentIngestRequest(BaseModel):
    url: str


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    url: str
    title: Optional[str] = None
    status: str
    error_message: Optional[str] = None
    created_at: datetime


class ConversationCreate(BaseModel):
    document_id: int


class ConversationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    document_id: int
    title: str
    created_at: datetime
    updated_at: datetime
    document_url: Optional[str] = None


class MessageCreate(BaseModel):
    message: str


class Source(BaseModel):
    title: str
    section: str
    preview: str
    relevance_score: float
    source: str
    boosted: bool


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    role: str
    content: str
    citations: Optional[List[dict]] = None
    created_at: datetime
