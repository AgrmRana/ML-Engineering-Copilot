from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None


class ProjectResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    status: str
    created_at: str
    updated_at: Optional[str] = None
    document_count: Optional[int] = None
    metadata: Optional[dict] = None


class DocumentResponse(BaseModel):
    id: int
    project_id: int
    filename: str
    file_type: str
    file_size: Optional[int]
    chunk_count: int
    status: str
    created_at: str


class ConversationCreate(BaseModel):
    project_id: Optional[int] = None
    title: str = "New Conversation"


class ConversationResponse(BaseModel):
    id: int
    project_id: Optional[int]
    title: str
    status: str
    created_at: str


class MessageCreate(BaseModel):
    content: str = Field(..., min_length=1)
    use_rag: bool = True


class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    sources: Optional[List[dict]] = None
    confidence: Optional[float] = None
    created_at: str


class ChatResponse(BaseModel):
    user_message: MessageResponse
    assistant_message: MessageResponse


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    project_id: Optional[int] = None
    document_types: Optional[List[str]] = None
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResult(BaseModel):
    content: str
    metadata: dict
    score: float
