from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
from contextlib import asynccontextmanager

from database.connection import get_db, init_db
from services import ProjectService, DocumentService, ConversationService
from services.workflows import (
    ProjectSummaryWorkflow,
    DocumentationAssistantWorkflow,
    ModelValidationWorkflow,
    ExplainabilityWorkflow,
    RepositoryReviewWorkflow,
    InterviewAssistantWorkflow,
)
from api.schemas import (
    ProjectCreate, ProjectResponse,
    DocumentResponse,
    ConversationCreate, ConversationResponse,
    MessageCreate, ChatResponse,
    SearchRequest, SearchResult
)
from config.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Initialize database on startup
    init_db()
    yield
    # Cleanup on shutdown if needed
    pass


app = FastAPI(
    title="ML Workspace AI",
    description="Enterprise ML Engineering Copilot API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check
@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "ml-workspace-ai"}


# Project endpoints
@app.post("/api/projects", response_model=ProjectResponse)
def create_project(project: ProjectCreate, db: Session = Depends(get_db)):
    """Create a new project."""
    service = ProjectService(db)
    result = service.create_project(project.name, project.description)
    return result


@app.get("/api/projects/{project_id}", response_model=ProjectResponse)
def get_project(project_id: int, db: Session = Depends(get_db)):
    """Get project details."""
    service = ProjectService(db)
    result = service.get_project(project_id)
    if not result:
        raise HTTPException(status_code=404, detail="Project not found")
    return result


@app.get("/api/projects", response_model=List[ProjectResponse])
def list_projects(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """List all projects."""
    service = ProjectService(db)
    return service.list_projects(skip=skip, limit=limit)


@app.delete("/api/projects/{project_id}")
def delete_project(project_id: int, db: Session = Depends(get_db)):
    """Delete a project."""
    service = ProjectService(db)
    if not service.delete_project(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    return {"message": "Project deleted successfully"}


# Document endpoints
@app.post("/api/projects/{project_id}/documents", response_model=DocumentResponse)
async def upload_document(
    project_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload a document to a project."""
    service = DocumentService(db)
    
    # Read file content
    content = await file.read()
    
    # Check file size
    if len(content) > settings.max_file_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds maximum of {settings.max_file_size_mb}MB"
        )
    
    result = service.upload_document(project_id, content, file.filename)
    return result


@app.get("/api/projects/{project_id}/documents", response_model=List[DocumentResponse])
def list_documents(project_id: int, db: Session = Depends(get_db)):
    """List documents in a project."""
    service = DocumentService(db)
    return service.list_documents(project_id)


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: int, db: Session = Depends(get_db)):
    """Delete a document."""
    service = DocumentService(db)
    if not service.delete_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": "Document deleted successfully"}


# Conversation endpoints
@app.post("/api/conversations", response_model=ConversationResponse)
def create_conversation(conversation: ConversationCreate, db: Session = Depends(get_db)):
    """Create a new conversation."""
    service = ConversationService(db)
    result = service.create_conversation(conversation.project_id, conversation.title)
    return result


@app.get("/api/conversations/{conversation_id}")
def get_conversation(conversation_id: int, db: Session = Depends(get_db)):
    """Get conversation with messages."""
    service = ConversationService(db)
    result = service.get_conversation(conversation_id)
    if not result:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return result


@app.get("/api/conversations", response_model=List[ConversationResponse])
def list_conversations(project_id: int = None, db: Session = Depends(get_db)):
    """List conversations."""
    service = ConversationService(db)
    return service.list_conversations(project_id=project_id)


@app.post("/api/conversations/{conversation_id}/messages", response_model=ChatResponse)
def send_message(
    conversation_id: int,
    message: MessageCreate,
    db: Session = Depends(get_db)
):
    """Send a message and get AI response."""
    if message.use_rag and (not settings.openai_api_key or settings.openai_api_key == "your_openai_api_key_here"):
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key not configured. Please set OPENAI_API_KEY in backend/.env to use AI features."
        )
    service = ConversationService(db)
    result = service.send_message(conversation_id, message.content, message.use_rag)
    return result


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: int, db: Session = Depends(get_db)):
    """Delete a conversation."""
    service = ConversationService(db)
    if not service.delete_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"message": "Conversation deleted successfully"}


# Search endpoint
@app.post("/api/search", response_model=List[SearchResult])
def search(search_request: SearchRequest, db: Session = Depends(get_db)):
    """Search documents using semantic search."""
    from retrieval import HierarchicalRetriever, VectorStore
    from config.settings import settings
    import os
    
    # Check if OpenAI API key is configured
    if not settings.openai_api_key or settings.openai_api_key == "your_openai_api_key_here":
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key not configured. Please set OPENAI_API_KEY in backend/.env to use search features."
        )
    
    vector_store = VectorStore()
    retriever = HierarchicalRetriever(vector_store, search_request.top_k)
    
    try:
        results = retriever.retrieve(
            query=search_request.query,
            project_id=search_request.project_id,
            document_types=search_request.document_types,
            min_score=settings.min_confidence_score
        )
        
        return [
            SearchResult(
                content=r["content"],
                metadata=r["metadata"],
                score=r["score"]
            )
            for r in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# Workflow endpoints
@app.post("/api/projects/{project_id}/workflows/summary")
def generate_project_summary(project_id: int):
    """Generate a comprehensive project summary."""
    if not settings.openai_api_key or settings.openai_api_key == "your_openai_api_key_here":
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key not configured. Please set OPENAI_API_KEY in backend/.env to use workflow features."
        )
    workflow = ProjectSummaryWorkflow()
    result = workflow.generate_summary(project_id)
    return result


@app.post("/api/projects/{project_id}/workflows/documentation/readme")
def improve_readme(project_id: int):
    """Generate an improved README."""
    if not settings.openai_api_key or settings.openai_api_key == "your_openai_api_key_here":
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key not configured. Please set OPENAI_API_KEY in backend/.env to use workflow features."
        )
    workflow = DocumentationAssistantWorkflow()
    result = workflow.improve_readme(project_id)
    return result


@app.post("/api/projects/{project_id}/workflows/documentation/architecture")
def generate_architecture_doc(project_id: int):
    """Generate architecture documentation."""
    if not settings.openai_api_key or settings.openai_api_key == "your_openai_api_key_here":
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key not configured. Please set OPENAI_API_KEY in backend/.env to use workflow features."
        )
    workflow = DocumentationAssistantWorkflow()
    result = workflow.generate_architecture_doc(project_id)
    return result


@app.post("/api/projects/{project_id}/workflows/documentation/model-card")
def generate_model_card(project_id: int):
    """Generate a model card."""
    if not settings.openai_api_key or settings.openai_api_key == "your_openai_api_key_here":
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key not configured. Please set OPENAI_API_KEY in backend/.env to use workflow features."
        )
    workflow = DocumentationAssistantWorkflow()
    result = workflow.generate_model_card(project_id)
    return result


@app.post("/api/projects/{project_id}/workflows/validation/report")
def generate_validation_report(project_id: int):
    """Generate a model validation report."""
    if not settings.openai_api_key or settings.openai_api_key == "your_openai_api_key_here":
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key not configured. Please set OPENAI_API_KEY in backend/.env to use workflow features."
        )
    workflow = ModelValidationWorkflow()
    result = workflow.generate_validation_report(project_id)
    return result


@app.post("/api/projects/{project_id}/workflows/validation/risks")
def identify_risks(project_id: int):
    """Identify model risks."""
    if not settings.openai_api_key or settings.openai_api_key == "your_openai_api_key_here":
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key not configured. Please set OPENAI_API_KEY in backend/.env to use workflow features."
        )
    workflow = ModelValidationWorkflow()
    result = workflow.identify_risks(project_id)
    return result


@app.post("/api/projects/{project_id}/workflows/explainability/shap")
def explain_shap(project_id: int):
    """Explain SHAP analysis."""
    if not settings.openai_api_key or settings.openai_api_key == "your_openai_api_key_here":
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key not configured. Please set OPENAI_API_KEY in backend/.env to use workflow features."
        )
    workflow = ExplainabilityWorkflow()
    result = workflow.explain_shap(project_id)
    return result


@app.post("/api/projects/{project_id}/workflows/explainability/feature-importance")
def explain_feature_importance(project_id: int):
    """Explain feature importance."""
    if not settings.openai_api_key or settings.openai_api_key == "your_openai_api_key_here":
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key not configured. Please set OPENAI_API_KEY in backend/.env to use workflow features."
        )
    workflow = ExplainabilityWorkflow()
    result = workflow.explain_feature_importance(project_id)
    return result


@app.post("/api/projects/{project_id}/workflows/explainability/metrics")
def explain_evaluation_metrics(project_id: int):
    """Explain evaluation metrics."""
    if not settings.openai_api_key or settings.openai_api_key == "your_openai_api_key_here":
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key not configured. Please set OPENAI_API_KEY in backend/.env to use workflow features."
        )
    workflow = ExplainabilityWorkflow()
    result = workflow.explain_evaluation_metrics(project_id)
    return result


@app.post("/api/projects/{project_id}/workflows/review/code")
def review_code_quality(project_id: int):
    """Review code quality."""
    if not settings.openai_api_key or settings.openai_api_key == "your_openai_api_key_here":
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key not configured. Please set OPENAI_API_KEY in backend/.env to use workflow features."
        )
    workflow = RepositoryReviewWorkflow()
    result = workflow.review_code_quality(project_id)
    return result


@app.post("/api/projects/{project_id}/workflows/review/engineering")
def review_engineering_practices(project_id: int):
    """Review engineering practices."""
    if not settings.openai_api_key or settings.openai_api_key == "your_openai_api_key_here":
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key not configured. Please set OPENAI_API_KEY in backend/.env to use workflow features."
        )
    workflow = RepositoryReviewWorkflow()
    result = workflow.review_engineering_practices(project_id)
    return result


@app.post("/api/projects/{project_id}/workflows/interview/questions")
def generate_interview_questions(project_id: int, role: str = "Data Scientist"):
    """Generate interview questions."""
    if not settings.openai_api_key or settings.openai_api_key == "your_openai_api_key_here":
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key not configured. Please set OPENAI_API_KEY in backend/.env to use workflow features."
        )
    workflow = InterviewAssistantWorkflow()
    result = workflow.generate_interview_questions(project_id, role)
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
