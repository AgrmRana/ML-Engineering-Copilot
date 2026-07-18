from typing import List, Optional
from sqlalchemy.orm import Session
from database.repositories import ProjectRepository, DocumentRepository
from processing.document_processor import DocumentProcessor
from retrieval import VectorStore, HierarchicalRetriever
from config.settings import settings


class ProjectService:
    """Service for project management operations."""
    
    def __init__(self, db: Session):
        self.db = db
        self.project_repo = ProjectRepository(db)
        self.document_repo = DocumentRepository(db)
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.retriever = HierarchicalRetriever(self.vector_store, settings.top_k_retrieval)
    
    def create_project(self, name: str, description: Optional[str] = None) -> dict:
        """Create a new project."""
        project = self.project_repo.create(name=name, description=description)
        return {
            "id": project.id,
            "name": project.name,
            "description": project.description,
            "status": project.status.value,
            "created_at": project.created_at.isoformat()
        }
    
    def get_project(self, project_id: int) -> Optional[dict]:
        """Get project details."""
        project = self.project_repo.get(project_id)
        if not project:
            return None
        
        documents = self.document_repo.list_by_project(project_id)
        
        return {
            "id": project.id,
            "name": project.name,
            "description": project.description,
            "status": project.status.value,
            "created_at": project.created_at.isoformat(),
            "updated_at": project.updated_at.isoformat(),
            "document_count": len(documents),
            "meta": project.meta
        }
    
    def list_projects(self, skip: int = 0, limit: int = 100) -> List[dict]:
        """List all projects."""
        projects = self.project_repo.list(skip=skip, limit=limit)
        return [
            {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "status": p.status.value,
                "created_at": p.created_at.isoformat()
            }
            for p in projects
        ]
    
    def delete_project(self, project_id: int) -> bool:
        """Delete a project and its associated data."""
        # Get all documents in project
        documents = self.document_repo.list_by_project(project_id)
        
        # Remove from vector store
        for doc in documents:
            # Delete chunks from vector store
            # This would need embedding IDs stored in database
            pass
        
        # Delete project (cascade will handle documents)
        return self.project_repo.delete(project_id)
    
    def update_project_status(self, project_id: int, status: str) -> Optional[dict]:
        """Update project status."""
        from database.models import ProjectStatus
        project = self.project_repo.update(project_id, status=ProjectStatus(status))
        if project:
            return {
                "id": project.id,
                "name": project.name,
                "status": project.status.value
            }
        return None
