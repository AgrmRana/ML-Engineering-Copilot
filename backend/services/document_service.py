from typing import List, Optional
from sqlalchemy.orm import Session
import os
import shutil
from pathlib import Path
from database.repositories import DocumentRepository
from processing.document_processor import DocumentProcessor
from retrieval import VectorStore
from config.settings import settings


class DocumentService:
    """Service for document processing and management."""
    
    def __init__(self, db: Session):
        self.db = db
        self.document_repo = DocumentRepository(db)
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.upload_dir = Path("./data/uploads")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    def upload_document(self, project_id: int, file_content: bytes, filename: str) -> dict:
        """Upload and process a document."""
        # Save file
        file_path = self.upload_dir / f"{project_id}_{filename}"
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        # Process document
        content, chunks, metadata = self.document_processor.process_file(
            str(file_path), filename
        )
        
        # Create document record
        document = self.document_repo.create(
            project_id=project_id,
            filename=filename,
            file_path=str(file_path),
            file_type=metadata["file_type"],
            file_size=metadata["file_size"],
            meta=metadata
        )
        
        # Add chunks to vector store
        chunk_ids = []
        for i, chunk in enumerate(chunks):
            chunk.metadata["document_id"] = document.id
            chunk.metadata["project_id"] = project_id
            chunk_ids.append(f"doc_{document.id}_chunk_{i}")
        
        self.vector_store.add_documents(chunks, ids=chunk_ids)
        
        # Update document with chunk count
        self.document_repo.update(document.id, chunk_count=len(chunks), status="completed")
        
        return {
            "id": document.id,
            "filename": document.filename,
            "file_type": document.file_type.value,
            "file_size": document.file_size,
            "chunk_count": len(chunks),
            "status": "completed"
        }
    
    def get_document(self, document_id: int) -> Optional[dict]:
        """Get document details."""
        document = self.document_repo.get(document_id)
        if not document:
            return None
        
        return {
            "id": document.id,
            "project_id": document.project_id,
            "filename": document.filename,
            "file_type": document.file_type.value,
            "file_size": document.file_size,
            "chunk_count": document.chunk_count,
            "status": document.status,
            "created_at": document.created_at.isoformat()
        }
    
    def list_documents(self, project_id: int) -> List[dict]:
        """List documents in a project."""
        documents = self.document_repo.list_by_project(project_id)
        return [
            {
                "id": d.id,
                "filename": d.filename,
                "file_type": d.file_type.value,
                "file_size": d.file_size,
                "chunk_count": d.chunk_count,
                "status": d.status,
                "created_at": d.created_at.isoformat()
            }
            for d in documents
        ]
    
    def delete_document(self, document_id: int) -> bool:
        """Delete a document and its chunks."""
        document = self.document_repo.get(document_id)
        if not document:
            return False
        
        # Delete file
        file_path = Path(document.file_path)
        if file_path.exists():
            file_path.unlink()
        
        # Delete from vector store (would need stored chunk IDs)
        # For now, just delete from database
        return self.document_repo.delete(document_id)
