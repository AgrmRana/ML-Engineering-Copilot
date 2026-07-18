from sqlalchemy.orm import Session
from typing import List, Optional
from database.models import Project, Document, DocumentChunk, Conversation, Message, Report
from datetime import datetime


class ProjectRepository:
    def __init__(self, db: Session):
        self.db = db
    
    def create(self, name: str, description: Optional[str] = None, meta: Optional[dict] = None) -> Project:
        project = Project(
            name=name,
            description=description,
            meta=meta or {}
        )
        self.db.add(project)
        self.db.commit()
        self.db.refresh(project)
        return project
    
    def get(self, project_id: int) -> Optional[Project]:
        return self.db.query(Project).filter(Project.id == project_id).first()
    
    def list(self, skip: int = 0, limit: int = 100) -> List[Project]:
        return self.db.query(Project).offset(skip).limit(limit).all()
    
    def update(self, project_id: int, **kwargs) -> Optional[Project]:
        project = self.get(project_id)
        if project:
            for key, value in kwargs.items():
                setattr(project, key, value)
            project.updated_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(project)
        return project
    
    def delete(self, project_id: int) -> bool:
        project = self.get(project_id)
        if project:
            self.db.delete(project)
            self.db.commit()
            return True
        return False


class DocumentRepository:
    def __init__(self, db: Session):
        self.db = db
    
    def create(self, project_id: int, filename: str, file_path: str, 
               file_type: str, file_size: int, meta: Optional[dict] = None) -> Document:
        document = Document(
            project_id=project_id,
            filename=filename,
            file_path=file_path,
            file_type=file_type,
            file_size=file_size,
            meta=meta or {}
        )
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)
        return document
    
    def get(self, document_id: int) -> Optional[Document]:
        return self.db.query(Document).filter(Document.id == document_id).first()
    
    def list_by_project(self, project_id: int) -> List[Document]:
        return self.db.query(Document).filter(Document.project_id == project_id).all()
    
    def update(self, document_id: int, **kwargs) -> Optional[Document]:
        document = self.get(document_id)
        if document:
            for key, value in kwargs.items():
                setattr(document, key, value)
            document.updated_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(document)
        return document
    
    def delete(self, document_id: int) -> bool:
        document = self.get(document_id)
        if document:
            self.db.delete(document)
            self.db.commit()
            return True
        return False


class ConversationRepository:
    def __init__(self, db: Session):
        self.db = db
    
    def create(self, project_id: Optional[int], title: str = "New Conversation") -> Conversation:
        conversation = Conversation(
            project_id=project_id,
            title=title
        )
        self.db.add(conversation)
        self.db.commit()
        self.db.refresh(conversation)
        return conversation
    
    def get(self, conversation_id: int) -> Optional[Conversation]:
        return self.db.query(Conversation).filter(Conversation.id == conversation_id).first()
    
    def list(self, project_id: Optional[int] = None, skip: int = 0, limit: int = 100) -> List[Conversation]:
        query = self.db.query(Conversation)
        if project_id:
            query = query.filter(Conversation.project_id == project_id)
        return query.offset(skip).limit(limit).all()
    
    def update(self, conversation_id: int, **kwargs) -> Optional[Conversation]:
        conversation = self.get(conversation_id)
        if conversation:
            for key, value in kwargs.items():
                setattr(conversation, key, value)
            conversation.updated_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(conversation)
        return conversation
    
    def delete(self, conversation_id: int) -> bool:
        conversation = self.get(conversation_id)
        if conversation:
            self.db.delete(conversation)
            self.db.commit()
            return True
        return False


class MessageRepository:
    def __init__(self, db: Session):
        self.db = db
    
    def create(self, conversation_id: int, role: str, content: str, 
               sources: Optional[list] = None, confidence: Optional[float] = None,
               meta: Optional[dict] = None) -> Message:
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            sources=sources,
            confidence=confidence,
            meta=meta or {}
        )
        self.db.add(message)
        self.db.commit()
        self.db.refresh(message)
        return message
    
    def list_by_conversation(self, conversation_id: int) -> List[Message]:
        return self.db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).order_by(Message.created_at).all()


class ReportRepository:
    def __init__(self, db: Session):
        self.db = db
    
    def create(self, project_id: int, report_type: str, title: str, 
               content: str, meta: Optional[dict] = None) -> Report:
        report = Report(
            project_id=project_id,
            report_type=report_type,
            title=title,
            content=content,
            meta=meta or {}
        )
        self.db.add(report)
        self.db.commit()
        self.db.refresh(report)
        return report
    
    def get(self, report_id: int) -> Optional[Report]:
        return self.db.query(Report).filter(Report.id == report_id).first()
    
    def list_by_project(self, project_id: int) -> List[Report]:
        return self.db.query(Report).filter(Report.project_id == project_id).all()
    
    def delete(self, report_id: int) -> bool:
        report = self.get(report_id)
        if report:
            self.db.delete(report)
            self.db.commit()
            return True
        return False
