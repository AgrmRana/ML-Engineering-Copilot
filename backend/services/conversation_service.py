from typing import List, Optional
from sqlalchemy.orm import Session
from database.repositories import ConversationRepository, MessageRepository
from retrieval.hierarchical_retriever import HierarchicalRetriever
from retrieval.vector_store import VectorStore
from llm.llm_client import LLMClient
from config.settings import settings


class ConversationService:
    """Service for conversation management and AI interactions."""
    
    def __init__(self, db: Session):
        self.db = db
        self.conversation_repo = ConversationRepository(db)
        self.message_repo = MessageRepository(db)
        self.vector_store = VectorStore()
        self.retriever = HierarchicalRetriever(self.vector_store, settings.top_k_retrieval)
        self.llm_client = LLMClient()
    
    def create_conversation(self, project_id: Optional[int] = None, title: str = "New Conversation") -> dict:
        """Create a new conversation."""
        conversation = self.conversation_repo.create(project_id=project_id, title=title)
        return {
            "id": conversation.id,
            "project_id": conversation.project_id,
            "title": conversation.title,
            "status": conversation.status.value,
            "created_at": conversation.created_at.isoformat()
        }
    
    def get_conversation(self, conversation_id: int) -> Optional[dict]:
        """Get conversation details with messages."""
        conversation = self.conversation_repo.get(conversation_id)
        if not conversation:
            return None
        
        messages = self.message_repo.list_by_conversation(conversation_id)
        
        return {
            "id": conversation.id,
            "project_id": conversation.project_id,
            "title": conversation.title,
            "status": conversation.status.value,
            "created_at": conversation.created_at.isoformat(),
            "messages": [
                {
                    "id": m.id,
                    "role": m.role,
                    "content": m.content,
                    "sources": m.sources,
                    "confidence": m.confidence,
                    "created_at": m.created_at.isoformat()
                }
                for m in messages
            ]
        }
    
    def list_conversations(self, project_id: Optional[int] = None) -> List[dict]:
        """List conversations."""
        conversations = self.conversation_repo.list(project_id=project_id)
        return [
            {
                "id": c.id,
                "project_id": c.project_id,
                "title": c.title,
                "status": c.status.value,
                "created_at": c.created_at.isoformat()
            }
            for c in conversations
        ]
    
    def send_message(
        self,
        conversation_id: int,
        content: str,
        use_rag: bool = True
    ) -> dict:
        """Send a message and get AI response."""
        # Save user message
        user_message = self.message_repo.create(
            conversation_id=conversation_id,
            role="user",
            content=content
        )
        
        # Get conversation history
        messages = self.message_repo.list_by_conversation(conversation_id)
        conversation_history = [
            {"role": m.role, "content": m.content}
            for m in messages[:-1]  # Exclude the just-added user message
        ]
        
        # Get conversation to check project
        conversation = self.conversation_repo.get(conversation_id)
        
        # Generate response
        if use_rag and conversation.project_id:
            # RAG-enabled response
            retrieved_docs = self.retriever.retrieve(
                query=content,
                project_id=conversation.project_id,
                min_score=settings.min_confidence_score
            )
            
            llm_response = self.llm_client.generate_with_sources(
                query=content,
                retrieved_docs=retrieved_docs,
                conversation_history=conversation_history
            )
            
            assistant_message = self.message_repo.create(
                conversation_id=conversation_id,
                role="assistant",
                content=llm_response["response"],
                sources=llm_response["sources"],
                confidence=sum(doc.get("score", 0) for doc in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0.0
            )
        else:
            # Direct LLM response without RAG
            response = self.llm_client.generate_response(
                query=content,
                conversation_history=conversation_history
            )
            
            assistant_message = self.message_repo.create(
                conversation_id=conversation_id,
                role="assistant",
                content=response
            )
        
        return {
            "user_message": {
                "id": user_message.id,
                "role": user_message.role,
                "content": user_message.content,
                "created_at": user_message.created_at.isoformat()
            },
            "assistant_message": {
                "id": assistant_message.id,
                "role": assistant_message.role,
                "content": assistant_message.content,
                "sources": assistant_message.sources,
                "confidence": assistant_message.confidence,
                "created_at": assistant_message.created_at.isoformat()
            }
        }
    
    def delete_conversation(self, conversation_id: int) -> bool:
        """Delete a conversation."""
        return self.conversation_repo.delete(conversation_id)
