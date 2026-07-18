from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from config.settings import settings


class VectorStore:
    """Vector database for semantic search."""
    
    def __init__(self, collection_name: str = "ml_workspace"):
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.openai_api_key,
            model=settings.openai_embedding_model
        )
        self.client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embeddings
        )
    
    def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None) -> List[str]:
        """Add documents to vector store."""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        self.vectorstore.add_documents(documents, ids=ids)
        return ids
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Search for similar documents."""
        return self.vectorstore.similarity_search(
            query=query,
            k=k,
            filter=filter
        )
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple[Document, float]]:
        """Search for similar documents with relevance scores."""
        return self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter
        )
    
    def delete(self, ids: List[str]) -> None:
        """Delete documents by IDs."""
        self.vectorstore.delete(ids)
    
    def clear_collection(self) -> None:
        """Clear all documents from collection."""
        self.client.delete_collection(name=self.collection_name)
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings
        )
