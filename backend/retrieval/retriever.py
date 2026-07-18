from typing import List, Dict, Any, Optional
from langchain.schema import Document
from .vector_store import VectorStore


class HierarchicalRetriever:
    """Hierarchical retrieval for better context."""
    
    def __init__(self, vector_store: VectorStore, top_k: int = 5):
        self.vector_store = vector_store
        self.top_k = top_k
    
    def retrieve(
        self,
        query: str,
        project_id: Optional[int] = None,
        document_types: Optional[List[str]] = None,
        min_score: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks with metadata."""
        # Build filter
        filter_dict = {}
        if project_id is not None:
            filter_dict["project_id"] = project_id
        if document_types:
            filter_dict["file_type"] = {"$in": document_types}
        
        # Search with scores
        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=self.top_k * 2,  # Get more to filter by score
            filter=filter_dict if filter_dict else None
        )
        
        # Filter by minimum score and format results
        formatted_results = []
        for doc, score in results:
            if score >= min_score:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score
                })
        
        return formatted_results[:self.top_k]
    
    def retrieve_by_document(
        self,
        query: str,
        document_id: int,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Retrieve chunks from a specific document."""
        return self.retrieve(
            query=query,
            filter={"document_id": document_id},
            top_k=top_k
        )
