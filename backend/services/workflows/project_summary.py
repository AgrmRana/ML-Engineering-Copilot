from typing import List, Dict, Any
from retrieval.hierarchical_retriever import HierarchicalRetriever
from retrieval.vector_store import VectorStore
from llm.llm_client import LLMClient


class ProjectSummaryWorkflow:
    """Generate comprehensive project summaries."""
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.retriever = HierarchicalRetriever(self.vector_store, top_k=10)
        self.llm_client = LLMClient()
    
    def generate_summary(self, project_id: int) -> Dict[str, Any]:
        """Generate a comprehensive project summary."""
        # Retrieve all relevant documents
        retrieved_docs = self.retriever.retrieve(
            query="project overview architecture components",
            project_id=project_id,
            min_score=0.5
        )
        
        if not retrieved_docs:
            return {
                "summary": "No documents found to generate summary.",
                "sections": {}
            }
        
        # Build context
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"[Document {i}: {doc['metadata'].get('filename', 'Unknown')}]\n{doc['content']}")
        
        context = "\n\n".join(context_parts)
        
        # Generate summary with structured sections
        system_prompt = """You are an expert ML engineer. Generate a comprehensive project summary with the following sections:
1. Project Overview
2. Architecture
3. Key Components
4. Technologies Used
5. Data Pipeline
6. Model Information
7. Evaluation Approach

Be concise and technical. Use markdown formatting."""
        
        summary = self.llm_client.generate_response(
            query="Generate a comprehensive project summary with the sections listed above.",
            context=context,
            system_prompt=system_prompt
        )
        
        return {
            "summary": summary,
            "document_count": len(retrieved_docs),
            "sources": [doc["metadata"].get("filename") for doc in retrieved_docs]
        }
