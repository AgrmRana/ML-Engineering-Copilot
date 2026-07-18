from typing import Dict, Any
from retrieval import HierarchicalRetriever, VectorStore
from llm.llm_client import LLMClient


class DocumentationAssistantWorkflow:
    """Generate documentation improvements and new docs."""
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.retriever = HierarchicalRetriever(self.vector_store, top_k=10)
        self.llm_client = LLMClient()
    
    def improve_readme(self, project_id: int) -> Dict[str, Any]:
        """Generate an improved README."""
        retrieved_docs = self.retriever.retrieve(
            query="README documentation setup installation",
            project_id=project_id,
            min_score=0.5
        )
        
        context = self._build_context(retrieved_docs)
        
        system_prompt = """You are a technical documentation expert. Generate a comprehensive README with:
1. Project Title and Description
2. Features
3. Installation Instructions
4. Usage Examples
5. Project Structure
6. Configuration
7. Contributing Guidelines

Use markdown formatting. Be clear and professional."""
        
        readme = self.llm_client.generate_response(
            query="Generate a comprehensive README file for this project.",
            context=context,
            system_prompt=system_prompt
        )
        
        return {"readme": readme, "sources": self._get_sources(retrieved_docs)}
    
    def generate_architecture_doc(self, project_id: int) -> Dict[str, Any]:
        """Generate architecture documentation."""
        retrieved_docs = self.retriever.retrieve(
            query="architecture design system components modules",
            project_id=project_id,
            min_score=0.5
        )
        
        context = self._build_context(retrieved_docs)
        
        system_prompt = """You are a software architect. Generate architecture documentation with:
1. System Overview
2. Architecture Diagram (describe in text)
3. Components and Their Responsibilities
4. Data Flow
5. Technology Stack
6. Design Patterns Used
7. Scalability Considerations

Use markdown formatting."""
        
        arch_doc = self.llm_client.generate_response(
            query="Generate comprehensive architecture documentation.",
            context=context,
            system_prompt=system_prompt
        )
        
        return {"architecture_doc": arch_doc, "sources": self._get_sources(retrieved_docs)}
    
    def generate_model_card(self, project_id: int) -> Dict[str, Any]:
        """Generate a model card."""
        retrieved_docs = self.retriever.retrieve(
            query="model training evaluation metrics performance",
            project_id=project_id,
            min_score=0.5
        )
        
        context = self._build_context(retrieved_docs)
        
        system_prompt = """You are an ML engineer. Generate a model card with:
1. Model Details (name, version, type)
2. Intended Use
3. Training Data
4. Performance Metrics
5. Limitations
6. Ethical Considerations
7. Usage Instructions

Use markdown formatting."""
        
        model_card = self.llm_client.generate_response(
            query="Generate a comprehensive model card.",
            context=context,
            system_prompt=system_prompt
        )
        
        return {"model_card": model_card, "sources": self._get_sources(retrieved_docs)}
    
    def _build_context(self, docs: list) -> str:
        """Build context from retrieved documents."""
        return "\n\n".join([
            f"[{doc['metadata'].get('filename', 'Unknown')}]\n{doc['content']}"
            for doc in docs
        ])
    
    def _get_sources(self, docs: list) -> list:
        """Extract source filenames."""
        return [doc["metadata"].get("filename") for doc in docs]
