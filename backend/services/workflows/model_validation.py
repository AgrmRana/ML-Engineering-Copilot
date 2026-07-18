from typing import Dict, Any
from retrieval.hierarchical_retriever import HierarchicalRetriever
from retrieval.vector_store import VectorStore
from llm.llm_client import LLMClient


class ModelValidationWorkflow:
    """Review projects for model validation issues."""
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.retriever = HierarchicalRetriever(self.vector_store, top_k=15)
        self.llm_client = LLMClient()
    
    def generate_validation_report(self, project_id: int) -> Dict[str, Any]:
        """Generate a comprehensive model validation report."""
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(
            query="model training validation evaluation data preprocessing",
            project_id=project_id,
            min_score=0.5
        )
        
        if not retrieved_docs:
            return {
                "report": "No documents found for validation review.",
                "findings": []
            }
        
        context = self._build_context(retrieved_docs)
        
        system_prompt = """You are a model validation expert. Review the project and identify:
1. Assumptions Made
2. Potential Risks
3. Validation Weaknesses
4. Possible Data Leakage
5. Overfitting Concerns
6. Monitoring Recommendations
7. Explainability Concerns
8. Missing Validation Steps

Be thorough and specific. Use markdown formatting with clear sections."""
        
        report = self.llm_client.generate_response(
            query="Conduct a comprehensive model validation review.",
            context=context,
            system_prompt=system_prompt
        )
        
        return {
            "report": report,
            "document_count": len(retrieved_docs),
            "sources": self._get_sources(retrieved_docs)
        }
    
    def identify_risks(self, project_id: int) -> Dict[str, Any]:
        """Identify specific model risks."""
        retrieved_docs = self.retriever.retrieve(
            query="risk bias fairness ethical concerns",
            project_id=project_id,
            min_score=0.5
        )
        
        context = self._build_context(retrieved_docs)
        
        system_prompt = """You are a model risk specialist. Identify:
1. Bias Risks
2. Fairness Concerns
3. Ethical Issues
4. Regulatory Compliance Issues
5. Security Vulnerabilities
6. Performance Risks

Be specific and provide actionable recommendations."""
        
        risks = self.llm_client.generate_response(
            query="Identify model risks and provide mitigation strategies.",
            context=context,
            system_prompt=system_prompt
        )
        
        return {"risk_analysis": risks, "sources": self._get_sources(retrieved_docs)}
    
    def _build_context(self, docs: list) -> str:
        """Build context from retrieved documents."""
        return "\n\n".join([
            f"[{doc['metadata'].get('filename', 'Unknown')}]\n{doc['content']}"
            for doc in docs
        ])
    
    def _get_sources(self, docs: list) -> list:
        """Extract source filenames."""
        return [doc["metadata"].get("filename") for doc in docs]
