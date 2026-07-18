from typing import Dict, Any
from retrieval.hierarchical_retriever import HierarchicalRetriever
from retrieval.vector_store import VectorStore
from llm.llm_client import LLMClient


class RepositoryReviewWorkflow:
    """Review code quality and engineering practices."""
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.retriever = HierarchicalRetriever(self.vector_store, top_k=15)
        self.llm_client = LLMClient()
    
    def review_code_quality(self, project_id: int) -> Dict[str, Any]:
        """Review code quality."""
        retrieved_docs = self.retriever.retrieve(
            query="code structure modules functions classes",
            project_id=project_id,
            min_score=0.5
        )
        
        context = self._build_context(retrieved_docs)
        
        system_prompt = """You are a senior software engineer. Review code quality:
1. Code Organization
2. Naming Conventions
3. Code Duplication
4. Complexity Issues
5. Best Practices
6. Specific Recommendations

Be constructive and specific. Use markdown formatting."""
        
        review = self.llm_client.generate_response(
            query="Review the code quality of this project.",
            context=context,
            system_prompt=system_prompt
        )
        
        return {"code_review": review, "sources": self._get_sources(retrieved_docs)}
    
    def review_engineering_practices(self, project_id: int) -> Dict[str, Any]:
        """Review software engineering practices."""
        retrieved_docs = self.retriever.retrieve(
            query="testing CI/CD version control documentation",
            project_id=project_id,
            min_score=0.5
        )
        
        context = self._build_context(retrieved_docs)
        
        system_prompt = """You are a DevOps engineer. Review engineering practices:
1. Testing Coverage
2. CI/CD Pipeline
3. Version Control Practices
4. Documentation Quality
5. Error Handling
6. Logging and Monitoring
7. Security Practices

Provide actionable improvements."""
        
        review = self.llm_client.generate_response(
            query="Review the software engineering practices.",
            context=context,
            system_prompt=system_prompt
        )
        
        return {"engineering_review": review, "sources": self._get_sources(retrieved_docs)}
    
    def review_testing(self, project_id: int) -> Dict[str, Any]:
        """Review testing approach."""
        retrieved_docs = self.retriever.retrieve(
            query="test unit test integration test pytest",
            project_id=project_id,
            min_score=0.5
        )
        
        context = self._build_context(retrieved_docs)
        
        system_prompt = """You are a QA engineer. Review testing:
1. Test Coverage
2. Test Types (unit, integration, E2E)
3. Test Quality
4. Missing Tests
5. Test Organization
6. Recommendations

Be thorough and specific."""
        
        review = self.llm_client.generate_response(
            query="Review the testing approach.",
            context=context,
            system_prompt=system_prompt
        )
        
        return {"testing_review": review, "sources": self._get_sources(retrieved_docs)}
    
    def review_documentation(self, project_id: int) -> Dict[str, Any]:
        """Review documentation quality."""
        retrieved_docs = self.retriever.retrieve(
            query="README docs documentation comments",
            project_id=project_id,
            min_score=0.5
        )
        
        context = self._build_context(retrieved_docs)
        
        system_prompt = """You are a technical writer. Review documentation:
1. README Quality
2. Code Comments
3. API Documentation
4. Architecture Documentation
5. Missing Documentation
6. Clarity and Completeness

Provide specific improvements."""
        
        review = self.llm_client.generate_response(
            query="Review the documentation quality.",
            context=context,
            system_prompt=system_prompt
        )
        
        return {"documentation_review": review, "sources": self._get_sources(retrieved_docs)}
    
    def _build_context(self, docs: list) -> str:
        """Build context from retrieved documents."""
        return "\n\n".join([
            f"[{doc['metadata'].get('filename', 'Unknown')}]\n{doc['content']}"
            for doc in docs
        ])
    
    def _get_sources(self, docs: list) -> list:
        """Extract source filenames."""
        return [doc["metadata"].get("filename") for doc in docs]
