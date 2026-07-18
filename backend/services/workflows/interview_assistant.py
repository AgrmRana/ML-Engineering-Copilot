from typing import Dict, Any
from retrieval.hierarchical_retriever import HierarchicalRetriever
from retrieval.vector_store import VectorStore
from llm.llm_client import LLMClient


class InterviewAssistantWorkflow:
    """Generate interview questions based on projects."""
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.retriever = HierarchicalRetriever(self.vector_store, top_k=10)
        self.llm_client = LLMClient()
    
    def generate_interview_questions(self, project_id: int, role: str = "Data Scientist") -> Dict[str, Any]:
        """Generate role-specific interview questions."""
        retrieved_docs = self.retriever.retrieve(
            query="project implementation algorithms techniques",
            project_id=project_id,
            min_score=0.5
        )
        
        context = self._build_context(retrieved_docs)
        
        system_prompt = f"""You are a technical interviewer for a {role} position. Generate interview questions based on this project:
1. Technical Questions (5-8 questions)
2. For each question, provide:
   - The question
   - Expected answer key points
   - Follow-up questions
   - Difficulty level (Easy/Medium/Hard)

Questions should test understanding of the project's implementation and underlying concepts.
Use markdown formatting with clear structure."""
        
        questions = self.llm_client.generate_response(
            query=f"Generate interview questions for a {role} position based on this project.",
            context=context,
            system_prompt=system_prompt
        )
        
        return {
            "role": role,
            "interview_questions": questions,
            "sources": self._get_sources(retrieved_docs)
        }
    
    def generate_ml_questions(self, project_id: int) -> Dict[str, Any]:
        """Generate ML-specific interview questions."""
        retrieved_docs = self.retriever.retrieve(
            query="model training algorithms evaluation metrics",
            project_id=project_id,
            min_score=0.5
        )
        
        context = self._build_context(retrieved_docs)
        
        system_prompt = """You are an ML interviewer. Generate ML interview questions:
1. Model Selection Questions
2. Feature Engineering Questions
3. Evaluation Questions
4. Optimization Questions
5. Deployment Questions

For each question, provide expected answers and follow-ups.
Focus on the ML techniques used in this project."""
        
        questions = self.llm_client.generate_response(
            query="Generate ML-specific interview questions.",
            context=context,
            system_prompt=system_prompt
        )
        
        return {"ml_questions": questions, "sources": self._get_sources(retrieved_docs)}
    
    def generate_engineering_questions(self, project_id: int) -> Dict[str, Any]:
        """Generate engineering interview questions."""
        retrieved_docs = self.retriever.retrieve(
            query="code architecture design patterns scalability",
            project_id=project_id,
            min_score=0.5
        )
        
        context = self._build_context(retrieved_docs)
        
        system_prompt = """You are a software engineering interviewer. Generate engineering questions:
1. Architecture Questions
2. Design Pattern Questions
3. Scalability Questions
4. Code Quality Questions
5. DevOps Questions

For each question, provide expected answers and follow-ups.
Focus on the engineering practices in this project."""
        
        questions = self.llm_client.generate_response(
            query="Generate software engineering interview questions.",
            context=context,
            system_prompt=system_prompt
        )
        
        return {"engineering_questions": questions, "sources": self._get_sources(retrieved_docs)}
    
    def generate_system_design_questions(self, project_id: int) -> Dict[str, Any]:
        """Generate system design interview questions."""
        retrieved_docs = self.retriever.retrieve(
            query="system architecture components data flow",
            project_id=project_id,
            min_score=0.5
        )
        
        context = self._build_context(retrieved_docs)
        
        system_prompt = """You are a system design interviewer. Generate system design questions:
1. Architecture Questions
2. Scalability Questions
3. Data Storage Questions
4. API Design Questions
5. Performance Questions

For each question, provide expected answers and follow-ups.
Base questions on the system design in this project."""
        
        questions = self.llm_client.generate_response(
            query="Generate system design interview questions.",
            context=context,
            system_prompt=system_prompt
        )
        
        return {"system_design_questions": questions, "sources": self._get_sources(retrieved_docs)}
    
    def _build_context(self, docs: list) -> str:
        """Build context from retrieved documents."""
        return "\n\n".join([
            f"[{doc['metadata'].get('filename', 'Unknown')}]\n{doc['content']}"
            for doc in docs
        ])
    
    def _get_sources(self, docs: list) -> list:
        """Extract source filenames."""
        return [doc["metadata"].get("filename") for doc in docs]
