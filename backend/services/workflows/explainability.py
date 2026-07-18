from typing import Dict, Any
from retrieval import HierarchicalRetriever, VectorStore
from llm.llm_client import LLMClient


class ExplainabilityWorkflow:
    """Explain ML model components and decisions."""
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.retriever = HierarchicalRetriever(self.vector_store, top_k=10)
        self.llm_client = LLMClient()
    
    def explain_shap(self, project_id: int) -> Dict[str, Any]:
        """Explain SHAP analysis if present."""
        retrieved_docs = self.retriever.retrieve(
            query="SHAP feature importance explainability",
            project_id=project_id,
            min_score=0.5
        )
        
        context = self._build_context(retrieved_docs)
        
        system_prompt = """You are an ML explainability expert. Explain the SHAP analysis:
1. What SHAP values represent
2. Top important features
3. Feature impact direction
4. Global vs local explanations
5. Interpretation of results

Be clear and educational. Use markdown formatting."""
        
        explanation = self.llm_client.generate_response(
            query="Explain the SHAP analysis in this project.",
            context=context,
            system_prompt=system_prompt
        )
        
        return {"shap_explanation": explanation, "sources": self._get_sources(retrieved_docs)}
    
    def explain_feature_importance(self, project_id: int) -> Dict[str, Any]:
        """Explain feature importance."""
        retrieved_docs = self.retriever.retrieve(
            query="feature importance feature selection",
            project_id=project_id,
            min_score=0.5
        )
        
        context = self._build_context(retrieved_docs)
        
        system_prompt = """You are an ML engineer. Explain feature importance:
1. Methodology used (gain, permutation, etc.)
2. Top features and their importance scores
3. Feature relationships
4. Feature engineering impact
5. Recommendations for feature selection

Be technical and specific."""
        
        explanation = self.llm_client.generate_response(
            query="Explain the feature importance analysis.",
            context=context,
            system_prompt=system_prompt
        )
        
        return {"feature_importance_explanation": explanation, "sources": self._get_sources(retrieved_docs)}
    
    def explain_evaluation_metrics(self, project_id: int) -> Dict[str, Any]:
        """Explain evaluation metrics."""
        retrieved_docs = self.retriever.retrieve(
            query="evaluation metrics accuracy precision recall F1 ROC AUC",
            project_id=project_id,
            min_score=0.5
        )
        
        context = self._build_context(retrieved_docs)
        
        system_prompt = """You are an ML evaluation expert. Explain the evaluation metrics:
1. Metrics used and their definitions
2. Metric values and interpretation
3. Trade-offs between metrics
4. Business relevance of metrics
5. Comparison to baselines

Be clear and contextual."""
        
        explanation = self.llm_client.generate_response(
            query="Explain the evaluation metrics used in this project.",
            context=context,
            system_prompt=system_prompt
        )
        
        return {"metrics_explanation": explanation, "sources": self._get_sources(retrieved_docs)}
    
    def explain_feature_engineering(self, project_id: int) -> Dict[str, Any]:
        """Explain feature engineering techniques."""
        retrieved_docs = self.retriever.retrieve(
            query="feature engineering preprocessing transformation",
            project_id=project_id,
            min_score=0.5
        )
        
        context = self._build_context(retrieved_docs)
        
        system_prompt = """You are an ML engineer. Explain feature engineering:
1. Techniques used (encoding, scaling, etc.)
2. Feature creation and transformation
3. Handling missing values
4. Feature selection methods
5. Impact on model performance

Be detailed and technical."""
        
        explanation = self.llm_client.generate_response(
            query="Explain the feature engineering approach.",
            context=context,
            system_prompt=system_prompt
        )
        
        return {"feature_engineering_explanation": explanation, "sources": self._get_sources(retrieved_docs)}
    
    def explain_modeling_decisions(self, project_id: int) -> Dict[str, Any]:
        """Explain modeling decisions."""
        retrieved_docs = self.retriever.retrieve(
            query="model selection hyperparameters architecture",
            project_id=project_id,
            min_score=0.5
        )
        
        context = self._build_context(retrieved_docs)
        
        system_prompt = """You are an ML architect. Explain modeling decisions:
1. Model choice and rationale
2. Hyperparameter tuning approach
3. Architecture decisions
4. Training strategy
5. Alternative models considered

Be comprehensive and justify decisions."""
        
        explanation = self.llm_client.generate_response(
            query="Explain the modeling decisions made in this project.",
            context=context,
            system_prompt=system_prompt
        )
        
        return {"modeling_decisions_explanation": explanation, "sources": self._get_sources(retrieved_docs)}
    
    def _build_context(self, docs: list) -> str:
        """Build context from retrieved documents."""
        return "\n\n".join([
            f"[{doc['metadata'].get('filename', 'Unknown')}]\n{doc['content']}"
            for doc in docs
        ])
    
    def _get_sources(self, docs: list) -> list:
        """Extract source filenames."""
        return [doc["metadata"].get("filename") for doc in docs]
