from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from config.settings import settings


class LLMClient:
    """Client for LLM interactions."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            openai_api_key=settings.openai_api_key,
            model=settings.openai_model,
            temperature=0.7
        )
    
    def generate_response(
        self,
        query: str,
        context: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate a response from the LLM."""
        messages = []
        
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        
        if context:
            messages.append(SystemMessage(content=f"Context:\n{context}"))
        
        if conversation_history:
            for msg in conversation_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
        
        messages.append(HumanMessage(content=query))
        
        response = self.llm.invoke(messages)
        return response.content
    
    def generate_with_sources(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Generate response with source citations."""
        # Format context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            metadata = doc.get("metadata", {})
            source = metadata.get("filename", "Unknown")
            context_parts.append(
                f"[Source {i}: {source}]\n{doc['content']}"
            )
        
        context = "\n\n".join(context_parts)
        
        system_prompt = """You are an expert ML engineering assistant. Answer questions based on the provided context.
Always cite your sources using [Source X] notation.
If the context doesn't contain enough information to answer the question, state that clearly.
Be precise and technical."""
        
        response = self.generate_response(
            query=query,
            context=context,
            conversation_history=conversation_history,
            system_prompt=system_prompt
        )
        
        # Extract sources from response
        sources = self._extract_sources(retrieved_docs)
        
        return {
            "response": response,
            "sources": sources,
            "context_used": len(retrieved_docs)
        }
    
    def _extract_sources(self, retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract source information from retrieved documents."""
        sources = []
        for doc in retrieved_docs:
            metadata = doc.get("metadata", {})
            sources.append({
                "filename": metadata.get("filename", "Unknown"),
                "file_type": metadata.get("file_type", "Unknown"),
                "chunk_index": metadata.get("chunk_index", 0),
                "score": doc.get("score", 0.0)
            })
        return sources
