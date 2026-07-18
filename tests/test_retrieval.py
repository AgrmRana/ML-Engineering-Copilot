import pytest
from retrieval.vector_store import VectorStore
from retrieval.hierarchical_retriever import HierarchicalRetriever
from langchain.schema import Document


def test_vector_store_initialization():
    """Test vector store initialization."""
    vector_store = VectorStore()
    assert vector_store is not None
    assert vector_store.embeddings is not None


def test_hierarchical_retriever_initialization():
    """Test hierarchical retriever initialization."""
    vector_store = VectorStore()
    retriever = HierarchicalRetriever(vector_store, top_k=5)
    assert retriever is not None
    assert retriever.top_k == 5


def test_retrieve_with_no_documents():
    """Test retrieval with no documents in vector store."""
    vector_store = VectorStore()
    retriever = HierarchicalRetriever(vector_store, top_k=5)
    
    results = retriever.retrieve(
        query="test query",
        project_id=1,
        min_score=0.7
    )
    
    # Should return empty list when no documents exist
    assert isinstance(results, list)
