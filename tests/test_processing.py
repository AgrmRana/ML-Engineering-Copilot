import pytest
from processing.chunker import SemanticChunker
from processing.document_processor import DocumentProcessor
from langchain.schema import Document


def test_semantic_chunker():
    """Test semantic chunking."""
    chunker = SemanticChunker(chunk_size=100, chunk_overlap=20)
    
    content = "This is a test document. It has multiple sentences. Each sentence should be chunked properly."
    metadata = {"filename": "test.txt", "file_type": "txt"}
    
    chunks = chunker.chunk_document(content, metadata)
    
    assert len(chunks) > 0
    assert all(isinstance(chunk, Document) for chunk in chunks)
    assert all("chunk_index" in chunk.metadata for chunk in chunks)


def test_code_chunking():
    """Test code-aware chunking for Python files."""
    chunker = SemanticChunker()
    
    code = """
def hello_world():
    print("Hello, World!")

class Calculator:
    def add(self, a, b):
        return a + b
"""
    metadata = {"filename": "test.py", "file_type": "python"}
    
    chunks = chunker.chunk_code(code, metadata)
    
    assert len(chunks) > 0
    assert all(chunk.metadata.get("is_code") for chunk in chunks)


def test_document_processor():
    """Test document processor initialization."""
    processor = DocumentProcessor()
    assert processor is not None
    assert processor.chunker is not None
