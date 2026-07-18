from typing import List, Dict, Any
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


class SemanticChunker:
    """Semantic chunking for better context preservation."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_document(self, content: str, metadata: Dict[str, Any]) -> List[Document]:
        """Split document into semantic chunks."""
        # Extract structure-aware chunks
        chunks = self.text_splitter.split_text(content)
        
        documents = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_index": i,
                "chunk_count": len(chunks)
            })
            documents.append(Document(page_content=chunk, metadata=chunk_metadata))
        
        return documents
    
    def chunk_code(self, content: str, metadata: Dict[str, Any]) -> List[Document]:
        """Chunk code files preserving function/class structure."""
        # Split by function/class definitions for Python
        if metadata.get("file_type") == "python":
            return self._chunk_python_code(content, metadata)
        
        # Default to regular chunking for other code
        return self.chunk_document(content, metadata)
    
    def _chunk_python_code(self, content: str, metadata: Dict[str, Any]) -> List[Document]:
        """Chunk Python code by function/class definitions."""
        chunks = []
        lines = content.split("\n")
        current_chunk = []
        current_indent = 0
        
        for line in lines:
            stripped = line.lstrip()
            if not stripped:
                current_chunk.append(line)
                continue
            
            # Detect function/class definition
            if stripped.startswith(("def ", "class ")):
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_indent = len(line) - len(stripped)
            else:
                indent = len(line) - len(stripped)
                # New top-level block
                if indent <= current_indent and current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = [line]
                    current_indent = indent
                else:
                    current_chunk.append(line)
        
        if current_chunk:
            chunks.append("\n".join(current_chunk))
        
        documents = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_index": i,
                "chunk_count": len(chunks),
                "is_code": True
            })
            documents.append(Document(page_content=chunk, metadata=chunk_metadata))
        
        return documents
