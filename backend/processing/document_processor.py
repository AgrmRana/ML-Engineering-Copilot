from typing import List, Dict, Any, Optional
from pathlib import Path
import pypdf
import docx
import pandas as pd
import nbformat
from langchain.schema import Document
from database.models import DocumentType
from .chunker import SemanticChunker


class DocumentProcessor:
    """Process various document types for ingestion."""
    
    def __init__(self, chunker: Optional[SemanticChunker] = None):
        self.chunker = chunker or SemanticChunker()
    
    def process_file(self, file_path: str, filename: str) -> tuple[str, List[Document], Dict[str, Any]]:
        """Process a file and return content, chunks, and metadata."""
        file_path_obj = Path(file_path)
        file_type = self._detect_file_type(filename, file_path_obj)
        
        content, metadata = self._extract_content(file_path_obj, file_type, filename)
        chunks = self._chunk_content(content, metadata, file_type)
        
        return content, chunks, metadata
    
    def _detect_file_type(self, filename: str, file_path: Path) -> DocumentType:
        """Detect document type from filename."""
        ext = file_path.suffix.lower()
        
        type_mapping = {
            ".pdf": DocumentType.PDF,
            ".md": DocumentType.MARKDOWN,
            ".txt": DocumentType.TXT,
            ".docx": DocumentType.DOCX,
            ".csv": DocumentType.CSV,
            ".json": DocumentType.JSON,
            ".py": DocumentType.PYTHON,
            ".ipynb": DocumentType.JUPYTER,
        }
        
        return type_mapping.get(ext, DocumentType.OTHER)
    
    def _extract_content(self, file_path: Path, file_type: DocumentType, filename: str) -> tuple[str, Dict[str, Any]]:
        """Extract text content from file based on type."""
        metadata = {
            "filename": filename,
            "file_type": file_type.value,
            "file_size": file_path.stat().st_size
        }
        
        if file_type == DocumentType.PDF:
            return self._extract_pdf(file_path, metadata)
        elif file_type == DocumentType.DOCX:
            return self._extract_docx(file_path, metadata)
        elif file_type == DocumentType.CSV:
            return self._extract_csv(file_path, metadata)
        elif file_type == DocumentType.JSON:
            return self._extract_json(file_path, metadata)
        elif file_type == DocumentType.JUPYTER:
            return self._extract_jupyter(file_path, metadata)
        else:  # TXT, MD, PYTHON, OTHER
            return self._extract_text(file_path, metadata)
    
    def _extract_pdf(self, file_path: Path, metadata: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """Extract text from PDF."""
        reader = pypdf.PdfReader(file_path)
        content = ""
        for page in reader.pages:
            content += page.extract_text() + "\n"
        
        metadata["page_count"] = len(reader.pages)
        return content, metadata
    
    def _extract_docx(self, file_path: Path, metadata: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """Extract text from DOCX."""
        doc = docx.Document(file_path)
        content = "\n".join(paragraph.text for paragraph in doc.paragraphs)
        return content, metadata
    
    def _extract_csv(self, file_path: Path, metadata: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """Extract text from CSV."""
        df = pd.read_csv(file_path)
        content = df.to_string(index=False)
        metadata["columns"] = df.columns.tolist()
        metadata["row_count"] = len(df)
        return content, metadata
    
    def _extract_json(self, file_path: Path, metadata: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """Extract text from JSON."""
        import json
        with open(file_path, 'r') as f:
            data = json.load(f)
        content = json.dumps(data, indent=2)
        return content, metadata
    
    def _extract_jupyter(self, file_path: Path, metadata: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """Extract code and markdown from Jupyter notebook."""
        notebook = nbformat.read(file_path, as_version=4)
        content = ""
        
        for cell in notebook.cells:
            if cell.cell_type == "code":
                content += f"```python\n{cell.source}\n```\n\n"
            elif cell.cell_type == "markdown":
                content += f"{cell.source}\n\n"
        
        metadata["cell_count"] = len(notebook.cells)
        return content, metadata
    
    def _extract_text(self, file_path: Path, metadata: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """Extract text from plain text files."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return content, metadata
    
    def _chunk_content(self, content: str, metadata: Dict[str, Any], file_type: DocumentType) -> List[Document]:
        """Chunk content based on file type."""
        if file_type in [DocumentType.PYTHON, DocumentType.JUPYTER]:
            return self.chunker.chunk_code(content, metadata)
        else:
            return self.chunker.chunk_document(content, metadata)
