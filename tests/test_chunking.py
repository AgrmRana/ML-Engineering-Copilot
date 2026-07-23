import chunking
from document_loader import RawDocument


def test_chunk_text_respects_token_budget():
    content = "\n\n".join(f"Paragraph {i} " + ("word " * 100) for i in range(20))
    doc = RawDocument(rel_path="notes.txt", file_type="text", content=content, metadata={})

    chunks = chunking.chunk_document(doc)

    assert len(chunks) > 1
    for c in chunks:
        assert chunking.count_tokens(c.text) <= chunking.CHUNK_TOKENS
    assert all(c.total_chunks == len(chunks) for c in chunks)


def test_chunk_python_splits_on_function_and_class_boundaries():
    code = '''"""Module docstring."""
import os


def top_level_function(x):
    return x + 1


class Calculator:
    """A simple calculator."""

    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b
'''
    doc = RawDocument(rel_path="calc.py", file_type="python", content=code, metadata={})

    chunks = chunking.chunk_document(doc)

    assert len(chunks) >= 1
    joined = "\n".join(c.text for c in chunks)
    assert "def top_level_function" in joined
    assert "def add" in joined
    assert "def subtract" in joined


def test_chunk_python_labels_methods_of_oversized_class_with_class_name():
    # A class large enough to exceed CHUNK_TOKENS forces per-method splitting.
    methods = "\n\n".join(
        f"    def method_{i}(self):\n        return {i}  " + ("# padding " * 40)
        for i in range(30)
    )
    code = f"class BigClass:\n{methods}\n"
    doc = RawDocument(rel_path="big.py", file_type="python", content=code, metadata={})

    chunks = chunking.chunk_document(doc)

    assert len(chunks) > 1
    assert any("class BigClass" in c.text for c in chunks)


def test_chunk_python_falls_back_to_text_on_syntax_error():
    bad_code = "def broken(:\n    pass"
    doc = RawDocument(rel_path="broken.py", file_type="python", content=bad_code, metadata={})

    chunks = chunking.chunk_document(doc)

    assert len(chunks) == 1
    assert "broken" in chunks[0].text


def test_chunk_notebook_preserves_cell_type_and_index():
    cells = [
        {"index": 0, "cell_type": "markdown", "source": "# Title\nSome intro text."},
        {"index": 1, "cell_type": "code", "source": "import numpy as np\nprint('hi')"},
    ]
    doc = RawDocument(rel_path="analysis.ipynb", file_type="jupyter", content="", metadata={"cells": cells})

    chunks = chunking.chunk_document(doc)

    assert len(chunks) >= 1
    joined = "\n".join(c.text for c in chunks)
    assert "numpy" in joined
    assert "Title" in joined
    assert "cell 0, markdown" in joined
    assert "cell 1, code" in joined


def test_chunk_markdown_keeps_headers_with_their_section():
    content = "# Section One\nSome content here.\n\n## Subsection\nMore content.\n\n# Section Two\nOther content."
    doc = RawDocument(rel_path="README.md", file_type="markdown", content=content, metadata={})

    chunks = chunking.chunk_document(doc)

    joined = "\n".join(c.text for c in chunks)
    assert "Section One" in joined
    assert "Section Two" in joined


def test_empty_document_produces_no_chunks():
    doc = RawDocument(rel_path="empty.txt", file_type="text", content="   \n\n  ", metadata={})
    assert chunking.chunk_document(doc) == []
