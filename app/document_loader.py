"""Reads uploaded files (or a zipped repository) into plain text + metadata.

Everything here happens against a temporary directory that is deleted as soon as
ingestion finishes -- nothing from an upload is kept on disk afterwards.
"""
import json
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import nbformat
import pandas as pd
import pypdf

from utils import IGNORED_DIR_NAMES, IGNORED_EXTENSIONS, MAX_FILE_SIZE_BYTES, format_size

EXTENSION_TO_TYPE = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".py": "python",
    ".ipynb": "jupyter",
    ".pdf": "pdf",
    ".csv": "csv",
    ".json": "json",
}


@dataclass
class RawDocument:
    rel_path: str
    file_type: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


def detect_file_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return EXTENSION_TO_TYPE.get(ext, "text")


def ingest_uploads(uploaded_files: list) -> tuple[list[RawDocument], list[str]]:
    """Ingest a list of Streamlit UploadedFile objects (files and/or .zip archives).

    Returns (documents, skipped_reasons). All extraction happens in a temp
    directory that is removed before this function returns.
    """
    docs: list[RawDocument] = []
    skipped: list[str] = []

    with tempfile.TemporaryDirectory(prefix="rag_upload_") as tmp:
        tmp_path = Path(tmp)
        for uploaded in uploaded_files:
            name = uploaded.name
            data = uploaded.getvalue()

            if name.lower().endswith(".zip"):
                zip_path = tmp_path / name
                zip_path.write_bytes(data)
                extract_dir = tmp_path / f"_extracted_{zip_path.stem}"
                extract_dir.mkdir(parents=True, exist_ok=True)
                try:
                    with zipfile.ZipFile(zip_path) as zf:
                        zf.extractall(extract_dir)
                except zipfile.BadZipFile:
                    skipped.append(f"{name} (not a valid zip file)")
                    continue
                sub_docs, sub_skipped = _walk_directory(extract_dir)
                docs.extend(sub_docs)
                skipped.extend(sub_skipped)
            else:
                file_path = tmp_path / name
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_bytes(data)
                doc, reason = _load_single(file_path, name)
                if doc:
                    docs.append(doc)
                elif reason:
                    skipped.append(f"{name} ({reason})")

    return docs, skipped


def _walk_directory(root: Path) -> tuple[list[RawDocument], list[str]]:
    docs: list[RawDocument] = []
    skipped: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        rel_parts = path.relative_to(root).parts
        if any(part in IGNORED_DIR_NAMES for part in rel_parts[:-1]):
            continue
        rel_path = "/".join(rel_parts)
        doc, reason = _load_single(path, rel_path)
        if doc:
            docs.append(doc)
        elif reason:
            skipped.append(f"{rel_path} ({reason})")
    return docs, skipped


def _load_single(path: Path, rel_path: str) -> tuple[Optional[RawDocument], Optional[str]]:
    if path.suffix.lower() in IGNORED_EXTENSIONS:
        return None, None  # expected noise (images, weights, archives) -- don't clutter the skip list

    try:
        size = path.stat().st_size
    except OSError:
        return None, "unreadable"

    if size == 0:
        return None, "empty"
    if size > MAX_FILE_SIZE_BYTES:
        return None, f"exceeds {format_size(MAX_FILE_SIZE_BYTES)} per-file limit"

    file_type = detect_file_type(rel_path)
    try:
        if file_type == "pdf":
            content, meta = _extract_pdf(path)
        elif file_type == "jupyter":
            content, meta = _extract_notebook(path)
        elif file_type == "csv":
            content, meta = _extract_csv(path)
        elif file_type == "json":
            content, meta = _extract_json(path)
        else:
            content, meta = _extract_text(path)
    except Exception as exc:  # malformed PDF/notebook/csv/etc -- skip, don't crash ingestion
        return None, f"failed to parse ({exc})"

    if not content.strip():
        return None, "no extractable text"

    meta.update({"filename": rel_path, "file_size": size})
    return RawDocument(rel_path=rel_path, file_type=file_type, content=content, metadata=meta), None


def _extract_text(path: Path) -> tuple[str, dict]:
    content = path.read_text(encoding="utf-8", errors="ignore")
    return content, {}


def _extract_pdf(path: Path) -> tuple[str, dict]:
    reader = pypdf.PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages), {"page_count": len(reader.pages)}


def _extract_notebook(path: Path) -> tuple[str, dict]:
    notebook = nbformat.read(path, as_version=4)
    cells = []
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type not in ("code", "markdown"):
            continue
        cells.append({"index": i, "cell_type": cell.cell_type, "source": cell.source})
    content = "\n\n".join(c["source"] for c in cells)
    return content, {"cells": cells, "cell_count": len(notebook.cells)}


def _extract_csv(path: Path) -> tuple[str, dict]:
    df = pd.read_csv(path)
    meta = {"rows": len(df), "columns": df.columns.astype(str).tolist()}

    if len(df) <= 50:
        content = df.to_string(index=False)
    else:
        # A full dump of a large CSV is expensive to embed and not very semantically
        # useful -- a schema + stats + sample is more informative per token spent.
        parts = [
            f"Shape: {df.shape[0]} rows x {df.shape[1]} columns",
            "Columns: " + ", ".join(df.columns.astype(str)),
            "Dtypes:\n" + df.dtypes.to_string(),
            "Summary statistics:\n" + df.describe(include="all").to_string(),
            "First 20 rows:\n" + df.head(20).to_string(index=False),
            "Last 5 rows:\n" + df.tail(5).to_string(index=False),
        ]
        content = "\n\n".join(parts)

    return content, meta


def _extract_json(path: Path) -> tuple[str, dict]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)
    return json.dumps(data, indent=2), {}
