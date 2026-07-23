"""Token-aware chunking.

Chunk size/overlap are measured in tokens against the embedding model's own
tokenizer (via `count_tokens`), not raw character counts -- a chunk is sized
by how much of the model's context it actually consumes.

Three strategies, chosen per file type:
  - Markdown: split on headers first (keeps sections intact), then pack
    paragraphs into token-bounded windows.
  - Python: split on `ast` top-level function/class boundaries instead of an
    indentation heuristic, so chunk boundaries always land on real syntactic
    edges. Methods of an oversized class are split out individually and
    labeled with their enclosing class name so a chunk never loses that
    context. Falls back to plain text chunking on a `SyntaxError`.
  - Everything else (text, CSV/JSON/PDF dumps): paragraph-level packing.
  - Notebooks: chunked per cell (preserving the code/markdown boundary and
    cell index), with tiny adjacent cells merged and oversized cells split,
    all via the same token-budget packer.
"""
import ast
import re
from dataclasses import dataclass

from document_loader import RawDocument
from utils import count_tokens, decode_tokens, encode_tokens

CHUNK_TOKENS = 200  # comfortably under the embedding model's 256-token max_seq_length
CHUNK_OVERLAP_TOKENS = 30

_HEADER_RE = re.compile(r"^#{1,6}\s+.*$", re.MULTILINE)


@dataclass
class Chunk:
    text: str
    filename: str
    file_type: str
    chunk_index: int
    total_chunks: int


def chunk_document(doc: RawDocument) -> list[Chunk]:
    if doc.file_type == "python":
        pieces = _chunk_python(doc.content)
    elif doc.file_type == "jupyter":
        pieces = _chunk_notebook(doc.metadata.get("cells", []))
    elif doc.file_type == "markdown":
        pieces = _chunk_markdown(doc.content)
    else:
        pieces = _chunk_text(doc.content)

    pieces = [p for p in pieces if p.strip()]
    total = len(pieces)
    return [
        Chunk(text=p, filename=doc.rel_path, file_type=doc.file_type, chunk_index=i, total_chunks=total)
        for i, p in enumerate(pieces)
    ]


# --- generic token-budget packing, shared by every strategy -----------------

def _pack_by_tokens(units: list[str], max_tokens: int, overlap_tokens: int, joiner: str = "\n\n") -> list[str]:
    """Greedily pack text units into token-bounded chunks with a token-based overlap."""
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = count_tokens(unit)

        if unit_tokens > max_tokens:
            if current:
                chunks.append(joiner.join(current))
                current, current_tokens = [], 0
            chunks.extend(_hard_split(unit, max_tokens, overlap_tokens))
            continue

        if current and current_tokens + unit_tokens > max_tokens:
            chunks.append(joiner.join(current))
            # carry the trailing ~overlap_tokens worth of units into the next chunk
            overlap_units: list[str] = []
            overlap_count = 0
            for u in reversed(current):
                t = count_tokens(u)
                if overlap_count + t > overlap_tokens:
                    break
                overlap_units.insert(0, u)
                overlap_count += t
            current, current_tokens = overlap_units, overlap_count

        current.append(unit)
        current_tokens += unit_tokens

    if current:
        chunks.append(joiner.join(current))
    return chunks


def _hard_split(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Last-resort splitter for a single unit that alone exceeds the token budget."""
    tokens = encode_tokens(text)
    if len(tokens) <= max_tokens:
        return [text]

    step = max(max_tokens - overlap_tokens, 1)
    windows = []
    for start in range(0, len(tokens), step):
        window = tokens[start:start + max_tokens]
        windows.append(decode_tokens(window))
        if start + max_tokens >= len(tokens):
            break
    return windows


# --- text / markdown ---------------------------------------------------------

def _chunk_text(content: str) -> list[str]:
    paragraphs = [p for p in re.split(r"\n\s*\n", content) if p.strip()]
    if not paragraphs:
        return []
    return _pack_by_tokens(paragraphs, CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS)


def _split_markdown_sections(content: str) -> list[str]:
    matches = list(_HEADER_RE.finditer(content))
    if not matches:
        return [content]

    sections = []
    if matches[0].start() > 0:
        sections.append(content[: matches[0].start()])
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        sections.append(content[m.start():end])
    return sections


def _chunk_markdown(content: str) -> list[str]:
    """Split on headers, but let small adjacent sections pack together up to the
    token budget -- otherwise a header-heavy file (many short sections) produces
    far more tiny chunks than the budget calls for."""
    sections = _split_markdown_sections(content)
    units: list[str] = []
    for section in sections:
        section = section.strip("\n")
        if not section.strip():
            continue
        if count_tokens(section) <= CHUNK_TOKENS:
            units.append(section)  # whole section packs as a single unit
        else:
            paragraphs = [p for p in re.split(r"\n\s*\n", section) if p.strip()]
            units.extend(paragraphs)

    if not units:
        return _chunk_text(content)
    return _pack_by_tokens(units, CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS)


# --- python (ast-based) -------------------------------------------------------

def _node_end_line(node: ast.AST, lines: list[str]) -> int:
    end_lineno = getattr(node, "end_lineno", None)
    return end_lineno if end_lineno else len(lines)


def _node_label(node: ast.AST) -> str:
    if isinstance(node, ast.ClassDef):
        return f"class {node.name}"
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return f"function {node.name}"
    return ""


def _split_class(node: ast.ClassDef, lines: list[str]) -> list[str]:
    """Split an oversized class into per-member chunks, each labeled with the class name."""
    units = []
    body = list(node.body)
    class_start = node.lineno - 1
    first_member_start = (body[0].lineno - 1) if body else _node_end_line(node, lines)

    header = "".join(lines[class_start:first_member_start])
    if header.strip():
        units.append(f"# class {node.name}\n{header}")

    cursor = first_member_start
    for member in body:
        start_idx = member.lineno - 1
        if start_idx > cursor:
            gap = "".join(lines[cursor:start_idx])
            if gap.strip():
                units.append(f"# class {node.name}\n{gap}")
        end_idx = _node_end_line(member, lines)
        text = "".join(lines[start_idx:end_idx])
        label = _node_label(member)
        prefix = f"class {node.name}" + (f", {label}" if label else "")
        units.append(f"# {prefix}\n{text}")
        cursor = end_idx

    return units


def _chunk_python(content: str) -> list[str]:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return _chunk_text(content)

    lines = content.splitlines(keepends=True)
    top_nodes = list(ast.iter_child_nodes(tree))
    if not top_nodes:
        return _chunk_text(content)

    units: list[str] = []
    cursor = 0
    for node in top_nodes:
        start_idx = getattr(node, "lineno", cursor + 1) - 1
        end_idx = _node_end_line(node, lines)

        if start_idx > cursor:
            gap = "".join(lines[cursor:start_idx])
            if gap.strip():
                units.append(gap)

        node_text = "".join(lines[start_idx:end_idx])
        if isinstance(node, ast.ClassDef) and count_tokens(node_text) > CHUNK_TOKENS:
            units.extend(_split_class(node, lines))
        else:
            label = _node_label(node)
            units.append(f"# {label}\n{node_text}" if label else node_text)

        cursor = end_idx

    if cursor < len(lines):
        tail = "".join(lines[cursor:])
        if tail.strip():
            units.append(tail)

    return _pack_by_tokens(units, CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS)


# --- notebooks (per-cell) -----------------------------------------------------

def _chunk_notebook(cells: list[dict]) -> list[str]:
    units = []
    for cell in cells:
        source = cell["source"].strip()
        if not source:
            continue
        if cell["cell_type"] == "code":
            units.append(f"# [cell {cell['index']}, code]\n```python\n{source}\n```")
        else:
            units.append(f"# [cell {cell['index']}, markdown]\n{source}")

    if not units:
        return []
    return _pack_by_tokens(units, CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS)
