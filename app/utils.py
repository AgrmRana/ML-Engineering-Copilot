"""Small shared helpers: token counting and file-filtering constants."""
from transformers import AutoTokenizer

# The embedding model's own tokenizer (all-MiniLM-L6-v2, max_seq_length=256).
# Chunking sizes chunks against this directly so nothing is silently truncated
# by the embedding model. It's also used to budget LLM prompt sizes -- an
# approximation there (Ollama models use their own tokenizers), but subword
# tokenizers produce similar token/word ratios, and `llm.py` sets a generous
# `num_ctx` with headroom, so the approximation never needs to be exact.
# Already a dependency of sentence-transformers, so this adds nothing new.
_TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


def count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text, add_special_tokens=False))


def encode_tokens(text: str) -> list[int]:
    return _TOKENIZER.encode(text, add_special_tokens=False)


def decode_tokens(tokens: list[int]) -> str:
    return _TOKENIZER.decode(tokens)


def format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


# Directories that never contain useful project content for RAG purposes.
IGNORED_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".ipynb_checkpoints",
    "node_modules",
    "venv",
    ".venv",
    "env",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    ".idea",
    ".vscode",
}

# Binary/non-informative file extensions that aren't worth extracting text from.
IGNORED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".ico", ".webp",
    ".pt", ".pth", ".h5", ".hdf5", ".onnx", ".pkl", ".pickle", ".joblib", ".npz", ".npy",
    ".zip", ".tar", ".gz", ".7z", ".rar",
    ".so", ".dylib", ".dll", ".exe", ".bin",
    ".parquet", ".feather",
    ".pyc",
}

# Per-file size cap so a single huge dataset/asset can't blow up ingestion time or cost.
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20MB
