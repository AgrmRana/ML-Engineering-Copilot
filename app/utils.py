"""Small shared helpers: token counting and file-filtering constants."""
import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens the way the embedding/chat models see them."""
    return len(_ENCODING.encode(text))


def encode_tokens(text: str) -> list[int]:
    return _ENCODING.encode(text)


def decode_tokens(tokens: list[int]) -> str:
    return _ENCODING.decode(tokens)


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
