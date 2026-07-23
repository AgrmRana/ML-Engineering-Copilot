"""Local embedding using Sentence Transformers.

No API calls, no cost. The all-MiniLM-L6-v2 model (22MB) runs entirely on
your machine and provides good semantic embeddings for retrieval.

Downloaded on first use and cached locally (~100MB disk space total).
"""
import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts locally, returning an (n, d) L2-normalized float32 matrix.

    Normalizing lets a plain inner product double as cosine similarity, which
    is what the FAISS IndexFlatIP in vector_store.py relies on.

    First call downloads the model (~22MB) from HuggingFace; subsequent calls
    use the cached copy. No internet required after first download.
    """
    if not texts:
        return np.zeros((0, 0), dtype="float32")

    model = _get_model()
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    matrix = np.array(vectors, dtype="float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
