"""Batched OpenAI embedding calls.

Defaults to text-embedding-3-small: cheaper and faster than -large, and
plenty of quality for a local single-session assistant -- there's no
production-scale recall requirement here that would justify the larger
model's cost.
"""
import os

import numpy as np
from openai import OpenAI

_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
_BATCH_SIZE = 100

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts, returning an (n, d) L2-normalized float32 matrix.

    Normalizing lets a plain inner product double as cosine similarity, which
    is what the FAISS IndexFlatIP in vector_store.py relies on.
    """
    if not texts:
        return np.zeros((0, 0), dtype="float32")

    client = _get_client()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i:i + _BATCH_SIZE]
        response = client.embeddings.create(model=_EMBEDDING_MODEL, input=batch)
        vectors.extend(item.embedding for item in response.data)

    matrix = np.array(vectors, dtype="float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
