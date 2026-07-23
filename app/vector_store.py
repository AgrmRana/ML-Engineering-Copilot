"""In-memory cosine-similarity index over document chunks.

Uses FAISS's IndexFlatIP over L2-normalized embeddings, so inner product IS
cosine similarity -- there is exactly one score, and higher always means more
similar. Nothing is written to disk; the index and every embedding disappear
when the process (or the session) ends.
"""
import faiss
import numpy as np

from chunking import Chunk
from embeddings import embed_texts


class VectorStore:
    def __init__(self) -> None:
        self._index: faiss.IndexFlatIP | None = None
        self._vectors: np.ndarray | None = None
        self.chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        vectors = embed_texts([c.text for c in chunks])
        if self._index is None:
            self._index = faiss.IndexFlatIP(vectors.shape[1])
            self._vectors = vectors
        else:
            self._vectors = np.vstack([self._vectors, vectors])

        self._index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return (chunk_index, cosine_score) pairs, best first."""
        if self._index is None or not self.chunks:
            return []

        k = min(k, len(self.chunks))
        scores, indices = self._index.search(query_vector.reshape(1, -1).astype("float32"), k)
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx != -1]

    def get_vector(self, idx: int) -> np.ndarray:
        return self._vectors[idx]
