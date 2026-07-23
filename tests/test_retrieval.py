import numpy as np
import pytest

import retrieval
import vector_store as vs_module
from chunking import Chunk

# Fixed 2D unit vectors (as angles from a shared origin) so cosine similarities
# are exact and hand-verifiable:
#   query = 0 deg
#   a     = 10 deg   (very relevant, selected first)
#   a_dup = 13 deg   (near-duplicate of "a": only 3 deg apart -> highly redundant)
#   b     = -40 deg  (moderately relevant, but far from "a" -> diverse)
_ANGLES = {"query": 0, "a": 10, "a_dup": 13, "b": -40}


def _unit(angle_degrees: float) -> np.ndarray:
    rad = np.radians(angle_degrees)
    return np.array([np.cos(rad), np.sin(rad)], dtype="float32")


_VECTORS = {key: _unit(angle) for key, angle in _ANGLES.items()}


def _fake_embed(texts: list[str]) -> np.ndarray:
    return np.array([_VECTORS[t] for t in texts], dtype="float32")


@pytest.fixture(autouse=True)
def patch_embeddings(monkeypatch):
    monkeypatch.setattr(vs_module, "embed_texts", _fake_embed)
    monkeypatch.setattr(retrieval, "embed_texts", _fake_embed)


def _make_chunk(key: str) -> Chunk:
    return Chunk(text=key, filename=f"{key}.txt", file_type="text", chunk_index=0, total_chunks=1)


def test_mmr_prefers_diversity_over_near_duplicate():
    store = vs_module.VectorStore()
    store.add([_make_chunk("a"), _make_chunk("a_dup"), _make_chunk("b")])

    results = retrieval.retrieve("query", store, k=2)
    texts = [chunk.text for chunk, _ in results]

    assert texts[0] == "a"  # most similar to the query, selected first
    assert "b" in texts  # diverse pick beats the near-duplicate "a_dup"
    assert "a_dup" not in texts


def test_plain_top_k_would_have_picked_the_duplicate():
    # Sanity check on the fixture itself: without MMR, top-2 by raw similarity
    # would be "a" and "a_dup" -- this is the behavior MMR is meant to avoid.
    store = vs_module.VectorStore()
    store.add([_make_chunk("a"), _make_chunk("a_dup"), _make_chunk("b")])

    query_vec = _fake_embed(["query"])[0]
    raw_hits = store.search(query_vec, k=2)
    raw_texts = [store.chunks[idx].text for idx, _ in raw_hits]

    assert raw_texts == ["a", "a_dup"]


def test_retrieve_on_empty_store_returns_empty():
    store = vs_module.VectorStore()
    assert retrieval.retrieve("query", store, k=3) == []


def test_build_context_respects_token_budget():
    long_text = "word " * 2000  # comfortably exceeds any reasonable budget alone
    results = [
        (Chunk(text=long_text, filename="big.txt", file_type="text", chunk_index=0, total_chunks=1), 0.9),
        (Chunk(text="short chunk", filename="small.txt", file_type="text", chunk_index=0, total_chunks=1), 0.8),
    ]

    context, sources = retrieval.build_context(results, token_budget=100)

    # The first (oversized) block is always included so context is never empty;
    # the second is dropped once the budget is already exceeded.
    assert len(sources) == 1
    assert sources[0]["filename"] == "big.txt"
    assert "small.txt" not in context
