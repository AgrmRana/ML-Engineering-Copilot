import numpy as np
import pytest

import vector_store as vs_module
from chunking import Chunk


def _fake_embed(texts: list[str]) -> np.ndarray:
    """Deterministic synthetic embeddings -- identical text always maps to the same
    vector, so similarity search is exact and no network call is made."""
    vectors = []
    for t in texts:
        seed = abs(hash(t)) % (2**32)
        rng = np.random.default_rng(seed)
        v = rng.normal(size=16).astype("float32")
        v /= np.linalg.norm(v)
        vectors.append(v)
    return np.array(vectors, dtype="float32")


@pytest.fixture(autouse=True)
def patch_embeddings(monkeypatch):
    monkeypatch.setattr(vs_module, "embed_texts", _fake_embed)


def _make_chunk(text: str) -> Chunk:
    return Chunk(text=text, filename="doc.txt", file_type="text", chunk_index=0, total_chunks=1)


def test_add_and_search_returns_most_similar_first():
    store = vs_module.VectorStore()
    chunks = [
        _make_chunk("apple banana fruit"),
        _make_chunk("rocket ship space"),
        _make_chunk("banana smoothie recipe"),
    ]
    store.add(chunks)
    assert len(store) == 3

    query_vec = _fake_embed(["banana smoothie recipe"])[0]
    hits = store.search(query_vec, k=2)

    assert len(hits) == 2
    top_idx, top_score = hits[0]
    assert store.chunks[top_idx].text == "banana smoothie recipe"
    assert top_score == pytest.approx(1.0, abs=1e-4)  # identical text -> identical vector


def test_search_caps_k_to_available_chunks():
    store = vs_module.VectorStore()
    store.add([_make_chunk("only one chunk")])

    query_vec = _fake_embed(["anything"])[0]
    hits = store.search(query_vec, k=10)

    assert len(hits) == 1


def test_search_on_empty_store_returns_empty():
    store = vs_module.VectorStore()
    query_vec = _fake_embed(["anything"])[0]
    assert store.search(query_vec, k=5) == []


def test_add_accumulates_across_multiple_calls():
    store = vs_module.VectorStore()
    store.add([_make_chunk("first batch")])
    store.add([_make_chunk("second batch")])
    assert len(store) == 2
