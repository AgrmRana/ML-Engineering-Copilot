"""Retrieval: over-fetch, diversify with MMR, and assemble a token-bounded context.

Deliberately contains no LLM calls -- retrieval and generation are separate
concerns so each can be tested, tuned, and reasoned about independently.

There is intentionally no fixed "minimum similarity" cutoff. The previous
version of this project filtered results with `score >= min_confidence_score`
against a Chroma distance (where *lower* is better), which was both backwards
and, even fixed, still an arbitrary magic number. Instead, retrieval always
returns its best-ranked, diversified candidates, and the prompt is
responsible for explicitly saying "not enough information" when the
retrieved context isn't actually relevant -- that's a property of generation,
not retrieval.
"""
from chunking import Chunk
from embeddings import embed_texts
from utils import count_tokens
from vector_store import VectorStore

FETCH_MULTIPLIER = 4  # over-fetch this many candidates per requested result, before MMR
MMR_LAMBDA = 0.5  # 0 = maximize diversity only, 1 = maximize relevance only
CONTEXT_TOKEN_BUDGET = 6000


def retrieve(query: str, store: VectorStore, k: int = 6) -> list[tuple[Chunk, float]]:
    """Return up to k relevant, mutually-diverse (chunk, cosine_score) pairs."""
    if len(store) == 0:
        return []

    query_vec = embed_texts([query])[0]
    fetch_k = min(len(store), max(k * FETCH_MULTIPLIER, k))
    hits = store.search(query_vec, fetch_k)
    if not hits:
        return []

    selected = _mmr_select(hits, store, k)
    return [(store.chunks[idx], score) for idx, score in selected]


def _mmr_select(hits: list[tuple[int, float]], store: VectorStore, k: int) -> list[tuple[int, float]]:
    """Maximal Marginal Relevance: greedily trade off relevance vs. redundancy."""
    sims_to_query = [score for _, score in hits]
    vectors = [store.get_vector(idx) for idx, _ in hits]

    selected: list[int] = []
    remaining = list(range(len(hits)))

    while remaining and len(selected) < k:
        if not selected:
            best = max(remaining, key=lambda i: sims_to_query[i])
        else:
            def mmr_score(i: int) -> float:
                redundancy = max(float(vectors[i] @ vectors[j]) for j in selected)
                return MMR_LAMBDA * sims_to_query[i] - (1 - MMR_LAMBDA) * redundancy

            best = max(remaining, key=mmr_score)
        selected.append(best)
        remaining.remove(best)

    return [hits[i] for i in selected]


def build_context(results: list[tuple[Chunk, float]], token_budget: int = CONTEXT_TOKEN_BUDGET) -> tuple[str, list[dict]]:
    """Assemble a numbered context block, stopping once the token budget is spent.

    Returns (context_text, sources) where sources carries enough metadata
    (filename, chunk position, score, snippet) for the UI to render a
    citation that's traceable back to the exact chunk that was used.
    """
    blocks: list[str] = []
    sources: list[dict] = []
    used_tokens = 0

    for n, (chunk, score) in enumerate(results, start=1):
        header = f"[{n}] {chunk.filename} (chunk {chunk.chunk_index + 1}/{chunk.total_chunks})"
        block = f"{header}\n{chunk.text}"
        block_tokens = count_tokens(block)

        if blocks and used_tokens + block_tokens > token_budget:
            break

        blocks.append(block)
        used_tokens += block_tokens
        sources.append({
            "n": n,
            "filename": chunk.filename,
            "chunk_index": chunk.chunk_index,
            "total_chunks": chunk.total_chunks,
            "score": round(score, 3),
            "snippet": chunk.text[:300],
        })

    return "\n\n".join(blocks), sources
