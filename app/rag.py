"""Grounded chat: one retrieval call, one LLM call, per user turn."""
from typing import Optional

from llm import chat
from prompts import render_rag_prompt
from retrieval import build_context, retrieve
from utils import count_tokens
from vector_store import VectorStore

TOP_K = 6
HISTORY_TOKEN_BUDGET = 2000


def answer_question(query: str, store: VectorStore, history: Optional[list[dict]] = None) -> dict:
    """Answer a user question grounded in retrieved context.

    Exactly one retrieval call and one LLM call happen here -- no chained or
    speculative extra requests.
    """
    results = retrieve(query, store, k=TOP_K)
    context, sources = build_context(results)
    system_prompt = render_rag_prompt(context)

    messages = _trim_history(history or []) + [{"role": "user", "content": query}]
    answer = chat(system_prompt, messages, temperature=0.0)

    return {"answer": answer, "sources": sources}


def _trim_history(history: list[dict]) -> list[dict]:
    """Keep only the most recent turns that fit in the history token budget."""
    trimmed: list[dict] = []
    used = 0
    for msg in reversed(history):
        tokens = count_tokens(msg["content"])
        if trimmed and used + tokens > HISTORY_TOKEN_BUDGET:
            break
        trimmed.insert(0, {"role": msg["role"], "content": msg["content"]})
        used += tokens
    return trimmed
