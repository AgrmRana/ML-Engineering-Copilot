"""Thin OpenAI chat-completion wrapper -- the only place that calls the chat API.

No LangChain: a single direct call is simpler, has fewer moving parts, and
keeps exactly what's sent to the model visible rather than hidden behind a
framework abstraction.
"""
import os

from openai import OpenAI

_CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def chat(system_prompt: str, messages: list[dict], temperature: float = 0.0) -> str:
    """One chat-completion call. `messages` is a list of {"role", "content"} dicts.

    Temperature defaults to 0.0: grounded, factual QA over retrieved context
    should be deterministic, not creative -- a high temperature only adds
    hallucination risk here.
    """
    client = _get_client()
    full_messages = [{"role": "system", "content": system_prompt}, *messages]
    response = client.chat.completions.create(
        model=_CHAT_MODEL,
        messages=full_messages,
        temperature=temperature,
    )
    return response.choices[0].message.content
