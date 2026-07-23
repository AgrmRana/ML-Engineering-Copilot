"""Local chat completion via Ollama -- the only place that calls the LLM.

No API, no cost, no LangChain. Requires Ollama running locally
(https://ollama.ai) with a model pulled, e.g. `ollama pull mistral`.
"""
import os

import requests

_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
_CHAT_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

# Ollama defaults to a small context window (historically 2048 tokens)
# regardless of what the model actually supports. retrieval.py can assemble up
# to ~6000 tokens of context plus ~2000 tokens of chat history, so without
# setting this explicitly, Ollama would silently truncate the prompt before
# the model ever saw most of the retrieved context. 12288 leaves headroom for
# prompt overhead and a long generated answer, and fits well within every
# model in the README's recommended list.
_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "12288"))


def chat(system_prompt: str, messages: list[dict], temperature: float = 0.0) -> str:
    """One chat-completion call against a local Ollama server.

    `messages` is a list of {"role", "content"} dicts. Temperature defaults
    to 0.0: grounded, factual QA over retrieved context should be
    deterministic, not creative -- a high temperature only adds
    hallucination risk here.
    """
    full_messages = [{"role": "system", "content": system_prompt}, *messages]

    try:
        response = requests.post(
            f"{_OLLAMA_URL}/api/chat",
            json={
                "model": _CHAT_MODEL,
                "messages": full_messages,
                "stream": False,
                "options": {"temperature": temperature, "num_ctx": _NUM_CTX},
            },
            timeout=120,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException:
        # Covers "Ollama isn't running" (ConnectionError) as well as "Ollama is
        # running but the model isn't pulled" (a 404 HTTPError) and any other
        # transport failure -- all of these should surface as a chat message,
        # not crash the app with an unhandled exception.
        return (
            "Could not get a response from Ollama at "
            f"{_OLLAMA_URL}. Make sure Ollama is running (`ollama serve`) "
            f"and the model is pulled (`ollama pull {_CHAT_MODEL}`)."
        )

    return response.json()["message"]["content"]


def check_ollama_status() -> tuple[bool, str]:
    """Check whether Ollama is reachable and the configured model is pulled.

    Returns (is_ready, message) for the sidebar status indicator.
    """
    try:
        response = requests.get(f"{_OLLAMA_URL}/api/tags", timeout=3)
        response.raise_for_status()
    except requests.exceptions.RequestException:
        return False, f"Ollama not reachable at {_OLLAMA_URL}. Run `ollama serve`."

    models = [m["name"] for m in response.json().get("models", [])]
    if not any(m.split(":")[0] == _CHAT_MODEL.split(":")[0] for m in models):
        return False, f"Model '{_CHAT_MODEL}' not found. Run `ollama pull {_CHAT_MODEL}`."

    return True, f"Ollama ready ({_CHAT_MODEL})."
