"""All application state lives in Streamlit's session_state -- nothing is
persisted. A fresh session (or a page reload after "Clear session") starts
completely empty, matching a local desktop app that forgets everything on
close.
"""
import streamlit as st

from vector_store import VectorStore


def init_session() -> None:
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStore()
    if "documents" not in st.session_state:
        st.session_state.documents = []  # [{filename, chunk_count, token_count}]
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # [{role, content, sources?}]
    if "skipped_files" not in st.session_state:
        st.session_state.skipped_files = []


def clear_session() -> None:
    st.session_state.vector_store = VectorStore()
    st.session_state.documents = []
    st.session_state.chat_history = []
    st.session_state.skipped_files = []


def get_store() -> VectorStore:
    return st.session_state.vector_store
