"""All Streamlit UI: sidebar upload/session controls, Chat/Search/Assistants tabs."""
import streamlit as st

import chunking
import document_loader
import session
import workflows
from llm import check_ollama_status
from rag import answer_question
from retrieval import retrieve
from utils import count_tokens


def render() -> None:
    st.set_page_config(page_title="ML Project Assistant", layout="wide")
    session.init_session()

    st.title("ML Project Assistant")
    st.caption(
        "Upload a machine learning project or repository, index it, and ask questions grounded in its "
        "actual contents. Everything lives in memory for this session only."
    )

    _render_sidebar()

    chat_tab, search_tab, assistants_tab = st.tabs(["Chat", "Search", "Assistants"])
    with chat_tab:
        _render_chat_tab()
    with search_tab:
        _render_search_tab()
    with assistants_tab:
        _render_assistants_tab()


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("Project")

        ollama_ready, ollama_message = check_ollama_status()
        if not ollama_ready:
            st.error(ollama_message)
        else:
            st.success(ollama_message)

        uploaded = st.file_uploader(
            "Upload files or a .zip of a repository",
            accept_multiple_files=True,
        )
        if st.button("Process files", disabled=not uploaded, use_container_width=True):
            _process_uploads(uploaded)

        st.divider()
        store = session.get_store()
        st.metric("Indexed chunks", len(store))
        if st.session_state.documents:
            st.caption("Indexed files")
            for doc in st.session_state.documents:
                st.text(f"{doc['filename']} — {doc['chunk_count']} chunks, {doc['token_count']} tokens")
        if st.session_state.skipped_files:
            with st.expander(f"Skipped {len(st.session_state.skipped_files)} file(s)"):
                for reason in st.session_state.skipped_files:
                    st.text(reason)

        st.divider()
        if st.button("Clear session", use_container_width=True):
            session.clear_session()
            st.rerun()


def _process_uploads(uploaded_files: list) -> None:
    with st.spinner("Reading files..."):
        docs, skipped = document_loader.ingest_uploads(uploaded_files)

    if not docs:
        st.warning("No supported files found to index.")
        st.session_state.skipped_files.extend(skipped)
        return

    with st.spinner(f"Chunking and embedding {len(docs)} file(s)..."):
        all_chunks = []
        new_doc_summaries = []
        for doc in docs:
            doc_chunks = chunking.chunk_document(doc)
            if not doc_chunks:
                continue
            all_chunks.extend(doc_chunks)
            new_doc_summaries.append({
                "filename": doc.rel_path,
                "chunk_count": len(doc_chunks),
                "token_count": sum(count_tokens(c.text) for c in doc_chunks),
            })
        session.get_store().add(all_chunks)

    st.session_state.documents.extend(new_doc_summaries)
    st.session_state.skipped_files.extend(skipped)
    st.success(f"Indexed {len(docs)} file(s) into {len(all_chunks)} chunks.")


def _render_chat_tab() -> None:
    store = session.get_store()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    _render_sources(msg["sources"])

    query = st.chat_input("Ask a question about the uploaded project...")
    if not query:
        return

    if len(store) == 0:
        st.warning("Upload and process a project first.")
        return

    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    history_for_model = [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history[:-1]
    ]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = answer_question(query, store, history=history_for_model)
        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("Sources"):
                _render_sources(result["sources"])

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })


def _render_search_tab() -> None:
    store = session.get_store()
    st.caption("Raw semantic search over indexed chunks, ranked by cosine similarity -- no LLM call.")

    query = st.text_input("Search query")
    top_k = st.slider("Results", min_value=1, max_value=20, value=8)

    if not query:
        return
    if len(store) == 0:
        st.warning("Upload and process a project first.")
        return

    results = retrieve(query, store, k=top_k)
    for chunk, score in results:
        with st.container(border=True):
            st.markdown(
                f"**{chunk.filename}** (chunk {chunk.chunk_index + 1}/{chunk.total_chunks}) — "
                f"score {score:.3f}"
            )
            st.code(chunk.text[:1000], language=None)


def _render_assistants_tab() -> None:
    store = session.get_store()
    st.caption("Grounded assistants built on the same retrieval pipeline as Chat.")

    options = {wf.label: key for key, wf in workflows.WORKFLOWS.items()}
    label = st.selectbox("Choose an assistant", list(options.keys()))
    key = options[label]

    role = None
    if key == "interview_questions":
        role = st.text_input("Target role", value="Data Scientist")

    if st.button("Run", use_container_width=True):
        if len(store) == 0:
            st.warning("Upload and process a project first.")
            return
        with st.spinner(f"Running {label}..."):
            result = workflows.run_workflow(key, store, role=role)
        st.markdown(result["text"])
        if result["sources"]:
            with st.expander("Sources"):
                _render_sources(result["sources"])


def _render_sources(sources: list[dict]) -> None:
    for s in sources:
        st.markdown(
            f"**[{s['n']}] {s['filename']}** (chunk {s['chunk_index'] + 1}/{s['total_chunks']}, "
            f"score {s['score']:.3f})"
        )
        st.caption(s["snippet"])
