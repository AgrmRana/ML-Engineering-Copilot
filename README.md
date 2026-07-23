# ML Project Assistant

A local, single-session RAG assistant for exploring machine learning projects and repositories — upload a
project, it gets chunked, embedded, and indexed in memory, and you can ask grounded questions about it,
search it semantically, or run a handful of specialized "assistant" workflows (summaries, documentation
generation, validation review, interview questions) over it. Think **NotebookLM for ML repos**, running
entirely on your machine, for the length of one session.

## Motivation and business problem

ML projects accumulate a lot of scattered documentation: a README, some notebooks, training scripts, a
CSV or two, maybe a PDF design doc. Understanding an unfamiliar ML codebase — your own from six months
ago, a teammate's, or a candidate's take-home — usually means reading all of that by hand. This project
demonstrates how far a fairly small, carefully-built RAG pipeline can go toward automating that first
pass: retrieval-augmented question answering grounded in the actual project contents, plus a few
higher-level workflows (project summary, README generation, validation review, mock interview questions)
built on the same retrieval primitive.

This is deliberately **not** an attempt at a production document-management platform. There's no
database, no auth, no multi-user support, no persistent history. It's built to showcase the AI
engineering — chunking, embeddings, vector search, retrieval, prompting, LLM orchestration — as clearly
and correctly as possible, with just enough surrounding application to make that usable and demoable.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Streamlit UI (app/ui.py)                       │
│        Chat  |  Search  |  Assistants        (single process)       │
└───────────────────────────────┬───────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐      ┌─────────────────┐      ┌────────────────┐
│ document_loader│ -->  │   chunking      │ -->  │  embeddings    │
│ (extract text)  │      │ (token-aware,   │      │ (local Sentence │
│                 │      │  AST/cell-aware)│      │  Transformers)  │
└───────────────┘      └─────────────────┘      └────────┬───────┘
                                                            ▼
                                                   ┌─────────────────┐
                                                   │  vector_store   │
                                                   │ (FAISS, in-RAM) │
                                                   └────────┬────────┘
                                                            ▼
                                                   ┌─────────────────┐
                                                   │   retrieval     │
                                                   │ (MMR + token-   │
                                                   │  budget context)│
                                                   └────────┬────────┘
                                                            ▼
                                          ┌─────────────────┴─────────────────┐
                                          ▼                                   ▼
                                  ┌───────────────┐                  ┌────────────────┐
                                  │     rag.py     │                  │  workflows.py  │
                                  │ (chat Q&A)      │                  │ (12 assistants)│
                                  └───────┬───────┘                  └────────┬───────┘
                                          └──────────────┬───────────────────┘
                                                          ▼
                                                    ┌──────────┐
                                                    │  llm.py  │
                                                    │ (Ollama, │
                                                    │  local)  │
                                                    └──────────┘
```

Everything runs in one Python process. There is no REST API, no client/server split, and no persistence
layer — the whole pipeline is a sequence of plain function calls over data held in
`st.session_state`.

### Folder structure

```
app/
    main.py            # Streamlit entry point (`streamlit run app/main.py`)
    ui.py               # Sidebar (upload/session) + Chat/Search/Assistants tabs
    document_loader.py  # File-type detection, per-type text extraction, zip/folder walking
    chunking.py         # Token-aware chunking (text, markdown, AST-based python, notebook cells)
    embeddings.py       # Local Sentence Transformers embeddings, L2-normalized
    vector_store.py     # FAISS IndexFlatIP wrapper (in-memory cosine similarity)
    retrieval.py        # Over-fetch + MMR diversification + token-budget context assembly
    llm.py               # Thin wrapper around a local Ollama chat model
    prompts.py            # RAG system prompt + one prompt per assistant workflow
    rag.py                 # Grounded chat: retrieve -> build context -> one LLM call
    workflows.py            # Data-driven assistant workflows (12 of them, one runner)
    session.py              # st.session_state-backed session (chunks, index, chat history)
    utils.py                 # Token counting, file-size formatting, ignore-pattern constants
data/                        # Empty except for .gitkeep -- transient zip-extraction scratch space only
tests/
    test_chunking.py         # AST python chunking, markdown packing, notebook cells, token budgets
    test_vector_store.py     # FAISS add/search correctness (synthetic vectors, no network)
    test_retrieval.py        # MMR diversification behavior (synthetic vectors, no network)
.env.example
requirements.txt
pytest.ini
```

Each module does one thing. There's no service layer, no repository pattern, no dependency-injection
framework — a Streamlit callback calls `document_loader`, then `chunking`, then `vector_store`, then
`retrieval`, then `rag`/`workflows`, in a straight line.

## The RAG pipeline

### 1. Document processing

`document_loader.py` accepts individual files **or a `.zip` of a whole repository**. A zip is extracted
into a temporary directory, walked recursively, and deleted as soon as ingestion finishes — nothing from
an upload survives past that function call. The walker skips directories that are never useful for RAG
(`.git`, `__pycache__`, `node_modules`, `venv`/`.venv`, `.ipynb_checkpoints`, `dist`/`build`, etc.) and
binary/non-informative extensions (images, model weights, archives), and enforces a 20MB per-file cap so
one huge asset can't blow up ingestion time or embedding cost. Skipped files are reported in the sidebar,
not silently dropped.

**Supported document types:**

| Type | Extraction |
|---|---|
| Markdown (`.md`) | Read as text, later chunked header-aware |
| Python (`.py`) | Read as text, later chunked via `ast` |
| Jupyter notebooks (`.ipynb`) | Parsed with `nbformat`; code and markdown cells kept separate, tagged with cell index/type |
| PDF (`.pdf`) | Text extracted per page with `pypdf` |
| CSV (`.csv`) | Small files: full table. Large files: shape, dtypes, `describe()`, and a head/tail sample instead of a full dump — a raw dump of a 50,000-row CSV is expensive to embed and not particularly useful to a semantic search anyway |
| JSON (`.json`) | Pretty-printed |
| Everything else (`.txt`, `.yaml`, `Dockerfile`, `LICENSE`, etc.) | Read as plain text |

`.docx` is intentionally **not** supported — it wasn't in the target document set for this project, and
dropping it keeps the dependency list smaller (see [Limitations](#limitations) for how to add it back).

### 2. Chunking strategy — the part most RAG demos get wrong

Chunk size and overlap are measured in **tokens**, against the *actual* embedding model's tokenizer
(`all-MiniLM-L6-v2`'s WordPiece tokenizer, loaded via `transformers.AutoTokenizer` — already a dependency
of `sentence-transformers`, so this adds nothing new), not raw character counts and not a stand-in
tokenizer from a different model family. `all-MiniLM-L6-v2` truncates its input at **256 tokens**
(`model.max_seq_length`); an earlier version of this project counted tokens with OpenAI's `tiktoken`
(cl100k_base) — a leftover from when embeddings were generated via the OpenAI API — while chunking to
~400 of those tokens per chunk. Because tiktoken and the MiniLM tokenizer segment text differently, a
"400-token" chunk by that count was often 380-420 *real* WordPiece tokens once it reached the embedding
model, well past the 256-token limit — the model silently truncated the excess, so the embedding
represented only part of the chunk while the full (untruncated) text still went into the LLM's context.
This version chunks at **~200 tokens per chunk with ~30 tokens (~15%) of overlap**, counted with the same
tokenizer that will embed the text, so nothing is silently dropped.

Three chunking strategies, chosen per file type:

- **Markdown**: split on headers first, so a chunk never straddles two unrelated sections. Small adjacent
  sections are then packed together up to the token budget (a header-heavy README with many short
  sections would otherwise produce dozens of near-empty chunks — packing avoids that waste). Oversized
  sections are split further by paragraph.
- **Python**: split using the `ast` module on top-level function/class boundaries, instead of comparing
  leading-whitespace counts line by line (the original approach, which breaks on decorators, multi-line
  strings, and nested `def`s). If a class is too large to fit in one chunk, it's split **per method**, and
  every resulting chunk is labeled with its enclosing class name (`# class Foo, function bar`) so a
  chunk never loses the context of which class it came from. Falls back to plain text chunking on a
  `SyntaxError` (e.g. a code fragment or non-Python-3 file).
- **Notebooks**: chunked **per cell**, preserving the code/markdown boundary and cell index, with tiny
  adjacent cells merged and oversized cells split — all through the same token-budget packer used
  everywhere else.

Every chunk carries `filename`, `file_type`, `chunk_index`, and `total_chunks_in_file` — this is what
later powers citations that point at a specific chunk instance, not just a bare filename.

### 3. Embeddings

`embeddings.py` uses a local **Sentence Transformers** model (`all-MiniLM-L6-v2`, ~22MB, 384 dimensions)
via the `sentence-transformers` library — everything runs on your machine, with no API key, no network
call, and no per-token cost. The model is downloaded once from HuggingFace on first use and cached
locally afterward. `all-MiniLM-L6-v2` is a standard, well-benchmarked choice for semantic similarity: fast
on CPU, small on disk, and good enough for a single-session assistant's retrieval needs — this is not a
system trying to optimize recall at billion-document scale. Every embedding is L2-normalized on the way
out, which is what makes a plain inner product equivalent to cosine similarity downstream.

### 4. Vector store

`vector_store.py` wraps a FAISS `IndexFlatIP` (inner product) over normalized vectors, entirely **in
memory** — nothing is written to disk, no SQLite file, no persist directory. This directly replaces the
previous ChromaDB setup, which persisted a collection to a local SQLite file across sessions (in tension
with the "everything disappears on close" model of a local desktop app) and — more importantly — had a
scoring bug: Chroma's `similarity_search_with_score` returns a **distance** (lower = better) by default,
but the retrieval code filtered with `score >= min_confidence_score`, silently treating a distance as if
it were a similarity. Using FAISS `IndexFlatIP` over normalized vectors sidesteps the ambiguity entirely:
there is exactly one score, and higher always means "more similar."

### 5. Retrieval

`retrieval.py` never calls the LLM — retrieval and generation are kept as separate, independently
testable concerns. For a query:

1. Over-fetch `4 * k` candidates by raw cosine similarity.
2. Apply **Maximal Marginal Relevance (MMR)** (`lambda = 0.5`, implemented directly in `numpy` — no extra
   dependency) to select `k` chunks that are relevant *and* mutually diverse. Without this step, a query
   that matches one section well tends to retrieve several near-duplicate chunks from that same section,
   crowding out other relevant context. `tests/test_retrieval.py` verifies this concretely: given a
   highly relevant chunk, a near-duplicate of it, and a moderately relevant-but-diverse chunk, plain
   top-k similarity picks the near-duplicate, MMR picks the diverse one instead.
3. Assemble a numbered context block **under a fixed token budget** (6000 tokens), stopping once the
   budget is spent rather than concatenating however many chunks were requested. The original project's
   "assistant" workflows retrieved up to 15 chunks and concatenated all of them unconditionally — this
   bounds both cost and the risk of diluting the model's attention with more text than it needs.

There is deliberately **no fixed minimum-similarity cutoff**. The earlier version filtered out anything
below `min_confidence_score = 0.7` — on top of being compared in the wrong direction (see above), a fixed
threshold on an unnormalized, ambiguous score is not a principled way to decide "there's no relevant
answer here." Instead, retrieval always returns its best-ranked, diversified candidates, and the prompt
(see below) is explicitly responsible for saying "the context doesn't contain this" when what was
retrieved isn't actually relevant. That's a property of generation, not retrieval, and it's a more honest
way to handle it than an arbitrary number.

### 6. Prompting and generation

`llm.py` is a single, direct wrapper around a local **Ollama** server's chat API (`POST
/api/chat`) — there is no LangChain, no hosted API, and no cost. Ollama runs an open-source model (this
project defaults to **Mistral 7B**, configurable via `OLLAMA_MODEL`) entirely on your machine. A framework
buys very little here and hides exactly what's being sent to the model, which matters when the whole
point is to demonstrate the mechanics clearly.

Every request explicitly sets **`num_ctx=12288`** (configurable via `OLLAMA_NUM_CTX`). Ollama defaults new
sessions to a much smaller context window (historically 2048 tokens) regardless of what the underlying
model actually supports, and this pipeline can assemble up to 6000 tokens of retrieved context plus 2000
tokens of chat history — without raising `num_ctx`, Ollama would silently truncate the prompt itself
before the model ever saw most of what was retrieved, independent of anything `retrieval.py` does
correctly. 12288 leaves headroom for prompt overhead and a long generated answer (the assistant workflows
in particular can produce lengthy structured documents) and comfortably fits every model in the
recommended list above. `chat()` also treats *any* request failure — Ollama not running, the configured
model not pulled (a 404), a timeout — as a recoverable condition and returns a plain-language message
instead of raising, so a misconfigured local model degrades to a clear in-chat error rather than crashing
the whole Streamlit app.

Every prompt in `prompts.py` shares the same grounding rules: answer only from the numbered context
sources, cite the source number for every claim (`[2]`), and explicitly say so when the context doesn't
cover the question rather than falling back on the model's own knowledge. Generation runs at
**`temperature=0.0`** — grounded, factual QA over retrieved context should be deterministic, not
creative; the original project used `0.7` for this, which only adds hallucination risk with no
corresponding benefit. Chat history passed to the model is itself token-budget-trimmed
(`rag.py::_trim_history`), so a long conversation doesn't silently grow the prompt without bound.

Each turn/workflow run makes **exactly one retrieval call and one LLM call** — no speculative or chained
extra requests.

## AI workflows

Built on the same retrieval pipeline as Chat, available from the Assistants tab:

- **Project Summary** — architecture, components, technologies, data pipeline, model, evaluation
- **Generate README**, **Architecture Documentation**, **Model Card** — documentation generation
- **Model Validation Report**, **Risk Analysis** — assumptions, data leakage, overfitting, bias/fairness/compliance risks
- **Explain SHAP Analysis**, **Explain Feature Importance**, **Explain Evaluation Metrics** — explainability
- **Code Quality Review**, **Engineering Practices Review** — repository review
- **Interview Questions** (role-parameterized) — mock technical interview questions based on the uploaded project

All twelve are defined as data (a `Workflow(label, query, prompt_template, k)` entry in `workflows.py`)
rather than as separate classes — the earlier version had six ~100-line classes that were all "retrieve →
build context → one LLM call" with a different prompt, which is now one dataclass table and one
`run_workflow()` function.

## Installation and setup

**Prerequisites:** Python 3.10+, [Ollama](https://ollama.ai) installed locally. No API key, no account,
no cost.

```bash
git clone <repository-url>
cd RAG-LLM
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env           # optional -- only needed to override defaults

# Pull a local chat model (one-time, ~4GB download)
ollama pull mistral
```

`sentence-transformers` downloads its embedding model (~22MB) automatically on first use; no separate
step is needed for that one.

## Running it

```bash
ollama serve                   # in one terminal, if not already running
streamlit run app/main.py      # in another terminal
```

This opens the app at `http://localhost:8501`. The sidebar shows whether Ollama is reachable and the
configured model is pulled.

### Uploading a project

In the sidebar, either:
- select individual files (`.py`, `.ipynb`, `.md`, `.csv`, `.pdf`, `.json`, `.txt`, ...), or
- select a **`.zip`** of an entire repository,

then click **Process files**. The sidebar reports how many files were indexed into how many chunks and
tokens, plus a list of anything that was skipped and why (unsupported binary, too large, empty, failed to
parse). Click **Clear session** at any point to wipe everything and start over — this is the equivalent
of closing and reopening the app.

### Example queries (Chat tab)

- "What does this project do, at a high level?"
- "Which file defines the training loop, and what algorithm does it use?"
- "What evaluation metrics are reported, and where?"
- "Are there any obvious data leakage risks in the preprocessing code?"
- "Summarize what `retrieval.py` does and cite the relevant chunk."

### Search tab

Raw semantic search over the indexed chunks, ranked by cosine similarity, with **no LLM call** — useful
for seeing exactly what the retriever considers relevant to a query, independent of how the model chooses
to answer.

## Limitations

- **Local model quality/speed trade-off.** Mistral 7B (via Ollama) and `all-MiniLM-L6-v2` (via Sentence
  Transformers) are free and run entirely offline, but are smaller and slower than hosted models like
  GPT-4 — expect noticeably slower responses (especially without a GPU) and occasionally less polished
  answers. Swap `OLLAMA_MODEL` for a larger local model (e.g. `llama3`) for better quality at the cost of
  more RAM/disk and slower inference.
- **Session-only, single-user.** There is no persistence, no accounts, and no way to resume a previous
  session — this mirrors a local desktop app, not a deployable multi-user service.
- **English-oriented chunking heuristics.** The markdown header regex and sentence/paragraph splitting
  assume roughly English/Markdown conventions.
- **No `.docx` support.** Dropped to keep dependencies minimal; would need `python-docx` and a paragraph
  extractor similar to the PDF/CSV ones in `document_loader.py`.
- **No cross-encoder reranking.** MMR improves diversity using the same embedding vectors already
  computed; a dedicated reranker (e.g. a cross-encoder model) could improve precision further at the cost
  of another dependency and extra latency.
- **In-memory FAISS index.** Fine for a single project/session; it is not built to scale to millions of
  chunks or to be shared across processes.
- **No conversation persistence across app restarts.** This is intentional (see the memory model above),
  not an oversight.

## Troubleshooting

- **"Ollama not reachable"** — make sure Ollama is installed and running: `ollama serve` in a terminal
  (it may already be running as a background service after installation on macOS/Windows).
- **"Model not found"** — run `ollama pull mistral` (or whichever model you set via `OLLAMA_MODEL`).
- **`faiss` fails to install** — `faiss-cpu` ships prebuilt wheels for common platforms (Linux, macOS,
  Windows on standard Python versions); if your platform lacks a wheel, installing via `conda install -c
  pytorch faiss-cpu` is the usual fallback.
- **First run is slow** — the embedding model downloads (~22MB) on first use, and Ollama loads the chat
  model into memory on its first request; both are one-time costs per machine/session.
- **Large repository takes a while to process** — local inference is slower than a hosted API,
  especially without a GPU. The 20MB per-file cap and directory ignore-list (`venv/`, `node_modules/`,
  etc.) exist specifically to keep this bounded.
- **A file didn't get indexed** — check the "Skipped files" expander in the sidebar; it lists every
  skipped file with a reason (unsupported type, too large, empty, or a parse failure).

## Future improvements

- Optional `.docx` support for design docs.
- A cross-encoder reranking pass on top of MMR for higher-precision retrieval on larger corpora.
- Streaming LLM responses in the Chat tab for lower perceived latency.
- Hybrid search (BM25 + embeddings) for queries that hinge on exact identifiers (function names, error
  codes) that embeddings alone can under-rank.
