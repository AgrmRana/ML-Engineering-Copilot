# ML Project Assistant

A local RAG assistant for exploring machine learning projects and repositories. Upload a project, it gets chunked, embedded, and indexed in memory, and you can ask grounded questions about it, run semantic search over it, or kick off one of a dozen specialized workflows (summaries, docs, validation review, interview prep). Basically NotebookLM for ML repos, except it runs entirely on your own machine for the length of one session.

I built this partly because I got tired of dumping code into a chat model and getting confident-sounding answers about functions it had never actually seen, and partly because I wanted to build a full retrieval pipeline myself instead of gluing together a framework — chunking, embeddings, the vector index, MMR, prompting, all of it.

---

## Quick start

```bash
git clone https://github.com/AgrmRana/ML-Engineering-Copilot.git
cd RAG-LLM
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
ollama pull mistral             # local LLM, ~4GB, one-time download

ollama serve                    # terminal 1
streamlit run app/main.py       # terminal 2, opens at localhost:8501
```

No API keys, no cloud calls, no per-token cost. Everything — embeddings and generation — runs locally.

---

## What it does

There are three tabs.

**Chat** is grounded Q&A. Ask a question, the system retrieves whatever's relevant from the indexed project and answers using only that, with citations back to the exact file and chunk. If the retrieved context doesn't actually cover your question, it's supposed to say so rather than fall back on the model's general knowledge — that's baked into the system prompt, not left up to chance.

**Search** is raw semantic search with no LLM involved. Type a query, get back the chunks that scored highest by cosine similarity, with their scores and a preview. This is mostly useful for debugging — if an answer in Chat feels off, this is where you check whether retrieval actually found the right stuff before blaming the model.

**Assistants** is a set of twelve prepared workflows that reuse the same retrieval pipeline with a different prompt each time: project summary, README generation, architecture doc, model card, validation report, risk analysis, SHAP explanation, feature importance, metrics explanation, code quality review, engineering practices review, and interview question generation (this last one takes a target role and generates questions an interviewer might actually ask about the uploaded project).

---

## How it actually works

### Ingestion

When you upload files or a zip, everything gets extracted to a temp directory that's wiped as soon as parsing finishes — nothing from the upload sticks around on disk. Each file type gets parsed differently:

- **Python** — chunked on real `ast` boundaries (function/class defs), not line counts
- **Notebooks** — parsed per cell, keeping code/markdown separation and cell index
- **Markdown** — split on headers, adjacent sections packed together
- **CSV** — small files get dumped whole; large ones get a schema + stats + sample instead, since embedding 50,000 rows of raw numbers is a waste of tokens and doesn't actually help retrieval
- **PDF** — text pulled per page with pypdf

Binaries, model weights, `.git`, `node_modules`, and anything over 20MB get skipped, and the sidebar tells you exactly what got skipped and why so nothing disappears silently.

### Chunking

This is the part I spent the most time getting right. Most RAG tutorials chunk by character count, which is a bit of an odd choice — the embedding model doesn't see characters, it sees tokens, and it truncates at 256 of them. So chunk size here is measured against the actual tokenizer the embedding model uses (WordPiece, same one as `all-MiniLM-L6-v2`), targeting around 200 tokens per chunk with roughly 30 tokens of overlap.

Python files get split on `ast` boundaries — functions and classes — instead of guessing based on indentation. If a class is too big to fit in one chunk, it gets split per method, and each piece still gets labeled with the class name so you never end up with an orphaned method that's lost its context. Markdown keeps headers attached to their sections. Notebooks keep cell boundaries and cell type. Everything falls back to plain paragraph chunking if something goes wrong (e.g. a Python file with a syntax error).

### Embeddings and the index

Embeddings come from `all-MiniLM-L6-v2` — small (22MB), fast, runs fine on CPU, no network calls. Vectors get L2-normalized so a plain inner product doubles as cosine similarity, and everything sits in a FAISS `IndexFlatIP` — exact search, in memory, nothing written to disk. At the scale this tool operates at (a single project, thousands of chunks at most), an exact flat index is both simpler and fast enough that there's no real reason to reach for something approximate.

### Retrieval

For a given query it over-fetches — pulls back 4x the requested number of candidates by raw similarity — and then runs Maximal Marginal Relevance to pick the final set. Plain top-k tends to return a handful of near-identical chunks if a topic gets discussed in five places in a row; MMR trades off relevance against redundancy so you get a diverse set of genuinely different chunks instead of five copies of the same idea. After that, chunks get packed into a numbered context block up to a 6,000-token budget.

One thing I deliberately don't do: filter results by a minimum similarity score. An earlier version of this had a confidence threshold, and it was comparing against the wrong kind of distance (backwards, as it turned out) — even once fixed, it was still just an arbitrary cutoff with no principled basis. Now retrieval always returns its best-ranked results, and it's the prompt's job to say "the context doesn't cover this" when that's actually true. Felt like the more honest way to split the responsibility.

### Generation

Runs through Ollama, locally, with Mistral 7B by default (configurable — Llama, Phi, orca-mini, whatever you've pulled). Temperature is 0 across the board since this is meant to be grounded, factual QA, not creative writing. One thing that's easy to miss: Ollama defaults to a tiny context window regardless of what the model can actually handle, so it'll silently truncate a long prompt unless you set `num_ctx` explicitly — this is set to 12,288 tokens to comfortably fit the assembled context plus chat history plus the model's response.

The system prompt is the same shape across every feature: answer only from the numbered sources, cite the source number for every claim, say explicitly when the sources don't cover something, don't lean on outside knowledge about what a typical ML project "usually" looks like.

---

## Architecture

```
Streamlit UI (app/ui.py)
  Chat | Search | Assistants
         |
         v
document_loader -> chunking -> embeddings -> vector_store (FAISS)
                                                    |
                                                    v
                                               retrieval
                                          (MMR + context budget)
                                              /         \
                                          rag.py      workflows.py
                                         (chat QA)   (12 assistants)
                                              \         /
                                                llm.py
                                              (Ollama)
```

Single Python process, no client/server split, no database, no persistence layer. Each module does one job and calls the next in a straight line.

```
app/
    main.py             entry point
    ui.py               sidebar + tabs
    document_loader.py  file parsing
    chunking.py         token-aware chunking (AST/markdown/notebook aware)
    embeddings.py       local Sentence Transformers wrapper
    vector_store.py     FAISS index
    retrieval.py        over-fetch, MMR, context assembly
    llm.py              Ollama client
    prompts.py          all system prompts
    rag.py              chat pipeline
    workflows.py        the 12 assistants
    session.py          Streamlit session state
    utils.py            token counting, file filtering

tests/                  15 tests covering chunking, retrieval, vector store
```

---

## Setup

**You need:** Python 3.10+, [Ollama](https://ollama.ai), ~10GB free disk, 8GB RAM (16 recommended for bigger repos).

```bash
git clone https://github.com/AgrmRana/ML-Engineering-Copilot.git
cd RAG-LLM
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull mistral        # or orca-mini / phi for something smaller and faster
```

Run it:

```bash
ollama serve                     # terminal 1
streamlit run app/main.py        # terminal 2 -> localhost:8501
```

Optional `.env` overrides:

```env
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=mistral
OLLAMA_NUM_CTX=12288
```

---

## A few design choices worth explaining

**Why fully local, no APIs.** No cost, no keys, no rate limits, works offline once the models are pulled, and nothing about your codebase ever leaves your machine. The trade-off is real — a 7B local model is noticeably weaker than the leading hosted models at following instructions precisely — but for this use case (grounded retrieval doing most of the heavy lifting) it's an acceptable one, and it means the whole thing can be demoed anywhere with zero setup friction.

**Why token-aware chunking instead of character-based.** Because the embedding model truncates at 256 tokens, and characters-per-token varies a lot between prose and code — a "1000 character chunk" could be anywhere from 200 to 350 tokens depending on what's in it. Measuring against the actual tokenizer means nothing gets silently cut off.

**Why MMR instead of plain top-k.** Because raw similarity search over a codebase tends to return several near-duplicate chunks when a topic comes up repeatedly, which wastes context budget without adding information. MMR costs almost nothing extra (a handful of dot products) and noticeably improves the diversity of what actually reaches the model.

**Why no minimum confidence threshold.** Mentioned above, but worth repeating: an arbitrary similarity cutoff is a magic number dressed up as a design decision. Retrieval returns its best candidates regardless of score; refusing to answer when there's not enough information is something the prompt handles, not something baked into the retriever.

**Why explicit `num_ctx`.** Ollama's default context window is small and will truncate a long prompt with zero warning. Since this pipeline can assemble up to 6,000 tokens of context, leaving this unset would mean the model silently never sees most of what was retrieved.

---

## What's not here

- No persistent history — everything lives in memory for the session and disappears on close
- English/markdown-centric chunking heuristics
- No reranking step (a cross-encoder would help precision but adds latency)
- In-memory index only — fine for one project, not built for scale
- No `.docx` support — kept dependencies minimal since `.md`/`.py`/`.ipynb` cover most ML repos

## Ideas for later

- Cross-encoder reranking for better precision
- Streaming responses instead of a spinner
- Hybrid search (BM25 + embeddings) for exact-match queries like function names or error codes
- Optional persistence across sessions
- Export conversations and generated docs

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "Ollama not reachable" | `ollama serve` in a separate terminal |
| "Model not found" | `ollama pull mistral` (or whatever you've configured) |
| First run is slow | one-time model download (~4GB) + embedding model cache (~22MB) |
| Large repo takes a while | local inference is slower than a hosted API; the 20MB file cap and ignore list help keep this reasonable |
| A file didn't get indexed | check the "Skipped files" expander in the sidebar for the reason |
| `faiss` won't install | try `conda install -c pytorch faiss-cpu` |

## Supported file types

`.py`, `.ipynb`, `.md`/`.markdown`, `.txt`/`.yaml`/other plain text, `.pdf`, `.csv`, `.json` — pretty much anything else gets read as plain text if it can be decoded, or skipped with a reason if it can't.

---

## Testing

```bash
pytest -v
```

15 tests covering token budget compliance, AST-based Python chunking, markdown header preservation, notebook cell handling, MMR diversification, and the FAISS index. All should pass.

---

This was built to be readable end to end — no microservices, no unnecessary abstraction layers, nothing hidden behind a framework. If you're looking at this before an interview or a code review: every design decision above has a reason behind it, and the code itself is short enough (roughly 1,200 lines across the `app/` directory) to actually read in one sitting.
