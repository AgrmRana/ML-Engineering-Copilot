# ML Project Assistant

**A local, single-session RAG assistant for exploring machine learning projects and repositories.**

Upload a project, it gets chunked, embedded, and indexed in memory, and you can ask grounded questions about it, search it semantically, or run specialized workflows (summaries, documentation generation, validation review, interview questions). Think **NotebookLM for ML repos**, running entirely on your machine, for the length of one session.

---

## Quick Start (60 seconds)

```bash
# 1. Clone and install (first time only)
git clone https://github.com/AgrmRana/ML-Engineering-Copilot.git
cd RAG-LLM
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
ollama pull mistral             # Download local LLM (~4GB, one-time)

# 2. Run (any time)
ollama serve                    # Terminal 1: Start Ollama
streamlit run app/main.py       # Terminal 2: Start the UI
# Opens at http://localhost:8501
```

**No API keys. No cloud. No cost. Everything runs locally.**

---

## What It Does: Three Ways to Explore

### 1. **Chat Tab** — Grounded Question Answering

**Experience:** Ask a natural question about the project. The system retrieves the most relevant code/docs and answers based *only* on what it found, with citations.

**Example workflow:**
```
You: "What does the training loop do?"
↓
System: Retrieves functions/cells related to training
↓
LLM: "The training loop (defined in train.py, lines 45-67) iterates
     over batches, computes loss using categorical cross-entropy [1],
     updates weights via Adam optimizer [2], and logs metrics every
     100 steps [3]."
     
[Sources shown below: exact file, chunk index, relevance score]
```

**Key features:**
- One retrieval call + one LLM call (efficient, predictable)
- Chat history is kept (within token budget)
- Every claim is cited to the actual source
- If context doesn't cover the question, the system says so explicitly

### 2. **Search Tab** — Raw Semantic Search

**Experience:** Find chunks related to a topic without involving the LLM. See exactly what the retriever considers relevant.

**Example:**
```
Query: "data preprocessing validation split"
↓
Results:
  1. preprocessing.py (chunk 2/5) — score 0.85
     "train_test_split(X, y, test_size=0.2...)"
  2. config.yaml (chunk 1/3) — score 0.72
     "validation_split: 0.2"
  3. README.md (chunk 3/7) — score 0.68
     "Data is split 80/20 for training and validation"
```

**Why this matters:** See the retriever working *independently* from the LLM. Useful for understanding what context *would* be retrieved for a query before the model answers.

### 3. **Assistants Tab** — Specialized Workflows

**12 pre-built assistants** for common tasks, each using the same retrieval pipeline but with different prompts:

| Workflow | What It Generates |
|----------|-------------------|
| **Project Summary** | Architecture, components, tech stack, data pipeline |
| **Generate README** | Installation, usage, project structure, contributing |
| **Architecture Doc** | System overview, components, data flow, design patterns |
| **Model Card** | Model details, intended use, training data, performance, limitations |
| **Validation Report** | Assumptions, risks, data leakage, overfitting concerns |
| **Risk Analysis** | Bias/fairness/compliance/security risks with mitigations |
| **Explain SHAP** | SHAP analysis interpretation and feature importance |
| **Explain Feature Importance** | Feature ranking, methodology, recommendations |
| **Explain Metrics** | Evaluation metrics, trade-offs, business relevance |
| **Code Quality Review** | Code organization, naming, complexity, best practices |
| **Engineering Review** | Testing, CI/CD, version control, security practices |
| **Interview Questions** | 5–8 tailored technical questions (parameterized by role) |

Example: Click "Interview Questions", set role to "Senior ML Engineer", get a list of questions the interviewer could ask about the uploaded project.

---

## Visual Walkthrough: How the App Works

### Initial State — Ready for Upload

```
┌─────────────────────────────────────────────────────┐
│ ML Project Assistant                                │
│ Upload a machine learning project or repository...  │
├──────────────────────────────────────────────────────┤
│                                                      │
│  ✓ Ollama ready (mistral)          | Chat / Search │
│                                    | Assistants    │
│  📁 Upload files or a .zip                          │
│    ├── Drag and drop or Browse                      │
│    └── Limit 200MB per file                         │
│                                                      │
│  📊 Indexed chunks: 0                               │
│                                                      │
│  [Process files]     [Clear session]                │
└──────────────────────────────────────────────────────┘
```

**What you see:**
- **Sidebar (left):** Ollama status indicator (green = ready), file upload widget, indexed chunk counter
- **Main (right):** Tab selector (Chat/Search/Assistants), empty state waiting for files
- **Status indicator:** Shows Ollama is running + model is pulled (critical for functionality)

---

### After Upload — Files Indexed

Once you drag-and-drop a project `.zip` or select files:

```
┌─────────────────────────────────────────────────────┐
│ ML Project Assistant                                │
├──────────────────────────────────────────────────────┤
│                                                      │
│  ✓ Ollama ready (mistral)          | Chat / Search │
│                                    | Assistants    │
│  📁 Uploaded: credit_risk_dataset  │                │
│                                    │ Ask a question │
│  📊 Indexed chunks: 18             │ about the      │
│                                    │ uploaded...    │
│  Indexed files:                    │                │
│  • train.py — 8 chunks, 1.2K tokens                │
│  • eval.py — 6 chunks, 945 tokens                  │
│  • README.md — 4 chunks, 623 tokens                │
│                                                      │
│  [Process more files]                              │
└──────────────────────────────────────────────────────┘
```

**What changed:**
- **Chunk counter:** Updated to show how many semantic chunks were created
- **File breakdown:** Each file shows its chunk count and token count (helps you see what got indexed)
- **Ready for queries:** You can now ask questions or run searches

---

### Tab 1: Chat — Ask Grounded Questions

```
┌─────────────────────────────────────────────────────┐
│ ML Project Assistant                                │
├──────────────────────────────────────────────────────┤
│                                                      │
│ Chat / Search / Assistants                          │
│                                                      │
│ 👤 You: "How is the model evaluated?"              │
│                                                      │
│ 🤖 Assistant:                                       │
│    The model is evaluated using:                    │
│    - Accuracy [1] on the test set                  │
│    - F1 score [1] weighted by class                │
│    - ROC-AUC [2] for probability ranking           │
│                                                      │
│    ▼ Sources (click to expand)                     │
│      [1] eval.py (chunk 2/3, score 0.89)          │
│      [2] metrics.py (chunk 1/2, score 0.85)       │
│                                                      │
│ 💬 Ask a question about the project...             │
└─────────────────────────────────────────────────────┘
```

**Key features visible:**
- **Chat history:** Shows your question + assistant's answer
- **Citations:** Every claim is numbered and traceable [1], [2], etc.
- **Source panel:** Click to see which file/chunk gave that answer
- **Input box:** Ready for your next question
- **Confidence score:** Shows how confident the retriever was (0.85 = very confident)

---

### Tab 2: Search — Semantic Search (No LLM)

```
┌─────────────────────────────────────────────────────┐
│ ML Project Assistant                                │
├──────────────────────────────────────────────────────┤
│                                                      │
│ Chat / Search / Assistants                          │
│                                                      │
│ Raw semantic search over indexed chunks, ranked    │
│ by cosine similarity -- no LLM call.               │
│                                                      │
│ Search query: [preprocessing validation]           │
│ Results: [●●●●●●●●────────] 8                     │
│                                                      │
│ ┌─────────────────────────────────────┐            │
│ │ train.py (chunk 2/5) — score 0.87   │            │
│ │ train_test_split(X, y, test_size...)│            │
│ └─────────────────────────────────────┘            │
│                                                      │
│ ┌─────────────────────────────────────┐            │
│ │ config.yaml (chunk 1/2) — score 0.73│            │
│ │ validation_split: 0.2                │            │
│ └─────────────────────────────────────┘            │
│                                                      │
│ ┌─────────────────────────────────────┐            │
│ │ README.md (chunk 3/4) — score 0.69  │            │
│ │ Data is split 80/20 train/val...    │            │
│ └─────────────────────────────────────┘            │
└─────────────────────────────────────────────────────┘
```

**Key features visible:**
- **No LLM involvement:** Just retrieve + rank semantically, instant results
- **Relevance scores:** See exactly how similar each chunk is to your query (0.87 = very similar)
- **Previews:** First 100 characters of each chunk visible
- **Slider control:** Adjust number of results (1–20)

**Why this is powerful:** You see the *raw* retrieval independently from generation. Debug retrieval quality without waiting for LLM.

---

### Tab 3: Assistants — Specialized Workflows

```
┌─────────────────────────────────────────────────────┐
│ ML Project Assistant                                │
├──────────────────────────────────────────────────────┤
│                                                      │
│ Chat / Search / Assistants                          │
│                                                      │
│ Grounded assistants built on the same retrieval    │
│ pipeline as Chat.                                   │
│                                                      │
│ Choose an assistant:                               │
│ ┌───────────────────────────────────┐             │
│ │ Project Summary                   │ ▼            │
│ └───────────────────────────────────┘             │
│                                                    │
│ [Or choose: Generate README, Architecture Doc,   │
│  Model Card, Validation Report, Risk Analysis,   │
│  Code Quality Review, Interview Questions, ...]   │
│                                                    │
│ Target role (for Interview Questions):            │
│ [Data Scientist____________________]              │
│                                                    │
│                    [Run]                          │
│                                                    │
│ ⏳ Running Project Summary...                     │
│                                                    │
│ 🤖 Output:                                        │
│    PROJECT OVERVIEW                              │
│    This is a credit risk classification model    │
│    using logistic regression trained on...       │
│    [continues with full summary, sources below] │
│                                                    │
│    ▼ Sources (click to expand)                   │
│      [1] README.md, [2] train.py, [3] eval.py   │
└─────────────────────────────────────────────────────┘
```

**Key features visible:**
- **Workflow selector:** 12 pre-built assistants (dropdown list)
- **Parameterization:** Some workflows (like Interview Questions) take a role input
- **Real-time generation:** Shows output as it's generated
- **Cited sources:** All outputs are grounded in retrieved context

---

## The AI Engineering: How It Works

### 1. Document Processing & Parsing

**What happens when you upload:**

```
your_project.zip
├── train.py              → Parse as Python code
├── analysis.ipynb        → Parse as Jupyter (separate code/markdown)
├── README.md             → Parse as Markdown
├── data.csv              → Parse as table + stats
└── paper.pdf             → Extract text from each page
```

**Smart extraction per type:**
- **Python:** Uses `ast` module to split on function/class boundaries (AST-aware, not line-count heuristics)
- **Notebooks:** Keeps cell structure and type (code vs. markdown) with cell indices
- **Markdown:** Splits on headers first, then packs sections intelligently
- **CSV:** For small files, includes full table; for large files, includes schema + stats + sample rows (more useful per token)
- **PDF:** Text extracted per page

**Automatic filtering:**
- Skips binaries, archives, model weights (`.pt`, `.pth`, `.h5`, `.pkl`)
- Skips directories like `.git`, `__pycache__`, `node_modules`, `venv`
- Skips files >20MB (per-file cap to keep ingestion fast)
- Reports everything skipped with reasons (sidebar)

### 2. Chunking Strategy (Token-Aware, Not Character-Aware)

**The problem:** Most RAG demos chunk by character count ("1000 chars, 200 char overlap") — arbitrary, doesn't match how LLMs actually see text.

**This project:** Chunks by *tokens*, using the **embedding model's own tokenizer** (WordPiece from `all-MiniLM-L6-v2`, max 256 tokens). 

**Why this matters:**
- Chunks are sized at ~200 tokens (⅔ of model's limit, safe margin)
- Overlap of ~30 tokens (15%)
- Nothing silently truncated by the embedding model
- Token budgets are predictable across retrieval and generation

**Three chunking strategies:**

1. **Markdown files:** Split on headers first, pack adjacent sections
   ```
   # Architecture
   ...(section content)...
   ## Components
   ...(section content)...
   ```
   → Keeps sections intact, prevents truncation mid-concept

2. **Python files:** Split on AST boundaries (function/class defs)
   ```python
   class DataLoader:
       def __init__(self): ...      [chunk]
       def load(self): ...          [chunk]
   def train_model(): ...           [chunk]
   ```
   → Clean, syntactically valid chunks; methods labeled with class name

3. **Notebooks:** Per-cell chunking
   ```
   [Cell 0, Markdown] # Exploratory Data Analysis
   [Cell 1, Code] import pandas as pd; df = pd.read_csv(...)
   ```
   → Preserves notebook structure and cell types

### 3. Embeddings & Vector Store

**Embedding model:** `all-MiniLM-L6-v2` (22MB, 384 dimensions)
- Runs entirely locally on CPU
- Fast (~500 chunks/sec on modern CPU)
- Well-benchmarked for semantic similarity
- No network calls, no API costs

**Vector store:** FAISS `IndexFlatIP` (in-memory)
- Pure in-memory, nothing persisted to disk
- Exact cosine similarity (no approximation)
- Instant search (<100ms for thousands of chunks)
- Cleared when app closes (matches "local desktop app" model)

### 4. Retrieval Pipeline

**For each query, three steps:**

1. **Over-fetch:** Retrieve `4 × k` candidates by raw similarity
   - Default k=6, so fetch top 24

2. **Maximal Marginal Relevance (MMR):** Keep top k, but diversify
   - Prevents returning 5 near-identical chunks from the same section
   - Balances relevance with diversity: `score = 0.5 × similarity - 0.5 × redundancy`
   - Returns 6 diverse, relevant chunks instead of 5 redundant ones

3. **Context Assembly:** Pack under token budget
   - Add chunks in order until reaching 6000-token budget
   - Stop early if budget is full (don't overload the LLM)
   - Assemble numbered context block: `[1] file.py\n...text...\n\n[2] other.py\n...text...`

**Example:**
```
Query: "how is the model evaluated?"
↓
Over-fetch: 24 candidates by similarity
↓
MMR: Select 6 with best balance of relevance & diversity
↓
Pack: Add to context until 6000 tokens
↓
Result: [1] eval.py (chunk 2/4)
        [2] metrics.py (chunk 1/3)
        ... (up to 6000 tokens)
```

### 5. Prompting & Generation

**LLM:** Ollama + Mistral 7B (local, free)
- Configurable via `OLLAMA_MODEL` (also supports Llama, Orca-mini, Phi, etc.)
- Context window explicitly set to 12,288 tokens (vs. Ollama's small default)
- Temperature=0.0 (deterministic, factual answers)

**System prompt (same for all queries):**
```
You are an AI assistant exploring machine learning projects.
Answer ONLY from the numbered context sources below.
Cite source numbers for every factual claim.
If sources don't cover the question, say so explicitly.
Do not use outside knowledge.
```

**Why this design:**
- Keeps retrieval and generation separate and testable
- Makes hallucination-resistance explicit
- Every answer is grounded in actual project content

---

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

**Everything runs in one Python process. No client/server split, no REST API, no persistence layer.**

### Folder Structure

```
app/
    main.py            # Streamlit entry point
    ui.py              # Sidebar, Chat/Search/Assistants tabs
    document_loader.py # File parsing (PDF, CSV, Jupyter, etc.)
    chunking.py        # Token-aware chunking with AST/markdown/notebook strategies
    embeddings.py      # Local Sentence Transformers wrapper
    vector_store.py    # FAISS in-memory index
    retrieval.py       # Over-fetch, MMR, context assembly
    llm.py             # Ollama chat client with error handling
    prompts.py         # System prompts for all workflows
    rag.py             # Chat QA pipeline (retrieve → assemble → chat)
    workflows.py       # 12 assistant workflows (data-driven)
    session.py         # Streamlit session state (temporary)
    utils.py           # Token counting, file filtering

tests/
    test_chunking.py   # AST, markdown, notebook, token budget tests
    test_retrieval.py  # MMR and context assembly tests
    test_vector_store.py # FAISS add/search tests

requirements.txt      # Python dependencies
.env.example          # Ollama config (optional overrides)
```

**Design principle:** Each module does one thing. No service layer, no repository pattern, no DI framework. A Streamlit callback calls `document_loader` → `chunking` → `vector_store` → `retrieval` → `rag` in a straight line.

---

## Installation & Setup

### Prerequisites
- **Python:** 3.10+
- **Ollama:** Download from [ollama.ai](https://ollama.ai)
- **Disk space:** ~10GB (model + cache)
- **RAM:** 8GB minimum (16GB recommended for large projects)

### Step-by-Step

```bash
# 1. Clone the repository
git clone https://github.com/AgrmRana/ML-Engineering-Copilot.git
cd RAG-LLM

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# or
.venv\Scripts\activate          # Windows

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Pull a local LLM (one-time download)
ollama pull mistral
# Or: ollama pull orca-mini  (smaller, ~2GB)
# Or: ollama pull phi        (smallest, ~1.6GB)
```

**That's it.** The embedding model (`all-MiniLM-L6-v2`, ~22MB) downloads automatically on first use.

### Run

**Terminal 1: Start Ollama**
```bash
ollama serve
# Ollama will be available at http://localhost:11434
```

**Terminal 2: Start the app**
```bash
streamlit run app/main.py
# Opens at http://localhost:8501
```

### Configuration (Optional)

Create a `.env` file in the project root to override defaults:

```env
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=mistral              # or orca-mini, phi, llama2, neural-chat
OLLAMA_NUM_CTX=12288              # Context window size
```

---

## Usage Walkthrough

### 1. Upload a Project

**Sidebar:**
- Click **"Browse files"** or drag-and-drop
- Select individual files or a `.zip` of a repository
- Click **"Process files"**

**What happens:**
```
Sidebar shows:
✓ Ollama ready (mistral)
  
Processing...
  
✓ Indexed 3 file(s) into 18 chunks

Indexed files:
  train.py — 8 chunks, 1,247 tokens
  eval.py — 6 chunks, 945 tokens
  README.md — 4 chunks, 623 tokens

Skipped files:
  models/ (not a supported type)
  .gitignore (empty)
```

### 2. Chat Tab — Ask Questions

**Type:** "What evaluation metrics does the model use?"

**System:**
1. Retrieves ~6 relevant chunks (training/evaluation code, metrics definitions)
2. Assembles context
3. Sends to Ollama with system prompt
4. Returns answer with citations

**Response might be:**
```
The model is evaluated using:
- Accuracy [1] on the test split
- F1 score [1] weighted by class frequency  
- ROC-AUC [2] to assess ranking performance across thresholds

[Sources]
[1] eval.py (chunk 2/3): score = metrics.accuracy(y_true, y_pred)
[2] metrics.py (chunk 1/2): roc_auc_score(y_true, y_pred_proba)
```

### 3. Search Tab — Explore Semantically

**Query:** "model training optimization"

**Results:**
```
Semantic search (no LLM call):

[1] train.py (chunk 3/8) — score 0.89
    def train_model(model, train_loader, optimizer, device):

[2] config.yaml (chunk 1/2) — score 0.84
    optimizer: adam
    learning_rate: 0.001

[3] README.md (chunk 2/4) — score 0.76
    Training uses Adam optimizer with initial LR of 1e-3
```

### 4. Assistants Tab — Generate Summaries/Docs

**Choose:** "Interview Questions"
- Set role: "Data Scientist"
- Click "Run"

**Output:**
```
1. Describe the data preprocessing pipeline and how you handle missing values.
   Key points: [extracted from code]
   Follow-up: What validation techniques did you use to ensure the preprocessed data was correct?
   Difficulty: Medium

2. The model achieves X% accuracy on test data. How would you approach improving it?
   ...
```

---

## Key Design Decisions & Rationale

### Local-Only (No APIs)

**Decision:** Use Ollama + local embeddings, never call cloud APIs.

**Why:**
- No cost, no API keys, no rate limits
- Works offline (after model download)
- Data never leaves your machine
- Demonstrates a complete AI system independently
- Interview-friendly: works everywhere, no account setup

**Trade-off:** Smaller models are slower and lower quality than GPT-4/Claude. Acceptable for a demo.

---

### Token-Aware Chunking

**Decision:** Size chunks against the embedding model's tokenizer, not character count.

**Why:**
- Chunks respect the model's actual input limit (256 tokens)
- No silent truncation by the embedding model
- Predictable token budgets throughout the pipeline
- Fixes a common RAG mistake: "400-token chunks" by the wrong tokenizer

**How it works:**
- Count tokens with `all-MiniLM-L6-v2`'s WordPiece tokenizer (already a dependency)
- Target ~200 tokens per chunk (⅔ of 256-token limit)
- Overlap of ~30 tokens (15%)

---

### Maximal Marginal Relevance (MMR)

**Decision:** Use MMR to diversify retrieval results, not just raw similarity.

**Why:**
- Prevents returning 5 near-identical chunks from the same section
- Balances relevance with diversity
- Simple, no extra dependencies (pure numpy)

**Example:**
```
Query: "How is data split?"

Without MMR (top-5 by similarity):
  preprocessing.py, chunk 1 (score 0.92)
  preprocessing.py, chunk 2 (score 0.91) ← near duplicate
  preprocessing.py, chunk 3 (score 0.89) ← near duplicate
  preprocessing.py, chunk 4 (score 0.88) ← near duplicate
  config.yaml, chunk 1 (score 0.72)

With MMR (top-5 diverse):
  preprocessing.py, chunk 1 (score 0.92)
  config.yaml, chunk 1 (score 0.72)
  README.md, chunk 2 (score 0.68)
  validation.py, chunk 1 (score 0.65)
  metadata.json, chunk 1 (score 0.61)
```

Better coverage of different aspects of the same topic.

---

### Explicit Context Window

**Decision:** Set Ollama's `num_ctx` explicitly (12,288 tokens).

**Why:**
- Ollama defaults to a tiny context window (~2048) regardless of model capability
- Our RAG pipeline assembles up to 6000 tokens of context
- Without explicit `num_ctx`, Ollama silently truncates the prompt
- Easy to miss if you don't know Ollama's behavior

**Result:** Longer conversations, full RAG context delivered to the model.

---

### Separation of Retrieval & Generation

**Decision:** Retrieval happens independently; no minimum-similarity filter.

**Why:**
- Retrieval just returns its best matches
- Generation is responsible for "I don't know" responses
- Makes each step testable, debuggable, tunable independently
- Honest: the prompt explicitly tells the LLM to reject its own knowledge if context doesn't support it

---

## Limitations

- **Local model quality:** Mistral 7B is good but slower and lower quality than GPT-4
- **Single-session only:** No persistent history; everything disappears on close
- **English-oriented:** Chunking heuristics assume English/Markdown conventions
- **No cross-encoder reranking:** Could improve precision but adds latency
- **In-memory index:** Fine for one project; not built for million-document scale
- **No `.docx` support:** Kept dependencies minimal; `.md`, `.py`, `.ipynb` cover most project docs

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Ollama not reachable" | Run `ollama serve` in a terminal |
| "Model not found" | Run `ollama pull mistral` (or your configured model) |
| "First run is slow" | Model downloads (~4GB) + embedding model cache (~22MB) = one-time cost |
| "Large repo takes a while" | Local inference is slower than cloud APIs; 20MB file cap + ignore-list minimize this |
| "A file didn't get indexed" | Check sidebar's "Skipped files" expander for the reason |
| "`faiss` fails to install" | Try `conda install -c pytorch faiss-cpu` |

---

## Supported File Types

| Type | How It's Processed |
|------|-------------------|
| `.py` | AST-parsed into function/class chunks with class labels |
| `.ipynb` | Parsed per cell (code/markdown separated, cell index tagged) |
| `.md` | Split on headers, adjacent sections packed |
| `.txt`, `.yaml`, `Dockerfile`, etc. | Read as plain text, split by paragraphs |
| `.pdf` | Text extracted per page with `pypdf` |
| `.csv` | Small: full table; Large: shape + dtypes + stats + sample rows |
| `.json` | Pretty-printed |

---

## Example Queries

### Chat Tab

- "What does this project do, at a high level?"
- "Which file defines the training loop, and what optimizer does it use?"
- "What are the evaluation metrics, and where are they calculated?"
- "Are there any data leakage risks in the preprocessing code?"
- "Explain the architecture: how do the main modules interact?"
- "Summarize the README. What's the installation process?"

### Assistants Tab

- **Project Summary:** Get a comprehensive overview in 2 minutes
- **Interview Questions:** Generate questions for a take-home evaluation
- **Code Quality Review:** Identify code smells and anti-patterns
- **Validation Report:** Check for leakage, overfitting, and missing validations
- **README Generator:** Auto-draft installation/usage docs from code

---

## Testing

Run the test suite to verify chunking, retrieval, and vector store correctness:

```bash
pytest -v
```

Tests include:
- Token budget compliance for all chunking strategies
- AST-based Python chunking correctness
- Markdown header preservation
- Notebook cell type preservation
- MMR diversification behavior
- FAISS index add/search correctness

All 15 tests should pass.

---

## Design & Implementation Philosophy

This project prioritizes **clarity over complexity**:

- **No microservices:** Single Python process with plain function calls
- **No frameworks where not needed:** Direct FAISS, direct Ollama API, direct Streamlit
- **No premature abstraction:** Repository patterns, DI containers, etc. removed
- **Explicit over implicit:** All chunking/retrieval logic visible, testable, tunable
- **Grounded in real AI practices:** MMR, token-aware chunking, separation of concerns — not toy examples

The goal: demonstrate AI engineering fundamentals (embeddings, retrieval, prompting, LLM orchestration) as clearly and correctly as possible, using ~1000 lines of well-organized Python.

---

## Future Enhancements

- **Cross-encoder reranking:** Dedicated reranker model for higher-precision retrieval
- **Streaming responses:** Show LLM output in real-time instead of waiting for full generation
- **Hybrid search:** BM25 + embeddings for exact-match queries (function names, error codes)
- **Optional `.docx` support:** For design documents and proposals
- **Multi-session persistence:** Optional SQLite for chat history across sessions
- **Export capabilities:** Save conversations, summaries, and generated docs

---

## Requirements

- Python 3.10+
- 8GB RAM (16GB recommended)
- ~10GB disk (models + cache)
- macOS, Linux, or Windows

---

## License

Open source. See repo for details.

---

## Contact & Questions

This project is optimized for learning and interviews. Questions about the architecture, design decisions, or how to extend it? Refer to the modular code structure — each file is self-contained and well-commented.
