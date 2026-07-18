# ML Workspace AI

An enterprise-grade AI platform for machine learning teams to understand, validate, document, and maintain ML projects.

## Business Problem

Machine learning teams produce large amounts of documentation: source code, notebooks, README files, design documents, validation reports, feature engineering documentation, SHAP outputs, monitoring reports, model cards, and evaluation reports. Reviewing all of this manually is time-consuming.

ML Workspace AI allows technical teams to quickly understand and review complete machine learning projects through intelligent document processing and retrieval-augmented generation.

## Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React)                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────┐ │
│  │ Dashboard│ │ Projects │ │  Search  │ │ Assistant │ │Reports│ │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────┘ │
└────────────────────────────┬────────────────────────────────┘
                             │ HTTP/REST API
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Backend (FastAPI)                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │   API    │ │ Services │ │ Workflows│ │  Schemas │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
└────────────────────────────┬────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   Database   │   │ Vector Store │   │    LLM       │
│  (PostgreSQL)│   │  (ChromaDB)  │   │  (OpenAI)    │
└──────────────┘   └──────────────┘   └──────────────┘
```

### Retrieval Pipeline

```
Document Upload
       │
       ▼
┌──────────────┐
│ Document     │
│ Processor    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Semantic     │
│ Chunker      │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Metadata     │
│ Extraction   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Embedding    │
│ Generation   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Vector Store │
│ (ChromaDB)   │
└──────────────┘
       │
       ▼
┌──────────────┐
│ Hierarchical │
│ Retrieval    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ LLM with     │
│ Context      │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Grounded     │
│ Response     │
└──────────────┘
```

### Folder Structure

```
ml-workspace-ai/
├── backend/
│   ├── api/                  # FastAPI endpoints
│   │   ├── main.py          # Main application with routes
│   │   └── schemas.py       # Pydantic models
│   ├── services/             # Business logic layer
│   │   ├── project_service.py
│   │   ├── document_service.py
│   │   ├── conversation_service.py
│   │   └── workflows/       # Specialized AI workflows
│   │       ├── project_summary.py
│   │       ├── documentation_assistant.py
│   │       ├── model_validation.py
│   │       ├── explainability.py
│   │       ├── repository_review.py
│   │       └── interview_assistant.py
│   ├── retrieval/            # RAG pipeline
│   │   ├── vector_store.py   # ChromaDB integration
│   │   └── retriever.py      # Hierarchical retrieval
│   ├── llm/                  # LLM orchestration
│   │   └── llm_client.py     # OpenAI integration
│   ├── database/             # Database layer
│   │   ├── models.py         # SQLAlchemy models
│   │   ├── connection.py     # Database connection
│   │   └── repositories.py   # Repository pattern
│   ├── processing/           # Document processing
│   │   ├── chunker.py       # Semantic chunking
│   │   └── document_processor.py
│   └── config/               # Configuration
│       └── settings.py       # Environment variables
├── frontend/                # React TypeScript application
│   ├── src/
│   │   ├── components/       # Reusable components
│   │   │   └── Layout.tsx
│   │   ├── pages/            # Page components
│   │   │   ├── Dashboard.tsx
│   │   │   ├── Projects.tsx
│   │   │   ├── Upload.tsx
│   │   │   ├── Documents.tsx
│   │   │   ├── Search.tsx
│   │   │   ├── Assistant.tsx
│   │   │   ├── Reports.tsx
│   │   │   └── Settings.tsx
│   │   ├── lib/              # Utilities
│   │   │   ├── api.ts        # API client
│   │   │   └── utils.ts      # Helper functions
│   │   ├── App.tsx           # Main app component
│   │   └── main.tsx          # Entry point
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   └── tailwind.config.js
├── tests/                   # Test suites
│   ├── test_api.py
│   ├── test_processing.py
│   ├── test_retrieval.py
│   └── conftest.py
├── docker-compose.yml        # PostgreSQL container
├── setup.sh                 # Unix setup script
├── setup.ps1                # Windows setup script
└── README.md
```

## Features

### Phase 1 (Core)
- **Document Ingestion**: Support for PDF, Markdown, TXT, DOCX, CSV, JSON, Python files, Jupyter notebooks
- **Semantic Search**: Production-quality RAG with semantic chunking and hierarchical retrieval
- **Grounded Answers**: Every response includes source citations, filenames, and confidence scores
- **Project Management**: Upload, index, and manage entire ML projects
- **Enterprise Dashboard**: Professional web interface with multiple views

### Phase 2 (Specialized Assistants)
- **Project Summary**: Comprehensive repository overviews
- **Documentation Assistant**: Generate READMEs, architecture docs, model cards
- **Model Validation Assistant**: Identify risks, data leakage, overfitting concerns
- **Explainability Assistant**: Explain SHAP, feature importance, evaluation metrics
- **Repository Review Assistant**: Code quality and engineering improvements
- **Interview Assistant**: Generate technical interview questions

## Installation

### Prerequisites
- Python 3.10+
- Node.js 18+
- PostgreSQL 14+
- OpenAI API key (or compatible LLM)

### Quick Start (Unix/Linux/macOS)

```bash
# Clone the repository
git clone <repository-url>
cd ml-workspace-ai

# Run setup script
chmod +x setup.sh
./setup.sh

# Start PostgreSQL (using Docker)
docker-compose up -d

# Configure environment
cd backend
cp .env.example .env
# Edit .env with your OpenAI API key and database URL

# Start backend
source venv/bin/activate
uvicorn api.main:app --reload

# In another terminal, start frontend
cd ../frontend
npm run dev
```

### Quick Start (Windows)

```powershell
# Clone the repository
git clone <repository-url>
cd ml-workspace-ai

# Run setup script
.\setup.ps1

# Start PostgreSQL (using Docker)
docker-compose up -d

# Configure environment
cd backend
Copy-Item .env.example .env
# Edit .env with your OpenAI API key and database URL

# Start backend
.\venv\Scripts\activate
uvicorn api.main:app --reload

# In another terminal, start frontend
cd ..\frontend
npm run dev
```

### Manual Installation

#### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your configuration
```

#### Frontend Setup
```bash
cd frontend
npm install
```

## Running Locally

1. **Start PostgreSQL**:
   ```bash
   docker-compose up -d
   ```
   Or use your local PostgreSQL instance.

2. **Configure environment**:
   Edit `backend/.env` with your:
   - OpenAI API key
   - Database URL
   - Other configuration options

3. **Start backend server**:
   ```bash
   cd backend
   source venv/bin/activate
   uvicorn api.main:app --reload
   ```
   Backend will be available at http://localhost:8000

4. **Start frontend**:
   ```bash
   cd frontend
   npm run dev
   ```
   Frontend will be available at http://localhost:3000

5. **Access the application**:
   Open http://localhost:3000 in your browser

## Testing

### Backend Tests
```bash
cd backend
pytest tests/ -v
```

### Frontend Tests
```bash
cd frontend
npm test
```

## Example Queries

- "Compare the architectures of the uploaded projects"
- "What feature engineering techniques are used in this repository?"
- "Generate interview questions for this ML project"
- "What documentation is missing from this project?"
- "Explain the evaluation methodology used"
- "What model validation issues exist in this codebase?"

## API Endpoints

### Projects
- `POST /api/projects` - Create a new project
- `GET /api/projects` - List all projects
- `GET /api/projects/{id}` - Get project details
- `DELETE /api/projects/{id}` - Delete a project

### Documents
- `POST /api/projects/{id}/documents` - Upload document to project
- `GET /api/projects/{id}/documents` - List project documents
- `DELETE /api/documents/{id}` - Delete document

### Conversations
- `POST /api/conversations` - Create conversation
- `GET /api/conversations` - List conversations
- `GET /api/conversations/{id}` - Get conversation with messages
- `POST /api/conversations/{id}/messages` - Send message
- `DELETE /api/conversations/{id}` - Delete conversation

### Search
- `POST /api/search` - Semantic search across documents

### Workflows
- `POST /api/projects/{id}/workflows/summary` - Generate project summary
- `POST /api/projects/{id}/workflows/documentation/readme` - Generate README
- `POST /api/projects/{id}/workflows/documentation/architecture` - Generate architecture docs
- `POST /api/projects/{id}/workflows/documentation/model-card` - Generate model card
- `POST /api/projects/{id}/workflows/validation/report` - Generate validation report
- `POST /api/projects/{id}/workflows/validation/risks` - Identify model risks
- `POST /api/projects/{id}/workflows/explainability/shap` - Explain SHAP analysis
- `POST /api/projects/{id}/workflows/explainability/feature-importance` - Explain feature importance
- `POST /api/projects/{id}/workflows/explainability/metrics` - Explain evaluation metrics
- `POST /api/projects/{id}/workflows/review/code` - Review code quality
- `POST /api/projects/{id}/workflows/review/engineering` - Review engineering practices
- `POST /api/projects/{id}/workflows/interview/questions` - Generate interview questions

## Design Decisions

### Architecture
- **Modular monolith**: Clear separation of concerns with independent modules
- **Service layer pattern**: Business logic isolated from API controllers
- **Repository pattern**: Database access abstracted through repositories
- **Workflow pattern**: Specialized AI workflows as reusable components

### Retrieval
- **Semantic chunking**: Content-aware splitting rather than fixed-length chunks
- **Hierarchical retrieval**: Document-level and chunk-level retrieval for better context
- **Metadata filtering**: Rich metadata extraction for precise filtering
- **Grounded responses**: Every answer includes source citations and confidence scores

### Frontend
- **TypeScript**: Type safety across the entire application
- **React Query**: Efficient data fetching and caching
- **Tailwind CSS**: Utility-first CSS for consistent styling
- **Lucide Icons**: Modern icon library
- **Component library**: Reusable UI components for consistency

### Database
- **PostgreSQL**: Reliable relational database for structured data
- **SQLAlchemy**: ORM for database operations
- **Repository pattern**: Clean data access layer
- **Migration support**: Database schema versioning with Alembic

### Vector Store
- **ChromaDB**: Open-source vector database for embeddings
- **Persistent storage**: Data persists across sessions
- **Metadata filtering**: Filter searches by project, document type, etc.

## Environment Variables

Key environment variables in `backend/.env`:

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/ml_workspace_ai

# OpenAI
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4-turbo-preview
OPENAI_EMBEDDING_MODEL=text-embedding-3-large

# Application
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:3000

# Vector Database
CHROMA_PERSIST_DIR=./data/chroma

# Document Processing
MAX_FILE_SIZE_MB=100
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Retrieval
TOP_K_RETRIEVAL=5
MIN_CONFIDENCE_SCORE=0.7
```

## Troubleshooting

### Database Connection Issues
- Ensure PostgreSQL is running: `docker-compose ps`
- Check database URL in `.env` file
- Verify database credentials

### OpenAI API Issues
- Verify API key is correct
- Check API key has sufficient credits
- Ensure model name is correct

### Frontend Build Issues
- Clear node_modules: `rm -rf node_modules && npm install`
- Clear cache: `npm run build -- --force`
- Check Node.js version: `node --version` (should be 18+)

### Vector Store Issues
- Check ChromaDB persist directory permissions
- Ensure sufficient disk space
- Clear vector store: `rm -rf backend/data/chroma`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Write tests for new functionality
5. Run tests: `pytest tests/ -v`
6. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- FastAPI for the excellent web framework
- LangChain for the LLM orchestration tools
- ChromaDB for the vector database
- OpenAI for the LLM and embedding models
- React and the React community for the frontend framework
