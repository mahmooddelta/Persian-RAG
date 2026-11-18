#      Persian RAG Question-Answering System

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Architecture: Clean](https://img.shields.io/badge/Architecture-Clean%20%2F%20Hexagonal-brightgreen.svg)](#architecture--design)
[![Vector DB: Qdrant](https://img.shields.io/badge/Vector%20Store-Qdrant%20Local-orange.svg)](https://qdrant.tech/)
[![API: FastAPI](https://img.shields.io/badge/REST%20API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)

A production-grade **Persian Retrieval-Augmented Generation (RAG) Question-Answering System**


## Directory Structure

```text
persian-rag-qa/
├── data/
│   └── pdfs/                           # Sample Persian PDF documents
│       ├── 01_ai_basics.pdf
│       ├── 02_llm.pdf
│       └── 03.rag.pdf
├── scripts/
│   ├── generate_sample_pdfs.py         # Generates 3 realistic multi-page Persian PDFs
│   └── run_evaluation.py               # Runs 7 test queries and outputs evaluation_results.md
├── src/
│   ├── domain/
│   │   ├── entities.py                 # DocumentPage, DocumentChunk, SourceCitation, RetrievalResult, QAResponse
│   │   ├── interfaces.py               # Ports: DocumentLoaderPort, TextNormalizerPort, TextSplitterPort, EmbeddingPort, VectorStorePort, LLMPort
│   │   └── exceptions.py               # Custom domain exceptions hierarchy
│   ├── application/
│   │   ├── ingest_use_case.py          # Orchestrates PDF loading -> Persian normalization -> chunking -> embedding -> vector storage
│   │   └── query_use_case.py           # Orchestrates query normalization -> embedding -> retrieval (top_k >= 3) -> grounded LLM generation
│   ├── infrastructure/
│   │   ├── config.py                   # Pydantic v2 BaseSettings loading from .env
│   │   ├── pdf_loader.py               # PyMuPDF page-by-page extractor preserving filename & page_number
│   │   ├── persian_normalizer.py       # Normalizes Arabic chars (ي->ی, ك->ک, ة->ه), digits, and ZWNJ/whitespaces
│   │   ├── text_splitter.py            # Recursive sentence-aware Persian chunker with configurable chunk_size & overlap
│   │   ├── embedding_client.py         # OpenAI-compatible Embeddings client with exponential-backoff retry & batching
│   │   ├── vector_store.py             # Qdrant local persistent storage (path="./qdrant_storage")
│   │   └── llm_client.py               # OpenAI-compatible Chat Completions client with strict Persian RAG grounding prompt
│   └── presentation/
│       ├── cli.py                      # Typer/Rich interactive CLI (ingest, ask, chat commands)
│       └── api.py                      # FastAPI REST endpoints (POST /ingest, POST /ask, GET /health)
├── tests/
│   ├── unit/                           # Unit tests for normalizer, splitter, and use cases with mocked ports
│   └── integration/                    # End-to-end RAG pipeline & FastAPI integration tests
├── .env.example
├── .gitignore
├── pyproject.toml
├── evaluation_results.md               # Generated evaluation report (7 test queries, 100% Top-1 accuracy)
└── README.md
```

---

## Quickstart & Installation

### Prerequisites
- **Python 3.11+**
- `uv` or `pip`

### 1. Create Virtual Environment & Install Dependencies
```powershell
# Create virtual environment
uv venv .venv

# Install package and dev dependencies
uv pip install pymupdf qdrant-client httpx fastapi uvicorn typer rich pytest pytest-asyncio pydantic pydantic-settings python-dotenv openai tenacity
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and set your OpenAI-compatible API credentials (supports OpenAI, OpenRouter, TogetherAI, Ollama, vLLM, etc.):
```powershell
cp .env.example .env
```

Example `.env` for **OpenAI / OpenRouter** or **Local Ollama**:
```ini
LLM_API_KEY=sk-...
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini

EMBEDDING_API_KEY=sk-...
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536

QDRANT_STORAGE_PATH=./qdrant_storage
QDRANT_COLLECTION_NAME=persian_rag_documents
CHUNK_SIZE=600
CHUNK_OVERLAP=120
DEFAULT_TOP_K=4
```

##  Usage: CLI & FastAPI REST Endpoints

### A. Interactive CLI (`src/presentation/cli.py`)

```powershell
# 1. Ingest all Persian PDFs in ./data/pdfs into local Qdrant storage
.\.venv\Scripts\python.exe -m src.presentation.cli ingest --path ./data/pdfs

# 2. Ask a single question in Persian with top-k retrieval and citations
.\.venv\Scripts\python.exe -m src.presentation.cli ask "ظرفیت پردازش گرافیکی کشور باید به چند پتافلاپس برسد؟" --top-k 4

# 3. Start an interactive multi-turn terminal Q&A session
.\.venv\Scripts\python.exe -m src.presentation.cli chat
```

### B. FastAPI REST Server (`src/presentation/api.py`)

Start the server with Uvicorn:
```powershell
.\.venv\Scripts\uvicorn.exe src.presentation.api:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI / Swagger documentation is available at `http://localhost:8000/docs`.

#### Endpoint Examples:

1. **Health Check (`GET /health`)**
```bash
curl -X GET http://localhost:8000/health
```
2. **Ingest Documents (`POST /ingest`)**
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"path": "./data/pdfs"}'
```
3. **Ask Grounded Question (`POST /ask`)**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "یادگیری ماشین چیست و چه تفاوتی با یادگیری عمیق دارد؟",
    "top_k": 4
  }'
```

---
