"""
FastAPI REST API Presentation Layer for the Persian RAG System.

Exposes:
- `GET /health`: Service readiness and vector database statistics.
- `POST /ingest`: Ingest a specific PDF file or directory of PDFs.
- `POST /ask`: Ask a Persian question and receive a grounded answer + citations.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.application.ingest_use_case import IngestDocumentsUseCase
from src.application.query_use_case import AnswerQueryUseCase
from src.domain.exceptions import (
    DocumentLoadError,
    DomainError,
    EmptyQueryError,
    EmbeddingGenerationError,
    LLMGenerationError,
    VectorStoreError,
)
from src.infrastructure.config import Settings, get_settings
from src.infrastructure.embedding_client import OpenAICompatibleEmbeddingClient
from src.infrastructure.llm_client import OpenAICompatibleLLMClient
from src.infrastructure.pdf_loader import PyMuPDFLoader
from src.infrastructure.persian_normalizer import PersianTextNormalizer
from src.infrastructure.text_splitter import RecursivePersianTextSplitter
from src.infrastructure.vector_store import QdrantVectorStore


# ==============================================================================
# Pydantic v2 Request / Response Schemas
# ==============================================================================


class HealthResponseSchema(BaseModel):
    """Schema returned by `GET /health`."""

    status: str = Field(default="healthy", description="Service health status.")
    collection_name: str = Field(description="Active Qdrant collection name.")
    indexed_chunks_count: int = Field(
        description="Total number of Persian chunks currently indexed."
    )
    llm_model: str = Field(description="Configured LLM model name.")
    embedding_model: str = Field(description="Configured Embedding model name.")


class IngestRequestSchema(BaseModel):
    """Schema accepted by `POST /ingest`."""

    path: str | None = Field(
        default=None,
        description="Optional path to a PDF file or directory. Defaults to configured PDF_DATA_DIR.",
    )


class IngestResponseSchema(BaseModel):
    """Schema returned by `POST /ingest`."""

    documents_processed: int
    pages_extracted: int
    chunks_created: int
    vectors_stored: int
    source_files: list[str]
    latency_ms: float


class AskRequestSchema(BaseModel):
    """Schema accepted by `POST /ask`."""

    query: str = Field(
        ...,
        min_length=1,
        description="Persian question to ask the RAG system.",
    )
    top_k: int = Field(
        default=4,
        ge=3,
        le=20,
        description="Number of relevant chunks to retrieve (minimum 3).",
    )
    score_threshold: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Optional minimum cosine similarity score threshold.",
    )
    source_filter: str | None = Field(
        default=None,
        description="Optional source PDF filename filter (e.g., 'ai_national_strategy.pdf').",
    )


class CitationSchema(BaseModel):
    """Schema representing a single source citation."""

    source_filename: str
    page_number: int
    chunk_id: str
    score: float
    snippet: str


class AskResponseSchema(BaseModel):
    """Schema returned by `POST /ask`."""

    query: str
    normalized_query: str
    answer: str
    citations: list[CitationSchema]
    latency_ms: float
    created_at: str


# ==============================================================================
# Dependency Injection Container Wiring
# ==============================================================================


class ServiceContainer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.document_loader = PyMuPDFLoader()
        self.text_normalizer = PersianTextNormalizer()
        self.text_splitter = RecursivePersianTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self.embedding_client = OpenAICompatibleEmbeddingClient(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
            model=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
            max_retries=settings.embedding_max_retries,
            timeout_seconds=settings.embedding_timeout_seconds,
        )
        self.vector_store = QdrantVectorStore(
            storage_path=settings.qdrant_storage_path,
            collection_name=settings.qdrant_collection_name,
            embedding_dimension=settings.embedding_dimension,
        )
        self.llm_client = OpenAICompatibleLLMClient(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout_seconds=settings.llm_timeout_seconds,
        )

    @property
    def ingest_use_case(self) -> IngestDocumentsUseCase:
        return IngestDocumentsUseCase(
            document_loader=self.document_loader,
            text_normalizer=self.text_normalizer,
            text_splitter=self.text_splitter,
            embedding_client=self.embedding_client,
            vector_store=self.vector_store,
        )

    @property
    def query_use_case(self) -> AnswerQueryUseCase:
        return AnswerQueryUseCase(
            text_normalizer=self.text_normalizer,
            embedding_client=self.embedding_client,
            vector_store=self.vector_store,
            llm_client=self.llm_client,
            default_top_k=self.settings.default_top_k,
            score_threshold=self.settings.min_score_threshold,
        )


@lru_cache(maxsize=1)
def get_container() -> ServiceContainer:
    return ServiceContainer(get_settings())


def create_app() -> FastAPI:
    app = FastAPI(
        title="Persian RAG Question-Answering API",
        description=(
            "Production-grade Persian Retrieval-Augmented Generation (RAG) REST API "
            "built with Clean Architecture, PyMuPDF, Qdrant, and OpenAI-compatible LLMs."
        ),
        version="1.0.0",
    )

    @app.get(
        "/health",
        response_model=HealthResponseSchema,
        tags=["Monitoring"],
        summary="Health check and vector store status",
    )
    def health_check(
        container: Annotated[ServiceContainer, Depends(get_container)],
    ) -> HealthResponseSchema:
        try:
            count = container.vector_store.count()
            return HealthResponseSchema(
                status="healthy",
                collection_name=container.settings.qdrant_collection_name,
                indexed_chunks_count=count,
                llm_model=container.settings.llm_model,
                embedding_model=container.settings.embedding_model,
            )
        except DomainError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc

    @app.post(
        "/ingest",
        response_model=IngestResponseSchema,
        tags=["RAG Pipeline"],
        summary="Ingest Persian PDF documents into Qdrant vector store",
    )
    def ingest_documents(
        request: IngestRequestSchema,
        container: Annotated[ServiceContainer, Depends(get_container)],
    ) -> IngestResponseSchema:
        target_path = (
            Path(request.path) if request.path else container.settings.pdf_data_dir
        )
        try:
            report = container.ingest_use_case.execute(target_path)
            return IngestResponseSchema(
                documents_processed=report.documents_processed,
                pages_extracted=report.pages_extracted,
                chunks_created=report.chunks_created,
                vectors_stored=report.vectors_stored,
                source_files=report.source_files,
                latency_ms=report.latency_ms,
            )
        except DocumentLoadError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except (EmbeddingGenerationError, VectorStoreError) as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=str(exc),
            ) from exc
        except DomainError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

    @app.post(
        "/ask",
        response_model=AskResponseSchema,
        tags=["RAG Pipeline"],
        summary="Ask a Persian question and get a grounded answer with citations",
    )
    def ask_question(
        request: AskRequestSchema,
        container: Annotated[ServiceContainer, Depends(get_container)],
    ) -> AskResponseSchema:
        try:
            qa_response = container.query_use_case.execute(
                query=request.query,
                top_k=request.top_k,
                score_threshold=request.score_threshold,
                source_filter=request.source_filter,
            )
            return AskResponseSchema(
                query=qa_response.query,
                normalized_query=qa_response.normalized_query,
                answer=qa_response.answer,
                citations=[
                    CitationSchema(
                        source_filename=c.source_filename,
                        page_number=c.page_number,
                        chunk_id=c.chunk_id,
                        score=c.score,
                        snippet=c.snippet,
                    )
                    for c in qa_response.citations
                ],
                latency_ms=qa_response.latency_ms,
                created_at=qa_response.created_at,
            )
        except EmptyQueryError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        except (EmbeddingGenerationError, LLMGenerationError, VectorStoreError) as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=str(exc),
            ) from exc
        except DomainError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

    return app


app = create_app()
