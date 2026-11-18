from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True, slots=True)
class DocumentPage:
    source_filename: str
    page_number: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_filename or not self.source_filename.strip():
            raise ValueError("source_filename must be a non-empty string.")
        if self.page_number < 1:
            raise ValueError(f"page_number must be >= 1, got {self.page_number}.")


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    chunk_id: str
    text: str
    source_filename: str
    page_number: int
    chunk_index: int
    start_char: int = 0
    end_char: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chunk_id or not self.chunk_id.strip():
            raise ValueError("chunk_id must be a non-empty string.")
        if not self.text or not self.text.strip():
            raise ValueError("DocumentChunk text must not be empty.")
        if self.page_number < 1:
            raise ValueError(f"page_number must be >= 1, got {self.page_number}.")
        if self.chunk_index < 0:
            raise ValueError(f"chunk_index must be >= 0, got {self.chunk_index}.")

    @staticmethod
    def build_deterministic_id(
        source_filename: str,
        page_number: int,
        chunk_index: int,
        text: str,
    ) -> str:
        raw_key = f"{source_filename}::p{page_number}::c{chunk_index}::{text.strip()}"
        digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        # Format first 32 hex chars as a standard UUID string (8-4-4-4-12) for Qdrant compatibility
        return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


@dataclass(frozen=True, slots=True)
class SourceCitation:
    source_filename: str
    page_number: int
    chunk_id: str
    score: float
    snippet: str


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    chunk: DocumentChunk
    score: float

    def to_citation(self, max_snippet_length: int = 220) -> SourceCitation:
        cleaned = self.chunk.text.strip().replace("\n", " ")
        snippet = (
            cleaned[:max_snippet_length].rstrip() + "..."
            if len(cleaned) > max_snippet_length
            else cleaned
        )
        return SourceCitation(
            source_filename=self.chunk.source_filename,
            page_number=self.chunk.page_number,
            chunk_id=self.chunk.chunk_id,
            score=round(self.score, 4),
            snippet=snippet,
        )


@dataclass(frozen=True, slots=True)
class QAResponse:
    query: str
    normalized_query: str
    answer: str
    citations: list[SourceCitation]
    retrieved_chunks: list[RetrievalResult] = field(default_factory=list)
    latency_ms: float = 0.0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass(frozen=True, slots=True)
class IngestionReport:
    documents_processed: int
    pages_extracted: int
    chunks_created: int
    vectors_stored: int
    source_files: list[str]
    latency_ms: float = 0.0

