from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

from src.domain.entities import DocumentChunk, DocumentPage, RetrievalResult


class DocumentLoaderPort(ABC):
    @abstractmethod
    def load_file(self, file_path: Path) -> list[DocumentPage]:
        pass

    @abstractmethod
    def load_directory(self, directory_path: Path) -> list[DocumentPage]:
        pass


class TextNormalizerPort(ABC):
    @abstractmethod
    def normalize(self, text: str) -> str:
        pass


class TextSplitterPort(ABC):
    @abstractmethod
    def split_pages(self, pages: Sequence[DocumentPage]) -> list[DocumentChunk]:
        pass


class EmbeddingPort(ABC):
    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        pass

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        pass

class VectorStorePort(ABC):
    @abstractmethod
    def upsert_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        pass

    @abstractmethod
    def search(
        self,
        query_vector: Sequence[float],
        top_k: int = 4,
        score_threshold: float | None = None,
        source_filter: str | None = None,
    ) -> list[RetrievalResult]:
        pass

    @abstractmethod
    def count(self) -> int:
        pass

    def clear_collection(self) -> None:
        pass


class LLMPort(ABC):
    @abstractmethod
    def generate_grounded_answer(
        self,
        query: str,
        context_chunks: Sequence[RetrievalResult],
    ) -> str:
        pass
