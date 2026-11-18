from __future__ import annotations


class DomainError(Exception):
    def __init__(self, message: str, details: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} ({self.details})"
        return self.message

class DocumentLoadError(DomainError):
    """Raised when a PDF document cannot be found, opened, or parsed."""

class TextNormalizationError(DomainError):
    """Raised when text normalization fails unexpectedly."""

class EmbeddingGenerationError(DomainError):
    """Raised when an embedding service fails to vectorize text chunks or queries."""


class VectorStoreError(DomainError):
    """Raised when vector indexing, storage, or similarity search fails."""

class LLMGenerationError(DomainError):
    """Raised when the language model completion request fails or returns invalid output."""

class EmptyQueryError(DomainError):
    """Raised when a user query is empty or contains no meaningful characters after normalization."""

