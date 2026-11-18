from __future__ import annotations

import time
from pathlib import Path

from src.domain.entities import DocumentPage, IngestionReport
from src.domain.exceptions import DocumentLoadError
from src.domain.interfaces import (
    DocumentLoaderPort,
    EmbeddingPort,
    TextNormalizerPort,
    TextSplitterPort,
    VectorStorePort,
)


class IngestDocumentsUseCase:

    def __init__(
        self,
        document_loader: DocumentLoaderPort,
        text_normalizer: TextNormalizerPort,
        text_splitter: TextSplitterPort,
        embedding_client: EmbeddingPort,
        vector_store: VectorStorePort,
    ) -> None:
        self._document_loader = document_loader
        self._text_normalizer = text_normalizer
        self._text_splitter = text_splitter
        self._embedding_client = embedding_client
        self._vector_store = vector_store

    def execute(self, target_path: Path) -> IngestionReport:

        start_time = time.perf_counter()

        if not target_path.exists():
            raise DocumentLoadError(
                f"Target ingestion path does not exist: {target_path}"
            )

        if target_path.is_file():
            raw_pages = self._document_loader.load_file(target_path)
        elif target_path.is_dir():
            raw_pages = self._document_loader.load_directory(target_path)
        else:
            raise DocumentLoadError(
                f"Unsupported path type for ingestion: {target_path}"
            )

        if not raw_pages:
            raise DocumentLoadError(
                f"No readable PDF pages found at path: {target_path}"
            )

        normalized_pages: list[DocumentPage] = []
        source_files_set: set[str] = set()

        for page in raw_pages:
            cleaned_text = self._text_normalizer.normalize(page.text)
            if cleaned_text:
                normalized_pages.append(
                    DocumentPage(
                        source_filename=page.source_filename,
                        page_number=page.page_number,
                        text=cleaned_text,
                        metadata=dict(page.metadata),
                    )
                )
                source_files_set.add(page.source_filename)

        if not normalized_pages:
            raise DocumentLoadError(
                f"All extracted pages were empty after Persian normalization: {target_path}"
            )

        chunks = self._text_splitter.split_pages(normalized_pages)
        if not chunks:
            raise DocumentLoadError(
                f"No text chunks could be created from pages in: {target_path}"
            )

        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = self._embedding_client.embed_documents(chunk_texts)
        vectors_stored = self._vector_store.upsert_chunks(chunks, embeddings)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        sorted_sources = sorted(source_files_set)

        return IngestionReport(
            documents_processed=len(sorted_sources),
            pages_extracted=len(normalized_pages),
            chunks_created=len(chunks),
            vectors_stored=vectors_stored,
            source_files=sorted_sources,
            latency_ms=elapsed_ms,
        )

