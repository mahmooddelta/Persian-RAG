from __future__ import annotations

import time

from src.domain.entities import QAResponse, RetrievalResult
from src.domain.exceptions import EmptyQueryError
from src.domain.interfaces import (
    EmbeddingPort,
    LLMPort,
    TextNormalizerPort,
    VectorStorePort,
)

NO_CONTEXT_FALLBACK_MESSAGE = (
    "اطلاعات کافی برای پاسخ به این پرسش در اسناد بازیابی‌شده یافت نشد. "
    "لطفاً پرسش خود را دقیق‌تر مطرح کنید یا اطمینان حاصل نمایید که اسناد مرتبط بارگذاری شده‌اند."
)

MINIMUM_TOP_K = 3


class AnswerQueryUseCase:
    """
    Application service that answers a Persian question grounded strictly in
    retrieved document chunks.
    """

    def __init__(
        self,
        text_normalizer: TextNormalizerPort,
        embedding_client: EmbeddingPort,
        vector_store: VectorStorePort,
        llm_client: LLMPort,
        default_top_k: int = 4,
        score_threshold: float | None = 0.15,
    ) -> None:
        self._text_normalizer = text_normalizer
        self._embedding_client = embedding_client
        self._vector_store = vector_store
        self._llm_client = llm_client
        self._default_top_k = max(default_top_k, MINIMUM_TOP_K)
        self._score_threshold = score_threshold

    def execute(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        source_filter: str | None = None,
    ) -> QAResponse:
        """
        Process a user question through the RAG pipeline and return a grounded `QAResponse`.

        Args:
            query: Raw user question in Persian.
            top_k: Number of chunks to retrieve (enforced >= 3 per specification).
            score_threshold: Minimum similarity score threshold for retrieved chunks.
            source_filter: Optional PDF filename to scope retrieval.

        Returns:
            `QAResponse` containing the Persian answer, citations, and retrieved chunks.

        Raises:
            EmptyQueryError: If the query is blank before or after normalization.
        """
        start_time = time.perf_counter()

        if not query or not query.strip():
            raise EmptyQueryError("Query cannot be empty.")

        normalized_query = self._text_normalizer.normalize(query)
        if not normalized_query:
            raise EmptyQueryError(
                "Query contained no valid characters after Persian normalization."
            )

        effective_top_k = max(top_k if top_k is not None else self._default_top_k, MINIMUM_TOP_K)
        effective_threshold = (
            score_threshold if score_threshold is not None else self._score_threshold
        )

        query_vector = self._embedding_client.embed_query(normalized_query)
        retrieved_results: list[RetrievalResult] = self._vector_store.search(
            query_vector=query_vector,
            top_k=effective_top_k,
            score_threshold=effective_threshold,
            source_filter=source_filter,
        )

        if not retrieved_results:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            return QAResponse(
                query=query,
                normalized_query=normalized_query,
                answer=NO_CONTEXT_FALLBACK_MESSAGE,
                citations=[],
                retrieved_chunks=[],
                latency_ms=elapsed_ms,
            )

        answer = self._llm_client.generate_grounded_answer(
            query=normalized_query,
            context_chunks=retrieved_results,
        )
        citations = [res.to_citation() for res in retrieved_results]
        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        return QAResponse(
            query=query,
            normalized_query=normalized_query,
            answer=answer,
            citations=citations,
            retrieved_chunks=retrieved_results,
            latency_ms=elapsed_ms,
        )
