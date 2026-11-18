from __future__ import annotations

from pathlib import Path
from typing import Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.domain.entities import DocumentChunk, RetrievalResult
from src.domain.exceptions import VectorStoreError
from src.domain.interfaces import VectorStorePort


class QdrantVectorStore(VectorStorePort):

    def __init__(
        self,
        storage_path: Path | str = Path("./qdrant_storage"),
        collection_name: str = "persian_rag_documents",
        embedding_dimension: int = 1536,
        client: QdrantClient | None = None,
    ) -> None:
        self._storage_path = str(storage_path)
        self._collection_name = collection_name
        self._embedding_dimension = embedding_dimension

        try:
            if client is not None:
                self._client = client
            elif self._storage_path == ":memory:":
                self._client = QdrantClient(location=":memory:")
            else:
                path_obj = Path(self._storage_path)
                path_obj.mkdir(parents=True, exist_ok=True)
                self._client = QdrantClient(path=str(path_obj))

            self._ensure_collection(self._embedding_dimension)
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to initialize Qdrant vector store at '{self._storage_path}'",
                details=str(exc),
            ) from exc

    def _get_existing_vector_size(self) -> int | None:
        info = self._client.get_collection(self._collection_name)
        vectors_cfg = info.config.params.vectors
        if isinstance(vectors_cfg, qmodels.VectorParams):
            return int(vectors_cfg.size)
        if isinstance(vectors_cfg, dict) and vectors_cfg:
            first_param = next(iter(vectors_cfg.values()))
            if hasattr(first_param, "size"):
                return int(first_param.size)
        size_attr = getattr(vectors_cfg, "size", None)
        return int(size_attr) if size_attr is not None else None

    def _ensure_collection(self, vector_size: int) -> None:
        exists = self._client.collection_exists(self._collection_name)
        if exists:
            existing_dim = self._get_existing_vector_size()
            if existing_dim is not None and existing_dim != vector_size:
                self._client.delete_collection(self._collection_name)
                exists = False

        if not exists:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=qmodels.VectorParams(
                    size=vector_size,
                    distance=qmodels.Distance.COSINE,
                ),
            )

    def upsert_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"Mismatch between chunks count ({len(chunks)}) and embeddings count ({len(embeddings)})."
            )

        if not chunks:
            return 0

        try:
            actual_dim = len(embeddings[0])
            self._embedding_dimension = actual_dim
            self._ensure_collection(actual_dim)

            points: list[qmodels.PointStruct] = []
            for chunk, vector in zip(chunks, embeddings):
                payload = {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "source_filename": chunk.source_filename,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "metadata": chunk.metadata,
                }
                points.append(
                    qmodels.PointStruct(
                        id=chunk.chunk_id,
                        vector=list(vector),
                        payload=payload,
                    )
                )

            self._client.upsert(
                collection_name=self._collection_name,
                points=points,
                wait=True,
            )
            return len(points)
        except Exception as exc:
            raise VectorStoreError(
                "Failed to upsert chunks into Qdrant collection.",
                details=str(exc),
            ) from exc

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int = 4,
        score_threshold: float | None = None,
        source_filter: str | None = None,
    ) -> list[RetrievalResult]:
        if top_k < 1:
            raise VectorStoreError(f"top_k must be >= 1, got {top_k}.")

        try:
            query_filter: qmodels.Filter | None = None
            if source_filter:
                query_filter = qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="source_filename",
                            match=qmodels.MatchValue(value=source_filter),
                        )
                    ]
                )

            candidate_limit = top_k if source_filter else max(top_k * 4, 16)

            response = self._client.query_points(
                collection_name=self._collection_name,
                query=list(query_vector),
                limit=candidate_limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=True,
            )

            candidates: list[RetrievalResult] = []
            for point in response.points:
                payload = point.payload or {}
                chunk = DocumentChunk(
                    chunk_id=str(payload.get("chunk_id", point.id)),
                    text=str(payload.get("text", "")),
                    source_filename=str(payload.get("source_filename", "unknown.pdf")),
                    page_number=int(payload.get("page_number", 1)),
                    chunk_index=int(payload.get("chunk_index", 0)),
                    start_char=int(payload.get("start_char", 0)),
                    end_char=int(payload.get("end_char", 0)),
                    metadata=dict(payload.get("metadata") or {}),
                )
                candidates.append(
                    RetrievalResult(
                        chunk=chunk,
                        score=float(point.score),
                    )
                )

            if source_filter or len(candidates) <= top_k:
                return candidates[:top_k]

            return self._diversify_across_sources(candidates, top_k=top_k)
        except Exception as exc:
            raise VectorStoreError(
                "Failed to execute similarity search in Qdrant.",
                details=str(exc),
            ) from exc

    @staticmethod
    def _diversify_across_sources(
        candidates: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        if not candidates:
            return []

        best_overall_score = candidates[0].score
        viable_min_score = best_overall_score * 0.70

        selected: list[RetrievalResult] = []
        selected_ids: set[str] = set()
        seen_sources: set[str] = set()

        for item in candidates:
            src = item.chunk.source_filename
            if src not in seen_sources and item.score >= viable_min_score:
                selected.append(item)
                selected_ids.add(item.chunk.chunk_id)
                seen_sources.add(src)
                if len(selected) >= top_k:
                    break

        if len(selected) < top_k:
            for item in candidates:
                if item.chunk.chunk_id not in selected_ids:
                    selected.append(item)
                    selected_ids.add(item.chunk.chunk_id)
                    if len(selected) >= top_k:
                        break

        selected.sort(key=lambda r: r.score, reverse=True)
        return selected

    def count(self) -> int:
        try:
            if not self._client.collection_exists(self._collection_name):
                return 0
            result = self._client.count(
                collection_name=self._collection_name,
                exact=True,
            )
            return int(result.count)
        except Exception as exc:
            raise VectorStoreError(
                "Failed to count points in Qdrant collection.",
                details=str(exc),
            ) from exc

    def clear_collection(self) -> None:
        try:
            if self._client.collection_exists(self._collection_name):
                self._client.delete_collection(self._collection_name)
            self._ensure_collection(self._embedding_dimension)
        except Exception as exc:
            raise VectorStoreError(
                "Failed to clear Qdrant collection.",
                details=str(exc),
            ) from exc

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass
