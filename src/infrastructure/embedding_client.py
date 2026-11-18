from __future__ import annotations

import time
from typing import Sequence

import httpx

from src.domain.exceptions import EmbeddingGenerationError
from src.domain.interfaces import EmbeddingPort


class OpenAICompatibleEmbeddingClient(EmbeddingPort):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "text-embedding-3-small",
        batch_size: int = 32,
        max_retries: int = 3,
        timeout_seconds: float = 30.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._batch_size = max(1, batch_size)
        self._max_retries = max(0, max_retries)
        self._timeout_seconds = timeout_seconds
        self._http_client = http_client

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        all_vectors: list[list[float]] = []
        for start_idx in range(0, len(texts), self._batch_size):
            batch = list(texts[start_idx : start_idx + self._batch_size])
            batch_vectors = self._request_embeddings_with_retry(batch)
            all_vectors.extend(batch_vectors)

        return all_vectors

    def embed_query(self, query: str) -> list[float]:
        if not query or not query.strip():
            raise EmbeddingGenerationError("Cannot embed an empty query.")

        vectors = self._request_embeddings_with_retry([query.strip()])
        if not vectors:
            raise EmbeddingGenerationError("Embedding API returned an empty vector list.")
        return vectors[0]

    def _request_embeddings_with_retry(
        self, batch_texts: list[str]
    ) -> list[list[float]]:
        endpoint = f"{self._base_url}/embeddings"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": self._model,
            "input": batch_texts,
        }

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                if self._http_client is not None:
                    response = self._http_client.post(
                        endpoint,
                        json=payload,
                        headers=headers,
                        timeout=self._timeout_seconds,
                    )
                else:
                    with httpx.Client(timeout=self._timeout_seconds) as client:
                        response = client.post(
                            endpoint,
                            json=payload,
                            headers=headers,
                        )

                if response.status_code in {429, 500, 502, 503, 504} and attempt < self._max_retries:
                    backoff = 0.5 * (2**attempt)
                    time.sleep(backoff)
                    continue

                response.raise_for_status()
                data = response.json()
                raw_items = data.get("data", [])
                if len(raw_items) != len(batch_texts):
                    raise EmbeddingGenerationError(
                        f"Expected {len(batch_texts)} embeddings, received {len(raw_items)}."
                    )

                # Ensure items are ordered by their `index` field
                sorted_items = sorted(raw_items, key=lambda item: item.get("index", 0))
                return [list(map(float, item["embedding"])) for item in sorted_items]

            except Exception as exc:
                last_error = exc
                if attempt < self._max_retries and isinstance(
                    exc, (httpx.TimeoutException, httpx.TransportError)
                ):
                    backoff = 0.5 * (2**attempt)
                    time.sleep(backoff)
                    continue
                break

        raise EmbeddingGenerationError(
            f"Failed to generate embeddings from '{endpoint}' using model '{self._model}'",
            details=str(last_error),
        ) from last_error

