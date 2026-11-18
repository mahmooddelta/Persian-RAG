from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    llm_api_key: str = Field(
        default="",
    )
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
    )
    llm_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
    )
    llm_max_tokens: int = Field(
        default=1024,
        ge=64,
        le=16384,
    )
    llm_timeout_seconds: float = Field(
        default=60.0,
        gt=0.0,
    )

    embedding_api_key: str = Field(
        default="",
    )
    embedding_base_url: str = Field(
        default="https://api.openai.com/v1",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
    )
    embedding_dimension: int = Field(
        default=1536,
        gt=0,
    )
    embedding_batch_size: int = Field(
        default=32,
        ge=1,
        le=2048,
    )
    embedding_max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
    )
    embedding_timeout_seconds: float = Field(
        default=30.0,
        gt=0.0,
    )

    qdrant_storage_path: Path = Field(
        default=Path("./qdrant_storage"),
    )
    qdrant_collection_name: str = Field(
        default="persian_rag_documents",
        min_length=1,
    )

    chunk_size: int = Field(
        default=600,
        ge=100,
        le=8000,
    )
    chunk_overlap: int = Field(
        default=120,
        ge=0,
        le=4000,
    )
    default_top_k: int = Field(
        default=4,
        ge=3,
        le=25,
    )
    min_score_threshold: float = Field(
        default=0.15,
        ge=-1.0,
        le=1.0,
    )

    pdf_data_dir: Path = Field(
        default=Path("./data/pdfs"),
    )

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap_smaller_than_chunk_size(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 600)
        if v >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({v}) must be strictly less than chunk_size ({chunk_size})."
            )
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

