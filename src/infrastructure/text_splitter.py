from __future__ import annotations

import re
from typing import Sequence

from src.domain.entities import DocumentChunk, DocumentPage
from src.domain.interfaces import TextSplitterPort


class RecursivePersianTextSplitter(TextSplitterPort):

    _SENTENCE_SPLIT_PATTERN = re.compile(
        r"(?<=[.؟!؛\n])\s+"
    )

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 120) -> None:
        if chunk_size < 50:
            raise ValueError(f"chunk_size must be >= 50, got {chunk_size}.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be in [0, chunk_size ({chunk_size}))."
            )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def chunk_overlap(self) -> int:
        return self._chunk_overlap

    def split_pages(self, pages: Sequence[DocumentPage]) -> list[DocumentChunk]:
        all_chunks: list[DocumentChunk] = []

        for page in pages:
            page_text = page.text.strip()
            if not page_text:
                continue

            segments = self._split_text_into_chunks(page_text)
            search_cursor = 0

            for idx, chunk_text in enumerate(segments):
                cleaned_chunk = chunk_text.strip()
                if not cleaned_chunk:
                    continue

                found_pos = page_text.find(cleaned_chunk[:30], max(0, search_cursor - self._chunk_overlap))
                start_char = found_pos if found_pos != -1 else search_cursor
                end_char = start_char + len(cleaned_chunk)
                search_cursor = max(search_cursor, end_char - self._chunk_overlap)

                chunk_id = DocumentChunk.build_deterministic_id(
                    source_filename=page.source_filename,
                    page_number=page.page_number,
                    chunk_index=idx,
                    text=cleaned_chunk,
                )

                chunk_metadata = dict(page.metadata)
                chunk_metadata.update(
                    {
                        "source_filename": page.source_filename,
                        "page_number": page.page_number,
                        "chunk_index": idx,
                    }
                )

                all_chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=cleaned_chunk,
                        source_filename=page.source_filename,
                        page_number=page.page_number,
                        chunk_index=idx,
                        start_char=start_char,
                        end_char=end_char,
                        metadata=chunk_metadata,
                    )
                )

        return all_chunks

    def _split_text_into_chunks(self, text: str) -> list[str]:
        if len(text) <= self._chunk_size:
            return [text]

        atomic_units = self._extract_atomic_units(text)
        chunks: list[str] = []
        current_units: list[str] = []
        current_length = 0

        for unit in atomic_units:
            unit_len = len(unit)
            separator_len = 1 if current_units else 0

            if current_length + separator_len + unit_len <= self._chunk_size:
                current_units.append(unit)
                current_length += separator_len + unit_len
            else:
                if current_units:
                    chunks.append(" ".join(current_units))

                overlap_units: list[str] = []
                overlap_len = 0
                for prev_unit in reversed(current_units):
                    added_len = len(prev_unit) + (1 if overlap_units else 0)
                    if overlap_len + added_len <= self._chunk_overlap:
                        overlap_units.insert(0, prev_unit)
                        overlap_len += added_len
                    else:
                        break

                while overlap_units and (overlap_len + 1 + unit_len > self._chunk_size):
                    removed = overlap_units.pop(0)
                    overlap_len -= len(removed) + (1 if overlap_units else 0)

                current_units = overlap_units + [unit]
                current_length = sum(len(u) for u in current_units) + max(
                    0, len(current_units) - 1
                )

        if current_units:
            chunks.append(" ".join(current_units))

        return chunks

    def _extract_atomic_units(self, text: str) -> list[str]:
        raw_sentences = [
            s.strip() for s in self._SENTENCE_SPLIT_PATTERN.split(text) if s.strip()
        ]
        units: list[str] = []

        for sentence in raw_sentences:
            if len(sentence) <= self._chunk_size:
                units.append(sentence)
            else:
                clauses = [c.strip() for c in re.split(r"(?<=،)\s+", sentence) if c.strip()]
                for clause in clauses:
                    if len(clause) <= self._chunk_size:
                        units.append(clause)
                    else:
                        units.extend(self._split_oversized_clause(clause))

        return units

    def _split_oversized_clause(self, clause: str) -> list[str]:
        words = clause.split(" ")
        sub_chunks: list[str] = []
        current_words: list[str] = []
        current_len = 0
        step_overlap = min(self._chunk_overlap, self._chunk_size // 3)

        for word in words:
            if len(word) > self._chunk_size:
                if current_words:
                    sub_chunks.append(" ".join(current_words))
                    current_words = []
                    current_len = 0
                step = self._chunk_size - step_overlap
                for i in range(0, len(word), step):
                    sub_chunks.append(word[i : i + self._chunk_size])
                continue

            add_len = len(word) + (1 if current_words else 0)
            if current_len + add_len <= self._chunk_size:
                current_words.append(word)
                current_len += add_len
            else:
                sub_chunks.append(" ".join(current_words))
                overlap_words: list[str] = []
                ov_len = 0
                for w in reversed(current_words):
                    w_len = len(w) + (1 if overlap_words else 0)
                    if ov_len + w_len <= step_overlap:
                        overlap_words.insert(0, w)
                        ov_len += w_len
                    else:
                        break
                current_words = overlap_words + [word]
                current_len = sum(len(w) for w in current_words) + max(
                    0, len(current_words) - 1
                )

        if current_words:
            sub_chunks.append(" ".join(current_words))

        return sub_chunks

