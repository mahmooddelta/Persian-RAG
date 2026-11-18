from __future__ import annotations

from pathlib import Path

import pymupdf as fitz  # PyMuPDF

from src.domain.entities import DocumentPage
from src.domain.exceptions import DocumentLoadError
from src.domain.interfaces import DocumentLoaderPort


class PyMuPDFLoader(DocumentLoaderPort):

    def load_file(self, file_path: Path) -> list[DocumentPage]:

        resolved = file_path.resolve()
        if not resolved.exists() or not resolved.is_file():
            raise DocumentLoadError(f"PDF file not found: {file_path}")

        if resolved.suffix.lower() != ".pdf":
            raise DocumentLoadError(
                f"Expected a '.pdf' file, got '{resolved.suffix}': {file_path}"
            )

        pages: list[DocumentPage] = []
        try:
            with fitz.open(resolved) as pdf_doc:
                total_pages = len(pdf_doc)
                doc_metadata = pdf_doc.metadata or {}
                title = doc_metadata.get("title") or resolved.stem

                for page_idx in range(total_pages):
                    page = pdf_doc.load_page(page_idx)

                    extracted_text = page.get_text("text", sort=False)
                    if extracted_text and extracted_text.strip():
                        pages.append(
                            DocumentPage(
                                source_filename=resolved.name,
                                page_number=page_idx + 1,
                                text=extracted_text,
                                metadata={
                                    "total_pages": total_pages,
                                    "document_title": title,
                                    "file_path": str(resolved),
                                },
                            )
                        )
        except Exception as exc:
            if isinstance(exc, DocumentLoadError):
                raise
            raise DocumentLoadError(
                f"Failed to read PDF file '{resolved.name}'",
                details=str(exc),
            ) from exc

        return pages

    def load_directory(self, directory_path: Path) -> list[DocumentPage]:
        resolved_dir = directory_path.resolve()
        if not resolved_dir.exists() or not resolved_dir.is_dir():
            raise DocumentLoadError(f"PDF directory not found: {directory_path}")

        pdf_files = sorted(resolved_dir.glob("*.pdf"))
        if not pdf_files:
            raise DocumentLoadError(
                f"No '.pdf' files found in directory: {directory_path}"
            )

        all_pages: list[DocumentPage] = []
        for pdf_file in pdf_files:
            all_pages.extend(self.load_file(pdf_file))

        return all_pages
