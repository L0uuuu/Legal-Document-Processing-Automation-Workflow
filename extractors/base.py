"""Abstract base class for all document extractors."""

import os
import re
import time
from abc import ABC, abstractmethod
from typing import Optional

from models.extraction import (
    DocumentFormat,
    ExtractionMethod,
    ExtractionResult,
    PageContent,
)


class BaseExtractor(ABC):
    """Base class that all extractors must inherit from."""

    def __init__(self):
        self.warnings: list[str] = []

    @abstractmethod
    def extract(self, file_path: str) -> ExtractionResult:
        """Extract text from a document file."""
        pass

    @abstractmethod
    def supported_formats(self) -> list[DocumentFormat]:
        """Return list of formats this extractor supports."""
        pass

    def validate_file(self, file_path: str) -> None:
        """Check that the file exists and is readable."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        if not os.path.isfile(file_path):
            raise ValueError(f"Path is not a file: {file_path}")
        if os.path.getsize(file_path) == 0:
            raise ValueError(f"File is empty: {file_path}")

    def detect_languages(self, text: str) -> dict[str, bool]:
        """Detect presence of Arabic and French in text."""
        # Arabic Unicode range: \u0600-\u06FF (basic), \u0750-\u077F (supplement)
        has_arabic = bool(re.search(r"[\u0600-\u06FF\u0750-\u077F]", text))

        # French indicators: accented characters + common French words
        french_patterns = [
            r"[àâäéèêëïîôùûüÿçœæ]",
            r"\b(les|des|une|dans|pour|avec|sur|par|est|sont|aux|cette|entre)\b",
        ]
        has_french = any(bool(re.search(p, text, re.IGNORECASE)) for p in french_patterns)

        return {"has_arabic": has_arabic, "has_french": has_french}

    def clean_text(self, text: str) -> str:
        """Basic text cleanup while preserving Arabic and French characters."""
        if not text:
            return ""

        # Normalize whitespace (preserve newlines for structure)
        text = re.sub(r"[^\S\n]+", " ", text)

        # Remove excessive blank lines (keep max 2)
        text = re.sub(r"\n{4,}", "\n\n\n", text)

        # Strip leading/trailing whitespace per line
        lines = [line.strip() for line in text.splitlines()]
        text = "\n".join(lines)

        return text.strip()

    def _build_result(
        self,
        file_path: str,
        file_format: DocumentFormat,
        extraction_method: ExtractionMethod,
        pages: list[PageContent],
        start_time: float,
        is_scanned: bool = False,
    ) -> ExtractionResult:
        """Helper to construct the final ExtractionResult."""
        full_text = "\n\n".join(p.text for p in pages if p.text.strip())
        full_text = self.clean_text(full_text)
        languages = self.detect_languages(full_text)

        # Calculate average OCR confidence if available
        confidences = [p.confidence for p in pages if p.confidence is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else None

        return ExtractionResult(
            file_path=file_path,
            file_name=os.path.basename(file_path),
            file_format=file_format,
            file_size_bytes=os.path.getsize(file_path),
            extraction_method=extraction_method,
            total_pages=len(pages),
            extraction_duration_seconds=round(time.time() - start_time, 3),
            pages=pages,
            full_text=full_text,
            avg_confidence=avg_confidence,
            is_scanned=is_scanned,
            has_arabic_content=languages["has_arabic"],
            has_french_content=languages["has_french"],
            warnings=self.warnings,
        )