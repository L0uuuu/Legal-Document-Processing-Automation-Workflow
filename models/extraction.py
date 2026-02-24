"""Data models for document extraction results."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DocumentFormat(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    UNKNOWN = "unknown"


class ExtractionMethod(str, Enum):
    NATIVE_PDF = "native_pdf"
    OCR = "ocr"
    DOCX_PARSE = "docx_parse"
    PLAIN_TEXT = "plain_text"


class PageContent(BaseModel):
    """Represents extracted content from a single page."""
    page_number: int
    text: str
    confidence: Optional[float] = None  # OCR confidence score (0-1)
    language_detected: Optional[str] = None
    has_arabic: bool = False
    has_french: bool = False


class ExtractionResult(BaseModel):
    """Complete result of document text extraction."""
    # Source info
    file_path: str
    file_name: str
    file_format: DocumentFormat
    file_size_bytes: int

    # Extraction metadata
    extraction_method: ExtractionMethod
    extraction_timestamp: datetime = Field(default_factory=datetime.utcnow)
    total_pages: int = 0
    extraction_duration_seconds: float = 0.0

    # Content
    pages: list[PageContent] = []
    full_text: str = ""

    # Quality indicators
    avg_confidence: Optional[float] = None  # For OCR
    is_scanned: bool = False
    has_arabic_content: bool = False
    has_french_content: bool = False
    warnings: list[str] = []

    def get_text(self) -> str:
        """Get the complete extracted text."""
        if self.full_text:
            return self.full_text
        return "\n\n".join(page.text for page in self.pages if page.text.strip())