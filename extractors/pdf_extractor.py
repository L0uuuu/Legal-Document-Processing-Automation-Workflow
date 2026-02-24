"""Extract text from native (non-scanned) PDF documents."""

import time

import fitz  # PyMuPDF
import pdfplumber

from models.extraction import (
    DocumentFormat,
    ExtractionMethod,
    PageContent,
    ExtractionResult,
)
from extractors.base import BaseExtractor


class PDFExtractor(BaseExtractor):
    """
    Extracts text from native PDFs using two strategies:
    1. PyMuPDF (fitz) — fast, good for most PDFs
    2. pdfplumber — fallback, better for table-heavy PDFs

    If a page yields very little text, it flags the document
    as potentially scanned (needs OCR).
    """

    # Minimum characters per page to consider it "readable" (not scanned)
    MIN_CHARS_PER_PAGE = 50

    def supported_formats(self) -> list[DocumentFormat]:
        return [DocumentFormat.PDF]

    def extract(self, file_path: str) -> ExtractionResult:
        self.validate_file(file_path)
        start_time = time.time()

        # Try PyMuPDF first (faster)
        pages = self._extract_with_pymupdf(file_path)

        # Check if extraction quality is good enough
        scanned_pages = sum(
            1 for p in pages if len(p.text.strip()) < self.MIN_CHARS_PER_PAGE
        )
        total_pages = len(pages)
        is_scanned = total_pages > 0 and (scanned_pages / total_pages) > 0.5

        if is_scanned:
            self.warnings.append(
                f"Document appears to be scanned ({scanned_pages}/{total_pages} pages "
                f"with <{self.MIN_CHARS_PER_PAGE} chars). Consider using OCR extractor."
            )

        # If PyMuPDF got poor results, try pdfplumber as fallback
        total_text = sum(len(p.text.strip()) for p in pages)
        if total_text < self.MIN_CHARS_PER_PAGE and not is_scanned:
            self.warnings.append("PyMuPDF extraction poor, falling back to pdfplumber.")
            pages = self._extract_with_pdfplumber(file_path)

        return self._build_result(
            file_path=file_path,
            file_format=DocumentFormat.PDF,
            extraction_method=ExtractionMethod.NATIVE_PDF,
            pages=pages,
            start_time=start_time,
            is_scanned=is_scanned,
        )

    def _extract_with_pymupdf(self, file_path: str) -> list[PageContent]:
        """Extract text using PyMuPDF (fitz)."""
        pages = []
        try:
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc[page_num]

                # Extract text preserving layout
                text = page.get_text("text")
                languages = self.detect_languages(text)

                pages.append(
                    PageContent(
                        page_number=page_num + 1,
                        text=self.clean_text(text),
                        has_arabic=languages["has_arabic"],
                        has_french=languages["has_french"],
                    )
                )
            doc.close()
        except Exception as e:
            self.warnings.append(f"PyMuPDF extraction error: {str(e)}")

        return pages

    def _extract_with_pdfplumber(self, file_path: str) -> list[PageContent]:
        """Extract text using pdfplumber (fallback)."""
        pages = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    languages = self.detect_languages(text)

                    pages.append(
                        PageContent(
                            page_number=page_num + 1,
                            text=self.clean_text(text),
                            has_arabic=languages["has_arabic"],
                            has_french=languages["has_french"],
                        )
                    )
        except Exception as e:
            self.warnings.append(f"pdfplumber extraction error: {str(e)}")

        return pages