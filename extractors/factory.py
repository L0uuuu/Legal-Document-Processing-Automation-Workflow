"""Factory to auto-detect document format and return the right extractor."""

import os
from typing import Optional

from models.extraction import DocumentFormat, ExtractionResult
from extractors.base import BaseExtractor
from extractors.pdf_extractor import PDFExtractor
from extractors.ocr_extractor import OCRExtractor
from extractors.docx_extractor import DOCXExtractor
from extractors.txt_extractor import TXTExtractor


# File extension to format mapping
EXTENSION_MAP: dict[str, DocumentFormat] = {
    ".pdf": DocumentFormat.PDF,
    ".docx": DocumentFormat.DOCX,
    ".doc": DocumentFormat.DOCX,
    ".txt": DocumentFormat.TXT,
    ".text": DocumentFormat.TXT,
}


class ExtractorFactory:
    """
    Automatically selects the correct extractor based on file type.

    Usage:
        factory = ExtractorFactory()
        result = factory.extract("path/to/document.pdf")

        # Or force OCR:
        result = factory.extract("path/to/scanned.pdf", force_ocr=True)
    """

    def __init__(self):
        self._extractors: dict[DocumentFormat, BaseExtractor] = {
            DocumentFormat.DOCX: DOCXExtractor(),
            DocumentFormat.TXT: TXTExtractor(),
        }
        self._pdf_extractor = PDFExtractor()
        self._ocr_extractor: Optional[OCRExtractor] = None

    def _get_ocr_extractor(self) -> OCRExtractor:
        """Lazy-initialize OCR extractor (Tesseract check is expensive)."""
        if self._ocr_extractor is None:
            self._ocr_extractor = OCRExtractor()
        return self._ocr_extractor

    def detect_format(self, file_path: str) -> DocumentFormat:
        """Detect document format from file extension."""
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        return EXTENSION_MAP.get(ext, DocumentFormat.UNKNOWN)

    def extract(
        self,
        file_path: str,
        force_ocr: bool = False,
        format_override: Optional[DocumentFormat] = None,
    ) -> ExtractionResult:
        """
        Extract text from a document.

        Args:
            file_path: Path to the document file.
            force_ocr: If True, skip native PDF extraction and go straight to OCR.
            format_override: Override auto-detected format.

        Returns:
            ExtractionResult with all extracted content and metadata.
        """
        # Detect format
        doc_format = format_override or self.detect_format(file_path)

        if doc_format == DocumentFormat.UNKNOWN:
            raise ValueError(
                f"Unsupported file format: {file_path}. "
                f"Supported: {list(EXTENSION_MAP.keys())}"
            )

        # PDF: decide between native extraction and OCR
        if doc_format == DocumentFormat.PDF:
            if force_ocr:
                return self._get_ocr_extractor().extract(file_path)

            # Try native extraction first
            result = self._pdf_extractor.extract(file_path)

            # If document appears scanned, auto-fallback to OCR
            if result.is_scanned:
                print(
                    f"⚠️  Document appears scanned. Attempting OCR fallback..."
                )
                try:
                    ocr_result = self._get_ocr_extractor().extract(file_path)
                    # Use OCR result if it got more text
                    if len(ocr_result.full_text) > len(result.full_text):
                        return ocr_result
                except Exception as e:
                    result.warnings.append(f"OCR fallback failed: {str(e)}")

            return result

        # Non-PDF formats
        extractor = self._extractors.get(doc_format)
        if not extractor:
            raise ValueError(f"No extractor available for format: {doc_format}")

        return extractor.extract(file_path)


# Convenience function
def extract_document(
    file_path: str,
    force_ocr: bool = False,
) -> ExtractionResult:
    """
    One-liner to extract text from any supported document.

    Usage:
        from src.extractors.factory import extract_document
        result = extract_document("path/to/file.pdf")
        print(result.full_text)
    """
    factory = ExtractorFactory()
    return factory.extract(file_path, force_ocr=force_ocr)