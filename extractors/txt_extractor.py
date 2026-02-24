"""Extract text from plain text (.txt) files."""

import time

from models.extraction import (
    DocumentFormat,
    ExtractionMethod,
    PageContent,
    ExtractionResult,
)
from extractors.base import BaseExtractor


class TXTExtractor(BaseExtractor):
    """
    Extracts text from plain .txt files.
    Handles multiple encodings common in Arabic/French documents.
    """

    # Encodings to try, in order of likelihood for Tunisian legal docs
    ENCODINGS = ["utf-8", "utf-8-sig", "windows-1256", "iso-8859-6", "iso-8859-1", "cp1252"]

    def supported_formats(self) -> list[DocumentFormat]:
        return [DocumentFormat.TXT]

    def extract(self, file_path: str) -> ExtractionResult:
        self.validate_file(file_path)
        start_time = time.time()

        text, encoding_used = self._read_with_encoding(file_path)

        if encoding_used != "utf-8":
            self.warnings.append(
                f"File was not UTF-8. Decoded using: {encoding_used}"
            )

        languages = self.detect_languages(text)

        pages = [
            PageContent(
                page_number=1,
                text=self.clean_text(text),
                has_arabic=languages["has_arabic"],
                has_french=languages["has_french"],
            )
        ]

        return self._build_result(
            file_path=file_path,
            file_format=DocumentFormat.TXT,
            extraction_method=ExtractionMethod.PLAIN_TEXT,
            pages=pages,
            start_time=start_time,
        )

    def _read_with_encoding(self, file_path: str) -> tuple[str, str]:
        """Try multiple encodings to read the file."""
        for encoding in self.ENCODINGS:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    content = f.read()
                return content, encoding
            except (UnicodeDecodeError, UnicodeError):
                continue

        # Last resort: read with error replacement
        self.warnings.append("Could not detect encoding. Using UTF-8 with replacement.")
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        return content, "utf-8-fallback"