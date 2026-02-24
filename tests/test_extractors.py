"""Tests for document extractors."""

import os
import tempfile

import pytest

from extractors.factory import ExtractorFactory, extract_document
from extractors.txt_extractor import TXTExtractor
from extractors.base import BaseExtractor
from models.extraction import DocumentFormat


class TestTXTExtractor:
    """Test plain text extraction."""

    def test_extract_french_text(self):
        content = (
            "Article 2.- Des autorisations d'exercice de la médecine "
            "vétérinaire peuvent être accordées dans les conditions "
            "prévues par la présente loi."
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(content)
            f.flush()

            extractor = TXTExtractor()
            result = extractor.extract(f.name)

            assert result.file_format == DocumentFormat.TXT
            assert result.has_french_content is True
            assert result.total_pages == 1
            assert "autorisations" in result.full_text

        os.unlink(f.name)

    def test_extract_arabic_text(self):
        content = "يمكن منح تراخيص لممارسة الطب البيطري في الشروط المنصوص عليها"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(content)
            f.flush()

            extractor = TXTExtractor()
            result = extractor.extract(f.name)

            assert result.has_arabic_content is True
            assert "تراخيص" in result.full_text

        os.unlink(f.name)

    def test_extract_bilingual_text(self):
        content = (
            "Article 2.- Des autorisations d'exercice\n"
            "المادة 2.- يمكن منح تراخيص"
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(content)
            f.flush()

            extractor = TXTExtractor()
            result = extractor.extract(f.name)

            assert result.has_arabic_content is True
            assert result.has_french_content is True

        os.unlink(f.name)

    def test_empty_file_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            pass  # empty file

        extractor = TXTExtractor()
        with pytest.raises(ValueError, match="empty"):
            extractor.extract(f.name)

        os.unlink(f.name)

    def test_missing_file_raises(self):
        extractor = TXTExtractor()
        with pytest.raises(FileNotFoundError):
            extractor.extract("/nonexistent/file.txt")


class TestExtractorFactory:
    """Test the factory auto-detection logic."""

    def test_detect_pdf_format(self):
        factory = ExtractorFactory()
        assert factory.detect_format("doc.pdf") == DocumentFormat.PDF
        assert factory.detect_format("DOC.PDF") == DocumentFormat.PDF

    def test_detect_docx_format(self):
        factory = ExtractorFactory()
        assert factory.detect_format("doc.docx") == DocumentFormat.DOCX

    def test_detect_txt_format(self):
        factory = ExtractorFactory()
        assert factory.detect_format("doc.txt") == DocumentFormat.TXT

    def test_unknown_format(self):
        factory = ExtractorFactory()
        assert factory.detect_format("doc.xyz") == DocumentFormat.UNKNOWN

    def test_unsupported_format_raises(self):
        factory = ExtractorFactory()
        with pytest.raises(ValueError, match="Unsupported"):
            factory.extract("document.xyz")

    def test_convenience_function(self):
        content = "Article premier - Dispositions générales"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(content)
            f.flush()

            result = extract_document(f.name)
            assert "Dispositions" in result.full_text

        os.unlink(f.name)