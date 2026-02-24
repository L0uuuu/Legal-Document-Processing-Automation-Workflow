"""Extract text from scanned PDFs using Tesseract OCR."""

import time
from typing import Optional

import fitz  # PyMuPDF — to convert PDF pages to images
import pytesseract
from PIL import Image

from models.extraction import (
    DocumentFormat,
    ExtractionMethod,
    PageContent,
    ExtractionResult,
)
from extractors.base import BaseExtractor


class OCRExtractor(BaseExtractor):
    """
    Extracts text from scanned PDFs by:
    1. Converting each page to an image (via PyMuPDF)
    2. Running Tesseract OCR with Arabic + French language packs

    Prerequisites:
        - Tesseract installed: sudo apt install tesseract-ocr
        - Language packs: sudo apt install tesseract-ocr-ara tesseract-ocr-fra
    """

    # Default languages: French + Arabic
    DEFAULT_LANGUAGES = "fra+ara"

    # DPI for PDF-to-image conversion (higher = better quality, slower)
    DEFAULT_DPI = 300

    def __init__(
        self,
        languages: Optional[str] = None,
        dpi: int = DEFAULT_DPI,
        tesseract_cmd: Optional[str] = None,
    ):
        super().__init__()
        self.languages = languages or self.DEFAULT_LANGUAGES
        self.dpi = dpi

        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        # Verify Tesseract is available
        self._verify_tesseract()

    def _verify_tesseract(self) -> None:
        """Check that Tesseract is installed and languages are available."""
        try:
            available_langs = pytesseract.get_languages()
            for lang in self.languages.split("+"):
                if lang not in available_langs:
                    self.warnings.append(
                        f"Tesseract language '{lang}' not installed. "
                        f"Install with: sudo apt install tesseract-ocr-{lang}"
                    )
        except Exception as e:
            self.warnings.append(
                f"Tesseract not found or not configured: {str(e)}. "
                f"Install with: sudo apt install tesseract-ocr"
            )

    def supported_formats(self) -> list[DocumentFormat]:
        return [DocumentFormat.PDF]

    def extract(self, file_path: str) -> ExtractionResult:
        self.validate_file(file_path)
        start_time = time.time()

        pages = self._ocr_pdf(file_path)

        return self._build_result(
            file_path=file_path,
            file_format=DocumentFormat.PDF,
            extraction_method=ExtractionMethod.OCR,
            pages=pages,
            start_time=start_time,
            is_scanned=True,
        )

    def _ocr_pdf(self, file_path: str) -> list[PageContent]:
        """Convert each PDF page to image, then OCR it."""
        pages = []

        try:
            doc = fitz.open(file_path)

            for page_num in range(len(doc)):
                page = doc[page_num]

                # Convert page to image at specified DPI
                # Matrix controls the zoom/DPI: 300 DPI ≈ zoom of 300/72
                zoom = self.dpi / 72
                matrix = fitz.Matrix(zoom, zoom)
                pixmap = page.get_pixmap(matrix=matrix)

                # Convert to PIL Image
                img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)

                # Run OCR
                ocr_data = pytesseract.image_to_data(
                    img,
                    lang=self.languages,
                    output_type=pytesseract.Output.DICT,
                )

                # Extract text and calculate confidence
                text = pytesseract.image_to_string(img, lang=self.languages)
                confidence = self._calculate_confidence(ocr_data)
                languages = self.detect_languages(text)

                pages.append(
                    PageContent(
                        page_number=page_num + 1,
                        text=self.clean_text(text),
                        confidence=confidence,
                        has_arabic=languages["has_arabic"],
                        has_french=languages["has_french"],
                    )
                )

            doc.close()

        except Exception as e:
            self.warnings.append(f"OCR extraction error: {str(e)}")

        return pages

    def _calculate_confidence(self, ocr_data: dict) -> float:
        """Calculate average OCR confidence from Tesseract output."""
        confidences = [
            int(conf)
            for conf in ocr_data.get("conf", [])
            if str(conf).lstrip("-").isdigit() and int(conf) > 0
        ]

        if not confidences:
            return 0.0

        return round(sum(confidences) / len(confidences) / 100, 3)  # Normalize to 0-1