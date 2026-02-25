"""Step 2: Regex-based rough splitting of articles."""

import re
import unicodedata
from typing import Optional

from models.parsing import RoughArticle


# Article marker pattern — flexible to handle OCR errors
ARTICLE_PATTERN = re.compile(
    r"(?:A|a)?rticle\s+"
    r"(premier|1er|\d+\s*(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies)?)"
    r"\s*[\.\-–:]",
    re.IGNORECASE,
)

# Chapter pattern
CHAPTER_PATTERN = re.compile(
    r"(?:CHAPITRE|Chapitre)\s+(PREMIER|PREMIÈRE|[IVXLC]+|\d+)"
    r"(?:\s*[-–:]\s*(.+?))?$",
    re.MULTILINE,
)

# Section pattern
SECTION_PATTERN = re.compile(
    r"(?:SECTION|Section)\s+(PREMIÈRE|PREMIERE|PREMIER|[IVXLC]+|\d+)"
    r"(?:\s*[-–:]\s*(.+?))?$",
    re.MULTILINE,
)


def rough_split(text: str) -> list[RoughArticle]:
    """
    Split document text into rough article chunks using regex.

    Returns a list of RoughArticle objects, each containing:
    - The raw article text
    - The detected chapter/section context
    - Start/end positions in the original text
    """
    article_matches = list(ARTICLE_PATTERN.finditer(text))

    if not article_matches:
        return []

    articles: list[RoughArticle] = []
    current_chapter = None
    current_section = None

    for idx, match in enumerate(article_matches):
        # ── Determine content boundaries ──
        content_start = match.start()
        content_end = (
            article_matches[idx + 1].start()
            if idx + 1 < len(article_matches)
            else len(text)
        )

        raw_text = text[content_start:content_end].strip()

        # ── Scan for chapter/section changes ──
        # Look at text BETWEEN previous article end and this article start
        if idx == 0:
            between_text = text[:match.start()]
        else:
            between_text = text[article_matches[idx - 1].end():match.start()]

        # Check between text for chapter/section
        chapter, section = _detect_structure(
            between_text, current_chapter, current_section
        )
        current_chapter = chapter
        current_section = section

        # Also check INSIDE the article text for chapter/section
        # (sometimes they appear inline in OCR output)
        chapter_in, section_in = _detect_structure(raw_text, None, None)
        if chapter_in:
            current_chapter = chapter_in
        if section_in:
            current_section = section_in

        # ── Clean article text: remove structural headers ──
        cleaned_text = _strip_structural_headers(raw_text)

        articles.append(
            RoughArticle(
                article_marker=match.group(0),
                raw_text=cleaned_text,
                chapter_detected=current_chapter,
                section_detected=current_section,
                start_pos=content_start,
                end_pos=content_end,
            )
        )

    return articles


def _detect_structure(
    text: str,
    current_chapter: Optional[str],
    current_section: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """Detect chapter/section headers in a text block."""
    chapter = current_chapter
    section = current_section

    # Check for chapter
    chapter_match = CHAPTER_PATTERN.search(text)
    if chapter_match:
        number = chapter_match.group(1)
        title = chapter_match.group(2).strip() if chapter_match.group(2) else ""
        if title:
            chapter = f"CHAPITRE {number} - {title}"
        else:
            chapter = f"CHAPITRE {number}"

    # Check for section
    section_match = SECTION_PATTERN.search(text)
    if section_match:
        number = section_match.group(1)
        title = section_match.group(2).strip() if section_match.group(2) else ""
        if title:
            section = f"SECTION {number} - {title}"
        else:
            section = f"SECTION {number}"

    return chapter, section


def _strip_structural_headers(text: str) -> str:
    """Remove chapter/section headers from article text."""
    text = CHAPTER_PATTERN.sub("", text)
    text = SECTION_PATTERN.sub("", text)
    # Clean extra blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_chapter(name: str) -> str:
    """
    Normalize chapter name for indexing.
    'CHAPITRE PREMIER - DISPOSITIONS GENERALES'
    → 'chapitre_premier_dispositions_generales'
    """
    normalized = unicodedata.normalize("NFKD", name)
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    normalized = normalized.lower()
    normalized = re.sub(r"[-–:]+", " ", normalized)
    normalized = re.sub(r"[^a-z0-9\s]", "", normalized)
    normalized = re.sub(r"\s+", "_", normalized.strip())
    return normalized