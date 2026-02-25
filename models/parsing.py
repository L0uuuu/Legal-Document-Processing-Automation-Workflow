"""Data models for Phase 3: Document structure parsing & metadata extraction."""

import hashlib
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field


class HeaderResult(BaseModel):
    """Metadata extracted from the document header — shared across all articles."""
    jurisdiction: str = "TUNISIA"
    institution: Optional[str] = None
    institution_primary: Optional[str] = None
    institution_secondary: Optional[str] = None
    institutions: list[str] = []
    law_type: Optional[str] = None
    law_number: Optional[str] = None
    year: Optional[int] = None
    title_french: Optional[str] = None
    title_arabic: Optional[str] = None
    publication_date: Optional[str] = None
    effective_date: Optional[str] = None
    gazette_name: Optional[str] = None
    gazette_number: Optional[str] = None
    gazette_date: Optional[str] = None
    gazette_page: Optional[int] = None
    parent_law_id: Optional[str] = None
    preamble_text: str = ""


class RoughArticle(BaseModel):
    """A rough article chunk from regex splitting (Step 2)."""
    article_marker: str = ""
    raw_text: str = ""
    chapter_detected: Optional[str] = None
    section_detected: Optional[str] = None
    start_pos: int = 0
    end_pos: int = 0


class ParsedArticle(BaseModel):
    """Final parsed article with all metadata (Step 3 + Step 4)."""
    # ── Structural ──
    jurisdiction: str = "TUNISIA"
    institution: Optional[str] = None
    institution_primary: Optional[str] = None
    institution_secondary: Optional[str] = None
    institutions: list[str] = []
    law_type: Optional[str] = None
    law_number: Optional[str] = None
    year: Optional[int] = None
    title_french: Optional[str] = None
    title_arabic: Optional[str] = None
    chapter: Optional[str] = None
    chapter_normalized: Optional[str] = None
    section: Optional[str] = None
    article_number: Optional[str] = None
    article_order: int = 0

    # ── Content ──
    content_french: str = ""
    content_arabic: str = ""
    content_combined: str = ""
    summary_french: str = ""
    summary_arabic: str = ""
    search_content: str = ""
    content_hash_sha256: str = ""

    # ── Linking ──
    parent_law_id: Optional[str] = None
    preceding_article_id: Optional[str] = None
    following_article_id: Optional[str] = None

    # ── Semantic ──
    keywords: list[str] = []
    legal_domains: list[str] = []
    business_impact: str = "LOW"
    target_audience: list[str] = []
    related_laws: list[str] = []

    # ── Entities ──
    entity_names: list[str] = []
    entity_types: list[str] = []
    entity_ids: list[str] = []

    # ── Relations ──
    relation_target_ids: list[str] = []
    relation_types: list[str] = []

    # ── Boolean flags ──
    has_obligations: bool = False
    has_penalties: bool = False
    has_deadlines: bool = False
    has_exceptions: bool = False
    is_abrogation: bool = False
    is_transitional: bool = False
    ambiguity_level: str = "LOW"

    # ── Status (defaults) ──
    status: str = "ACTIVE"
    status_scope: str = "FULL"
    version: int = 1
    effective_date: Optional[str] = None
    publication_date: Optional[str] = None
    repeal_date: Optional[str] = None
    superseded_by_id: Optional[str] = None
    supersedes_id: Optional[str] = None
    last_checked: Optional[str] = None
    next_check: Optional[str] = None

    # ── Gazette ──
    gazette_name: Optional[str] = None
    gazette_number: Optional[str] = None
    gazette_date: Optional[str] = None
    gazette_page: Optional[int] = None
    source_url: Optional[str] = None

    def compute_hash(self):
        """Compute SHA-256 hash of combined content."""
        content = self.content_french + self.content_arabic
        self.content_hash_sha256 = hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()

    def compute_combined(self):
        """Build content_combined and search_content."""
        parts = []
        if self.content_french:
            parts.append(self.content_french)
        if self.content_arabic:
            parts.append(self.content_arabic)
        self.content_combined = " ".join(parts)

        search_parts = []
        if self.law_type and self.law_number:
            search_parts.append(f"{self.law_type} {self.law_number}")
        if self.title_french:
            search_parts.append(self.title_french)
        search_parts.extend(self.keywords)
        if self.content_french:
            search_parts.append(self.content_french[:200])
        if self.content_arabic:
            search_parts.append(self.content_arabic[:200])
        self.search_content = ", ".join(search_parts)


class ParsingResult(BaseModel):
    """Complete result from Phase 3."""
    header: HeaderResult
    articles: list[ParsedArticle] = []
    total_articles: int = 0
    rough_chunks: int = 0
    llm_calls: int = 0
    parsing_duration_seconds: float = 0.0
    model_used: str = ""
    warnings: list[str] = []