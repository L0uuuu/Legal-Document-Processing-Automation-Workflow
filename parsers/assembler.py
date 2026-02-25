"""Step 4: Assemble final article objects — merge header, link articles, compute fields."""

import unicodedata
import re
from datetime import datetime, timezone
from typing import Optional

from models.parsing import HeaderResult, ParsedArticle
from parsers.rough_splitter import normalize_chapter


def assemble_articles(
    header: HeaderResult,
    raw_extractions: list[Optional[dict]],
    warnings: list[str],
) -> list[ParsedArticle]:
    """
    Take the header + per-article LLM extractions and produce
    final ParsedArticle objects with all fields populated.
    """
    articles: list[ParsedArticle] = []

    for idx, extraction in enumerate(raw_extractions):
        order = idx + 1

        if extraction is None:
            warnings.append(f"Article {order}: LLM extraction failed, skipped.")
            continue

        article = _build_article(header, extraction, order)
        articles.append(article)

    # Link articles (preceding/following)
    _link_articles(articles, header.parent_law_id)

    # Compute derived fields
    for article in articles:
        article.compute_combined()
        article.compute_hash()

    return articles


def _build_article(
    header: HeaderResult,
    data: dict,
    order: int,
) -> ParsedArticle:
    """Build a ParsedArticle from header + LLM extraction data."""

    # Chapter normalization
    chapter = data.get("chapter") or None
    chapter_normalized = None
    if chapter:
        chapter_normalized = normalize_chapter(chapter)

    # Now timestamp for last_checked
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return ParsedArticle(
        # ── Header fields (shared) ──
        jurisdiction="TUNISIA",
        institution=header.institution,
        institution_primary=header.institution_primary,
        institution_secondary=header.institution_secondary,
        institutions=header.institutions,
        law_type=header.law_type,
        law_number=header.law_number,
        year=header.year,
        title_french=header.title_french,
        title_arabic=header.title_arabic,

        # ── Structural (from LLM) ──
        chapter=chapter,
        chapter_normalized=chapter_normalized,
        section=data.get("section"),
        article_number=data.get("article_number"),
        article_order=order,

        # ── Content (from LLM) ──
        content_french=data.get("content_french", ""),
        content_arabic=data.get("content_arabic", ""),
        summary_french=data.get("summary_french", ""),
        summary_arabic=data.get("summary_arabic", ""),

        # ── Linking ──
        parent_law_id=header.parent_law_id,

        # ── Semantic (from LLM) ──
        keywords=data.get("keywords", []),
        legal_domains=data.get("legal_domains", []),
        business_impact=data.get("business_impact", "LOW"),
        target_audience=data.get("target_audience", []),
        related_laws=data.get("related_laws", []),

        # ── Entities (from LLM) ──
        entity_names=data.get("entity_names", []),
        entity_types=data.get("entity_types", []),
        entity_ids=data.get("entity_ids", []),

        # ── Relations (from LLM) ──
        relation_target_ids=data.get("relation_target_ids", []),
        relation_types=data.get("relation_types", []),

        # ── Booleans (from LLM) ──
        has_obligations=data.get("has_obligations", False),
        has_penalties=data.get("has_penalties", False),
        has_deadlines=data.get("has_deadlines", False),
        has_exceptions=data.get("has_exceptions", False),
        is_abrogation=data.get("is_abrogation", False),
        is_transitional=data.get("is_transitional", False),
        ambiguity_level=data.get("ambiguity_level", "LOW"),

        # ── Status (defaults) ──
        status="ACTIVE",
        status_scope="FULL",
        version=1,
        effective_date=header.effective_date,
        publication_date=header.publication_date,
        repeal_date=None,
        superseded_by_id=None,
        supersedes_id=None,
        last_checked=now_iso,
        next_check=None,

        # ── Gazette ──
        gazette_name=header.gazette_name,
        gazette_number=header.gazette_number,
        gazette_date=header.gazette_date,
        gazette_page=header.gazette_page,
        source_url=None,
    )


def _link_articles(articles: list[ParsedArticle], parent_law_id: Optional[str]) -> None:
    """Set preceding/following article IDs."""
    for idx, article in enumerate(articles):
        art_num = article.article_number or str(article.article_order)

        if idx > 0:
            prev_num = articles[idx - 1].article_number or str(articles[idx - 1].article_order)
            if parent_law_id:
                article.preceding_article_id = f"{parent_law_id}-art-{prev_num}"
            else:
                article.preceding_article_id = f"art-{prev_num}"

        if idx < len(articles) - 1:
            next_num = articles[idx + 1].article_number or str(articles[idx + 1].article_order)
            if parent_law_id:
                article.following_article_id = f"{parent_law_id}-art-{next_num}"
            else:
                article.following_article_id = f"art-{next_num}"