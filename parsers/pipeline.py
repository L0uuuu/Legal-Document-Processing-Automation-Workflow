"""Phase 3 Pipeline: Orchestrates header extraction → rough split → article extraction → assembly."""

import time
from typing import Optional

from models.parsing import ParsingResult
from parsers.header_extractor import HeaderExtractor
from parsers.rough_splitter import rough_split
from parsers.article_extractor import ArticleExtractor
from parsers.assembler import assemble_articles


class ParsingPipeline:
    """
    Phase 3: AI-powered document parsing & metadata extraction.

    Step 1: Header extraction (1 LLM call)
    Step 2: Rough split (regex, no LLM)
    Step 3: Per-article extraction (1 LLM call per article)
    Step 4: Assembly & linking (no LLM)
    """

    def __init__(
        self,
        model: str = "gemma3:4b",
        timeout: int = 180,
    ):
        self.model = model
        self.timeout = timeout
        self.header_extractor = HeaderExtractor(model=model, timeout=timeout)
        self.article_extractor = ArticleExtractor(model=model, timeout=timeout)

    def parse(
        self,
        french_text: str,
        arabic_text: Optional[str] = None,
    ) -> ParsingResult:
        """Run the full Phase 3 parsing pipeline."""
        start_time = time.time()
        warnings: list[str] = []

        print(f"\n{'='*60}")
        print(f"🔍 Phase 3: Document Parsing & Metadata Extraction")
        print(f"{'='*60}")
        print(f"   Model: {self.model}")
        print(f"   Input: {len(french_text):,} chars French", end="")
        if arabic_text:
            print(f", {len(arabic_text):,} chars Arabic")
        else:
            print()

        # ── Step 1: Header Extraction ──────────────────────
        print(f"\n   {'─'*50}")
        print(f"   📋 Step 1: Header Extraction")
        print(f"   {'─'*50}")

        header = self.header_extractor.extract(french_text)

        # ── Step 2: Rough Split ────────────────────────────
        print(f"\n   {'─'*50}")
        print(f"   ✂️  Step 2: Rough Article Splitting")
        print(f"   {'─'*50}")

        rough_articles = rough_split(french_text)
        print(f"   │  Found {len(rough_articles)} articles")

        if not rough_articles:
            warnings.append("No articles found in document text.")
            duration = round(time.time() - start_time, 3)
            return ParsingResult(
                header=header,
                articles=[],
                total_articles=0,
                rough_chunks=0,
                llm_calls=1,
                parsing_duration_seconds=duration,
                model_used=self.model,
                warnings=warnings,
            )

        for i, art in enumerate(rough_articles):
            marker = art.article_marker[:30]
            chapter = art.chapter_detected or "—"
            print(f"   │  [{i+1}] {marker:<30} Chapter: {chapter}")

        # ── Step 3: Per-Article AI Extraction ──────────────
        print(f"\n   {'─'*50}")
        print(f"   🤖 Step 3: AI Article Extraction")
        print(f"   {'─'*50}")

        raw_extractions = []
        for idx, article in enumerate(rough_articles):
            order = idx + 1
            extraction = self.article_extractor.extract(
                article=article,
                header=header,
                article_order=order,
            )
            raw_extractions.append(extraction)

        # ── Step 4: Assembly & Linking ─────────────────────
        print(f"\n   {'─'*50}")
        print(f"   🔗 Step 4: Assembly & Linking")
        print(f"   {'─'*50}")

        articles = assemble_articles(header, raw_extractions, warnings)

        # Summary
        duration = round(time.time() - start_time, 3)
        total_calls = 1 + self.article_extractor.total_calls  # 1 for header

        successful = len([a for a in raw_extractions if a is not None])
        failed = len([a for a in raw_extractions if a is None])

        print(f"\n   ✅ Assembled {len(articles)} articles")
        print(f"   │  Successful: {successful}")
        print(f"   │  Failed: {failed}")

        for art in articles:
            art_id = f"{header.parent_law_id}-art-{art.article_number}" if header.parent_law_id else f"art-{art.article_number}"
            print(f"   │  {art_id}: {art.summary_french[:60]}..." if art.summary_french else f"   │  {art_id}")

        print(f"\n{'='*60}")
        print(f"✅ Phase 3 Complete — {duration:.1f}s")
        print(f"{'='*60}")
        print(f"   Articles: {len(articles)}")
        print(f"   LLM calls: {total_calls}")
        print(f"   Duration: {duration:.1f}s")

        if warnings:
            print(f"\n   ⚠️  Warnings:")
            for w in warnings:
                print(f"   - {w}")

        return ParsingResult(
            header=header,
            articles=articles,
            total_articles=len(articles),
            rough_chunks=len(rough_articles),
            llm_calls=total_calls,
            parsing_duration_seconds=duration,
            model_used=self.model,
            warnings=warnings,
        )