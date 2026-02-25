"""CLI entry point for the legal document processing pipeline."""

import argparse
import json
import sys

from extractors.factory import extract_document
from cleaners.pipeline import CleaningPipeline
from cleaners.llm_cleaner import LLMCleanerConfig
from parsers.pipeline import ParsingPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Legal Document Processing Pipeline (Phases 1-3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Phase 1 only (extraction)
  python main.py document.pdf --phase 1

  # Phase 1 + 2 (extraction + cleaning)
  python main.py document.pdf --phase 2

  # Phase 1 + 2 + 3 (full pipeline)
  python main.py document.pdf --phase 3

  # Full pipeline, output ONLY the parsed articles JSON
  python main.py document.pdf --phase 3 --articles-only -o articles.json

  # Skip LLM correction in Phase 2
  python main.py document.pdf --phase 3 --no-llm

  # Custom model
  python main.py document.pdf --phase 3 --model gemma3:4b

  # Save output
  python main.py document.pdf --phase 3 -o result.json
        """,
    )

    parser.add_argument("file", help="Path to the document file")
    parser.add_argument("--phase", type=int, default=3, choices=[1, 2, 3],
                        help="Run up to this phase (default: 3)")
    parser.add_argument("--ocr", action="store_true",
                        help="Force OCR extraction for scanned PDFs")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM correction in Phase 2 (rule-based only)")
    parser.add_argument("--model", default="gemma3:4b",
                        help="Ollama model for Phase 2+3 (default: gemma3:4b)")
    parser.add_argument("--timeout", type=int, default=180,
                        help="Timeout per LLM call in seconds (default: 180)")
    parser.add_argument("--output", "-o",
                        help="Save result to JSON file")
    parser.add_argument("--text-only", action="store_true",
                        help="Output only the cleaned text")
    parser.add_argument("--articles-only", action="store_true",
                        help="Output ONLY the parsed articles array from Phase 3 (no extraction/cleaning data)")

    args = parser.parse_args()

    try:
        # ── Phase 1: Extraction ───────────────────────────
        print(f"{'='*60}")
        print(f"📄 Phase 1: Document Extraction")
        print(f"{'='*60}")
        print(f"   File: {args.file}")
        print(f"   OCR: {'Forced' if args.ocr else 'Auto-detect'}")
        print()

        extraction_result = extract_document(args.file, force_ocr=args.ocr)

        print(f"   ✅ Extracted {len(extraction_result.full_text):,} characters")
        print(f"   Format: {extraction_result.file_format.value}")
        print(f"   Pages: {extraction_result.total_pages}")
        print(f"   Scanned: {'Yes' if extraction_result.is_scanned else 'No'}")

        if extraction_result.warnings:
            for w in extraction_result.warnings:
                print(f"   ⚠️  {w}")

        if args.phase < 2:
            if args.text_only:
                print(extraction_result.full_text)
            else:
                _print_preview("Extracted Text", extraction_result.full_text)
            _save_output(args.output, {
                "phase": 1,
                "extraction": extraction_result.model_dump(mode="json"),
            })
            return

        # ── Phase 2: Cleaning ─────────────────────────────
        print(f"\n{'='*60}")
        print(f"🧹 Phase 2: Text Cleaning & Correction")
        print(f"{'='*60}")
        print()

        llm_config = LLMCleanerConfig(
            model=args.model,
            timeout_seconds=args.timeout,
        )

        cleaning_pipeline = CleaningPipeline(
            llm_config=llm_config,
            skip_llm=args.no_llm,
        )

        cleaning_result = cleaning_pipeline.clean(extraction_result.full_text)

        if args.phase < 3:
            if args.text_only:
                print(cleaning_result.cleaned_text)
            else:
                if cleaning_result.french_text:
                    _print_preview("Cleaned French Text", cleaning_result.french_text)
                if cleaning_result.arabic_text:
                    _print_preview("Cleaned Arabic Text", cleaning_result.arabic_text)
            _save_output(args.output, {
                "phase": 2,
                "extraction": extraction_result.model_dump(mode="json"),
                "cleaning": cleaning_result.model_dump(mode="json"),
            })
            return

        # ── Phase 3: Parsing & Metadata Extraction ────────
        parsing_pipeline = ParsingPipeline(
            model=args.model,
            timeout=args.timeout,
        )

        parsing_result = parsing_pipeline.parse(
            french_text=cleaning_result.french_text,
            arabic_text=cleaning_result.arabic_text if cleaning_result.arabic_text else None,
        )

        # ── Output ────────────────────────────────────────
        if args.articles_only:
            # Output ONLY the articles array — clean JSON for database/API
            articles_data = [a.model_dump(mode="json") for a in parsing_result.articles]

            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(articles_data, f, ensure_ascii=False, indent=2, default=str)
                print(f"\n💾 Saved {len(articles_data)} articles to: {args.output}")
            else:
                print(json.dumps(articles_data, ensure_ascii=False, indent=2, default=str))
            return

        # Full output (all phases)
        if not args.text_only:
            print(f"\n{'='*60}")
            print(f"📋 Parsed Articles:")
            print(f"{'='*60}")
            for art in parsing_result.articles:
                print(f"\n  ── Article {art.article_number} (order: {art.article_order}) ──")
                if art.chapter:
                    print(f"  Chapter: {art.chapter}")
                if art.section:
                    print(f"  Section: {art.section}")
                print(f"  Summary: {art.summary_french[:100]}...")
                print(f"  Keywords: {art.keywords}")
                print(f"  Impact: {art.business_impact}")
                print(f"  Obligations: {art.has_obligations} | Penalties: {art.has_penalties}")
                print(f"  Prev: {art.preceding_article_id}")
                print(f"  Next: {art.following_article_id}")

        _save_output(args.output, {
            "phase": 3,
            "extraction": extraction_result.model_dump(mode="json"),
            "cleaning": cleaning_result.model_dump(mode="json"),
            "parsing": {
                "header": parsing_result.header.model_dump(mode="json"),
                "articles": [a.model_dump(mode="json") for a in parsing_result.articles],
                "total_articles": parsing_result.total_articles,
                "llm_calls": parsing_result.llm_calls,
                "model_used": parsing_result.model_used,
                "parsing_duration_seconds": parsing_result.parsing_duration_seconds,
                "warnings": parsing_result.warnings,
            },
        })

    except FileNotFoundError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        raise


def _print_preview(title: str, text: str, max_chars: int = 1500):
    """Print a text preview."""
    print(f"\n{'='*60}")
    print(f"📝 {title} (first {max_chars} chars):")
    print(f"{'='*60}")
    print(text[:max_chars])
    if len(text) > max_chars:
        print(f"\n... ({len(text) - max_chars:,} more characters)")


def _save_output(output_path: str, data: dict):
    """Save result to JSON file if path is provided."""
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n💾 Saved to: {output_path}")


if __name__ == "__main__":
    main()