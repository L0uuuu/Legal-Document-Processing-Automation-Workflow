"""CLI entry point for the legal document processing pipeline."""

import argparse
import json
import sys

from extractors.factory import extract_document
from cleaners.pipeline import CleaningPipeline
from cleaners.llm_cleaner import LLMCleanerConfig


def main():
    parser = argparse.ArgumentParser(
        description="Legal Document Processing Pipeline (Phases 1-2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Phase 1 only (extraction)
  python main.py document.pdf --phase 1

  # Phase 1 + 2 (extraction + cleaning)
  python main.py document.pdf --phase 2

  # Full pipeline with OCR
  python main.py scanned.pdf --phase 2 --ocr

  # Skip LLM correction (rule-based only)
  python main.py document.pdf --phase 2 --no-llm

  # Custom model
  python main.py document.pdf --phase 2 --model gemma3:4b

  # Save output
  python main.py document.pdf --phase 2 -o result.json
        """,
    )

    parser.add_argument("file", help="Path to the document file")
    parser.add_argument("--phase", type=int, default=2, choices=[1, 2],
                        help="Run up to this phase (default: 2)")
    parser.add_argument("--ocr", action="store_true",
                        help="Force OCR extraction for scanned PDFs")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM correction (rule-based cleaning only)")
    parser.add_argument("--model", default="gemma3:4b",
                        help="Ollama model for correction (default: gemma3:4b)")
    parser.add_argument("--timeout", type=int, default=180,
                        help="Timeout per chunk in seconds (default: 180)")
    parser.add_argument("--output", "-o",
                        help="Save result to JSON file")
    parser.add_argument("--text-only", action="store_true",
                        help="Output only the cleaned text")

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

        pipeline = CleaningPipeline(
            llm_config=llm_config,
            skip_llm=args.no_llm,
        )

        cleaning_result = pipeline.clean(extraction_result.full_text)

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