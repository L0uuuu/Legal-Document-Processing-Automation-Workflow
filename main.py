"""CLI entry point for document text extraction."""

import argparse
import json
import sys

from extractors.factory import extract_document


def main():
    parser = argparse.ArgumentParser(
        description="Extract text from legal documents (PDF, DOCX, TXT)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py document.pdf
  python main.py document.pdf --ocr
  python main.py document.pdf --output result.json
  python main.py document.docx --text-only
        """,
    )

    parser.add_argument("file", help="Path to the document file")
    parser.add_argument("--ocr", action="store_true", help="Force OCR extraction (for scanned PDFs)")
    parser.add_argument("--output", "-o", help="Save result to JSON file")
    parser.add_argument("--text-only", action="store_true", help="Output only the extracted text")

    args = parser.parse_args()

    try:
        print(f"📄 Processing: {args.file}")
        print(f"   Method: {'OCR (forced)' if args.ocr else 'Auto-detect'}")
        print()

        result = extract_document(args.file, force_ocr=args.ocr)

        # Text-only mode
        if args.text_only:
            print(result.get_text())
            return

        # Print summary
        print(f"✅ Extraction Complete")
        print(f"   Format:       {result.file_format.value}")
        print(f"   Method:       {result.extraction_method.value}")
        print(f"   Pages:        {result.total_pages}")
        print(f"   Characters:   {len(result.full_text):,}")
        print(f"   Duration:     {result.extraction_duration_seconds}s")
        print(f"   Scanned:      {'Yes' if result.is_scanned else 'No'}")
        print(f"   Has Arabic:   {'Yes' if result.has_arabic_content else 'No'}")
        print(f"   Has French:   {'Yes' if result.has_french_content else 'No'}")

        if result.avg_confidence is not None:
            print(f"   OCR Confidence: {result.avg_confidence:.1%}")

        if result.warnings:
            print(f"\n⚠️  Warnings:")
            for warning in result.warnings:
                print(f"   - {warning}")

        # Preview
        print(f"\n{'='*60}")
        print(f"📝 Text Preview (first 1000 chars):")
        print(f"{'='*60}")
        print(result.full_text[:1000])
        if len(result.full_text) > 1000:
            print(f"\n... ({len(result.full_text) - 1000:,} more characters)")

        # Save to JSON
        if args.output:
            output_data = result.model_dump(mode="json")
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"\n💾 Saved to: {args.output}")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()