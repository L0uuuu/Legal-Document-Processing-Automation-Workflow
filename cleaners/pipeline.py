"""
Phase 2 Pipeline: Orchestrates rule-based cleaning → LLM correction → validation.

Usage:
    from cleaners.pipeline import CleaningPipeline

    pipeline = CleaningPipeline()
    result = pipeline.clean(raw_text)
    print(result.cleaned_text)
"""

import time
from typing import Optional

from models.cleaning import (
    ChangeRecord,
    CleaningLayer,
    CleaningResult,
)
from cleaners.rule_cleaner import RuleCleaner
from cleaners.llm_cleaner import LLMCleaner, LLMCleanerConfig


class CleaningPipeline:
    """
    Two-layer cleaning pipeline:
        Layer 1: Rule-based (regex) — fast, deterministic
        Layer 2: LLM (Gemma3:4b via ollama library) — contextual word correction
    """

    def __init__(
        self,
        llm_config: Optional[LLMCleanerConfig] = None,
        skip_llm: bool = False,
    ):
        self.rule_cleaner = RuleCleaner()
        self.skip_llm = skip_llm

        if not skip_llm:
            if llm_config is None:
                llm_config = LLMCleanerConfig(model="gemma3:4b")
            self.llm_cleaner = LLMCleaner(config=llm_config)
        else:
            self.llm_cleaner = None

    def clean(self, text: str) -> CleaningResult:
        """Run the full cleaning pipeline."""
        start_time = time.time()
        changes: list[ChangeRecord] = []
        warnings: list[str] = []

        original_text = text
        original_char_count = len(text)

        # ── Layer 1: Rule-Based Cleaning ─────────────────────
        print("🧹 Layer 1: Rule-based cleaning...")
        rule_result = self.rule_cleaner.clean(text)

        if rule_result.noise_lines_removed > 0:
            changes.append(
                ChangeRecord(
                    layer=CleaningLayer.RULE_BASED,
                    original=f"[{rule_result.noise_lines_removed} noise lines]",
                    corrected="[removed]",
                    rule_applied="noise_line_removal",
                )
            )

        print(
            f"   ✓ Removed {rule_result.noise_lines_removed} noise lines, "
            f"{rule_result.noise_chars_removed} noise characters"
        )
        print(
            f"   ✓ Separated: {len(rule_result.french_text):,} chars French, "
            f"{len(rule_result.arabic_text):,} chars Arabic"
        )

        # ── Layer 2: LLM Correction ──────────────────────────
        llm_available = False
        chunks_processed = 0
        chunks_corrected = 0
        words_corrected = 0

        french_text = rule_result.french_text
        arabic_text = rule_result.arabic_text

        if self.llm_cleaner and not self.skip_llm:
            print(f"\n🤖 Layer 2: LLM correction")
            print(f"   Model: {self.llm_cleaner.config.model}")
            print(f"   Timeout: {self.llm_cleaner.config.timeout_seconds}s per chunk")
            print(f"   Max chunk size: {self.llm_cleaner.config.max_chunk_chars} chars")
            print(f"   Checking Ollama connection... ", end="", flush=True)

            if self.llm_cleaner.is_available():
                llm_result = self.llm_cleaner.clean(
                    french_text=rule_result.french_text,
                    arabic_text=rule_result.arabic_text,
                )

                llm_available = llm_result.llm_available
                french_text = llm_result.cleaned_french
                arabic_text = llm_result.cleaned_arabic
                chunks_processed = llm_result.chunks_processed
                chunks_corrected = llm_result.chunks_modified
                warnings.extend(llm_result.warnings)

                if llm_result.chunks_rejected > 0:
                    warnings.append(
                        f"{llm_result.chunks_rejected} chunk(s) rejected by validation "
                        f"(kept original text)"
                    )

                words_corrected = self._count_word_changes(
                    rule_result.french_text + rule_result.arabic_text,
                    french_text + arabic_text,
                )

                if llm_result.chunks_modified > 0:
                    changes.append(
                        ChangeRecord(
                            layer=CleaningLayer.LLM_CORRECTION,
                            original=f"[{llm_result.chunks_modified} chunks modified]",
                            corrected=f"[{words_corrected} words corrected]",
                            rule_applied=f"llm_{self.llm_cleaner.config.model}",
                        )
                    )
            else:
                print(f"\n   💡 To fix: run 'ollama serve' then 'ollama pull {self.llm_cleaner.config.model}'")
                warnings.append("Ollama not available. Skipping LLM correction.")
        else:
            print("\n🔇 Layer 2: Skipped (LLM disabled via --no-llm)")

        # ── Assemble Final Result ─────────────────────────────
        cleaned_parts = []
        if french_text.strip():
            cleaned_parts.append(french_text.strip())
        if arabic_text.strip():
            cleaned_parts.append(arabic_text.strip())
        cleaned_text = "\n\n".join(cleaned_parts)

        duration = round(time.time() - start_time, 3)

        result = CleaningResult(
            original_text=original_text,
            cleaned_text=cleaned_text,
            french_text=french_text.strip(),
            arabic_text=arabic_text.strip(),
            original_char_count=original_char_count,
            cleaned_char_count=len(cleaned_text),
            noise_chars_removed=rule_result.noise_chars_removed,
            words_corrected=words_corrected,
            lines_removed=rule_result.noise_lines_removed,
            changes=changes,
            chunks_processed=chunks_processed,
            chunks_corrected_by_llm=chunks_corrected,
            cleaning_duration_seconds=duration,
            llm_available=llm_available,
            warnings=warnings,
        )

        # Final summary
        print(f"\n{'='*60}")
        print(f"✅ Phase 2 Complete — {duration:.1f}s")
        print(f"{'='*60}")
        print(result.get_change_summary())

        if warnings:
            print(f"\n⚠️  Warnings:")
            for w in warnings:
                print(f"   - {w}")

        return result

    def _count_word_changes(self, original: str, corrected: str) -> int:
        """Count approximate number of words that changed."""
        orig_words = original.lower().split()
        corr_words = corrected.lower().split()
        orig_set = set(orig_words)
        corr_set = set(corr_words)
        return len(orig_set.symmetric_difference(corr_set))