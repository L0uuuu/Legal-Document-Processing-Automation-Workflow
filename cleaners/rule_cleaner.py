"""
Layer 1: Rule-based text cleaning for OCR output.

Handles deterministic fixes:
- Noise character removal
- Arabic/French text separation
- Quote normalization
- Stray symbol cleanup
- Whitespace normalization
"""

import re
import unicodedata
from dataclasses import dataclass, field


@dataclass
class RuleCleanerResult:
    """Result from rule-based cleaning pass."""
    cleaned_text: str
    french_text: str = ""
    arabic_text: str = ""
    noise_lines_removed: int = 0
    noise_chars_removed: int = 0
    changes: list[dict] = field(default_factory=list)


class RuleCleaner:
    """
    Deterministic, regex-based text cleaner.
    Handles ~60-70% of OCR noise without any LLM.
    """

    # Arabic Unicode ranges
    ARABIC_PATTERN = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")

    # French/Latin text pattern
    LATIN_PATTERN = re.compile(r"[a-zA-ZÀ-ÿœæŒÆ]")

    # Lines that are pure noise: only 1-3 non-alphanumeric characters
    NOISE_LINE_PATTERN = re.compile(r"^[\s\W]{0,5}$")

    # Stray isolated characters/numbers that are OCR artifacts
    STRAY_ARTIFACT_PATTERNS = [
        re.compile(r"\.\s*\d{1,3}\s*$"),          # ". 06" at end of line
        re.compile(r"^\s*[&@#%^*]+\s*$"),          # Lines of just symbols
        re.compile(r"^\s*[A-Z]{1,2}\s*$"),         # Isolated 1-2 uppercase letters like "LS", "N"
        re.compile(r"\s+\d{1,2}-\d{1,2}\s+\d+$"), # "5-2 116" trailing noise
        re.compile(r"^\s*[\.\,\;\:]+\s*$"),        # Lines of just punctuation
    ]

    # Smart quotes and similar character normalization
    CHAR_REPLACEMENTS = {
        "\u201c": '"',   # " left double quote
        "\u201d": '"',   # " right double quote
        "\u2018": "'",   # ' left single quote
        "\u2019": "'",   # ' right single quote
        "\u2013": "-",   # – en dash
        "\u2014": "-",   # — em dash
        "\u00a0": " ",   # non-breaking space
        "\u200f": "",    # right-to-left mark (stray)
        "\u200e": "",    # left-to-right mark (stray)
        "\u200b": "",    # zero-width space
        "\ufeff": "",    # BOM
    }

    def clean(self, text: str) -> RuleCleanerResult:
        """Run all rule-based cleaning passes."""
        original_len = len(text)
        changes = []

        # Pass 1: Character normalization
        text = self._normalize_characters(text)

        # Pass 2: Remove noise lines
        text, noise_count = self._remove_noise_lines(text)

        # Pass 3: Clean stray artifacts within lines
        text = self._clean_stray_artifacts(text)

        # Pass 4: Remove Arabic fragments mixed into French lines
        #         and separate Arabic text
        french_text, arabic_text = self._separate_languages(text)

        # Pass 5: Normalize whitespace
        french_text = self._normalize_whitespace(french_text)
        arabic_text = self._normalize_whitespace(arabic_text)

        # Pass 6: Fix common OCR punctuation issues
        french_text = self._fix_punctuation(french_text)

        # Combine for the cleaned_text (French first, then Arabic)
        cleaned_parts = []
        if french_text.strip():
            cleaned_parts.append(french_text.strip())
        if arabic_text.strip():
            cleaned_parts.append(arabic_text.strip())
        cleaned_text = "\n\n".join(cleaned_parts)

        noise_removed = original_len - len(cleaned_text)

        return RuleCleanerResult(
            cleaned_text=cleaned_text,
            french_text=french_text.strip(),
            arabic_text=arabic_text.strip(),
            noise_lines_removed=noise_count,
            noise_chars_removed=max(0, noise_removed),
            changes=changes,
        )

    def _normalize_characters(self, text: str) -> str:
        """Replace smart quotes, stray Unicode marks, etc."""
        for old, new in self.CHAR_REPLACEMENTS.items():
            text = text.replace(old, new)

        # Normalize Unicode (NFC form — composed characters)
        text = unicodedata.normalize("NFC", text)

        return text

    def _remove_noise_lines(self, text: str) -> tuple[str, int]:
        """Remove lines that are pure noise (symbols, isolated chars)."""
        lines = text.splitlines()
        cleaned_lines = []
        removed_count = 0

        for line in lines:
            stripped = line.strip()

            # Empty lines: keep (they're structural)
            if not stripped:
                cleaned_lines.append("")
                continue

            # Check if line is noise
            if self._is_noise_line(stripped):
                removed_count += 1
                continue

            cleaned_lines.append(line)

        return "\n".join(cleaned_lines), removed_count

    def _is_noise_line(self, line: str) -> bool:
        """Determine if a line is OCR noise."""
        # Pure whitespace/symbols (1-5 chars)
        if self.NOISE_LINE_PATTERN.match(line):
            return True

        # Check against known artifact patterns
        for pattern in self.STRAY_ARTIFACT_PATTERNS:
            if pattern.match(line):
                return True

        # Line has no actual letters (Arabic or Latin), just numbers/symbols
        has_letters = bool(self.ARABIC_PATTERN.search(line)) or bool(self.LATIN_PATTERN.search(line))
        if not has_letters and len(line) < 10:
            return True

        return False

    def _clean_stray_artifacts(self, text: str) -> str:
        """Remove stray artifacts within lines (not entire line removal)."""
        lines = text.splitlines()
        cleaned = []

        for line in lines:
            # Remove trailing noise like ". 06", "& ", stray numbers
            line = re.sub(r"\.\s*\d{1,3}\s*$", "", line)
            line = re.sub(r"\s+[&@#]+\s*$", "", line)
            line = re.sub(r"\s+\d{1,2}-\d{1,2}\s+\d{2,3}\s*$", "", line)

            # Remove isolated single characters surrounded by spaces
            # (but NOT "à", "a", "y", "l'" which are valid French)
            line = re.sub(r"\s+[B-DF-HJ-NP-TV-Z]\s+", " ", line)

            cleaned.append(line)

        return "\n".join(cleaned)

    def _separate_languages(self, text: str) -> tuple[str, str]:
        """
        Separate French and Arabic content.

        Strategy:
        - For each line, determine if it's primarily Arabic or French
        - Remove inline Arabic fragments from French lines (OCR noise)
        - Collect Arabic lines separately
        """
        french_lines = []
        arabic_lines = []

        for line in text.splitlines():
            if not line.strip():
                french_lines.append("")
                arabic_lines.append("")
                continue

            arabic_chars = len(self.ARABIC_PATTERN.findall(line))
            latin_chars = len(self.LATIN_PATTERN.findall(line))
            total = arabic_chars + latin_chars

            if total == 0:
                # No real text, keep in French section (might be numbers/structure)
                french_lines.append(line)
                continue

            arabic_ratio = arabic_chars / total if total > 0 else 0

            if arabic_ratio > 0.6:
                # Primarily Arabic line
                arabic_lines.append(line)
            elif arabic_ratio < 0.15:
                # Primarily French — remove any stray Arabic chars
                cleaned_line = self.ARABIC_PATTERN.sub("", line)
                cleaned_line = re.sub(r"\s{2,}", " ", cleaned_line).strip()
                french_lines.append(cleaned_line)
            else:
                # Mixed line — try to split
                # Keep French part, move Arabic to arabic_lines
                french_part = self.ARABIC_PATTERN.sub("", line)
                arabic_part = self.LATIN_PATTERN.sub("", line)
                french_part = re.sub(r"\s{2,}", " ", french_part).strip()
                arabic_part = re.sub(r"\s{2,}", " ", arabic_part).strip()

                if french_part:
                    french_lines.append(french_part)
                if arabic_part:
                    arabic_lines.append(arabic_part)

        return "\n".join(french_lines), "\n".join(arabic_lines)

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace while preserving paragraph structure."""
        # Replace tabs with spaces
        text = text.replace("\t", " ")

        # Collapse multiple spaces (not newlines)
        text = re.sub(r"[^\S\n]+", " ", text)

        # Remove spaces at start/end of lines
        text = re.sub(r"^ +| +$", "", text, flags=re.MULTILINE)

        # Collapse 3+ blank lines into 2
        text = re.sub(r"\n{4,}", "\n\n\n", text)

        return text

    def _fix_punctuation(self, text: str) -> str:
        """Fix common OCR punctuation errors in French legal text."""
        # Fix "1" → "1er" (premier)
        text = re.sub(r'\b1"\s', "1er ", text)

        # Fix missing space after period in article references
        text = re.sub(r"\.(?=[A-Z])", ". ", text)

        # Fix ".-" pattern (common in articles: "Article 2.-")
        text = re.sub(r"\.\s*-\s*", ".- ", text)

        # Fix spaces before punctuation (French typography)
        text = re.sub(r"\s+([,\.])", r"\1", text)

        # Ensure space after punctuation
        text = re.sub(r"([,\.])(?=[a-zA-ZÀ-ÿ])", r"\1 ", text)

        return text