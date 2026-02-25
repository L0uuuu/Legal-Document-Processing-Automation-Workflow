"""
Layer 2: LLM-based text correction using Gemma3:4b via Ollama.

Uses the `ollama` Python library for reliable connections.
"""

import re
import time
import threading
from dataclasses import dataclass, field
from typing import Optional

import ollama

from cleaners.prompts import (
    FRENCH_CORRECTION_PROMPT,
    ARABIC_CORRECTION_PROMPT,
)


@dataclass
class LLMCleanerConfig:
    """Configuration for the LLM cleaner."""
    model: str = "gemma3:4b"
    temperature: float = 0.1
    max_chunk_chars: int = 800
    timeout_seconds: int = 180
    max_retries: int = 2
    max_length_change: float = 0.50
    max_change_ratio: float = 0.70


@dataclass
class LLMChunkResult:
    """Result of LLM correction for a single chunk."""
    original: str
    corrected: str
    was_modified: bool = False
    was_rejected: bool = False
    rejection_reason: Optional[str] = None
    duration_seconds: float = 0.0
    token_count: int = 0


@dataclass
class LLMCleanerResult:
    """Complete result from LLM cleaning pass."""
    cleaned_french: str
    cleaned_arabic: str
    chunks_processed: int = 0
    chunks_modified: int = 0
    chunks_rejected: int = 0
    chunks_unchanged: int = 0
    total_tokens: int = 0
    total_duration_seconds: float = 0.0
    llm_available: bool = True
    warnings: list[str] = field(default_factory=list)


class LLMCleaner:
    """
    Uses Gemma3:4b via the ollama Python library to fix OCR errors
    that rule-based cleaning can't handle.
    """

    SYSTEM_PROMPT = (
        "You are an OCR error corrector. "
        "Respond ONLY with the corrected text. "
        "Do NOT explain. Do NOT add any commentary. "
        "Output ONLY the corrected text, nothing else."
    )

    def __init__(self, config: Optional[LLMCleanerConfig] = None):
        self.config = config or LLMCleanerConfig()
        self._is_available: Optional[bool] = None

    def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        if self._is_available is not None:
            return self._is_available

        try:
            models = ollama.list()
            model_names = [m.model for m in models.models]
            self._is_available = any(
                self.config.model in name for name in model_names
            )

            if self._is_available:
                self._log(f"✅ Connected! Model '{self.config.model}' found.")
            else:
                self._log(f"⚠️  Model '{self.config.model}' not found.")
                self._log(f"   Available models: {model_names}")
                self._is_available = self._try_pull_model()

        except Exception as e:
            self._log(f"❌ Cannot connect to Ollama: {e}")
            self._log(f"   Make sure Ollama is running: ollama serve")
            self._is_available = False

        return self._is_available

    def _try_pull_model(self) -> bool:
        """Try a simple query to see if model auto-loads."""
        try:
            ollama.chat(
                model=self.config.model,
                messages=[{"role": "user", "content": "test"}],
                options={"temperature": 0.1, "num_predict": 1},
            )
            self._log(f"   ✅ Model loaded successfully!")
            return True
        except Exception as e:
            self._log(f"   ❌ Failed: {e}")
            return False

    def clean(self, french_text: str, arabic_text: str) -> LLMCleanerResult:
        """Clean both French and Arabic text using LLM correction."""
        start_time = time.time()

        if not self.is_available():
            return LLMCleanerResult(
                cleaned_french=french_text,
                cleaned_arabic=arabic_text,
                llm_available=False,
                warnings=[
                    f"Ollama not available or model '{self.config.model}' not found. "
                    f"Skipping LLM correction."
                ],
            )

        total_chunks = 0
        total_modified = 0
        total_rejected = 0
        total_unchanged = 0
        total_tokens = 0
        warnings = []

        cleaned_french = french_text
        if french_text.strip():
            self._log_header("French text")
            cleaned_french, stats = self._process_text(
                french_text, language="french"
            )
            total_chunks += stats["chunks"]
            total_modified += stats["modified"]
            total_rejected += stats["rejected"]
            total_unchanged += stats["unchanged"]
            total_tokens += stats["tokens"]
            warnings.extend(stats["warnings"])

        cleaned_arabic = arabic_text
        if arabic_text.strip():
            self._log_header("Arabic text")
            cleaned_arabic, stats = self._process_text(
                arabic_text, language="arabic"
            )
            total_chunks += stats["chunks"]
            total_modified += stats["modified"]
            total_rejected += stats["rejected"]
            total_unchanged += stats["unchanged"]
            total_tokens += stats["tokens"]
            warnings.extend(stats["warnings"])

        total_duration = round(time.time() - start_time, 3)

        self._log(
            f"\n   📊 LLM Summary: {total_chunks} chunks | "
            f"✏️  {total_modified} fixed | "
            f"⏭️  {total_unchanged} unchanged | "
            f"❌ {total_rejected} rejected | "
            f"🔤 {total_tokens} tokens | "
            f"⏱️  {total_duration:.1f}s total"
        )

        return LLMCleanerResult(
            cleaned_french=cleaned_french,
            cleaned_arabic=cleaned_arabic,
            chunks_processed=total_chunks,
            chunks_modified=total_modified,
            chunks_rejected=total_rejected,
            chunks_unchanged=total_unchanged,
            total_tokens=total_tokens,
            total_duration_seconds=total_duration,
            llm_available=True,
            warnings=warnings,
        )

    def _process_text(
        self, text: str, language: str
    ) -> tuple[str, dict]:
        """Process text by splitting into chunks and correcting each."""
        chunks = self._split_into_chunks(text)
        corrected_chunks = []
        stats = {
            "chunks": len(chunks),
            "modified": 0,
            "rejected": 0,
            "unchanged": 0,
            "tokens": 0,
            "warnings": [],
        }

        self._log(
            f"   Split into {len(chunks)} chunks "
            f"(max {self.config.max_chunk_chars} chars each)"
        )

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                corrected_chunks.append(chunk)
                continue

            chunk_num = i + 1
            chunk_preview = chunk[:70].replace("\n", " ").strip()
            self._log(
                f"\n   ┌─ Chunk {chunk_num}/{len(chunks)} "
                f"({len(chunk)} chars)"
            )
            self._log(f'   │  "{chunk_preview}..."')
            self._log(f"   │  Sending to {self.config.model}...", end="")

            result = self._correct_chunk(chunk, language)
            stats["tokens"] += result.token_count

            if result.was_rejected:
                corrected_chunks.append(result.original)
                stats["rejected"] += 1
                self._log(f" ❌ REJECTED ({result.duration_seconds:.1f}s)")
                self._log(f"   │  Reason: {result.rejection_reason}")
                if result.rejection_reason:
                    stats["warnings"].append(
                        f"Chunk {chunk_num} ({language}): {result.rejection_reason}"
                    )
            elif result.was_modified:
                corrected_chunks.append(result.corrected)
                stats["modified"] += 1
                self._log(
                    f" ✏️  FIXED ({result.duration_seconds:.1f}s, "
                    f"{result.token_count} tokens)"
                )
                self._show_diff(result.original, result.corrected)
            else:
                corrected_chunks.append(result.corrected)
                stats["unchanged"] += 1
                self._log(
                    f" ✅ OK ({result.duration_seconds:.1f}s, "
                    f"{result.token_count} tokens)"
                )

            self._log(f"   └─")

        return "\n\n".join(corrected_chunks), stats

    def _show_diff(self, original: str, corrected: str):
        """Show a compact diff of what changed."""
        orig_words = original.split()
        corr_words = corrected.split()

        changes_shown = 0
        max_changes_to_show = 5

        for idx in range(min(len(orig_words), len(corr_words))):
            if orig_words[idx] != corr_words[idx]:
                if changes_shown < max_changes_to_show:
                    self._log(
                        f'   │  ✏️  "{orig_words[idx]}" → "{corr_words[idx]}"'
                    )
                    changes_shown += 1

        if changes_shown == 0 and len(orig_words) != len(corr_words):
            self._log(
                f"   │  Word count: {len(orig_words)} → {len(corr_words)}"
            )

        remaining = (
            sum(1 for a, b in zip(orig_words, corr_words) if a != b)
            - changes_shown
        )
        if remaining > 0:
            self._log(f"   │  ... and {remaining} more changes")

    def _split_into_chunks(self, text: str) -> list[str]:
        """Split text into chunks for LLM processing."""
        paragraphs = re.split(r"\n\s*\n", text)

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) > self.config.max_chunk_chars:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                if len(para) > self.config.max_chunk_chars:
                    sentence_chunks = self._split_long_paragraph(para)
                    chunks.extend(sentence_chunks)
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _split_long_paragraph(self, text: str) -> list[str]:
        """Split a long paragraph on sentence boundaries."""
        sentences = re.split(r"(?<=\.)\s+(?=[A-ZÀ-Ý])", text)

        chunks = []
        current = ""

        for sentence in sentences:
            if len(current) + len(sentence) > self.config.max_chunk_chars:
                if current.strip():
                    chunks.append(current.strip())
                current = sentence
            else:
                current = current + " " + sentence if current else sentence

        if current.strip():
            chunks.append(current.strip())

        return chunks

    def _correct_chunk(self, chunk: str, language: str) -> LLMChunkResult:
        """Send a single chunk to the LLM for correction, then validate."""
        start_time = time.time()

        if language == "arabic":
            prompt = ARABIC_CORRECTION_PROMPT.format(chunk=chunk)
        else:
            prompt = FRENCH_CORRECTION_PROMPT.format(chunk=chunk)

        corrected, token_count = self._call_ollama(prompt)
        duration = round(time.time() - start_time, 3)

        # If LLM call failed, keep original
        if corrected is None:
            return LLMChunkResult(
                original=chunk,
                corrected=chunk,
                was_modified=False,
                was_rejected=True,
                rejection_reason="LLM call failed or timed out",
                duration_seconds=duration,
                token_count=0,
            )

        # Validate the correction
        validation = self._validate_correction(chunk, corrected)

        if not validation["accepted"]:
            return LLMChunkResult(
                original=chunk,
                corrected=chunk,
                was_modified=False,
                was_rejected=True,
                rejection_reason=validation["reason"],
                duration_seconds=duration,
                token_count=token_count,
            )

        was_modified = corrected.strip() != chunk.strip()

        return LLMChunkResult(
            original=chunk,
            corrected=corrected.strip(),
            was_modified=was_modified,
            duration_seconds=duration,
            token_count=token_count,
        )

    def _validate_correction(self, original: str, corrected: str) -> dict:
        """
        Validate that the LLM correction is reasonable.

        3 checks:
        1. Length change — reject if text grew/shrank by more than 50%
        2. Legal markers — article numbers, chapter headers must be preserved
        3. Hallucination — reject if LLM added commentary instead of correcting
        """
        original_clean = original.strip()
        corrected_clean = corrected.strip()

        if len(original_clean) == 0:
            return {"accepted": True, "reason": None}

        # ── Check 1: Length change ────────────────────────
        orig_len = len(original_clean)
        corr_len = len(corrected_clean)
        length_ratio = abs(orig_len - corr_len) / max(orig_len, 1)

        if length_ratio > self.config.max_length_change:
            return {
                "accepted": False,
                "reason": f"Text length changed too much ({orig_len} → {corr_len}, "
                          f"{length_ratio:.0%} change). Possible hallucination.",
            }

        # ── Check 2: Legal markers preserved ──────────────
        legal_markers = [
            r"Article\s+\d+",
            r"Article\s+premier",
            r"CHAPITRE",
            r"SECTION",
        ]

        for pattern in legal_markers:
            orig_matches = re.findall(pattern, original_clean, re.IGNORECASE)
            corr_matches = re.findall(pattern, corrected_clean, re.IGNORECASE)

            if len(orig_matches) != len(corr_matches):
                return {
                    "accepted": False,
                    "reason": f"Legal marker count changed: '{pattern}' "
                              f"({len(orig_matches)} → {len(corr_matches)})",
                }

        # Check article numbers specifically
        orig_numbers = re.findall(r"Article\s+(\d+)", original_clean)
        corr_numbers = re.findall(r"Article\s+(\d+)", corrected_clean)

        if orig_numbers != corr_numbers:
            return {
                "accepted": False,
                "reason": f"Article numbers changed: {orig_numbers} → {corr_numbers}",
            }

        # ── Check 3: Hallucination detection ──────────────
        hallucination_patterns = [
            r"^(Here|Voici|The corrected|Le texte corrigé|I have|J'ai)",
            r"(Note:|NB:|Explanation:|Remarque:)",
            r"(I corrected|I fixed|I changed|J'ai corrigé|J'ai modifié)",
            r"^(Okay|Let me|Let's|I need|First|Hmm|Wait|Alright|Sure)",
            r"(OCR error|OCR correction|the original|l'original)",
        ]

        for pattern in hallucination_patterns:
            if re.search(pattern, corrected_clean, re.IGNORECASE):
                return {
                    "accepted": False,
                    "reason": f"LLM added commentary instead of just correcting text.",
                }

        return {"accepted": True, "reason": None}

    def _call_ollama(self, prompt: str) -> tuple[Optional[str], int]:
        """Call ollama.chat() with timeout using threading."""
        for attempt in range(self.config.max_retries + 1):
            result = []
            error = []

            def target():
                try:
                    messages = [
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ]

                    response = ollama.chat(
                        model=self.config.model,
                        messages=messages,
                        options={
                            "temperature": self.config.temperature,
                            "num_predict": 2048,
                        },
                    )
                    result.append(response)
                except Exception as e:
                    error.append(str(e))

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()

            elapsed = 0
            dot_interval = 3
            while thread.is_alive() and elapsed < self.config.timeout_seconds:
                thread.join(timeout=dot_interval)
                elapsed += dot_interval
                if thread.is_alive():
                    self._log(".", end="")

            if thread.is_alive():
                self._log(f" ⏰ timeout ({self.config.timeout_seconds}s)!", end="")
                if attempt < self.config.max_retries:
                    self._log(f" ⟳ retry {attempt + 1}...", end="")
                    continue
                return None, 0

            if error:
                self._log(f" 💥 {error[0][:60]}", end="")
                if attempt < self.config.max_retries:
                    self._log(f" ⟳ retry {attempt + 1}...", end="")
                    continue
                return None, 0

            if result:
                response = result[0]
                raw_text = response["message"]["content"].strip()

                token_count = response.get("eval_count", 0)
                if token_count == 0:
                    token_count = len(raw_text) // 4

                # Remove markdown code blocks if wrapped
                response_text = re.sub(r"^```\w*\n?", "", raw_text)
                response_text = re.sub(r"\n?```$", "", response_text)
                response_text = response_text.strip()

                if response_text:
                    return response_text, token_count

                self._log(f" ⚠️  empty response", end="")

            if attempt < self.config.max_retries:
                self._log(f" ⟳ retry {attempt + 1}...", end="")
                continue

        return None, 0

    def _log(self, message: str, end: str = "\n"):
        print(message, end=end, flush=True)

    def _log_header(self, text_type: str):
        self._log(f"\n   {'─'*50}")
        self._log(f"   🔤 Processing: {text_type}")
        self._log(f"   {'─'*50}")