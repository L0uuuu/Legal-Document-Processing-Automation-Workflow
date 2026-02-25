"""Data models for text cleaning results."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class CleaningLayer(str, Enum):
    RULE_BASED = "rule_based"
    LLM_CORRECTION = "llm_correction"


class ChangeRecord(BaseModel):
    """Tracks a single change made during cleaning."""
    layer: CleaningLayer
    line_number: Optional[int] = None
    original: str
    corrected: str
    rule_applied: Optional[str] = None  # Which rule or "llm_qwen3"
    confidence: Optional[float] = None  # LLM confidence if applicable


class ChunkCleaningResult(BaseModel):
    """Result of cleaning a single chunk/paragraph."""
    chunk_index: int
    original: str
    cleaned: str
    changes_made: int = 0
    llm_used: bool = False
    llm_confidence: Optional[float] = None


class CleaningResult(BaseModel):
    """Complete result of the text cleaning pipeline."""
    # Source
    original_text: str
    cleaned_text: str

    # Language separation
    french_text: str = ""
    arabic_text: str = ""

    # Metrics
    original_char_count: int = 0
    cleaned_char_count: int = 0
    noise_chars_removed: int = 0
    words_corrected: int = 0
    lines_removed: int = 0

    # Change log
    changes: list[ChangeRecord] = []
    chunks_processed: int = 0
    chunks_corrected_by_llm: int = 0

    # Quality
    cleaning_duration_seconds: float = 0.0
    llm_available: bool = False
    warnings: list[str] = []
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def get_change_summary(self) -> str:
        """Human-readable summary of changes made."""
        lines = [
            f"Cleaning Summary:",
            f"  Characters: {self.original_char_count:,} → {self.cleaned_char_count:,} "
            f"({self.noise_chars_removed:,} noise removed)",
            f"  Lines removed: {self.lines_removed}",
            f"  Words corrected by LLM: {self.words_corrected}",
            f"  Chunks processed: {self.chunks_processed} "
            f"({self.chunks_corrected_by_llm} needed LLM)",
            f"  Duration: {self.cleaning_duration_seconds:.2f}s",
        ]
        return "\n".join(lines)