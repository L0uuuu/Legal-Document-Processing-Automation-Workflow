"""Step 3: Per-article AI extraction of structure + metadata."""

import re
import json
import time
import threading
from typing import Optional

import ollama

from models.parsing import RoughArticle, HeaderResult
from parsers.prompts import ARTICLE_EXTRACTION_PROMPT


class ArticleExtractor:
    """Extract structured metadata from each article using LLM."""

    def __init__(self, model: str = "gemma3:4b", timeout: int = 180):
        self.model = model
        self.timeout = timeout
        self.total_calls = 0

    def extract(
        self,
        article: RoughArticle,
        header: HeaderResult,
        article_order: int,
    ) -> Optional[dict]:
        """
        Send one article to the LLM and extract all metadata.
        Returns parsed dict or None if failed.
        """
        prompt = ARTICLE_EXTRACTION_PROMPT.format(
            law_type=header.law_type or "Unknown",
            law_number=header.law_number or "Unknown",
            year=header.year or "Unknown",
            chapter=article.chapter_detected or "Non détecté",
            section=article.section_detected or "Non détectée",
            article_text=article.raw_text,
        )

        chunk_preview = article.raw_text[:70].replace("\n", " ").strip()
        print(f"\n   ┌─ Article {article_order} ({len(article.raw_text)} chars)")
        print(f'   │  "{chunk_preview}..."')
        print(f"   │  Sending to {self.model}...", end="", flush=True)

        start_time = time.time()
        raw_response = self._call_llm(prompt)
        duration = round(time.time() - start_time, 1)
        self.total_calls += 1

        if raw_response is None:
            print(f" ❌ FAILED ({duration}s)")
            return None

        parsed = self._parse_json(raw_response)

        if parsed is None:
            print(f" ❌ JSON PARSE FAILED ({duration}s)")
            # Show what we got for debugging
            preview = raw_response[:150].replace("\n", "\\n")
            print(f'   │  Raw: "{preview}"')
            return None

        print(f" ✅ OK ({duration}s)")

        # Show what was extracted
        art_num = parsed.get("article_number", "?")
        keywords = parsed.get("keywords", [])[:3]
        impact = parsed.get("business_impact", "?")
        print(f"   │  Article: {art_num}")
        print(f"   │  Keywords: {keywords}")
        print(f"   │  Impact: {impact}")
        print(f"   └─")

        return parsed

    def _parse_json(self, text: str) -> Optional[dict]:
        """Extract and parse JSON from LLM response (handles Qwen thinking)."""
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Find the outermost JSON object
        # This handles Qwen's thinking: "blah blah blah {actual json}"
        brace_depth = 0
        json_start = None

        for i, char in enumerate(text):
            if char == "{":
                if brace_depth == 0:
                    json_start = i
                brace_depth += 1
            elif char == "}":
                brace_depth -= 1
                if brace_depth == 0 and json_start is not None:
                    json_candidate = text[json_start:i + 1]
                    try:
                        return json.loads(json_candidate)
                    except json.JSONDecodeError:
                        # Try fixing common issues
                        fixed = self._fix_json(json_candidate)
                        try:
                            return json.loads(fixed)
                        except json.JSONDecodeError:
                            json_start = None
                            continue

        return None

    def _fix_json(self, text: str) -> str:
        """Fix common JSON issues from LLM output."""
        # Remove trailing commas
        text = re.sub(r",\s*}", "}", text)
        text = re.sub(r",\s*]", "]", text)
        # Fix single quotes → double quotes (careful with apostrophes)
        # Only do this if there are no double quotes at all
        if '"' not in text and "'" in text:
            text = text.replace("'", '"')
        return text

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call Ollama with timeout."""
        result = []
        error = []

        def target():
            try:
                response = ollama.chat(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Tu es un expert juridique tunisien. "
                                "Réponds UNIQUEMENT avec du JSON valide. "
                                "Pas d'explication, pas de commentaire. "
                                "Le premier caractère de ta réponse doit être {."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    options={"temperature": 0.1, "num_predict": 4096},
                )
                result.append(response["message"]["content"].strip())
            except Exception as e:
                error.append(str(e))

        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()

        elapsed = 0
        while thread.is_alive() and elapsed < self.timeout:
            thread.join(timeout=3)
            elapsed += 3
            if thread.is_alive():
                print(".", end="", flush=True)

        if thread.is_alive():
            print(f" ⏰ timeout ({self.timeout}s)", end="", flush=True)
            return None

        if error:
            print(f" 💥 {error[0][:60]}", end="", flush=True)
            return None

        return result[0] if result else None