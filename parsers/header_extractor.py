"""Step 1: Extract document header metadata using LLM."""

import re
import json
import time
import threading
from typing import Optional

import ollama

from models.parsing import HeaderResult
from parsers.prompts import HEADER_EXTRACTION_PROMPT


class HeaderExtractor:
    """Extract header metadata from the document preamble using LLM."""

    def __init__(self, model: str = "gemma3:4b", timeout: int = 180):
        self.model = model
        self.timeout = timeout

    def extract(self, text: str) -> HeaderResult:
        """Extract header from the first ~2000 chars of the document."""
        header_text = self._get_header_text(text)

        print(f"   📋 Extracting header metadata...")
        print(f"   │  Header text: {len(header_text)} chars")

        prompt = HEADER_EXTRACTION_PROMPT.format(header_text=header_text)
        raw_response = self._call_llm(prompt)

        if raw_response is None:
            print(f"   │  ⚠️  LLM failed, using regex fallback")
            return self._regex_fallback(header_text)

        parsed = self._parse_json(raw_response)

        if parsed is None:
            print(f"   │  ⚠️  JSON parse failed, using regex fallback")
            return self._regex_fallback(header_text)

        result = self._build_result(parsed, header_text)
        print(f"   │  ✅ {result.law_type} n° {result.law_number} ({result.year})")
        print(f"   │  Title: {(result.title_french or '')[:80]}...")
        print(f"   │  Institutions: {result.institutions}")
        print(f"   │  Source: {result.source_name} n°{result.source_number}")
        print(f"   │  Parent ID: {result.parent_document_id}")

        return result

    def _get_header_text(self, text: str) -> str:
        """Get text before the first article."""
        match = re.search(
            r"(?:A|a)?rticle\s+(premier|1er|1)\s*[\.\-–:]",
            text,
            re.IGNORECASE,
        )
        if match:
            return text[: match.start()].strip()
        return text[:2000].strip()

    def _build_result(self, data: dict, header_text: str) -> HeaderResult:
        """Build HeaderResult from parsed JSON."""
        institutions = data.get("institutions", [])
        institution_str = data.get("institution", "")

        if not institutions and institution_str:
            institutions = [i.strip() for i in institution_str.split(",") if i.strip()]

        primary = institutions[0] if institutions else None
        secondary = institutions[1] if len(institutions) > 1 else None

        law_type = data.get("law_type")
        law_number = data.get("law_number")
        parent_id = self._build_parent_id(law_type, law_number)

        pub_date = data.get("publication_date")
        eff_date = data.get("effective_date") or pub_date

        return HeaderResult(
            jurisdiction="TUNISIA",
            institution=institution_str or (", ".join(institutions) if institutions else None),
            institution_primary=primary,
            institution_secondary=secondary,
            institutions=institutions,
            law_type=law_type,
            law_number=str(law_number) if law_number else None,
            year=data.get("year"),
            title_french=data.get("title_french"),
            title_arabic=data.get("title_arabic"),
            publication_date=pub_date,
            effective_date=eff_date,
            source_name=data.get("source_name"),
            source_number=str(data["source_number"]) if data.get("source_number") else None,
            source_date=data.get("source_date"),
            parent_document_id=parent_id,
            preamble_text=header_text,
        )

    def _build_parent_id(self, law_type: Optional[str], law_number: Optional[str]) -> Optional[str]:
        """Build parent ID like 'tn-loi-66-27' or 'tn-loi-14'."""
        if not law_type or not law_number:
            return None

        type_normalized = (
            law_type.lower()
            .replace(" ", "-")
            .replace("é", "e")
            .replace("è", "e")
            .replace("ê", "e")
            .replace("à", "a")
            .replace("û", "u")
        )
        return f"tn-{type_normalized}-{law_number}"

    def _regex_fallback(self, text: str) -> HeaderResult:
        """Fallback: extract basic header info with regex."""
        law_number = None
        match = re.search(r"n°\s*(\d{2,4}[-–]?\d*)", text, re.IGNORECASE)
        if match:
            law_number = re.sub(r"\s*[-–]\s*", "-", match.group(1))

        law_type = None
        type_patterns = [
            ("Décret-loi", r"[Dd]écret[-\s]loi"),
            ("Loi organique", r"[Ll]oi\s+organique"),
            ("Loi", r"[Ll]oi"),
            ("Décret", r"[Dd]écret"),
            ("Arrêté", r"[Aa]rrêté"),
            ("Circulaire", r"[Cc]irculaire"),
        ]
        for lt, pattern in type_patterns:
            if re.search(pattern, text):
                law_type = lt
                break

        year = None
        if law_number and "-" in law_number:
            parts = law_number.split("-")
            y = int(parts[0])
            year = (1900 + y) if y > 30 else (2000 + y) if y < 100 else y
        elif law_number:
            # Try finding year from date
            match = re.search(r"(\d{4})", text)
            if match:
                year = int(match.group(1))

        parent_id = self._build_parent_id(law_type, law_number)

        return HeaderResult(
            law_type=law_type,
            law_number=law_number,
            year=year,
            parent_document_id=parent_id,
            preamble_text=text,
        )

    def _parse_json(self, text: str) -> Optional[dict]:
        """Extract and parse JSON from LLM response."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Find JSON in response (handles Qwen thinking)
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
                    candidate = text[json_start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        cleaned = re.sub(r",\s*}", "}", candidate)
                        cleaned = re.sub(r",\s*]", "]", cleaned)
                        try:
                            return json.loads(cleaned)
                        except json.JSONDecodeError:
                            json_start = None
                            continue

        return None

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
                    options={"temperature": 0.1, "num_predict": 2048},
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

        print("", flush=True)

        if thread.is_alive():
            print(f"   │  ⏰ Timeout ({self.timeout}s)")
            return None

        if error:
            print(f"   │  💥 {error[0][:80]}")
            return None

        return result[0] if result else None