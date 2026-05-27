"""Cheap source-language hints from script ratios (Latin vs Devanagari).

Used for prompt branching (English vs Hindi-forward Hinglish). Not a full language-ID model.
"""

from __future__ import annotations

import re
from typing import Literal

SourceLang = Literal["en", "hi", "mixed", "unknown"]

# Devanagari block (Hindi and related scripts)
_DEVA_RE = re.compile(r"[\u0900-\u097F]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def detect_source_lang(text: str) -> SourceLang:
    """
    Classify excerpt language profile for tutoring prompts.

    - en: mostly Latin (English/Latin-alphabet source).
    - hi: mostly Devanagari — Hindi-forward spoken style in prompts.
    - mixed: substantial both scripts.
    - unknown: too little signal.
    """
    if not text or not text.strip():
        return "unknown"

    deva = len(_DEVA_RE.findall(text))
    latin = len(_LATIN_RE.findall(text))
    total = deva + latin
    if total < 12:
        return "unknown"

    r_dev = deva / total
    r_lat = latin / total

    if r_lat >= 0.55 and r_dev <= 0.35:
        return "en"
    if r_dev >= 0.55 and r_lat <= 0.35:
        return "hi"
    if r_dev > 0.2 and r_lat > 0.2:
        return "mixed"
    if r_dev > r_lat:
        return "hi"
    if r_lat > r_dev:
        return "en"
    return "mixed"


def document_primary_lang(text: str) -> SourceLang:
    """Single label for the whole extract (session defaults)."""
    return detect_source_lang(text)


def resolve_segment_lang(override: str, detected: SourceLang) -> SourceLang:
    """User override from upload form, or per-segment detection when ``auto``."""
    o = (override or "auto").strip().lower()
    if o in ("en", "hi", "mixed"):
        return o  # type: ignore[return-value]
    return detected
