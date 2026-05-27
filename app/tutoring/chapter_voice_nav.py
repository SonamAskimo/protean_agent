"""Resolve spoken chapter / module / unit jumps without relying on the LLM."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

# 1-based spoken ordinals → 0-based chapter index
_EN_WORD_TO_INT: dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}

_ROMAN_MAP = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}


def _int_to_roman(n: int) -> str:
    if n <= 0 or n >= 4000:
        return ""
    vals = [
        (1000, "m"), (900, "cm"), (500, "d"), (400, "cd"),
        (100, "c"), (90, "xc"), (50, "l"), (40, "xl"),
        (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i"),
    ]
    out = []
    for v, s in vals:
        while n >= v:
            out.append(s)
            n -= v
    return "".join(out)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]+", " ", (s or "").lower())).strip()


def _roman_to_int(tok: str) -> int | None:
    t = (tok or "").strip().lower()
    if not t or not re.fullmatch(r"[ivxlcdm]+", t):
        return None
    total = 0
    prev = 0
    for ch in reversed(t):
        v = _ROMAN_MAP.get(ch, 0)
        if v < prev:
            total -= v
        else:
            total += v
            prev = v
    return total if 0 < total < 5000 else None


def _parse_1based_index(tok: str) -> int | None:
    t = (tok or "").strip().lower()
    if not t:
        return None
    if t.isdigit():
        n = int(t)
        return n if n > 0 else None
    ri = _roman_to_int(t)
    if ri is not None:
        return ri
    return _EN_WORD_TO_INT.get(t)


def _to_zero_based(one_based: int | None) -> int | None:
    if one_based is None or one_based < 1:
        return None
    return one_based - 1


def _looks_like_content_question(low: str) -> bool:
    if re.search(
        r"\b(what|why|how|when|where|explain|meaning|meanings|tell\s+me|describe|"
        r"difference|define|summarize|summary|outline|compare|quiz|test|example|"
        r"kya|kaun|kaise|kyun|kyon|matlab|samjhao|samjhado)\b",
        low,
        re.I,
    ):
        if re.search(r"\bwhat\s+chapter\s+are\s+we\b", low, re.I):
            return False
        if _has_nav_cue(low):
            return False
        return True
    return False


def _has_chapter_scope(low: str) -> bool:
    return bool(
        re.search(
            r"\b(chapter|chapters|module|modules|unit|units|lesson|lessons|अध्याय)\b",
            low,
            re.I,
        )
    )


def _has_nav_cue(low: str) -> bool:
    patterns = [
        r"\b(go|jump|switch|move)\s+to\b",
        r"\b(let'?s|lets)\s+(go\s+to|start|begin|open)\b",
        r"\blet\s+us\s+(go\s+to|start|begin)\b",
        r"\b(open|teach|padhao|chalo|chaliye|shuru|शुरू)\s+(?:the\s+)?(chapter|module|unit|lesson)\b",
        r"\b(start|begin)\s+(?:with|from|teaching)\s+(?:the\s+)?(chapter|module|unit|lesson)\b",
        r"\b(chapter|module|unit|lesson)\b[^.]{0,48}\b(start|shuru|padhao|chalo|begin)\b",
        r"\b(next|agla|agle|agli|pichla|picchla|pichle|previous|first|last|pehla|pahla|akhri|aakhri)\s+"
        r"(chapter|module|unit|lesson|अध्याय)\b",
    ]
    return any(re.search(p, low, re.I) for p in patterns)


def _relative_chapter_delta(low: str) -> int | None:
    if re.search(
        r"\b(next|agla|agle|agli|अगला)\s+(chapter|module|unit|lesson|अध्याय)\b",
        low,
        re.I,
    ):
        return 1
    if re.search(
        r"\b(previous|pichla|picchla|pichle|prev)\s+(chapter|module|unit|lesson|अध्याय)\b",
        low,
        re.I,
    ):
        return -1
    if re.search(r"\b(first|pehla|pahla|पहला)\s+(chapter|module|unit|lesson)\b", low, re.I):
        return -999  # sentinel: first
    if re.search(r"\b(last|akhri|aakhri|अंतिम)\s+(chapter|module|unit|lesson)\b", low, re.I):
        return 999  # sentinel: last
    return None


_NUM_PATTERNS = [
    (
        "with_cue",
        r"\b(?:go|jump|switch|move)\s+to\s+(?:the\s+)?"
        r"(chapter|module|unit|lesson)\s*[:\#]?\s*([0-9]+|[ivxlcdm]+|[a-z]{3,20})\b",
    ),
    (
        "with_cue",
        r"\b(?:start|begin|open|teach|padhao|chalo|shuru)\s+(?:with\s+)?(?:the\s+)?"
        r"(chapter|module|unit|lesson)\s*[:\#]?\s*([0-9]+|[ivxlcdm]+|[a-z]{3,20})\b",
    ),
    (
        "bare",
        r"\b(chapter|module|unit|lesson)\s*(?:number|no\.?)?\s*[:\#]?\s*"
        r"([0-9]+|[ivxlcdm]+)\b",
    ),
    (
        "with_cue",
        r"\b(chapter|module|unit)\s+([0-9]+|[ivxlcdm]+|[a-z]{3,20})\s+"
        r"(?:please|now|start|shuru|padhao|chalo)\b",
    ),
]


def _match_numbered(low: str) -> tuple[str, str, int] | None:
    """Return (kind, match_kind_flag, spoken_number) or None.

    ``spoken_number`` is what the user said (the number literal in the utterance),
    NOT a zero-based index.  Caller decides whether to treat it as an ordinal
    or to match it against chapter titles like "Module N".
    """
    for flag, pat in _NUM_PATTERNS:
        m = re.search(pat, low, re.I)
        if not m:
            continue
        kind = (m.group(1) or "").lower()
        tok = m.group(2) or ""
        n = _parse_1based_index(tok)
        if n is None:
            continue
        return (kind, flag, n)
    return None


_TITLE_KINDS = ("module", "chapter", "unit", "lesson")


def _title_number_index(
    kind: str,
    spoken: int,
    chapters: list[dict[str, Any]],
    jumpable: list[int],
) -> int | None:
    """Find a chapter whose title contains the spoken number after a label word.

    The student's spoken ``<kind>`` is IGNORED on purpose: books commonly label
    sections "Module N", "Chapter N" or "Unit N" even when the student says
    "chapter 3" (they mean the unit labeled 3, not the 3rd entry in the list).
    We match the same number behind any of the label words so "chapter 3" and
    "module 3" both resolve to a title like "Module 3 – …".

    Falls back to a loose "number appears in title" match when exactly one
    jumpable chapter contains the number.
    """
    if spoken is None or spoken <= 0:
        return None
    roman = _int_to_roman(spoken)
    value_variants = [str(spoken)]
    if roman:
        value_variants.append(roman)

    for kind_word in _TITLE_KINDS:
        for val in value_variants:
            pat = rf"\b{kind_word}s?\s*[-:#]?\s*{re.escape(val)}\b"
            hits: list[int] = []
            for i in jumpable:
                if i >= len(chapters):
                    continue
                title = str(chapters[i].get("title") or "").lower()
                if re.search(pat, title, re.I):
                    hits.append(i)
            if len(hits) == 1:
                return hits[0]

    loose_pat = rf"(?<!\d){spoken}(?!\d)"
    loose_hits: list[int] = []
    for i in jumpable:
        if i >= len(chapters):
            continue
        title = str(chapters[i].get("title") or "").lower()
        if re.search(loose_pat, title):
            loose_hits.append(i)
    if len(loose_hits) == 1:
        return loose_hits[0]

    return None


def _fuzzy_title_index(low: str, chapters: list[dict[str, Any]], jumpable: list[int]) -> int | None:
    blob = _norm(low)
    if len(blob) < 5:
        return None
    best_i: int | None = None
    best_r = 0.72
    for i in jumpable:
        if i >= len(chapters):
            continue
        title = _norm(str(chapters[i].get("title") or ""))
        if len(title) < 4:
            continue
        if title in blob and len(title) >= 6:
            return i
        r = SequenceMatcher(None, blob, title).ratio()
        if r > best_r:
            best_r = r
            best_i = i
    return best_i


def analyze_chapter_voice_nav(
    text: str,
    chapters: list[dict[str, Any]],
    jumpable: list[int],
    current_chapter_index: int,
) -> tuple[int | None, bool]:
    """Return (resolved_0based_index_or_none, treat_as_chapter_navigation).

    When the second value is True, upstream should avoid same-segment resume hints
    and may hand off to the LLM + ``jump_to_chapter`` if the first value is None.
    """
    raw = (text or "").strip()
    if not raw or not jumpable:
        return (None, False)

    low = raw.lower()
    if _looks_like_content_question(low):
        return (None, False)

    rel = _relative_chapter_delta(low)
    if rel is not None:
        order = sorted(jumpable)
        cur = current_chapter_index if current_chapter_index in order else order[0]
        pos = order.index(cur)
        if rel == -999:
            return (order[0], True)
        if rel == 999:
            return (order[-1], True)
        npos = pos + rel
        if 0 <= npos < len(order):
            return (order[npos], True)
        return (None, True)

    matched = _match_numbered(low)
    if matched is not None:
        kind, flag, spoken = matched

        implicit_unit = kind in ("module", "unit")
        has_cue = _has_nav_cue(low)
        if flag == "bare" and not has_cue and not implicit_unit:
            # Bare "chapter 4" without a cue → let the LLM decide (might be a meta question).
            return (None, False)

        title_hit = _title_number_index(kind, spoken, chapters, jumpable)
        if title_hit is not None:
            return (title_hit, True)

        ordinal = _to_zero_based(spoken)
        if ordinal is not None and ordinal in jumpable:
            return (ordinal, True)
        return (None, True)

    if _has_nav_cue(low) and _has_chapter_scope(low):
        fuzzy = _fuzzy_title_index(low, chapters, jumpable)
        if fuzzy is not None:
            return (fuzzy, True)
        return (None, True)

    return (None, False)
