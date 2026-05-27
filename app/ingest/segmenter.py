from __future__ import annotations

import re
from dataclasses import dataclass

from .pdf_extract import Paragraph


@dataclass(frozen=True)
class SegmentInput:
    segment_id: str
    pages: list[int]
    source_text: str


# Split sentences using ONLY full stops.
# We intentionally do NOT split on "!", "?" or newlines to avoid producing
# disconnected fragments from noisy PDF extraction.
#
# Devanagari danda (।) and English full stop (.) are sentence boundaries.
_SENT_RE = re.compile(r"(?<=[।.])\s+")

# Target segment size: aim for 2-3 sentences (more natural teaching units).
_MIN_SENTENCES = 2
_MAX_SENTENCES = 3



def _split_into_sentences(text: str) -> list[str]:
    raw = _SENT_RE.split(text)
    # Keep small fragments only if they are real-ish sentences.
    return [s.strip() for s in raw if s.strip() and len(s.strip()) >= 3]


def build_segments(
    paragraphs: list[Paragraph], *, max_segments: int = 120
) -> list[SegmentInput]:
    """
    Splits text into sentence-level segments.

    Segment boundaries are built to follow a stable ~2-3 sentence teaching unit.
    This reduces "disconnected" segments that can happen when we only rely on
    character thresholds on noisy PDF extractions.
    """
    all_sentences: list[tuple[str, int]] = []  # (sentence, page)
    for p in paragraphs:
        t = p.text.strip()
        if not t:
            continue
        for sent in _split_into_sentences(t):
            all_sentences.append((sent, p.page))

    segments: list[SegmentInput] = []
    sid = 0
    buf: list[str] = []
    buf_pages: list[int] = []
    buf_len = 0

    def flush() -> None:
        nonlocal sid, buf, buf_pages, buf_len
        if not buf:
            return
        sid += 1
        segments.append(
            SegmentInput(
                segment_id=f"s_{sid:04d}",
                pages=sorted(set(buf_pages)),
                source_text=" ".join(buf).strip(),
            )
        )
        buf = []
        buf_pages = []
        buf_len = 0

    for sent, page in all_sentences:
        if len(segments) >= max_segments:
            break

        sent_len = len(sent)

        # Start new segment.
        if not buf:
            buf = [sent]
            buf_pages = [page]
            buf_len = sent_len

            # If a single sentence is extremely long, still allow it to be
            # its own segment (can't split it further with full-stop rule).
            if sent_len >= 2000:
                flush()
            continue

        # Enforce max sentences: flush before adding if we already have enough.
        if len(buf) >= _MAX_SENTENCES:
            flush()
            buf = [sent]
            buf_pages = [page]
            buf_len = sent_len
            if sent_len >= 2000:
                flush()
            continue

        # No character budget: rely purely on 2–3 sentence grouping.
        if len(buf) >= _MIN_SENTENCES and len(buf) >= _MAX_SENTENCES:
            flush()
            buf = [sent]
            buf_pages = [page]
            buf_len = sent_len
            if sent_len >= 2000:
                flush()
            continue

        # Add sentence normally.
        buf.append(sent)
        buf_pages.append(page)
        buf_len += sent_len + 1

        # Flush as soon as we reach the max sentences.
        if len(buf) >= _MAX_SENTENCES:
            flush()

    if len(segments) < max_segments:
        flush()

    return segments[:max_segments]
