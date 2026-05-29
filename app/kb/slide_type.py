"""Slide-type hints for mindful PPT / image-PDF teaching."""

from __future__ import annotations

import re

_TITLE_MARKERS = (
    "title or introductory",
    "title slide",
    "introductory slide",
    "cover slide",
    "opening slide",
)


def infer_slide_type(description: str) -> str:
    """Return ``title`` or ``content`` from a slide description caption."""
    low = (description or "").strip().lower()
    if not low:
        return "content"
    if any(m in low for m in _TITLE_MARKERS):
        return "title"
    return "content"


def _extract_title_topic(description: str) -> str:
    """Pull a short topic line from common Protean deck footer patterns."""
    text = description or ""
    # Quoted lines in descriptions often hold the real topic.
    quoted = re.findall(r'"([^"]{8,120})"', text)
    for q in reversed(quoted):
        if "prosure" in q.lower() or "journey" in q.lower() or "feature" in q.lower():
            return q.strip()
    for line in text.splitlines():
        line = line.strip().strip('"')
        if len(line) < 12:
            continue
        low = line.lower()
        if "prosure" in low or ("product" in low and "journey" in low):
            return line
    return "this Protean training deck"


def teaching_source_text(description: str, *, slide_type: str) -> str | None:
    """Return a tutor-focused source_text for title slides; None keeps original."""
    if slide_type != "title":
        return None
    topic = _extract_title_topic(description)
    return (
        "TITLE SLIDE (framing only — do not describe logo, colors, layout, or branding).\n\n"
        f"Topic for the student: {topic}.\n\n"
        "Teach this slide in one or two sentences: what this session covers, then a short "
        "check-in (e.g. \"Ready to continue?\"). When the student confirms, call "
        "`navigate_segment` with direction \"next\" — do not narrate visual design details."
    )


def enrich_segment(segment: dict, description: str = "") -> dict:
    """Set ``slide_type`` and optionally replace ``source_text`` for title slides."""
    desc = description or segment.get("source_text") or ""
    st = segment.get("slide_type") or infer_slide_type(desc)
    segment["slide_type"] = st
    replacement = teaching_source_text(desc, slide_type=st)
    if replacement:
        segment["source_text"] = replacement
    return segment
