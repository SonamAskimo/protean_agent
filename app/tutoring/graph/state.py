from __future__ import annotations

from typing import NotRequired, TypedDict


class SegmentDict(TypedDict, total=False):
    segment_id: str
    pages: list[int]
    source_text: NotRequired[str]
    """Primary excerpt text for this teaching unit."""
    source_lang: NotRequired[str]
    """en | hi | mixed | unknown — guides spoken style."""
    slide_index: NotRequired[int]
    slide_url: NotRequired[str]
    slide_file: NotRequired[str]


class TutorState(TypedDict, total=False):
    # --- set once at session init, never mutated ---
    segments: list[SegmentDict]
    total_segments: int
    chapter_title: str
    selected_chapter_index: int
    chapters: list[dict]  # TOC / chapter titles from the PDF
    jumpable_chapter_indices: list[int]  # indices with extracted segments (voice jump + tools)
    chapter_preview: str  # first ~1500 chars for opening summary
    subject: str  # free-text: "Physics", "Sales training", etc.
    content_language: str  # auto | en | hi | mixed — override for all segments
    content_type: str  # pdf | ppt — ppt enables slide UI + Live vision

    # --- mutated by the graph on every turn ---
    current_segment_idx: int
    phase: str       # "greeting" | "teaching" | "done"
    user_input: str  # latest transcribed student utterance
    intent: str      # classified intent for this turn
    system_prompt: str  # dynamically built prompt sent to the LLM
