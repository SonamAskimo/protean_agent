"""LangGraph node functions — each returns a *partial* state update dict."""

from __future__ import annotations

from ..chapter_voice_nav import analyze_chapter_voice_nav
from .state import TutorState

# ── keyword sets (lowercase, mixed Hindi / English / Hinglish) ──

# Explicit "move forward" phrases — ONLY multi-word segment/paragraph requests.
# Single words like "next", "skip", "okay" are intentionally excluded to avoid
# accidental advances when the student is just talking.
_CONTINUE = {
    "next slide",
    "next paragraph",
    "next segment",
    "next part",
    "agla paragraph",
    "agle paragraph",
    "agla segment",
    "agle segment",
    "agli line",
}

# Explicit "move backward" phrases.
_GO_BACK = {
    "previous slide",
    "previous paragraph",
    "previous segment",
    "previous part",
    "last paragraph",
    "last segment",
    "last part",
    "pichla paragraph",
    "pichla segment",
    "pichle paragraph",
    "picchla paragraph",
}

# Short-utterance fallbacks. When the whole utterance is ≤ 3 words and contains
# one of these tokens we treat it as navigation. This catches ASR drops like
# "next segment" → "segment", or a curt "Next!".
_SHORT_CONTINUE_TOKENS = {
    "next", "segment", "paragraph", "part",
    "aage", "agla", "agle", "agli", "aagey",
    "skip", "chalo", "continue", "proceed",
}
_SHORT_BACK_TOKENS = {
    "previous", "prev", "back",
    "pichla", "pichle", "picchla", "pichhla",
    "last",
}


def _short_navigation(text: str) -> str | None:
    """Return 'continue'/'go_back' when a curt utterance clearly intends nav."""
    raw = (text or "").lower()
    tokens = [t.strip(" .,!?;:—–-\"'") for t in raw.split()]
    tokens = [t for t in tokens if t]
    if not tokens or len(tokens) > 3:
        return None
    toks = set(tokens)
    # Check back-tokens first so "previous segment" (which also contains
    # 'segment') is classified as go_back, not continue.
    if toks & _SHORT_BACK_TOKENS:
        return "go_back"
    if toks & _SHORT_CONTINUE_TOKENS:
        return "continue"
    return None

_REPEAT = {
    "repeat", "again", "dobara", "phir se",
    "दोबारा", "फिर से", "wapas bolo", "ek baar aur",
}
_SIMPLER = {
    "simple", "easy", "aasan", "samjhao", "simply",
    "आसान", "सरल", "समझाओ", "aur simple",
}
_DONE = {
    # NOTE: do NOT include short acknowledgments like "thank you", "thanks",
    # or "dhanyavaad" here — students routinely use them as mid-lesson
    # acknowledgments ("thank you, that makes sense — let's continue").
    # Treating them as goodbye flips `phase` to `"done"` which used to
    # kill auto-advance and prevent any further quiz / segment progress.
    "bye", "goodbye", "enough", "stop", "exit", "quit",
    "bas", "khatam",
    "बस",
}


def _match(words: set[str], text: str) -> bool:
    low = text.lower()
    return bool(words & set(low.split())) or any(kw in low for kw in words if " " in kw)


def is_segment_navigation(user_input: str) -> bool:
    """True when this utterance advances or goes back a segment (interrupt + fast UI)."""
    u = (user_input or "").strip()
    if not u:
        return False
    low = u.lower()
    if _match(_CONTINUE, low) or _match(_GO_BACK, low):
        return True
    return _short_navigation(low) is not None


# Pure affirmations / backchannels. When the WHOLE utterance is made of these
# (and it is not a question), the student is acknowledging — not asking a doubt.
# Such turns must NOT suppress auto-advance, otherwise the lesson stalls and the
# student is forced to explicitly say "next slide" every time.
_ACK_TOKENS = {
    "yeah", "yea", "yes", "yep", "yup", "ya", "yah",
    "ok", "okay", "k", "kk",
    "mhm", "mmm", "mm", "hmm", "hm", "mhmm", "uh", "uhhuh",
    "got", "it", "i", "you", "understood", "understand",
    "sure", "right", "alright", "cool", "good", "fine", "great", "nice",
    "done", "perfect", "clear",
    "thik", "theek", "hai", "haan", "han", "ha", "ji", "achha", "acha",
    "samajh", "samjh", "gaya", "gya", "gaye", "liya",
    "thanks", "thank", "thx",
    "and", "so", "now", "the", "this", "that",
}


def is_acknowledgment(user_input: str) -> bool:
    """True when the utterance is only affirmation/backchannel words (no question)."""
    u = (user_input or "").strip()
    if not u or "?" in u:
        return False
    tokens = [t.strip(" .,!;:—–-\"'") for t in u.lower().split()]
    tokens = [t for t in tokens if t]
    if not tokens or len(tokens) > 6:
        return False
    return all(t in _ACK_TOKENS for t in tokens)


# ── node functions ──


def classify_intent(state: TutorState) -> dict:
    if state.get("phase") == "greeting":
        return {"intent": "greeting"}

    user = (state.get("user_input") or "").strip()
    if not user:
        return {"intent": "continue"}

    if _match(_CONTINUE, user):
        return {"intent": "continue"}
    if _match(_GO_BACK, user):
        return {"intent": "go_back"}
    short_nav = _short_navigation(user)
    if short_nav:
        return {"intent": short_nav}
    if _match(_REPEAT, user):
        return {"intent": "repeat"}
    if _match(_SIMPLER, user):
        return {"intent": "simpler"}
    if _match(_DONE, user):
        return {"intent": "done"}

    chapters = state.get("chapters") or []
    jumpable = state.get("jumpable_chapter_indices") or []
    if chapters and jumpable:
        _, chapter_nav = analyze_chapter_voice_nav(
            user,
            chapters,
            jumpable,
            int(state.get("selected_chapter_index", -1) or -1),
        )
        if chapter_nav:
            return {"intent": "chapter_jump"}

    return {"intent": "question"}


def advance_segment(state: TutorState) -> dict:
    idx = state.get("current_segment_idx", 0) + 1
    total = state.get("total_segments", 0)
    if idx >= total:
        return {"current_segment_idx": total - 1, "phase": "done"}
    return {"current_segment_idx": idx, "phase": "teaching"}


def go_back_segment(state: TutorState) -> dict:
    return {"current_segment_idx": max(0, state.get("current_segment_idx", 0) - 1)}


def mark_done(state: TutorState) -> dict:
    return {"phase": "done"}


def build_prompt(state: TutorState) -> dict:
    from ..prompts import build_system_prompt
    return {"system_prompt": build_system_prompt(state)}
