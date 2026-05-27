"""
Gemini Live API bridge for the PDF tutor — no LiveKit.

Browser WebSocket  <->  this module  <->  Google Gemini BidiGenerateContent
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import websockets
from fastapi import WebSocket, WebSocketDisconnect
from openai import OpenAI

from ..core.paths import SESSIONS as SESSIONS_DIR
from ..core.paths import UPLOADS as UPLOADS_DIR
from ..tutoring.chapter_voice_nav import analyze_chapter_voice_nav
from ..tutoring.graph.nodes import is_segment_navigation
from ..tutoring.prompts import build_segment_injection
from ..tutoring.tutor_llm import GraphRunner

logger = logging.getLogger("gemini-live")
logger.setLevel(logging.INFO)
# uvicorn's default root handler runs at WARNING, so add our own INFO-level
# stderr handler (with propagate=False to avoid double-logging).
if not any(getattr(h, "_gemini_live_handler", False) for h in logger.handlers):
    _h = logging.StreamHandler(sys.stderr)
    _h.setLevel(logging.INFO)
    _h.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
    _h._gemini_live_handler = True  # type: ignore[attr-defined]
    logger.addHandler(_h)
    logger.propagate = False

GEMINI_WS_BASE = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)
OUTPUT_SAMPLE_RATE = 24000


def cleanup_session_uploads(session_id: str) -> None:
    """Remove upload artifacts for a teaching session (PPT tree under uploads/<id>/ and uploads/<id>.pdf).

    Uses short retries to handle Windows file locks (browser may still hold slide JPEGs briefly).
    Safe to call twice.
    """
    sid = (session_id or "").strip()
    if not sid:
        return
    subdir = UPLOADS_DIR / sid
    pdf = UPLOADS_DIR / f"{sid}.pdf"
    last_err: BaseException | None = None
    for attempt in range(4):
        try:
            if subdir.is_dir():
                shutil.rmtree(subdir)
            if pdf.is_file():
                pdf.unlink()
            logger.info("cleaned uploads for session %s", sid)
            return
        except OSError as exc:
            last_err = exc
            time.sleep(0.2 * (attempt + 1))

    if subdir.is_dir():
        shutil.rmtree(subdir, ignore_errors=True)
    try:
        if pdf.is_file():
            pdf.unlink(missing_ok=True)  # py3.8+ — 3.12 ok
    except OSError:
        pass
    if last_err:
        logger.warning(
            "uploads cleanup for session %s fell back to ignore_errors (%s)", sid, last_err
        )



_POLL_INTERVAL = 0.5
_COVERED_IDLE_S = 1.0
_FALLBACK_IDLE_S = 5.0
_RESUME_HINT_MAX_CHARS = 900
_RESUME_HINT_DELAY_S = 1.5

_RAG_EMBEDDING_MODEL = "text-embedding-3-small"
_RAG_MAX_CHARS_PER_CHUNK = 900
_RAG_OVERLAP_CHARS = 180
_RAG_TOP_K = 3
_RAG_QUERY_EMBED_TIMEOUT_S = 1.2


def _normalize_segments(raw: list[dict]) -> list[dict]:
    out: list[dict] = []
    for s in raw:
        d = dict(s)
        txt = (d.get("source_text") or "").strip()
        d["source_text"] = txt
        d.setdefault("source_lang", "unknown")
        out.append(d)
    return out


def _clean_for_rag(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-zA-Z0-9]{3,}", (text or "").lower())}


def _chunk_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    text = _clean_for_rag(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    out: list[str] = []
    i = 0
    step = max(1, chunk_size - overlap)
    while i < len(text):
        piece = text[i : i + chunk_size].strip()
        if piece:
            out.append(piece)
        i += step
    return out


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _l2(vec: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


class GeminiLiveTutorSession:
    """One tutoring call: orchestrates LangGraph state + Gemini Live WebSocket."""

    def __init__(self, session: dict, *, api_key: str) -> None:
        self._session = session
        self._api_key = api_key
        self._room_name = session.get("room_name") or ""
        _sid = str(session.get("session_id") or "").strip()
        self._session_id = _sid or (self._room_name or "").removeprefix("tutor-").strip()
        self._content_type = str(session.get("content_type") or "pdf").strip().lower()
        slides_raw = session.get("slides_dir") or ""
        self._slides_dir = Path(slides_raw) if slides_raw else (UPLOADS_DIR / self._session_id / "slides")
        self._slide_vision_after_tool: int | None = None

        segs = _normalize_segments(session.get("segments") or [])
        self._rag_segs = _normalize_segments(session.get("rag_segments") or [])
        self._chapters: list[dict] = list(session.get("chapters") or [])
        self._chapter_segments_map: dict[int, list[dict]] = {}
        for row in session.get("chapter_segments") or []:
            try:
                idx = int(row.get("chapter_index"))
            except Exception:
                continue
            self._chapter_segments_map[idx] = _normalize_segments(row.get("segments") or [])

        initial_state = {
            "segments": segs,
            "total_segments": len(segs),
            "chapter_title": session.get("chapter_title", ""),
            "selected_chapter_index": session.get("selected_chapter_index", -1),
            "chapters": self._chapters,
            "jumpable_chapter_indices": sorted(self._chapter_segments_map.keys()),
            "chapter_preview": session.get("chapter_preview", ""),
            "subject": session.get("subject", ""),
            "content_language": session.get("content_language", "auto"),
            "content_type": self._content_type,
            "current_segment_idx": 0,
            "phase": "greeting",
            "user_input": "",
            "intent": "",
            "system_prompt": "",
        }
        self._runner = GraphRunner(initial_state)
        self._initial_prompt = self._runner.process_turn("")

        self._browser_ws: WebSocket | None = None
        self._gemini_ws: Any = None
        self._gemini_ready = asyncio.Event()
        self._closed = False

        self._auto_task: asyncio.Task | None = None
        self._last_activity_time = time.monotonic()
        # Estimated wall-clock time at which the browser will finish playing the
        # audio we've forwarded so far. Gemini streams audio much faster than
        # the browser plays it, so we use this (not chunk-arrival time) as the
        # "user just heard the agent" marker for auto-advance.
        self._audio_play_until_ts = time.monotonic()
        self._turn_lock = asyncio.Lock()
        # Buffer for streaming user-speech transcription. Gemini emits
        # inputTranscription.text as small chunks ("next" then " seg" then
        # "ment"); we accumulate them and dispatch one _process_user_input
        # call with the full sentence when the user turn ends.
        self._pending_user_text = ""
        self._pending_user_dispatched_for_turn = False
        self._logged_first_server_content = False
        # Wall-clock timestamp of the last tool-driven segment navigation.
        # The transcription path (_process_user_input) checks this so it doesn't
        # double-advance when the model both calls navigate_segment *and* the
        # final transcript "next segment" arrives a beat later.
        self._last_tool_nav_ts = 0.0
        self._segment_rev = 0
        self._last_sent_segment_idx: int | None = None
        self._assistant_segment_text = ""

        self._openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        self._openai = OpenAI(api_key=self._openai_key) if self._openai_key else None
        self._rag_ready = False
        self._rag_chunks: list[dict] = []
        self._rag_vecs: list[list[float]] = []
        self._rag_norms: list[float] = []

        self._last_user_speech_ts = 0.0
        self._user_speaking_now = False

        self._model = (
            (os.getenv("GEMINI_MODEL") or "").strip() or "gemini-3.1-flash-live-preview"
        )
        self._voice = (os.getenv("GEMINI_VOICE") or "").strip() or "Kore"
        self._temperature = float(os.getenv("GEMINI_TEMPERATURE") or "0.25")
        self._silence_ms = int(os.getenv("GEMINI_END_OF_SPEECH_SILENCE_MS") or "700")

    async def _send_browser(self, payload: dict) -> None:
        if self._browser_ws and not self._closed:
            try:
                await self._browser_ws.send_json(payload)
            except Exception:
                logger.debug("browser send failed", exc_info=True)

    async def _push_segment_metadata(self, *, force: bool = False) -> None:
        idx = int(self._runner.state.get("current_segment_idx", 0) or 0)
        if not force and self._last_sent_segment_idx == idx:
            return
        self._segment_rev += 1
        raw_ch = self._runner.state.get("selected_chapter_index", -1)
        chapter_idx = int(raw_ch) if isinstance(raw_ch, (int, float, str)) else -1
        chapter_title = ""
        if 0 <= chapter_idx < len(self._chapters):
            chapter_title = str(self._chapters[chapter_idx].get("title") or "")
        payload: dict[str, Any] = {
            "type": "segment",
            "seg": idx,
            "rev": self._segment_rev,
            "chapter_index": chapter_idx,
            "chapter_title": chapter_title,
            "content_type": self._content_type,
        }
        segments = self._runner.state.get("segments") or []
        if 0 <= idx < len(segments):
            seg = segments[idx]
            if seg.get("slide_url"):
                payload["slide_url"] = seg["slide_url"]
        await self._send_browser(payload)
        self._last_sent_segment_idx = idx

    async def _send_slide_vision(self, segment_idx: int) -> None:
        """Send the current slide JPEG to Gemini Live (PPT sessions only)."""
        if self._content_type != "ppt" or self._closed:
            return
        segments = self._runner.state.get("segments") or []
        if not (0 <= segment_idx < len(segments)):
            return
        seg = segments[segment_idx]
        fname = str(seg.get("slide_file") or "").strip()
        if not fname:
            url = str(seg.get("slide_url") or "")
            if "/slides/" in url:
                fname = url.rsplit("/slides/", 1)[-1].split("?")[0]
        if not fname:
            logger.warning("ppt vision: no slide_file for seg=%d", segment_idx)
            return
        path = self._slides_dir / fname
        if not path.is_file():
            logger.warning("ppt vision: missing file %s", path)
            return
        import base64

        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        # Gemini Live deprecated media_chunks; static slide JPEGs use `video`.
        await self._gemini_send_await(
            {
                "realtimeInput": {
                    "video": {
                        "mimeType": "image/jpeg",
                        "data": b64,
                    }
                }
            }
        )
        logger.info("ppt vision sent: seg=%d file=%s (%d bytes b64)", segment_idx, fname, len(b64))

    # ── RAG ──

    async def _build_rag_index(self) -> None:
        if not self._openai:
            return
        segs = self._rag_segs or (self._runner.state.get("segments") or [])
        chunk_rows: list[dict] = []
        for idx, seg in enumerate(segs):
            seg_text = (seg.get("source_text") or "").strip()
            if not seg_text:
                continue
            for part in _chunk_text(
                seg_text,
                chunk_size=_RAG_MAX_CHARS_PER_CHUNK,
                overlap=_RAG_OVERLAP_CHARS,
            ):
                chunk_rows.append({"segment_idx": idx, "text": part})
        if not chunk_rows:
            return
        try:
            resp = await asyncio.to_thread(
                self._openai.embeddings.create,
                model=_RAG_EMBEDDING_MODEL,
                input=[c["text"] for c in chunk_rows],
            )
        except Exception:
            logger.exception("RAG build failed")
            return
        self._rag_vecs = [list(d.embedding) for d in resp.data]
        self._rag_norms = [_l2(v) for v in self._rag_vecs]
        self._rag_chunks = chunk_rows
        self._rag_ready = True
        logger.info("RAG ready: %d chunks", len(chunk_rows))

    async def _retrieve_rag_context(self, query: str) -> list[dict]:
        if not self._rag_ready or not query.strip():
            return []
        qtok = _tokens(query)
        if not qtok:
            return []
        lex_scored: list[tuple[float, int]] = []
        for i, row in enumerate(self._rag_chunks):
            ttok = _tokens(row["text"])
            if not ttok:
                continue
            overlap = len(qtok & ttok)
            if overlap:
                lex_scored.append((overlap / max(1, len(qtok)), i))
        lex_scored.sort(key=lambda x: x[0], reverse=True)
        lex_scored = lex_scored[:12]
        sem_scores: dict[int, float] = {}
        if self._openai and self._rag_vecs and lex_scored:
            try:
                q_resp = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._openai.embeddings.create,
                        model=_RAG_EMBEDDING_MODEL,
                        input=[query],
                    ),
                    timeout=_RAG_QUERY_EMBED_TIMEOUT_S,
                )
                q = list(q_resp.data[0].embedding)
                qn = _l2(q)
                if qn > 0:
                    for _, i in lex_scored:
                        dn = self._rag_norms[i]
                        if dn:
                            sem_scores[i] = _dot(q, self._rag_vecs[i]) / (qn * dn)
            except Exception:
                pass
        final: list[tuple[float, int]] = []
        for lex, i in lex_scored:
            score = (0.65 * lex) + (0.35 * max(0.0, sem_scores.get(i, 0.0)))
            if score >= 0.1:
                final.append((score, i))
        final.sort(key=lambda x: x[0], reverse=True)
        out: list[dict] = []
        for s, i in final[:_RAG_TOP_K]:
            row = self._rag_chunks[i]
            out.append({"score": s, "segment_idx": row["segment_idx"], "text": row["text"]})
        return out

    # ── Gemini wire protocol ──

    async def _gemini_send_await(self, message: dict) -> None:
        if self._gemini_ws and not self._closed:
            await self._gemini_ws.send(json.dumps(message))

    def _navigate_segment_tool_declaration(self) -> dict:
        return {
            "name": "navigate_segment",
            "description": (
                "Move to the next or previous segment when the student explicitly asks "
                "to navigate within the CURRENT chapter (e.g. \"next segment\", "
                "\"next paragraph\", \"previous segment\", \"go back\", \"skip\", "
                "\"aage chalo\", \"pichla segment\", \"agla paragraph\"). Call this "
                "IMMEDIATELY as the first thing in your response — BEFORE speaking "
                "the acknowledgement. "
                "Do NOT use this for chapter switches (use jump_to_chapter)."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "direction": {
                        "type": "STRING",
                        "enum": ["next", "previous"],
                        "description": (
                            "Use 'next' to move forward one segment, "
                            "'previous' to move back."
                        ),
                    },
                },
                "required": ["direction"],
            },
        }

    def _gemini_tool_declarations(self) -> list[dict]:
        return [
            {
                "name": "retrieve_chapter_context",
                "description": (
                    "Search the full uploaded chapter for excerpts relevant to the "
                    "student's question."
                ),
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "query": {
                            "type": "STRING",
                            "description": "Natural-language search query",
                        }
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "jump_to_chapter",
                "description": "Jump to chapter by 0-based index.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "chapter_index": {"type": "INTEGER"},
                    },
                    "required": ["chapter_index"],
                },
            },
            self._navigate_segment_tool_declaration(),
        ]

    def _build_setup_message(self) -> dict:
        return {
            "setup": {
                "model": f"models/{self._model}",
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "temperature": self._temperature,
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {"voiceName": self._voice},
                        },
                    },
                },
                "systemInstruction": {"parts": [{"text": self._initial_prompt}]},
                "tools": [
                    {
                        "functionDeclarations": self._gemini_tool_declarations(),
                    }
                ],
                "inputAudioTranscription": {},
                "outputAudioTranscription": {},
                "realtimeInputConfig": {
                    "automaticActivityDetection": {
                        "disabled": False,
                        "silenceDurationMs": self._silence_ms,
                        "prefixPaddingMs": 300,
                    },
                    "activityHandling": "START_OF_ACTIVITY_INTERRUPTS",
                },
            },
        }

    async def _inject_text(self, text: str) -> None:
        trimmed = (text or "").strip()
        if not trimmed:
            return
        await self._gemini_send_await({"realtimeInput": {"text": trimmed}})

    async def _inject_lesson_context(
        self, *, prev_idx: int, new_idx: int, prev_phase: str, new_phase: str
    ) -> None:
        payload = build_segment_injection(
            self._runner.state,
            prev_segment_idx=prev_idx,
            new_segment_idx=new_idx,
            prev_phase=prev_phase,
            new_phase=new_phase,
        )
        await self._inject_text(payload)
        await self._push_segment_metadata(force=True)
        if self._content_type == "ppt":
            await self._send_slide_vision(new_idx)

    # ── tool handlers ──

    def _bump_audio_play_end(self, b64_audio: str) -> None:
        """Track when the browser is estimated to finish playing the audio so far.

        Gemini streams PCM 16-bit 24 kHz mono (= 48000 bytes/sec). A base64
        string of length L decodes to ~3*L/4 bytes (minus padding).
        """
        if not b64_audio:
            return
        padding = b64_audio.count("=")
        byte_count = max(0, (len(b64_audio) * 3) // 4 - padding)
        if byte_count <= 0:
            return
        duration = byte_count / 48000.0
        now = time.monotonic()
        if self._audio_play_until_ts < now:
            # Browser queue is empty; account for AudioContext startup latency.
            self._audio_play_until_ts = now + 0.05
        self._audio_play_until_ts += duration
        # Pin _last_activity_time to when the user actually finishes hearing
        # us so the auto-continue loop's idle timer reflects the listener's
        # experience, not how fast Gemini emits audio.
        if self._audio_play_until_ts > self._last_activity_time:
            self._last_activity_time = self._audio_play_until_ts

    async def _handle_tool_call(self, tool_call: dict) -> None:
        calls = tool_call.get("functionCalls") or []
        responses = []
        for fc in calls:
            name = fc.get("name") or ""
            args = fc.get("args") or {}
            fc_id = fc.get("id")
            result: dict[str, Any] = {"result": "ok"}

            if name == "retrieve_chapter_context":
                query = str(args.get("query") or "")
                hits = await self._retrieve_rag_context(query)
                if not hits:
                    result = {"result": "No relevant chapter context found."}
                else:
                    lines = [
                        f"[Segment {int(h['segment_idx']) + 1} | score={h['score']:.2f}] {h['text']}"
                        for h in hits
                    ]
                    result = {"result": "\n".join(lines)}

            elif name == "jump_to_chapter":
                try:
                    ch_idx = int(args.get("chapter_index"))
                except Exception:
                    result = {"result": "Invalid chapter index."}
                else:
                    msg = await self._switch_to_chapter(ch_idx)
                    result = {"result": msg}

            elif name == "navigate_segment":
                direction = str(args.get("direction") or "").strip().lower()
                if direction not in {"next", "previous"}:
                    result = {"result": "Invalid direction; use 'next' or 'previous'."}
                else:
                    msg = await self._navigate_segment(direction)
                    result = {"result": msg}

            responses.append({"id": fc_id, "name": name, "response": result})

        await self._gemini_send_await({"toolResponse": {"functionResponses": responses}})
        if self._slide_vision_after_tool is not None:
            idx = self._slide_vision_after_tool
            self._slide_vision_after_tool = None
            await self._send_slide_vision(idx)

    async def _navigate_segment(self, direction: str) -> str:
        """Fast-path segment navigation triggered by the ``navigate_segment`` tool.

        Per Gemini Live's function-calling spec, we MUST respond to every
        ``toolCall`` with a ``toolResponse``. The tool response is also the
        right place to hand the model new context — anything we send as a
        ``realtimeInput.text`` between the toolCall and the toolResponse gets
        treated as a fresh user activity (``activityHandling`` =
        ``START_OF_ACTIVITY_INTERRUPTS``), which previously caused the model
        to re-fire ``navigate_segment`` in a tight 0.6s loop because the
        injection's "NAVIGATION: student went to the PREVIOUS segment" phrase
        looked like another nav request.

        So this handler:
          1. updates server state and pushes segment metadata to the browser
             (UI updates immediately);
          2. embeds the new segment text + teaching guidance directly inside
             the tool response string — the model receives it as a normal
             function result and continues its turn teaching the new segment.
        """
        async with self._turn_lock:
            prev_idx = int(self._runner.state.get("current_segment_idx", 0) or 0)
            prev_phase = str(self._runner.state.get("phase") or "")
            segments = self._runner.state.get("segments") or []
            total = len(segments)
            if total == 0:
                return "No segments available in this chapter."

            if direction == "next" and prev_idx + 1 >= total:
                logger.info("navigate_segment: at end of chapter (seg=%d)", prev_idx)
                self._last_tool_nav_ts = time.monotonic()
                await self._push_segment_metadata(force=True)
                return (
                    "Already on the last segment of this chapter. "
                    "Stay here and offer the next chapter if appropriate. "
                    "Do NOT call navigate_segment again."
                )
            if direction == "previous" and prev_idx - 1 < 0:
                logger.info("navigate_segment: at start of chapter (seg=%d)", prev_idx)
                self._last_tool_nav_ts = time.monotonic()
                await self._push_segment_metadata(force=True)
                return (
                    "Already on the first segment of this chapter. "
                    "Stay here and continue teaching it. "
                    "Do NOT call navigate_segment again."
                )

            new_idx = prev_idx + 1 if direction == "next" else prev_idx - 1
            new_phase = "teaching"

            self._runner.state["current_segment_idx"] = new_idx
            self._runner.state["phase"] = new_phase
            self._runner.state["intent"] = (
                "continue" if direction == "next" else "go_back"
            )
            self._assistant_segment_text = ""
            self._last_activity_time = time.monotonic()
            self._last_tool_nav_ts = time.monotonic()
            logger.info(
                "navigate_segment: %s → seg %d→%d (phase %r→%r)",
                direction, prev_idx, new_idx, prev_phase, new_phase,
            )

            # UI side — fast.
            await self._push_segment_metadata(force=True)

            # Build the new segment context and return it as the tool result.
            # We deliberately DO NOT call `_inject_lesson_context` here: that
            # would emit a separate `realtimeInput.text` between toolCall and
            # toolResponse and the model would re-fire the tool. Embedding
            # the context in the tool response keeps the conversation single-
            # turn and consumed naturally by the model.
            context_payload = build_segment_injection(
                self._runner.state,
                prev_segment_idx=prev_idx,
                new_segment_idx=new_idx,
                prev_phase=prev_phase,
                new_phase=new_phase,
            )
            self._slide_vision_after_tool = new_idx
            return (
                f"Moved to slide {new_idx + 1} of {total}. A JPEG of this slide will "
                f"arrive in your vision stream immediately after this tool result. "
                f"Do NOT call navigate_segment again — explain what is visible on "
                f"the slide (diagrams, labels, layout), using notes only as supplement.\n\n"
                f"{context_payload}"
            )

    def _chapter_switch_precheck(self, chapter_index: int) -> tuple[str | None, str, list[dict]]:
        if chapter_index not in self._chapter_segments_map:
            return ("Chapter not available in this session.", "", [])
        segs = self._chapter_segments_map[chapter_index]
        if not segs:
            return ("Selected chapter has no extracted segments.", "", [])
        title = ""
        if 0 <= chapter_index < len(self._chapters):
            title = str(self._chapters[chapter_index].get("title") or "").strip()
        return (None, title, segs)

    async def _switch_to_chapter(self, chapter_index: int) -> str:
        err, chapter_title, segs = self._chapter_switch_precheck(chapter_index)
        if err:
            return err
        async with self._turn_lock:
            cur = int(self._runner.state.get("selected_chapter_index", -1) or -1)
            if cur == chapter_index:
                return f"Already on {chapter_title or f'chapter {chapter_index + 1}'}."
            self._runner.state["segments"] = segs
            self._runner.state["total_segments"] = len(segs)
            self._runner.state["current_segment_idx"] = 0
            self._runner.state["phase"] = "teaching"
            self._runner.state["intent"] = "continue"
            self._runner.state["selected_chapter_index"] = chapter_index
            if chapter_title:
                self._runner.state["chapter_title"] = chapter_title
            self._assistant_segment_text = ""
            self._last_activity_time = time.monotonic()
            await self._inject_lesson_context(
                prev_idx=0, new_idx=0, prev_phase="teaching", new_phase="teaching"
            )
            return f"Switched to {chapter_title or f'chapter {chapter_index + 1}'}."

    # ── user input / transcripts ──

    @staticmethod
    def _is_echoed_instruction(text: str) -> bool:
        low = text.lower()
        markers = (
            "context update",
            "resume —",
            "resume -",
            "--- segment ",
            "--- end segment ---",
        )
        return any(m in low for m in markers)

    async def _dispatch_pending_user_text(self, *, reason: str) -> None:
        """Flush the accumulated user-speech transcription to the runner.

        Called from two places in the Gemini message loop:
        * ``inputTranscription.finished == True``
        * ``serverContent.turnComplete == True`` (fallback, since some
          responses skip the explicit ``finished`` flag and only signal the
          end of the user's turn via turnComplete on the model turn).
        """
        text = self._pending_user_text.strip()
        self._pending_user_text = ""
        if not text:
            return
        if self._pending_user_dispatched_for_turn:
            # Avoid double-firing when both ``finished`` and ``turnComplete``
            # arrive for the same user turn.
            return
        if self._is_echoed_instruction(text):
            logger.info("dropped echoed instruction (reason=%s): %r", reason, text[:80])
            return
        self._pending_user_dispatched_for_turn = True
        self._user_speaking_now = False
        logger.info("dispatching user input (reason=%s): %r", reason, text[:160])
        asyncio.create_task(self._process_user_input(text))

    async def _process_user_input(self, user_input: str) -> None:
        async with self._turn_lock:
            self._last_activity_time = time.monotonic()
            jumpable = self._runner.state.get("jumpable_chapter_indices") or []
            cur_ch = int(self._runner.state.get("selected_chapter_index", -1) or -1)
            resolved, _ = analyze_chapter_voice_nav(
                user_input, self._chapters, jumpable, cur_ch
            )
            if resolved is not None and resolved != cur_ch:
                await self._switch_to_chapter(resolved)
                return

            prev_idx = int(self._runner.state.get("current_segment_idx", 0) or 0)
            prev_phase = str(self._runner.state.get("phase") or "")

            # Explicit segment navigation ("next/previous segment", "go back",
            # short forms like "next" / "previous", Hinglish variants) MUST
            # always reach the LangGraph runner so we (a) update
            # current_segment_idx, (b) inject the new segment's source_text,
            # and (c) push the new segment to the UI. Otherwise Gemini hears
            # the audio and responds from its current-segment-only system
            # prompt, which means it hallucinates the next/previous segment.
            is_nav = is_segment_navigation(user_input)

            # If the navigate_segment tool already handled this exact turn
            # (the tool call arrives a beat before the final transcript), the
            # runner would advance a SECOND time. Skip the LangGraph turn in
            # that case but keep going for non-nav inputs.
            if is_nav and (time.monotonic() - self._last_tool_nav_ts) < 5.0:
                logger.info(
                    "skipping LangGraph nav for %r — handled by navigate_segment tool %.2fs ago",
                    user_input[:80],
                    time.monotonic() - self._last_tool_nav_ts,
                )
                return

            self._runner.process_turn(user_input)
            new_idx = int(self._runner.state.get("current_segment_idx", 0) or 0)
            new_phase = str(self._runner.state.get("phase") or "")
            intent = str(self._runner.state.get("intent") or "")
            logger.info(
                "process_turn: %r → intent=%s seg %d→%d phase %r→%r",
                user_input[:80], intent, prev_idx, new_idx, prev_phase, new_phase,
            )
            if new_idx != prev_idx or new_phase != prev_phase:
                self._assistant_segment_text = ""
                await self._inject_lesson_context(
                    prev_idx=prev_idx,
                    new_idx=new_idx,
                    prev_phase=prev_phase,
                    new_phase=new_phase,
                )
            elif is_nav and new_idx == prev_idx:
                # User asked to navigate but we're at a boundary (segment 0
                # for "previous", last segment for "next"). Refresh metadata
                # so the UI sticks to the correct segment and the agent gets
                # a gentle nudge.
                logger.info("nav at boundary (seg=%d intent=%s)", prev_idx, intent)
                await self._push_segment_metadata(force=True)
                await self._inject_text(
                    "NAVIGATION: you are already at the boundary of this chapter — "
                    "stay on the current segment and continue teaching it briefly, "
                    "then offer to move to the next chapter if appropriate."
                )
            else:
                intent = str(self._runner.state.get("intent") or "")
                if intent in ("question", "simpler") and len(self._assistant_segment_text) >= 40:
                    await asyncio.sleep(_RESUME_HINT_DELAY_S)
                    tail = self._assistant_segment_text[-_RESUME_HINT_MAX_CHARS:]
                    await self._inject_text(
                        f"RESUME — answer briefly, then continue the SAME segment from:\n\n{tail}"
                    )

    def _segment_content_covered(self) -> bool:
        idx = int(self._runner.state.get("current_segment_idx", 0) or 0)
        segments = self._runner.state.get("segments") or []
        if idx >= len(segments):
            return False
        source = (segments[idx].get("source_text") or "").strip()
        agent_text = self._assistant_segment_text
        if len(source) < 30 or len(agent_text) < 50:
            return len(source) < 30
        source_norm = " ".join(source.lower().split())
        agent_norm = " ".join(agent_text.lower().split())
        tail = source_norm[max(0, len(source_norm) * 6 // 10) :]
        words = tail.split()
        if len(words) < 4:
            return True
        ngrams = [" ".join(words[i : i + 4]) for i in range(len(words) - 3)]
        return any(ng in agent_norm for ng in ngrams)

    async def _auto_continue_loop(self) -> None:
        await self._gemini_ready.wait()
        await asyncio.sleep(3.5)
        self._last_activity_time = time.monotonic()
        done_logged = False
        try:
            while not self._closed:
                await asyncio.sleep(_POLL_INTERVAL)
                if self._runner.state.get("phase") == "done":
                    if not done_logged:
                        logger.info("auto-advance: phase=done, pausing")
                        done_logged = True
                    self._last_activity_time = time.monotonic()
                    continue
                if done_logged:
                    done_logged = False
                    self._last_activity_time = time.monotonic()

                idle = time.monotonic() - self._last_activity_time
                covered = self._segment_content_covered()
                required = _COVERED_IDLE_S if covered else _FALLBACK_IDLE_S
                if idle < required:
                    continue

                cur_idx = int(self._runner.state.get("current_segment_idx", 0) or 0)
                cur_phase = str(self._runner.state.get("phase") or "")

                async with self._turn_lock:
                    idle = time.monotonic() - self._last_activity_time
                    covered = self._segment_content_covered()
                    required = _COVERED_IDLE_S if covered else _FALLBACK_IDLE_S
                    if idle < required:
                        continue
                    prev_idx = cur_idx
                    prev_phase = cur_phase
                    self._runner.process_turn("")
                    new_idx = int(self._runner.state.get("current_segment_idx", 0) or 0)
                    new_phase = str(self._runner.state.get("phase") or "")
                    if new_idx == prev_idx and prev_phase == "teaching" and new_phase == "greeting":
                        self._runner.state["phase"] = "teaching"
                        new_phase = "teaching"
                    if new_idx != prev_idx or new_phase != prev_phase:
                        logger.info(
                            "auto-advance: seg %d→%d phase %r→%r",
                            prev_idx, new_idx, prev_phase, new_phase,
                        )
                        self._assistant_segment_text = ""
                        await self._inject_lesson_context(
                            prev_idx=prev_idx,
                            new_idx=new_idx,
                            prev_phase=prev_phase,
                            new_phase=new_phase,
                        )
                        self._last_activity_time = time.monotonic()
        except asyncio.CancelledError:
            return

    # ── Gemini message loop ──

    async def _handle_gemini_message(self, raw: str) -> None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            print(
                f"[gemini-live] non-JSON message ({len(raw)} chars): {raw[:400]!r}",
                file=sys.stderr,
                flush=True,
            )
            return

        # NOTE: Gemini Live sends top-level keys as empty dicts (e.g.
        # `{"setupComplete": {}}`) which are falsy in Python — DO NOT use
        # `if data.get(key):`, use `key in data` instead.
        if "setupComplete" in data:
            print("[gemini-live] setupComplete received", file=sys.stderr, flush=True)
            logger.info("Gemini setupComplete received")
            self._gemini_ready.set()
            await self._send_browser({"type": "ready"})
            await self._push_segment_metadata(force=True)
            await self._inject_text(
                "Begin the tutoring session now per your greeting instructions. "
                "Start speaking immediately."
            )
            if self._content_type == "ppt":
                await self._send_slide_vision(0)
            return
        if "goAway" in data:
            logger.warning("Gemini goAway: %s", data["goAway"])
            await self._send_browser(
                {"type": "error", "message": "Gemini asked us to disconnect."}
            )
            return

        if "toolCall" in data and data["toolCall"]:
            await self._handle_tool_call(data["toolCall"])
            return

        content = data.get("serverContent")
        if not content:
            return

        if not self._logged_first_server_content:
            self._logged_first_server_content = True
            top_keys = list(content.keys()) if isinstance(content, dict) else []
            logger.info("first serverContent received; top-level keys=%s", top_keys)

        if content.get("interrupted"):
            self._user_speaking_now = True
            # Browser is about to clear its audio queue, so reset our estimate.
            self._audio_play_until_ts = time.monotonic()
            await self._send_browser({"type": "interrupted"})

        in_tx = content.get("inputTranscription") or {}
        in_chunk = str(in_tx.get("text") or "")
        if in_chunk:
            if self._pending_user_dispatched_for_turn and not self._pending_user_text:
                # Fresh chunk after a dispatched turn → start a new buffer.
                self._pending_user_dispatched_for_turn = False
            self._pending_user_text += in_chunk
            self._last_user_speech_ts = time.monotonic()
            self._user_speaking_now = True
            logger.info(
                "input transcription chunk: %r finished=%s buffer=%r",
                in_chunk[:80], in_tx.get("finished"), self._pending_user_text[:160],
            )
        if in_tx.get("finished"):
            await self._dispatch_pending_user_text(reason="transcription_finished")

        out_tx = content.get("outputTranscription") or {}
        if out_tx.get("text"):
            chunk = str(out_tx["text"])
            self._assistant_segment_text += chunk
            if out_tx.get("finished"):
                self._last_activity_time = time.monotonic()

        if content.get("turnComplete"):
            self._user_speaking_now = False
            # Some Gemini Live responses skip inputTranscription.finished and
            # only signal end-of-user-turn via turnComplete on the *model*
            # turn. Flush whatever we've buffered so user navigation isn't
            # silently dropped.
            await self._dispatch_pending_user_text(reason="turn_complete")

        for part in (content.get("modelTurn") or {}).get("parts") or []:
            audio_b64 = part.get("inlineData", {}).get("data")
            if audio_b64:
                await self._send_browser({"type": "audio", "data": audio_b64})
                self._bump_audio_play_end(audio_b64)

    async def _gemini_reader(self) -> None:
        assert self._gemini_ws is not None
        logger.info("gemini reader started")
        try:
            async for raw in self._gemini_ws:
                if self._closed:
                    break
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                await self._handle_gemini_message(raw)
        except websockets.ConnectionClosed as exc:
            logger.warning("Gemini WebSocket closed (code=%s reason=%r)", exc.code, exc.reason)
            await self._send_browser(
                {
                    "type": "error",
                    "message": f"Gemini connection closed (code={exc.code}): {exc.reason or 'unknown'}",
                }
            )
        except Exception:
            logger.exception("Gemini reader error")
            await self._send_browser(
                {"type": "error", "message": "Gemini reader crashed — see server log."}
            )
        finally:
            self._closed = True
            logger.info("gemini reader exiting")
            # Wake up the browser reader (blocked on receive) so the gather() resolves.
            if self._browser_ws is not None:
                try:
                    await self._browser_ws.close()
                except Exception:
                    pass

    async def _browser_reader(self) -> None:
        assert self._browser_ws is not None
        import base64
        logger.info("browser reader started")
        audio_chunks = 0
        try:
            while not self._closed:
                msg = await self._browser_ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    logger.info("browser disconnected (code=%s)", msg.get("code"))
                    break
                if "bytes" in msg and msg["bytes"]:
                    b64 = base64.b64encode(msg["bytes"]).decode("ascii")
                    await self._gemini_send_await(
                        {
                            "realtimeInput": {
                                "audio": {
                                    "mimeType": "audio/pcm;rate=16000",
                                    "data": b64,
                                }
                            }
                        }
                    )
                    audio_chunks += 1
                    continue
                if "text" not in msg or msg["text"] is None:
                    continue
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    logger.debug("browser sent invalid json: %r", msg["text"][:80])
                    continue
                mtype = data.get("type")
                if mtype == "audio":
                    await self._gemini_send_await(
                        {
                            "realtimeInput": {
                                "audio": {
                                    "mimeType": data.get("mimeType")
                                    or "audio/pcm;rate=16000",
                                    "data": data.get("data") or "",
                                }
                            }
                        }
                    )
                    audio_chunks += 1
                    if audio_chunks in (1, 50, 500):
                        logger.info("browser audio: %d chunks forwarded", audio_chunks)
                elif mtype == "jump_chapter":
                    logger.info("browser jump_chapter: %s", data.get("chapter_index"))
                    await self._switch_to_chapter(int(data.get("chapter_index", -1)))
                elif mtype == "end":
                    logger.info("browser sent end")
                    break
        except WebSocketDisconnect:
            logger.info("browser WebSocketDisconnect")
        except Exception:
            logger.exception("browser reader error")
        finally:
            self._closed = True
            logger.info("browser reader exiting (audio_chunks=%d)", audio_chunks)
            if self._gemini_ws:
                try:
                    await self._gemini_ws.close()
                except Exception:
                    pass

    async def run(self, browser_ws: WebSocket) -> None:
        self._browser_ws = browser_ws
        print(
            f"[gemini-live] starting session room={self._room_name} "
            f"model={self._model} voice={self._voice} key_prefix={self._api_key[:6]}",
            file=sys.stderr,
            flush=True,
        )
        logger.info(
            "starting Gemini Live session (room=%s model=%s voice=%s)",
            self._room_name, self._model, self._voice,
        )
        url = f"{GEMINI_WS_BASE}?key={self._api_key}"
        rag_task = asyncio.create_task(self._build_rag_index())
        try:
            try:
                gemini_ws = await asyncio.wait_for(
                    websockets.connect(url, max_size=32 * 1024 * 1024),
                    timeout=15.0,
                )
            except Exception as exc:
                print(
                    f"[gemini-live] connect FAILED: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                logger.exception("failed to connect to Gemini Live")
                await self._send_browser(
                    {
                        "type": "error",
                        "message": f"Could not connect to Gemini Live: {exc}",
                    }
                )
                return
            print("[gemini-live] connected; sending setup", file=sys.stderr, flush=True)
            logger.info("connected to Gemini Live; sending setup")
            self._gemini_ws = gemini_ws
            try:
                await self._gemini_send_await(self._build_setup_message())
                self._auto_task = asyncio.create_task(self._auto_continue_loop())
                # IMPORTANT: do NOT await rag_task before starting readers. The
                # gemini reader must run so setupComplete is processed and the
                # browser gets `ready`; the RAG index continues building in the
                # background and is only used by the retrieve_chapter_context tool.
                await asyncio.gather(
                    self._gemini_reader(),
                    self._browser_reader(),
                )
            finally:
                try:
                    await gemini_ws.close()
                except Exception:
                    pass
        except Exception:
            logger.exception("Gemini Live session crashed")
            await self._send_browser(
                {"type": "error", "message": "Gemini Live session crashed — see server log."}
            )
        finally:
            self._closed = True
            if self._auto_task and not self._auto_task.done():
                self._auto_task.cancel()
            if not rag_task.done():
                rag_task.cancel()
                try:
                    await rag_task
                except (asyncio.CancelledError, Exception):
                    pass
            await self._finalize_session()
            logger.info("Gemini Live session ended (room=%s)", self._room_name)

    async def _finalize_session(self) -> None:
        if self._room_name:
            path = SESSIONS_DIR / f"{self._room_name}.json"
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
        if self._session_id:
            await asyncio.to_thread(cleanup_session_uploads, self._session_id)


async def run_gemini_live_session(
    browser_ws: WebSocket, session: dict, *, api_key: str
) -> None:
    tutor = GeminiLiveTutorSession(session, api_key=api_key)
    await tutor.run(browser_ws)
