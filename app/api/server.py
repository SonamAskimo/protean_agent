"""
FastAPI server — PDF/PPT upload, session creation, Gemini Live WebSocket.

    uvicorn app.server:app --reload --port 8080
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import uuid
import asyncio
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from openai import OpenAI

from ..core.paths import KB, ROOT, SESSIONS, UPLOADS, WEB
from ..ingest.lang_detect import detect_source_lang, resolve_segment_lang
from ..ingest.pdf_extract import count_pages, detect_chapters, extract_paragraphs_by_mode, paragraphs_to_plain_text
from ..ingest.ppt_extract import build_ppt_segment_dicts, ingest_pptx
from ..ingest.segmenter import build_segments
from ..kb import store as kb_store
from ..kb.ingest import run_ingestion as kb_run_ingestion
from ..live.gemini_live_session import cleanup_session_uploads, run_gemini_live_session

load_dotenv(ROOT / ".env")

UPLOADS.mkdir(exist_ok=True)
SESSIONS.mkdir(exist_ok=True)

app = FastAPI(title="Protean — Gemini Live Agent")
logger = logging.getLogger("protean-server")

_SLIDE_FILE_RE = re.compile(r"^slide_\d{3}\.jpg$", re.IGNORECASE)
_SUBJECT_INFER_TIMEOUT_S = 2.8
_SUBJECT_INFER_MODEL = "gpt-4o-mini"
_OPENAI_CLIENT: OpenAI | None = None
_OPENAI_CLIENT_INIT = False


def _is_production_env() -> bool:
    raw = (
        os.getenv("APP_ENV")
        or os.getenv("ENV")
        or os.getenv("TUTOR_ENV")
        or os.getenv("FASTAPI_ENV")
        or ""
    )
    v = str(raw).strip().lower()
    return v in {"prod", "production", "live"}


def _infer_subject_label(*, subject: str, chapter_title: str, meta: dict, data: dict) -> str:
    explicit = (subject or "").strip()
    if explicit:
        return explicit
    inferred = (
        str(meta.get("inferred_subject") or "").strip()
        or str(data.get("inferred_subject") or "").strip()
    )
    if inferred:
        return inferred
    hint_blob = " ".join(
        [
            str(chapter_title or ""),
            str(meta.get("pdf_filename") or ""),
            str(data.get("pdf_filename") or ""),
            str(data.get("chapter_preview") or "")[:600],
        ]
    ).lower()
    rules = [
        ("network security", "Network Security"),
        ("cyber security", "Cyber Security"),
        ("cybersecurity", "Cyber Security"),
        ("cryptography", "Cryptography"),
        ("computer network", "Computer Networks"),
        ("machine learning", "Machine Learning"),
        ("data science", "Data Science"),
        ("python", "Python Programming"),
        ("java", "Java Programming"),
    ]
    for needle, label in rules:
        if needle in hint_blob:
            return label
    return "Unspecified"


def _get_openai_client() -> OpenAI | None:
    global _OPENAI_CLIENT, _OPENAI_CLIENT_INIT
    if _OPENAI_CLIENT_INIT:
        return _OPENAI_CLIENT
    _OPENAI_CLIENT_INIT = True
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        _OPENAI_CLIENT = None
        return None
    try:
        _OPENAI_CLIENT = OpenAI(api_key=api_key)
    except Exception:
        _OPENAI_CLIENT = None
    return _OPENAI_CLIENT


async def _infer_subject_from_pdf(
    *,
    explicit_subject: str,
    pdf_filename: str,
    extracted_text: str,
    chapter_titles: list[str],
) -> str:
    explicit = (explicit_subject or "").strip()
    if explicit:
        return explicit

    preview = (extracted_text or "").strip()[:5000]
    if not preview:
        return "Unspecified"

    client = _get_openai_client()
    if client is not None:
        prompt = (
            "Infer the most precise high-level subject of this study material.\n"
            "Return strict JSON only: {\"subject\":\"...\"}\n"
            "Rules: 2-6 words, title case, no punctuation, no explanation.\n\n"
            f"Filename: {pdf_filename}\n"
            f"Chapter titles: {', '.join(chapter_titles[:20])}\n"
            f"Content preview:\n{preview}"
        )
        try:
            resp = await asyncio.wait_for(
                asyncio.to_thread(
                    client.chat.completions.create,
                    model=_SUBJECT_INFER_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                ),
                timeout=_SUBJECT_INFER_TIMEOUT_S,
            )
            body = (resp.choices[0].message.content or "").strip()
            obj = json.loads(body)
            guessed = str(obj.get("subject") or "").strip()
            if guessed:
                return guessed
        except Exception:
            logger.warning("subject inference via OpenAI failed; using heuristic fallback")

    # Heuristic fallback when OPENAI_API_KEY unavailable or call fails.
    combined = f"{pdf_filename} {' '.join(chapter_titles)} {preview}".lower()
    rules = [
        ("network security", "Network Security"),
        ("cyber security", "Cyber Security"),
        ("cybersecurity", "Cyber Security"),
        ("cryptography", "Cryptography"),
        ("ethical hacking", "Ethical Hacking"),
        ("computer network", "Computer Networks"),
        ("data structure", "Data Structures"),
        ("operating system", "Operating Systems"),
        ("database", "Database Systems"),
        ("artificial intelligence", "Artificial Intelligence"),
        ("machine learning", "Machine Learning"),
        ("data science", "Data Science"),
    ]
    for needle, label in rules:
        if needle in combined:
            return label
    return "Unspecified"


def _build_segment_dicts(
    seg_inputs,
    *,
    lang_opt: str,
    chapter_id: str = "",
    chapter_title: str = "",
    chapter_index: int = -1,
) -> list[dict]:
    out: list[dict] = []
    for s in seg_inputs:
        detected = detect_source_lang(s.source_text)
        sl = resolve_segment_lang(lang_opt, detected)
        text = s.source_text
        out.append(
            {
                "segment_id": s.segment_id,
                "pages": s.pages,
                "source_text": text,
                "source_lang": sl,
                "chapter_id": chapter_id,
                "chapter_title": chapter_title,
                "chapter_index": chapter_index,
            }
        )
    return out


def _paras_for_range(paras: list, start_page: int, end_page: int) -> list:
    lo, hi = sorted((start_page, end_page))
    return [p for p in paras if lo <= p.page <= hi]


def _needs_ocr_fallback(paras: list, total_pages: int) -> bool:
    """Heuristic: when simple extraction yields too little usable text, switch to OCR."""
    if not paras:
        return True
    texts = [(p.text or "").strip() for p in paras]
    total_chars = sum(len(t) for t in texts)
    nonempty = sum(1 for t in texts if len(t) >= 20)
    avg_chars_per_page = total_chars / max(1, total_pages)
    sparse_ratio = 1.0 - (nonempty / max(1, total_pages))
    if total_chars < 600:
        return True
    if avg_chars_per_page < 140:
        return True
    if sparse_ratio > 0.65:
        return True
    return False


# ── routes ──

@app.get("/")
def index():
    return FileResponse(str(WEB / "index.html"))


@app.post("/api/books")
async def inspect_book(
    pdf: UploadFile = File(...),
):
    """Upload a PDF and return detected chapter boundaries."""
    if not pdf.filename or not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only .pdf files are accepted")
    session_id = uuid.uuid4().hex[:10]
    dst = UPLOADS / f"{session_id}.pdf"
    with dst.open("wb") as f:
        shutil.copyfileobj(pdf.file, f)
    total_pages = count_pages(str(dst))
    chapters = detect_chapters(str(dst))
    return JSONResponse(
        {
            "session_id": session_id,
            "pdf_filename": pdf.filename,
            "total_pages": total_pages,
            "chapters": [
                {
                    "chapter_id": c.chapter_id,
                    "title": c.title,
                    "level": c.level,
                    "start_page": c.start_page,
                    "end_page": c.end_page,
                    "source": c.source,
                }
                for c in chapters
            ],
        }
    )


@app.post("/api/sessions")
async def create_session(
    pdf: UploadFile = File(...),
    start_page: int = Form(...),
    end_page: int = Form(...),
    chapter_index: int = Form(-1),
    chapter_title: str = Form(""),
    subject: str = Form(""),
    content_language: str = Form("auto"),
    tts_pace: float = Form(1.0),
    extraction_mode: str = Form("simple"),
):
    if not pdf.filename or not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only .pdf files are accepted")

    session_id = uuid.uuid4().hex[:10]
    room_name = f"tutor-{session_id}"

    dst = UPLOADS / f"{session_id}.pdf"
    with dst.open("wb") as f:
        shutil.copyfileobj(pdf.file, f)

    total_pages = count_pages(str(dst))
    if start_page < 1 or end_page < 1 or start_page > total_pages:
        raise HTTPException(400, f"Page range invalid (PDF has {total_pages} pages)")

    requested_mode = (extraction_mode or "simple").strip().lower()
    if requested_mode not in {"simple", "ocr", "auto"}:
        raise HTTPException(400, "extraction_mode must be 'simple', 'ocr' or 'auto'")

    lang_opt = (content_language or "auto").strip().lower()
    if lang_opt not in {"auto", "en", "hi", "mixed"}:
        raise HTTPException(
            400, "content_language must be auto, en, hi, or mixed",
        )

    mode_used = "simple"
    try:
        rag_paras = extract_paragraphs_by_mode(
            str(dst),
            1,
            total_pages,
            mode="simple",
            ocr_lang=os.environ.get("OCR_TESS_LANG", "eng"),
        )
    except Exception:
        rag_paras = []

    force_ocr = requested_mode == "ocr"
    if force_ocr or _needs_ocr_fallback(rag_paras, total_pages):
        try:
            rag_paras = extract_paragraphs_by_mode(
                str(dst),
                1,
                total_pages,
                mode="ocr",
                ocr_lang=os.environ.get("OCR_TESS_LANG", "eng"),
            )
            mode_used = "ocr"
        except Exception as exc:
            if force_ocr or not rag_paras:
                raise HTTPException(
                    500,
                    "OCR extraction failed. Ensure Tesseract is installed and OCR language packs are available.",
                ) from exc
            mode_used = "simple"

    chapters = detect_chapters(str(dst))
    if not chapters:
        chapters = []
    chapters_payload = [
        {
            "chapter_id": ch.chapter_id,
            "title": ch.title,
            "level": ch.level,
            "start_page": ch.start_page,
            "end_page": ch.end_page,
            "source": ch.source,
        }
        for ch in chapters
    ]

    selected_chapter_idx = -1
    if chapters_payload and 0 <= chapter_index < len(chapters_payload):
        selected_chapter_idx = chapter_index
        start_page = int(chapters_payload[chapter_index]["start_page"])
        end_page = int(chapters_payload[chapter_index]["end_page"])
        chapter_title = str(chapters_payload[chapter_index]["title"])

    paras = _paras_for_range(rag_paras, start_page, end_page)
    extracted_text = paragraphs_to_plain_text(paras, page_markers=True)
    seg_inputs = build_segments(paras, max_segments=120)

    if not seg_inputs:
        raise HTTPException(400, "No extractable text found in the given page range")

    selected_chapter_id = ""
    selected_chapter_title = chapter_title
    if 0 <= selected_chapter_idx < len(chapters_payload):
        selected_chapter_id = str(chapters_payload[selected_chapter_idx]["chapter_id"])
        selected_chapter_title = str(chapters_payload[selected_chapter_idx]["title"])
    segments = _build_segment_dicts(
        seg_inputs,
        lang_opt=lang_opt,
        chapter_id=selected_chapter_id,
        chapter_title=selected_chapter_title,
        chapter_index=selected_chapter_idx,
    )
    chapter_preview = " ".join(p.text for p in paras)[:1500]
    inferred_subject = await _infer_subject_from_pdf(
        explicit_subject=(subject or "").strip(),
        pdf_filename=pdf.filename or "",
        extracted_text=extracted_text,
        chapter_titles=[str(ch.get("title") or "") for ch in chapters_payload],
    )

    if not paras:
        raise HTTPException(400, "No extractable text found in the selected chapter/page range")
    rag_seg_inputs = build_segments(rag_paras, max_segments=400)
    rag_segments = _build_segment_dicts(rag_seg_inputs, lang_opt=lang_opt)
    logger.info(
        "pdf extracted: session=%s file=%s mode=%s pages=%d-%d total_pages=%d segments=%d extracted_chars=%d",
        session_id,
        pdf.filename,
        mode_used,
        start_page,
        end_page,
        total_pages,
        len(seg_inputs),
        len(extracted_text or ""),
    )
    if extracted_text:
        logger.info(
            "pdf extracted preview: session=%s text=%r",
            session_id,
            extracted_text[:500],
        )
    logger.info(
        "rag prepared: session=%s rag_segments=%d chapter_count=%d",
        session_id,
        len(rag_segments),
        len(chapters_payload),
    )

    chapter_segments: list[dict] = []
    for i, ch in enumerate(chapters_payload):
        ch_paras = _paras_for_range(rag_paras, int(ch["start_page"]), int(ch["end_page"]))
        if not ch_paras:
            continue
        ch_seg_inputs = build_segments(ch_paras, max_segments=120)
        if not ch_seg_inputs:
            continue
        ch_segs = _build_segment_dicts(
            ch_seg_inputs,
            lang_opt=lang_opt,
            chapter_id=str(ch["chapter_id"]),
            chapter_title=str(ch["title"]),
            chapter_index=i,
        )
        chapter_segments.append(
            {
                "chapter_index": i,
                "chapter_id": ch["chapter_id"],
                "title": ch["title"],
                "start_page": ch["start_page"],
                "end_page": ch["end_page"],
                "segment_count": len(ch_segs),
                "segments": ch_segs,
            }
        )

    session = {
        "session_id": session_id,
        "room_name": room_name,
        "content_type": "pdf",
        "pdf_filename": pdf.filename,
        "start_page": start_page,
        "end_page": end_page,
        "total_pages": total_pages,
        "chapter_title": chapter_title,
        "selected_chapter_index": selected_chapter_idx,
        "chapters": chapters_payload,
        "chapter_segments": chapter_segments,
        "subject": (subject or "").strip(),
        "inferred_subject": inferred_subject,
        "content_language": lang_opt,
        "chapter_preview": chapter_preview,
        "segment_count": len(segments),
        "segments": segments,
        "rag_segment_count": len(rag_segments),
        "rag_segments": rag_segments,
        "tts_pace": tts_pace,
        "extraction_mode": mode_used,
    }
    (SESSIONS / f"{room_name}.json").write_text(
        json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Extraction step only. The client should call /api/sessions/{session_id}/token
    # when they are ready to connect to the agent.
    segments_for_ui = [
        {
            "pages": s["pages"],
            "source_text": s["source_text"],
            "source_lang": s.get("source_lang", "unknown"),
        }
        for s in segments
    ]

    # Full plain-text extract can be large and is unnecessary in production UI.
    # Expose it only when APP_ENV / ENV / etc. is not production so dev builds
    # can verify OCR / extraction without opening server logs.
    dev_mode = not _is_production_env()
    response_body: dict = {
        "session_id": session_id,
        "room_name": room_name,
        "content_type": "pdf",
        "segment_count": len(segments),
        "total_pages": total_pages,
        "start_page": start_page,
        "end_page": end_page,
        "subject": session.get("subject", ""),
        "content_language": lang_opt,
        "selected_chapter_index": selected_chapter_idx,
        "chapters": chapters_payload,
        "chapter_segments": chapter_segments,
        "segments": segments_for_ui,
        "tts_pace": tts_pace,
        "extraction_mode": mode_used,
        "development_mode": dev_mode,
    }
    if dev_mode:
        response_body["extracted_text"] = extracted_text

    return JSONResponse(response_body)


def _slides_dir(session_id: str) -> Path:
    sid = (session_id or "").strip()
    if not sid or not re.fullmatch(r"[a-f0-9]{10}", sid):
        raise HTTPException(400, "Invalid session id")
    return UPLOADS / sid / "slides"


@app.post("/api/sessions/ppt")
async def create_ppt_session(
    pptx: UploadFile = File(...),
    subject: str = Form(""),
    content_language: str = Form("auto"),
    tts_pace: float = Form(1.0),
):
    """Create a tutoring session from a PowerPoint deck (1 slide = 1 segment)."""
    filename = pptx.filename or ""
    if not filename.lower().endswith(".pptx"):
        raise HTTPException(400, "Only .pptx files are accepted")

    session_id = uuid.uuid4().hex[:10]
    room_name = f"tutor-{session_id}"
    lang_opt = (content_language or "auto").strip().lower()
    if lang_opt not in {"auto", "en", "hi", "mixed"}:
        raise HTTPException(400, "content_language must be auto, en, hi, or mixed")

    session_upload = UPLOADS / session_id
    session_upload.mkdir(parents=True, exist_ok=True)
    pptx_path = session_upload / "deck.pptx"
    with pptx_path.open("wb") as f:
        shutil.copyfileobj(pptx.file, f)

    slides_dir = session_upload / "slides"
    try:
        slide_rows = await asyncio.to_thread(
            ingest_pptx,
            pptx_path,
            slides_dir=slides_dir,
            session_id=session_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            503,
            "LibreOffice is required to process PowerPoint files. Install LibreOffice "
            "or set SOFFICE_PATH to the soffice executable.",
        ) from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc

    segments = build_ppt_segment_dicts(
        slide_rows,
        session_id=session_id,
        lang_opt=lang_opt,
        subject=(subject or "").strip(),
    )
    if not segments:
        raise HTTPException(400, "No slides found in the presentation")

    slide_count = len(segments)
    deck_title = Path(filename).stem
    chapter_preview = " ".join(
        (s.get("source_text") or "")[:200] for s in segments[:3]
    )[:1500]

    session = {
        "session_id": session_id,
        "room_name": room_name,
        "content_type": "ppt",
        "ppt_filename": filename,
        "pdf_filename": "",
        "start_page": 1,
        "end_page": slide_count,
        "total_pages": slide_count,
        "chapter_title": deck_title,
        "selected_chapter_index": -1,
        "chapters": [],
        "chapter_segments": [],
        "subject": (subject or "").strip(),
        "inferred_subject": (subject or "").strip() or deck_title,
        "content_language": lang_opt,
        "chapter_preview": chapter_preview,
        "segment_count": slide_count,
        "segments": segments,
        "rag_segment_count": 0,
        "rag_segments": [],
        "tts_pace": tts_pace,
        "extraction_mode": "ppt",
        "slides_dir": str(slides_dir),
    }
    (SESSIONS / f"{room_name}.json").write_text(
        json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info(
        "ppt ingested: session=%s file=%s slides=%d",
        session_id,
        filename,
        slide_count,
    )

    dev_mode = not _is_production_env()
    segments_for_ui = [
        {
            "pages": s["pages"],
            "slide_index": s.get("slide_index"),
            "slide_url": s.get("slide_url"),
            "source_text": s["source_text"],
            "source_lang": s.get("source_lang", "unknown"),
        }
        for s in segments
    ]
    response_body: dict = {
        "session_id": session_id,
        "room_name": room_name,
        "content_type": "ppt",
        "segment_count": slide_count,
        "total_pages": slide_count,
        "start_page": 1,
        "end_page": slide_count,
        "subject": session.get("subject", ""),
        "content_language": lang_opt,
        "selected_chapter_index": -1,
        "chapters": [],
        "chapter_segments": [],
        "segments": segments_for_ui,
        "tts_pace": tts_pace,
        "extraction_mode": "ppt",
        "development_mode": dev_mode,
    }
    if dev_mode:
        response_body["extracted_text"] = "\n\n---\n\n".join(
            f"Slide {i + 1}:\n{s.get('source_text', '')}"
            for i, s in enumerate(segments)
        )
    return JSONResponse(response_body)


@app.get("/api/sessions/{session_id}/slides/{filename}")
def get_session_slide(session_id: str, filename: str):
    """Serve a rendered slide JPEG for PPT sessions."""
    if not _SLIDE_FILE_RE.match(filename or ""):
        raise HTTPException(400, "Invalid slide filename")
    path = _slides_dir(session_id) / filename
    if not path.is_file():
        raise HTTPException(404, "Slide not found")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/sessions/{session_id}/live-config")
def get_live_config(session_id: str):
    """Client metadata for Gemini Live (no secrets)."""
    room_name = f"tutor-{session_id}"
    path = SESSIONS / f"{room_name}.json"
    if not path.exists():
        raise HTTPException(404, "Session not found")
    data = json.loads(path.read_text(encoding="utf-8"))
    model = (os.getenv("GEMINI_MODEL") or "").strip() or "gemini-3.1-flash-live-preview"
    voice = (os.getenv("GEMINI_VOICE") or "").strip() or "Kore"
    chapters = data.get("chapters") or []
    jumpable = [
        int(row.get("chapter_index"))
        for row in (data.get("chapter_segments") or [])
        if isinstance(row.get("chapter_index"), int)
    ]
    chapter_options = []
    for i, ch in enumerate(chapters):
        if jumpable and i not in jumpable:
            continue
        chapter_options.append(
            {
                "index": i,
                "title": str(ch.get("title") or f"Chapter {i + 1}"),
                "start_page": ch.get("start_page"),
                "end_page": ch.get("end_page"),
            }
        )
    content_type = str(data.get("content_type") or "pdf").strip().lower()
    return {
        "session_id": session_id,
        "room_name": room_name,
        "content_type": content_type,
        "model": model,
        "voice": voice,
        "ws_path": f"/ws/tutor/{session_id}",
        "output_sample_rate": 24000,
        "selected_chapter_index": data.get("selected_chapter_index", -1),
        "chapters": chapter_options,
    }


@app.websocket("/ws/tutor/{session_id}")
async def tutor_live_websocket(websocket: WebSocket, session_id: str):
    """Browser audio <-> server <-> Gemini Live API (replaces LiveKit + Ultravox)."""
    sid = (session_id or "").strip()
    room_name = f"tutor-{sid}"
    path = SESSIONS / f"{room_name}.json"

    cleanup_uploads_after_live = False

    if not path.exists():
        await websocket.close(code=4404, reason="Session not found")
        return

    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        await websocket.accept()
        await websocket.send_json(
            {"type": "error", "message": "GEMINI_API_KEY is not configured on the server."}
        )
        await websocket.close(code=1011)
        return

    await websocket.accept()
    try:
        session = json.loads(path.read_text(encoding="utf-8"))
        is_kb = bool(session.get("kb_file_id"))
        cleanup_uploads_after_live = not is_kb
        await run_gemini_live_session(websocket, session, api_key=api_key)
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Gemini live session failed for %s", sid)
        try:
            await websocket.send_json({"type": "error", "message": "Live session failed."})
        except Exception:
            pass
    finally:
        if cleanup_uploads_after_live:
            await asyncio.to_thread(cleanup_session_uploads, sid)


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    room_name = f"tutor-{session_id}"
    path = SESSIONS / f"{room_name}.json"
    if not path.exists():
        raise HTTPException(404, "Session not found")
    data = json.loads(path.read_text(encoding="utf-8"))
    data.pop("segments", None)
    data.pop("chapter_preview", None)
    return data


# ── Knowledge Base routes ──

@app.get("/kb")
def kb_page():
    return FileResponse(str(WEB / "kb.html"))


@app.get("/api/kb/files")
def kb_list_files():
    return JSONResponse(kb_store.list_files())


@app.get("/api/kb/files/{file_id}")
def kb_get_file(file_id: str):
    meta = kb_store.get_file(file_id)
    if not meta:
        raise HTTPException(404, "File not found in knowledge base")
    return JSONResponse(meta)


@app.get("/api/kb/files/{file_id}/slides/{filename}")
def kb_get_slide(file_id: str, filename: str):
    if not _SLIDE_FILE_RE.match(filename or ""):
        raise HTTPException(400, "Invalid slide filename")
    path = kb_store.slides_dir(file_id) / filename
    if not path.is_file():
        raise HTTPException(404, "Slide not found")
    return FileResponse(path, media_type="image/jpeg")


def _bg_ingest(file_id: str) -> None:
    """Run the async ingestion inside a fresh event loop (BackgroundTasks runs sync)."""
    asyncio.run(kb_run_ingestion(file_id))


@app.post("/api/kb/upload")
async def kb_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    kind: str = Form(""),
):
    """Upload a file into the knowledge base and start background ingestion."""
    filename = file.filename or ""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        file_type = "pdf"
        ext = ".pdf"
    elif lower.endswith(".pptx"):
        file_type = "ppt"
        ext = ".pptx"
        kind = ""
    else:
        raise HTTPException(400, "Only .pdf and .pptx files are accepted")

    if file_type == "pdf" and kind not in ("text", "image"):
        raise HTTPException(400, "For PDF uploads, 'kind' must be 'text' or 'image'")

    file_id = uuid.uuid4().hex[:12]
    meta = kb_store.create_file_entry(
        file_id, name=filename, file_type=file_type, kind=kind,
    )

    dst = kb_store.file_dir(file_id) / f"original{ext}"
    with dst.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    background_tasks.add_task(_bg_ingest, file_id)

    logger.info("kb upload: id=%s name=%s type=%s kind=%s", file_id, filename, file_type, kind)
    return JSONResponse(meta, status_code=202)


@app.delete("/api/kb/files/{file_id}")
def kb_delete_file(file_id: str):
    removed = kb_store.delete_file(file_id)
    if not removed:
        raise HTTPException(404, "File not found in knowledge base")
    return JSONResponse({"deleted": True, "id": file_id})


@app.post("/api/sessions/from-kb")
async def create_session_from_kb(
    file_id: str = Form(...),
    content_language: str = Form("auto"),
    tts_pace: float = Form(1.0),
):
    """Create a tutoring session from a precomputed KB file."""
    meta = kb_store.get_file(file_id)
    if not meta:
        raise HTTPException(404, "File not found in knowledge base")
    if meta.get("status") != "ready":
        raise HTTPException(400, f"File is not ready (status: {meta.get('status')})")

    content = kb_store.get_content(file_id)
    if not content:
        raise HTTPException(500, "Content data missing for this file")

    session_id = uuid.uuid4().hex[:10]
    room_name = f"tutor-{session_id}"
    content_type = content.get("content_type", "pdf")
    segments = content.get("segments") or []
    if not segments:
        raise HTTPException(400, "No segments found in the knowledge base file")

    total_pages = content.get("total_pages", len(segments))
    chapters = content.get("chapters") or []
    chapter_segments = content.get("chapter_segments") or []
    inferred_subject = content.get("inferred_subject", meta.get("name", ""))

    is_ppt_like = content_type == "ppt" or meta.get("kind") == "image"

    if is_ppt_like:
        for seg in segments:
            old_url = seg.get("slide_url", "")
            if old_url and f"/api/kb/files/{file_id}/" in old_url:
                pass
            elif seg.get("slide_file"):
                seg["slide_url"] = f"/api/kb/files/{file_id}/slides/{seg['slide_file']}"

    slides_dir_path = str(kb_store.slides_dir(file_id)) if is_ppt_like else ""

    session = {
        "session_id": session_id,
        "room_name": room_name,
        "content_type": content_type if content_type == "ppt" else ("ppt" if meta.get("kind") == "image" else "pdf"),
        "pdf_filename": meta.get("name", "") if content_type == "pdf" else "",
        "ppt_filename": meta.get("name", "") if content_type == "ppt" else "",
        "start_page": 1,
        "end_page": total_pages,
        "total_pages": total_pages,
        "chapter_title": inferred_subject,
        "selected_chapter_index": -1,
        "chapters": chapters,
        "chapter_segments": chapter_segments,
        "subject": inferred_subject,
        "inferred_subject": inferred_subject,
        "content_language": content_language,
        "chapter_preview": content.get("chapter_preview", ""),
        "segment_count": len(segments),
        "segments": segments,
        "rag_segment_count": len(content.get("rag_segments") or []),
        "rag_segments": content.get("rag_segments") or [],
        "tts_pace": tts_pace,
        "extraction_mode": content.get("extraction_mode", "kb"),
        "slides_dir": slides_dir_path,
        "kb_file_id": file_id,
    }
    (SESSIONS / f"{room_name}.json").write_text(
        json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    actual_content_type = session["content_type"]
    segments_for_ui = []
    for s in segments:
        entry: dict = {
            "pages": s.get("pages", []),
            "source_text": s.get("source_text", ""),
            "source_lang": s.get("source_lang", "unknown"),
        }
        if s.get("slide_index") is not None:
            entry["slide_index"] = s["slide_index"]
        if s.get("slide_url"):
            entry["slide_url"] = s["slide_url"]
        segments_for_ui.append(entry)

    return JSONResponse({
        "session_id": session_id,
        "room_name": room_name,
        "content_type": actual_content_type,
        "segment_count": len(segments),
        "total_pages": total_pages,
        "start_page": 1,
        "end_page": total_pages,
        "subject": inferred_subject,
        "content_language": content_language,
        "selected_chapter_index": -1,
        "chapters": chapters,
        "chapter_segments": chapter_segments,
        "segments": segments_for_ui,
        "tts_pace": tts_pace,
        "extraction_mode": session["extraction_mode"],
        "development_mode": not _is_production_env(),
        "kb_file_id": file_id,
    })


if WEB.is_dir():
    app.mount("/static", StaticFiles(directory=str(WEB)), name="static")
