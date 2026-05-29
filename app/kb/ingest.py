"""Background ingestion pipelines for the Knowledge Base.

Three flows:
  1. Text PDF   — extract paragraphs, segment, detect chapters (reuses existing ingest modules)
  2. Image PDF  — render each page to JPEG, caption via Gemini non-live
  3. PPT        — render slides to JPEG + extract text, caption via Gemini non-live
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from pathlib import Path

import fitz  # PyMuPDF

from ..ingest.lang_detect import detect_source_lang, resolve_segment_lang
from ..ingest.pdf_extract import (
    count_pages,
    detect_chapters,
    extract_paragraphs_by_mode,
    paragraphs_to_plain_text,
)
from ..ingest.ppt_extract import ingest_pptx
from ..ingest.segmenter import build_segments
from .slide_type import enrich_segment
from . import store

logger = logging.getLogger("kb-ingest")

_JPEG_QUALITY = 88
_JPEG_MAX_WIDTH = 1920
_CAPTION_MODEL = None


def _get_caption_model():
    global _CAPTION_MODEL
    if _CAPTION_MODEL is not None:
        return _CAPTION_MODEL
    try:
        import google.generativeai as genai

        api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
        if not api_key:
            logger.warning("GEMINI_API_KEY not set — image captioning will be skipped")
            return None
        genai.configure(api_key=api_key)
        model_name = (
            (os.getenv("GEMINI_CAPTION_MODEL") or "").strip() or "gemini-2.0-flash"
        )
        _CAPTION_MODEL = genai.GenerativeModel(model_name)
        return _CAPTION_MODEL
    except Exception:
        logger.exception("Failed to initialise Gemini caption model")
        return None


async def _caption_image(image_path: Path) -> str:
    """Send a slide/page image to Gemini non-live and get a text description."""
    model = _get_caption_model()
    if model is None:
        return "(Image description unavailable — GEMINI_API_KEY not configured)"

    import google.generativeai as genai

    img_bytes = image_path.read_bytes()
    b64 = base64.b64encode(img_bytes).decode()
    image_part = {"mime_type": "image/jpeg", "data": b64}

    prompt = (
        "You are an expert educational content analyzer. "
        "Describe this slide/page image in detail for a student who cannot see it. "
        "Include all text visible on the image, diagrams, charts, equations, "
        "and any visual relationships. Be thorough but concise."
    )

    try:
        resp = await asyncio.to_thread(
            model.generate_content, [prompt, image_part]
        )
        return (resp.text or "").strip() or "(No description generated)"
    except Exception as exc:
        logger.warning("Caption failed for %s: %s", image_path.name, exc)
        return f"(Caption failed: {exc})"


def _render_pdf_pages_to_jpegs(pdf_path: str, out_dir: Path) -> list[Path]:
    """Render every page of a PDF to slide_NNN.jpg (reuse the PPT approach)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(fitz, "TOOLS"):
        fitz.TOOLS.mupdf_display_errors(False)
    doc = fitz.open(pdf_path)
    paths: list[Path] = []
    try:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            pix = page.get_pixmap(dpi=220, alpha=False)
            if pix.width > _JPEG_MAX_WIDTH:
                scale = _JPEG_MAX_WIDTH / pix.width
                mat = fitz.Matrix(scale, scale)
                pix = page.get_pixmap(matrix=mat, alpha=False)
            out = out_dir / f"slide_{i + 1:03d}.jpg"
            pix.save(str(out), jpg_quality=_JPEG_QUALITY)
            paths.append(out)
    finally:
        doc.close()
    return paths


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
        out.append({
            "segment_id": s.segment_id,
            "pages": s.pages,
            "source_text": s.source_text,
            "source_lang": sl,
            "chapter_id": chapter_id,
            "chapter_title": chapter_title,
            "chapter_index": chapter_index,
        })
    return out


def _paras_for_range(paras: list, start_page: int, end_page: int) -> list:
    lo, hi = sorted((start_page, end_page))
    return [p for p in paras if lo <= p.page <= hi]


def _needs_ocr_fallback(paras: list, total_pages: int) -> bool:
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


# ---------------------------------------------------------------------------
# Flow 1: Text-embedded PDF
# ---------------------------------------------------------------------------

async def ingest_text_pdf(file_id: str) -> None:
    """Extract text, segment, detect chapters — same as current PDF flow."""
    try:
        meta = store.get_file(file_id)
        if not meta:
            return
        entry_dir = store.file_dir(file_id)
        pdf_path = str(entry_dir / "original.pdf")

        total_pages = await asyncio.to_thread(count_pages, pdf_path)
        lang_opt = "auto"
        ocr_lang = os.environ.get("OCR_TESS_LANG", "eng")

        rag_paras = await asyncio.to_thread(
            extract_paragraphs_by_mode, pdf_path, 1, total_pages,
            mode="simple", ocr_lang=ocr_lang,
        )
        mode_used = "simple"

        if _needs_ocr_fallback(rag_paras, total_pages):
            try:
                rag_paras = await asyncio.to_thread(
                    extract_paragraphs_by_mode, pdf_path, 1, total_pages,
                    mode="ocr", ocr_lang=ocr_lang,
                )
                mode_used = "ocr"
            except Exception:
                if not rag_paras:
                    raise

        chapters_raw = await asyncio.to_thread(detect_chapters, pdf_path)
        chapters_payload = [
            {
                "chapter_id": ch.chapter_id,
                "title": ch.title,
                "level": ch.level,
                "start_page": ch.start_page,
                "end_page": ch.end_page,
                "source": ch.source,
            }
            for ch in (chapters_raw or [])
        ]

        extracted_text = paragraphs_to_plain_text(rag_paras, page_markers=True)
        seg_inputs = build_segments(rag_paras, max_segments=120)
        segments = _build_segment_dicts(seg_inputs, lang_opt=lang_opt)

        rag_seg_inputs = build_segments(rag_paras, max_segments=400)
        rag_segments = _build_segment_dicts(rag_seg_inputs, lang_opt=lang_opt)

        chapter_segments: list[dict] = []
        for i, ch in enumerate(chapters_payload):
            ch_paras = _paras_for_range(
                rag_paras, int(ch["start_page"]), int(ch["end_page"])
            )
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
            chapter_segments.append({
                "chapter_index": i,
                "chapter_id": ch["chapter_id"],
                "title": ch["title"],
                "start_page": ch["start_page"],
                "end_page": ch["end_page"],
                "segment_count": len(ch_segs),
                "segments": ch_segs,
            })

        chapter_preview = " ".join(p.text for p in rag_paras)[:1500]
        inferred_subject = meta.get("name", "Unspecified")

        content = {
            "content_type": "pdf",
            "total_pages": total_pages,
            "segments": segments,
            "chapters": chapters_payload,
            "chapter_segments": chapter_segments,
            "rag_segments": rag_segments,
            "extracted_text": extracted_text,
            "chapter_preview": chapter_preview,
            "inferred_subject": inferred_subject,
            "extraction_mode": mode_used,
        }
        store.save_content(file_id, content)
        logger.info(
            "text-pdf ingested: file_id=%s pages=%d segments=%d mode=%s",
            file_id, total_pages, len(segments), mode_used,
        )
    except Exception as exc:
        logger.exception("text-pdf ingest failed: file_id=%s", file_id)
        store.set_status(file_id, "error", str(exc))


# ---------------------------------------------------------------------------
# Flow 2: Image-based PDF
# ---------------------------------------------------------------------------

async def ingest_image_pdf(file_id: str) -> None:
    """Render each page to JPEG, caption via Gemini, build segments."""
    try:
        meta = store.get_file(file_id)
        if not meta:
            return
        entry_dir = store.file_dir(file_id)
        pdf_path = str(entry_dir / "original.pdf")
        slides_out = store.slides_dir(file_id)

        image_paths = await asyncio.to_thread(
            _render_pdf_pages_to_jpegs, pdf_path, slides_out
        )
        if not image_paths:
            raise ValueError("No pages could be rendered from the PDF")

        segments: list[dict] = []
        for i, img_path in enumerate(image_paths):
            description = await _caption_image(img_path)
            slide_url = f"/api/kb/files/{file_id}/slides/{img_path.name}"
            seg = {
                "segment_id": f"s_{i + 1:04d}",
                "pages": [i + 1],
                "slide_index": i,
                "slide_url": slide_url,
                "slide_file": img_path.name,
                "source_text": description,
                "source_lang": "en",
                "chapter_id": "",
                "chapter_title": meta.get("name", "Document"),
                "chapter_index": -1,
            }
            enrich_segment(seg, description)
            segments.append(seg)

        content = {
            "content_type": "pdf",
            "kind": "image",
            "total_pages": len(image_paths),
            "segments": segments,
            "chapters": [],
            "chapter_segments": [],
            "rag_segments": [],
            "extracted_text": "",
            "chapter_preview": "",
            "inferred_subject": meta.get("name", "Unspecified"),
            "extraction_mode": "image",
        }
        store.save_content(file_id, content)
        logger.info(
            "image-pdf ingested: file_id=%s pages=%d", file_id, len(image_paths)
        )
    except Exception as exc:
        logger.exception("image-pdf ingest failed: file_id=%s", file_id)
        store.set_status(file_id, "error", str(exc))


# ---------------------------------------------------------------------------
# Flow 3: PPT
# ---------------------------------------------------------------------------

async def ingest_ppt(file_id: str) -> None:
    """Convert PPTX to slide JPEGs, extract text, caption via Gemini."""
    try:
        meta = store.get_file(file_id)
        if not meta:
            return
        entry_dir = store.file_dir(file_id)
        pptx_path = entry_dir / "original.pptx"
        slides_out = store.slides_dir(file_id)

        slide_rows = await asyncio.to_thread(
            ingest_pptx,
            pptx_path,
            slides_dir=slides_out,
            session_id=file_id,
        )

        if not slide_rows:
            raise ValueError("No slides found in the presentation")

        segments: list[dict] = []
        for s in slide_rows:
            description = await _caption_image(s.image_path)
            combined_text = s.source_text
            if description and not description.startswith("("):
                combined_text = f"{description}\n\n---\n[Slide text]\n{s.source_text}"

            slide_url = f"/api/kb/files/{file_id}/slides/{s.image_path.name}"
            detected = detect_source_lang(combined_text)
            sl = resolve_segment_lang("auto", detected)
            seg = {
                "segment_id": f"s_{s.slide_num:04d}",
                "pages": [s.slide_num],
                "slide_index": s.slide_index,
                "slide_url": slide_url,
                "slide_file": s.image_path.name,
                "source_text": combined_text,
                "source_lang": sl,
                "chapter_id": "",
                "chapter_title": meta.get("name", "Presentation"),
                "chapter_index": -1,
            }
            enrich_segment(seg, combined_text)
            segments.append(seg)

        deck_title = Path(meta.get("name", "deck")).stem
        chapter_preview = " ".join(
            (seg.get("source_text") or "")[:200] for seg in segments[:3]
        )[:1500]

        content = {
            "content_type": "ppt",
            "total_pages": len(segments),
            "segments": segments,
            "chapters": [],
            "chapter_segments": [],
            "rag_segments": [],
            "extracted_text": "",
            "chapter_preview": chapter_preview,
            "inferred_subject": deck_title,
            "extraction_mode": "ppt",
        }
        store.save_content(file_id, content)
        logger.info(
            "ppt ingested: file_id=%s slides=%d", file_id, len(segments)
        )
    except Exception as exc:
        logger.exception("ppt ingest failed: file_id=%s", file_id)
        store.set_status(file_id, "error", str(exc))


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

async def run_ingestion(file_id: str) -> None:
    """Route to the correct ingestion flow based on file metadata."""
    meta = store.get_file(file_id)
    if not meta:
        logger.error("run_ingestion: file_id=%s not found", file_id)
        return

    file_type = meta.get("type", "")
    kind = meta.get("kind", "")

    if file_type == "ppt":
        await ingest_ppt(file_id)
    elif file_type == "pdf" and kind == "image":
        await ingest_image_pdf(file_id)
    elif file_type == "pdf":
        await ingest_text_pdf(file_id)
    else:
        logger.error("Unknown file type/kind: %s/%s for %s", file_type, kind, file_id)
        store.set_status(file_id, "error", f"Unknown file type: {file_type}/{kind}")
