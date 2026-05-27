from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Literal
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image
import pytesseract


@dataclass(frozen=True)
class Paragraph:
    id: str
    page: int  # 1-indexed
    text: str


@dataclass(frozen=True)
class Chapter:
    chapter_id: str
    title: str
    level: int
    start_page: int  # 1-indexed inclusive
    end_page: int  # 1-indexed inclusive
    source: str  # "toc" | "heuristic"


_MULTI_SPACE = re.compile(r"[ \t]+")
_CHAPTER_LINE_RE = re.compile(
    r"^(?:chapter|chap|unit|module|lesson|part)\s*[\divxlcdm]+(?:\s*[-:.)]\s*|\s+)(.+)?$",
    re.IGNORECASE,
)
_ALL_CAPS_RE = re.compile(r"^[A-Z][A-Z0-9 ,\-:&()]{5,}$")


def _clean(text: str) -> str:
    text = text.replace("\u00ad", "")  # soft hyphen
    text = text.replace("\r", "\n")
    text = _MULTI_SPACE.sub(" ", text)
    # preserve paragraph boundaries; collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_title(text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    t = re.sub(r"^[\-:.)\s]+", "", t)
    return t


def _sanitize_outline_chapters(toc: list[list], total_pages: int) -> list[Chapter]:
    raw: list[tuple[int, str, int]] = []
    for item in toc:
        if len(item) < 3:
            continue
        try:
            level = int(item[0] or 1)
            title = _normalize_title(str(item[1] or ""))
            page = int(item[2] or 1)
        except Exception:
            continue
        if not title:
            continue
        page = max(1, min(page, total_pages))
        raw.append((max(1, level), title, page))

    if not raw:
        return []

    chapters: list[Chapter] = []
    starts_seen: set[tuple[int, str]] = set()
    for idx, (level, title, start_page) in enumerate(raw):
        dedupe_key = (start_page, title.lower())
        if dedupe_key in starts_seen:
            continue
        starts_seen.add(dedupe_key)
        next_page = total_pages + 1
        for j in range(idx + 1, len(raw)):
            cand = raw[j][2]
            if cand > start_page:
                next_page = cand
                break
        end_page = max(start_page, min(total_pages, next_page - 1))
        chapters.append(
            Chapter(
                chapter_id=f"ch_{len(chapters) + 1:03d}",
                title=title,
                level=level,
                start_page=start_page,
                end_page=end_page,
                source="toc",
            )
        )
    return chapters


def _detect_heading_like_line(text: str) -> str | None:
    line = _normalize_title(text.split("\n", 1)[0])
    if len(line) < 6 or len(line) > 120:
        return None
    if _CHAPTER_LINE_RE.match(line):
        return line
    if _ALL_CAPS_RE.match(line):
        return line.title()
    return None


def _heuristic_chapters(doc: fitz.Document) -> list[Chapter]:
    rows: list[tuple[str, int]] = []
    for p in range(doc.page_count):
        page = doc.load_page(p)
        raw = _clean(page.get_text("text"))
        if not raw:
            continue
        hit = _detect_heading_like_line(raw)
        if not hit:
            continue
        rows.append((hit, p + 1))

    if not rows:
        return []

    chapters: list[Chapter] = []
    for idx, (title, start_page) in enumerate(rows):
        next_start = rows[idx + 1][1] if idx + 1 < len(rows) else (doc.page_count + 1)
        end_page = max(start_page, min(doc.page_count, next_start - 1))
        chapters.append(
            Chapter(
                chapter_id=f"ch_{idx + 1:03d}",
                title=title,
                level=1,
                start_page=start_page,
                end_page=end_page,
                source="heuristic",
            )
        )
    return chapters


def detect_chapters(pdf_path: str) -> list[Chapter]:
    """Detect chapter ranges using PDF outline/bookmarks with a text fallback."""
    doc = fitz.open(pdf_path)
    try:
        total = doc.page_count
        toc = doc.get_toc(simple=True) or []
        chapters = _sanitize_outline_chapters(toc, total)
        if chapters:
            return chapters
        return _heuristic_chapters(doc)
    finally:
        doc.close()


def extract_paragraphs(pdf_path: str, start_page: int, end_page: int) -> list[Paragraph]:
    if start_page > end_page:
        start_page, end_page = end_page, start_page

    doc = fitz.open(pdf_path)
    try:
        n = doc.page_count
        start_page = max(1, min(start_page, n))
        end_page = max(1, min(end_page, n))

        paras: list[Paragraph] = []
        pid = 0
        for p in range(start_page - 1, end_page):
            page = doc.load_page(p)
            # "text" keeps reading order reasonably for most textbooks
            raw = page.get_text("text")
            cleaned = _clean(raw)
            if not cleaned:
                continue

            # Split into paragraphs using blank lines.
            blocks = [b.strip() for b in cleaned.split("\n\n") if b.strip()]
            for b in blocks:
                # Avoid ultra-short noise lines
                if len(b) < 10:
                    continue
                pid += 1
                paras.append(Paragraph(id=f"p_{pid:05d}", page=p + 1, text=b))
        return paras
    finally:
        doc.close()


def extract_paragraphs_ocr(
    pdf_path: str,
    start_page: int,
    end_page: int,
    *,
    lang: str = "eng",
    zoom: float = 2.0,
    psm: int = 6,
) -> list[Paragraph]:
    """
    OCR extraction for scanned PDFs using Tesseract.

    Returns paragraphs in the same Paragraph schema as extract_paragraphs().
    """
    if start_page > end_page:
        start_page, end_page = end_page, start_page

    doc = fitz.open(pdf_path)
    try:
        n = doc.page_count
        start_page = max(1, min(start_page, n))
        end_page = max(1, min(end_page, n))

        paras: list[Paragraph] = []
        pid = 0
        for p in range(start_page - 1, end_page):
            page = doc.load_page(p)
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png")))

            raw = pytesseract.image_to_string(
                img,
                lang=lang,
                config=f"--psm {psm}",
            )
            cleaned = _clean(raw)
            if not cleaned:
                continue

            # OCR output often has unstable paragraph breaks. Keep one paragraph per page.
            if len(cleaned) < 10:
                continue

            pid += 1
            paras.append(Paragraph(id=f"p_{pid:05d}", page=p + 1, text=cleaned))
        return paras
    finally:
        doc.close()


def extract_paragraphs_by_mode(
    pdf_path: str,
    start_page: int,
    end_page: int,
    *,
    mode: Literal["simple", "ocr"] = "simple",
    ocr_lang: str = "eng",
) -> list[Paragraph]:
    if mode == "ocr":
        return extract_paragraphs_ocr(
            pdf_path,
            start_page,
            end_page,
            lang=ocr_lang,
        )
    return extract_paragraphs(pdf_path, start_page, end_page)


def count_pages(pdf_path: str) -> int:
    doc = fitz.open(pdf_path)
    try:
        return doc.page_count
    finally:
        doc.close()


def paragraphs_to_plain_text(paragraphs: list[Paragraph], *, page_markers: bool = True) -> str:
    """Join extracted paragraphs for downstream LLM or file export."""
    parts: list[str] = []
    for para in paragraphs:
        if page_markers:
            parts.append(f"[Page {para.page}]\n{para.text}")
        else:
            parts.append(para.text)
    return "\n\n".join(parts)


def extract_pages_to_text_file(
    pdf_path: str,
    start_page: int,
    end_page: int,
    output_path: str | Path,
    *,
    page_markers: bool = True,
    file_header: bool = True,
) -> tuple[int, int, int]:
    """
    Extract selectable text from PDF page range and write UTF-8 plain text.
    Returns (paragraph_count, resolved_start_page, resolved_end_page).
    """
    s, e = start_page, end_page
    if s > e:
        s, e = e, s
    n = count_pages(pdf_path)
    resolved_start = max(1, min(s, n))
    resolved_end = max(1, min(e, n))

    paras = extract_paragraphs(pdf_path, start_page, end_page)
    body = paragraphs_to_plain_text(paras, page_markers=page_markers)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if file_header:
        name = Path(pdf_path).name
        header = (
            f"# PDF extract\n# file: {name}\n# pages: {resolved_start}-{resolved_end}\n"
            f"# paragraphs: {len(paras)}\n\n"
        )
        text = header + body
    else:
        text = body
    out.write_text(text, encoding="utf-8")
    return len(paras), resolved_start, resolved_end
