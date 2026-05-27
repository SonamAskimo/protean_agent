"""Filesystem-backed Knowledge Base storage.

Layout on disk
--------------
knowledge_base/
    index.json              # [{id, name, type, kind, status, created_at}, ...]
    <file_id>/
        meta.json           # full metadata dict for one file
        original.pdf|.pptx  # uploaded original
        content.json        # precomputed segments / slide descriptions
        slides/             # only for ppt and image-pdf
            slide_001.jpg
            ...
"""

from __future__ import annotations

import json
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core.paths import KB

_lock = threading.Lock()

KB.mkdir(exist_ok=True)
_INDEX = KB / "index.json"
if not _INDEX.exists():
    _INDEX.write_text("[]", encoding="utf-8")


def _read_index() -> list[dict]:
    try:
        return json.loads(_INDEX.read_text(encoding="utf-8"))
    except Exception:
        return []


def _write_index(entries: list[dict]) -> None:
    _INDEX.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


def list_files() -> list[dict]:
    return _read_index()


def get_file(file_id: str) -> dict | None:
    meta_path = KB / file_id / "meta.json"
    if not meta_path.is_file():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def get_content(file_id: str) -> dict | None:
    content_path = KB / file_id / "content.json"
    if not content_path.is_file():
        return None
    return json.loads(content_path.read_text(encoding="utf-8"))


def file_dir(file_id: str) -> Path:
    return KB / file_id


def slides_dir(file_id: str) -> Path:
    return KB / file_id / "slides"


def create_file_entry(
    file_id: str,
    *,
    name: str,
    file_type: str,
    kind: str = "",
) -> dict:
    """Create a new KB entry with status ``processing``.

    Returns the meta dict that was written.
    """
    entry_dir = KB / file_id
    entry_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).isoformat()
    meta: dict[str, Any] = {
        "id": file_id,
        "name": name,
        "type": file_type,
        "kind": kind,
        "status": "processing",
        "error": None,
        "created_at": now,
    }
    (entry_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with _lock:
        entries = _read_index()
        entries = [e for e in entries if e.get("id") != file_id]
        entries.append({
            "id": file_id,
            "name": name,
            "type": file_type,
            "kind": kind,
            "status": "processing",
            "created_at": now,
        })
        _write_index(entries)

    return meta


def save_content(file_id: str, content: dict) -> None:
    """Write precomputed content and mark the entry as ready."""
    entry_dir = KB / file_id
    entry_dir.mkdir(parents=True, exist_ok=True)
    (entry_dir / "content.json").write_text(
        json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    set_status(file_id, "ready")


def set_status(file_id: str, status: str, error_msg: str | None = None) -> None:
    meta_path = KB / file_id / "meta.json"
    if not meta_path.is_file():
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["status"] = status
    if error_msg is not None:
        meta["error"] = error_msg
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with _lock:
        entries = _read_index()
        for e in entries:
            if e.get("id") == file_id:
                e["status"] = status
                break
        _write_index(entries)


def delete_file(file_id: str) -> bool:
    entry_dir = KB / file_id
    removed = False
    if entry_dir.is_dir():
        shutil.rmtree(entry_dir, ignore_errors=True)
        removed = True

    with _lock:
        entries = _read_index()
        before = len(entries)
        entries = [e for e in entries if e.get("id") != file_id]
        _write_index(entries)
        if len(entries) < before:
            removed = True

    return removed
