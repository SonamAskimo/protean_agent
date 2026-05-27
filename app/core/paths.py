"""Project root and runtime data directories (sibling to ``app/``)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
UPLOADS = ROOT / "uploads"
SESSIONS = ROOT / "sessions"
WEB = ROOT / "web"
KB = ROOT / "knowledge_base"
