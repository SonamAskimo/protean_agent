"""Compatibility entrypoint for ASGI servers.

Run:
    uvicorn app.server:app --reload --port 8080
"""

from .api.server import app

