"""Standalone Gemini Live connectivity smoke test.

Usage:
    python scripts/test_gemini_connect.py            # minimal payload
    python scripts/test_gemini_connect.py --full     # the app's real setup
    python scripts/test_gemini_connect.py --send-image path/to/slide.jpg
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
from pathlib import Path

import websockets
from dotenv import load_dotenv

# Allow package imports from the project root when running from scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

GEMINI_WS_BASE = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)


def build_minimal_setup(model: str) -> dict:
    return {
        "setup": {
            "model": f"models/{model}",
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "temperature": 0.7,
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {"voiceName": "Kore"},
                    },
                },
            },
            "systemInstruction": {"parts": [{"text": "You are a helpful assistant."}]},
            "inputAudioTranscription": {},
            "outputAudioTranscription": {},
            "realtimeInputConfig": {
                "automaticActivityDetection": {
                    "disabled": False,
                    "silenceDurationMs": 700,
                    "prefixPaddingMs": 300,
                },
                "activityHandling": "START_OF_ACTIVITY_INTERRUPTS",
            },
        },
    }


def build_app_setup(model: str) -> dict:
    """Use the actual setup our app sends, minus any per-session segments."""
    from app.live.gemini_live_session import GeminiLiveTutorSession

    fake_session = {
        "room_name": "tutor-test",
        "segments": [
            {"segment_id": "s_0001", "pages": [1], "source_text": "Hello world."}
        ],
        "rag_segments": [],
        "chapter_segments": [],
        "chapters": [],
        "subject": "Test",
        "content_language": "en",
        "chapter_title": "Test",
        "chapter_preview": "test",
        "selected_chapter_index": -1,
    }
    tutor = GeminiLiveTutorSession(fake_session, api_key="x")
    return tutor._build_setup_message()


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--send-image", metavar="PATH", help="After setup, send one JPEG media_chunk")
    args = parser.parse_args()
    use_full = args.full
    image_path = Path(args.send_image).resolve() if args.send_image else None
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        print("GEMINI_API_KEY missing in .env", file=sys.stderr)
        return 1
    model = (os.getenv("GEMINI_MODEL") or "gemini-3.1-flash-live-preview").strip()
    print(f"connecting model={model} key_prefix={api_key[:6]} mode={'FULL' if use_full else 'minimal'}", flush=True)
    url = f"{GEMINI_WS_BASE}?key={api_key}"
    try:
        async with websockets.connect(url, max_size=32 * 1024 * 1024) as ws:
            setup = build_app_setup(model) if use_full else build_minimal_setup(model)
            print(f"connected; sending setup ({len(json.dumps(setup))} bytes)", flush=True)
            await ws.send(json.dumps(setup))
            print("waiting for setupComplete (up to 15s)...", flush=True)
            try:
                async with asyncio.timeout(15):
                    msg_count = 0
                    async for raw in ws:
                        msg_count += 1
                        if isinstance(raw, bytes):
                            raw = raw.decode("utf-8", errors="replace")
                        try:
                            data = json.loads(raw)
                            preview = json.dumps(data)[:1500]
                        except json.JSONDecodeError:
                            data = None
                            preview = raw[:1500]
                        print(f"[msg {msg_count}] {preview}", flush=True)
                        if isinstance(data, dict) and "setupComplete" in data:
                            print("OK setupComplete received -- payload is accepted", flush=True)
                            if image_path:
                                if not image_path.is_file():
                                    print(f"image not found: {image_path}", flush=True)
                                    return 5
                                b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
                                await ws.send(
                                    json.dumps(
                                        {
                                            "realtimeInput": {
                                                "video": {
                                                    "mimeType": "image/jpeg",
                                                    "data": b64,
                                                }
                                            }
                                        }
                                    )
                                )
                                print(
                                    f"sent video blob ({image_path.name}, {len(b64)} b64 chars)",
                                    flush=True,
                                )
                            return 0
                        if msg_count >= 5:
                            print("received 5 messages without setupComplete; stopping")
                            return 2
            except TimeoutError:
                print("TIMEOUT -- no setupComplete in 15s", flush=True)
                return 3
    except Exception as exc:
        print(f"CONNECT/PROTOCOL ERROR: {type(exc).__name__}: {exc}", flush=True)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
