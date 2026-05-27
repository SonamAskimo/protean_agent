# Protean Agent (Gemini Live)

Dedicated voice tutoring agent for **Protean** — PDF text lessons and PowerPoint slide decks. No segment quiz or analytics dashboard.

## Project layout

```
protean-agent/
├── app/                 # Python package
│   ├── server.py        # Compatibility entrypoint (uvicorn app.server:app)
│   ├── api/             # FastAPI routes / HTTP + WS
│   ├── live/            # Gemini Live websocket bridge
│   ├── tutoring/        # Prompts + LangGraph orchestration
│   ├── ingest/          # PDF/PPT extract + language detection
│   └── core/            # Shared paths/config helpers
├── web/
│   └── index.html       # Browser UI (served at /static)
├── samples/             # Example Protean training PDFs/PPTs (local testing)
├── scripts/             # Dev utilities (e.g. Gemini connectivity test)
├── uploads/             # Runtime uploads (gitignored)
├── sessions/            # Session JSON (gitignored)
├── requirements.txt
├── Dockerfile
└── .env.example
```

## Requirements

- Python 3.11+
- `GEMINI_API_KEY` in `.env`
- **PDF sessions**: PyMuPDF, optional Tesseract for OCR
- **PPT sessions**: [LibreOffice](https://www.libreoffice.org/) installed (provides `soffice`)

### LibreOffice (PowerPoint only)

PPT upload renders each slide to JPEG via LibreOffice headless:

1. Install LibreOffice on the machine running the server.
2. Ensure `soffice` is on `PATH`, **or** set in `.env`:

```env
SOFFICE_PATH=C:\Program Files\LibreOffice\program\soffice.exe
```

## Run

```powershell
cd protean-agent
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.server:app --reload --port 8080
```

Open http://127.0.0.1:8080

## Content types

| Upload | API | UI | Agent vision |
|--------|-----|-----|----------------|
| `.pdf` | `POST /api/sessions` | Text excerpt | No (text only) |
| `.pptx` | `POST /api/sessions/ppt` | Full-screen slide | Yes (`realtimeInput.video` JPEG per slide change) |

## Differences from `gemini-integrated`

- **No end-of-segment quiz** — no quiz tools, prompts, or auto-advance-after-quiz logic
- **No analytics** — no event ingestion, dashboard, or session scoring
- Branding and UI tuned for Protean

## Smoke test

```powershell
python scripts/test_gemini_connect.py
```

## Docker

```powershell
docker build -t protean-agent .
docker run --rm -p 8080:80 --env-file .env protean-agent
```
