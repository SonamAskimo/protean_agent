# Single-container image: FastAPI + Gemini Live WebSocket tutor (PDF/PPT ingest).
#
# LibreOffice (`soffice`) is installed for PPT→PDF conversion used by ppt_extract.py.
#
# Build:
#   docker build -t protean-agent .
# Run:
#   docker run --rm -p 8080:8080 --env-file .env protean-agent

FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    SOFFICE_PATH=/usr/bin/soffice

# ffmpeg/PyAV; OCR for PDF ingest; LibreOffice headless + fonts for PPT slide export
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        tesseract-ocr \
        libreoffice-impress \
        fonts-liberation \
        fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --no-compile -r /app/requirements.txt

COPY . /app/

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os,urllib.request;\
p=os.environ.get('PORT','8080');\
urllib.request.urlopen(f'http://127.0.0.1:{p}/', timeout=4)" || exit 1

ENTRYPOINT ["/docker-entrypoint.sh"]
