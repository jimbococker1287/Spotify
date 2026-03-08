FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    RUN_DIR=/app/outputs/models/champion \
    DATA_DIR=/app/data/raw \
    MAX_TOP_K=20 \
    AUTH_TOKEN=

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY spotify ./spotify

RUN python -m pip install --no-cache-dir -e . --no-deps

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD python -c "import os,sys,urllib.request;port=os.getenv('PORT','8000');url=f'http://127.0.0.1:{port}/health';r=urllib.request.urlopen(url,timeout=3);sys.exit(0 if r.status==200 else 1)"

CMD ["sh", "-c", "python -m spotify.predict_service --run-dir \"$RUN_DIR\" --data-dir \"$DATA_DIR\" --host \"$HOST\" --port \"$PORT\" --max-top-k \"$MAX_TOP_K\" --auth-token \"$AUTH_TOKEN\""]
