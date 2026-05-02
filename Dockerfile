FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    SERVICE_MODE=predict \
    RUN_DIR=/app/outputs/models/champion \
    DATA_DIR= \
    TASTE_OS_OUTPUT_DIR=/app/outputs/analysis/taste_os_service \
    TASTE_OS_STATE_DB= \
    MAX_TOP_K=20 \
    AUTH_TOKEN= \
    REQUEST_RATE_LIMIT=240

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY spotify ./spotify

RUN python -m pip install --no-cache-dir -e . --no-deps

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD python -c "import os,sys,urllib.request;port=os.getenv('PORT','8000');url=f'http://127.0.0.1:{port}/health';r=urllib.request.urlopen(url,timeout=3);sys.exit(0 if r.status==200 else 1)"

CMD ["sh", "-c", "set -- python -m spotify.service_api --app \"$SERVICE_MODE\" --run-dir \"$RUN_DIR\" --host \"$HOST\" --port \"$PORT\" --max-top-k \"$MAX_TOP_K\" --request-rate-limit \"$REQUEST_RATE_LIMIT\"; if [ -n \"$AUTH_TOKEN\" ]; then set -- \"$@\" --auth-token \"$AUTH_TOKEN\"; fi; if [ -n \"$DATA_DIR\" ]; then set -- \"$@\" --data-dir \"$DATA_DIR\"; fi; if [ -n \"$TASTE_OS_OUTPUT_DIR\" ]; then set -- \"$@\" --output-dir \"$TASTE_OS_OUTPUT_DIR\"; fi; if [ -n \"$TASTE_OS_STATE_DB\" ]; then set -- \"$@\" --state-db \"$TASTE_OS_STATE_DB\"; fi; exec \"$@\""]
