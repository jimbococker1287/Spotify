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
    TASTE_OS_DATABASE_URL= \
    MAX_TOP_K=20 \
    AUTH_TOKEN= \
    AUTH_MODE=auto \
    AUTH_SCOPE=mutations \
    JWT_SECRET= \
    JWKS_URL= \
    JWT_ISSUER= \
    JWT_AUDIENCE= \
    JWT_ALGORITHMS=RS256 \
    JWT_REQUIRED_SCOPES= \
    JWT_LEEWAY_SECONDS=0 \
    REQUEST_RATE_LIMIT=240 \
    RATE_LIMIT_BACKEND=memory \
    REDIS_URL= \
    REQUIRE_DEPLOYMENT_REGISTRY=0 \
    OTEL_ENABLED=1

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY spotify ./spotify

RUN python -m pip install --no-cache-dir -e . --no-deps

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD python -c "import os,sys,urllib.request;port=os.getenv('PORT','8000');url=f'http://127.0.0.1:{port}/readyz';r=urllib.request.urlopen(url,timeout=3);sys.exit(0 if r.status==200 else 1)"

CMD ["sh", "-c", "set -- python -m spotify.service_api --app \"$SERVICE_MODE\" --run-dir \"$RUN_DIR\" --host \"$HOST\" --port \"$PORT\" --max-top-k \"$MAX_TOP_K\" --request-rate-limit \"$REQUEST_RATE_LIMIT\" --rate-limit-backend \"$RATE_LIMIT_BACKEND\" --auth-mode \"$AUTH_MODE\" --auth-scope \"$AUTH_SCOPE\" --jwt-algorithms \"$JWT_ALGORITHMS\" --jwt-leeway-seconds \"$JWT_LEEWAY_SECONDS\"; if [ -n \"$AUTH_TOKEN\" ]; then set -- \"$@\" --auth-token \"$AUTH_TOKEN\"; fi; if [ -n \"$JWT_SECRET\" ]; then set -- \"$@\" --jwt-secret \"$JWT_SECRET\"; fi; if [ -n \"$JWKS_URL\" ]; then set -- \"$@\" --jwks-url \"$JWKS_URL\"; fi; if [ -n \"$JWT_ISSUER\" ]; then set -- \"$@\" --jwt-issuer \"$JWT_ISSUER\"; fi; if [ -n \"$JWT_AUDIENCE\" ]; then set -- \"$@\" --jwt-audience \"$JWT_AUDIENCE\"; fi; if [ -n \"$JWT_REQUIRED_SCOPES\" ]; then set -- \"$@\" --jwt-required-scopes \"$JWT_REQUIRED_SCOPES\"; fi; if [ -n \"$DATA_DIR\" ]; then set -- \"$@\" --data-dir \"$DATA_DIR\"; fi; if [ -n \"$TASTE_OS_OUTPUT_DIR\" ]; then set -- \"$@\" --output-dir \"$TASTE_OS_OUTPUT_DIR\"; fi; if [ -n \"$TASTE_OS_STATE_DB\" ]; then set -- \"$@\" --state-db \"$TASTE_OS_STATE_DB\"; fi; if [ -n \"$TASTE_OS_DATABASE_URL\" ]; then set -- \"$@\" --state-db-url \"$TASTE_OS_DATABASE_URL\"; fi; if [ -n \"$REDIS_URL\" ]; then set -- \"$@\" --redis-url \"$REDIS_URL\"; fi; if [ \"$REQUIRE_DEPLOYMENT_REGISTRY\" = \"1\" ]; then set -- \"$@\" --require-deployment-registry; fi; if [ \"$OTEL_ENABLED\" = \"0\" ]; then set -- \"$@\" --disable-otel; fi; exec \"$@\""]
