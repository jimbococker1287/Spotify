# Local Production-Smoke Stack

This stack gives you a near-production local validation path with:

- Postgres for Taste OS state
- Redis for shared rate limiting
- the production-style ASGI `predict` API
- the production-style ASGI `taste-os` API

It is useful for confirming that:

- the services boot with SQL + Redis attached
- JWT/token auth settings work
- the deployment registry `stable` channel resolves correctly
- `/readyz` and `/metrics/prometheus` behave as expected

## Setup

1. Publish a promoted run into the local stable channel:

```bash
python -m spotify.deployment_registry \
  --run-dir outputs/runs/<run_id> \
  --registry-root outputs/deployments/registry \
  --channel stable
```

2. Copy the env file:

```bash
cd deploy/local
cp .env.example .env
```

3. Fill in the auth values in `.env`.

4. Build and start the stack:

```bash
docker compose -f production-smoke.compose.yaml up --build
```

## Endpoints

- Prediction API: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Taste OS API: [http://127.0.0.1:8010](http://127.0.0.1:8010)
- Taste OS studio: [http://127.0.0.1:8010/taste-os](http://127.0.0.1:8010/taste-os)

## Quick Checks

```bash
curl -s http://127.0.0.1:8000/readyz
curl -s http://127.0.0.1:8010/readyz
curl -s http://127.0.0.1:8000/metrics/prometheus
```
