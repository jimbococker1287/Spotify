# Kubernetes Deployment Templates

These templates deploy the production-style ASGI services against a shared `outputs/` volume, a shared Redis rate-limit backend, and a SQL-backed Taste OS state store.

Expected supporting infrastructure:

- a persistent volume mounted at `/app/outputs`
- Redis reachable from the cluster
- a SQL database reachable from the cluster
- ingress / TLS configured separately

Recommended rollout flow:

1. Publish a promoted run into the local registry channel.
2. Mount the same shared `outputs/` path into the app pods.
3. Apply the config map, secrets, PVC, deployment, and service manifests.
4. Front the services with your preferred ingress / API gateway.

These manifests intentionally use placeholders like `CHANGE_ME` and example secret refs so you can adapt them to your cluster without editing application code.
