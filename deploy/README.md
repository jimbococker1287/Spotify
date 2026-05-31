# Deployment Assets

This directory now has four layers plus the serving smoke gates:

- `kubernetes/`: raw Kubernetes manifests for the two APIs
- `ecs/`: ECS/Fargate task definition examples
- `terraform/aws/`: AWS-first infrastructure baseline for VPC, ECS, ALB, EFS, Postgres, Redis, ECR, and secrets
- `local/`: local production-smoke stack with Postgres + Redis + both APIs
- `make release-readiness`: artifact-level verification that the active channel alias, release manifest, serving bundles, and deploy templates agree
- `make production-smoke`: in-process ASGI evidence for `/v1/readyz`, `/v1/metrics`, prediction, and Taste OS session requests

Recommended order:

1. Validate the wiring locally with `local/`
2. Publish a promoted run with `make deploy-release`
3. Run `make release-readiness EXTRA_ARGS='--run-dir outputs/deployments/registry/channels/stable --require-registry --strict'`
4. Run `make production-smoke EXTRA_ARGS='--run-dir outputs/deployments/registry/channels/stable --strict'`
5. Provision cloud infrastructure with `terraform/aws/`
6. Use `ecs/` or `kubernetes/` as the service-level deployment reference
