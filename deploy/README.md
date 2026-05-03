# Deployment Assets

This directory now has four layers:

- `kubernetes/`: raw Kubernetes manifests for the two APIs
- `ecs/`: ECS/Fargate task definition examples
- `terraform/aws/`: AWS-first infrastructure baseline for VPC, ECS, ALB, EFS, Postgres, Redis, ECR, and secrets
- `local/`: local production-smoke stack with Postgres + Redis + both APIs

Recommended order:

1. Validate the wiring locally with `local/`
2. Provision cloud infrastructure with `terraform/aws/`
3. Use `ecs/` or `kubernetes/` as the service-level deployment reference
