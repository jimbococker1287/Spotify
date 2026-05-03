# AWS Production Baseline

This Terraform stack provisions an AWS-first production baseline for the Spotify serving surfaces:

- VPC with 2 public and 2 private subnets
- Application Load Balancer
- ECS Fargate services for `predict` and `taste-os`
- EFS mounted at `/app/outputs`
- RDS Postgres for Taste OS state
- ElastiCache Redis for shared rate limiting
- ECR repository for the service image
- CloudWatch log groups
- Secrets Manager entries for API auth and the Taste OS database URL

This is designed to match the repo’s production assumptions:

- the services read promoted runs from `outputs/deployments/registry/channels/stable`
- both APIs mount the same shared `/app/outputs`
- Taste OS persists state in Postgres
- rate limiting uses Redis

## What You Still Need To Provide

- an AWS account and credentials
- a real container image pushed to ECR
- a promoted run published into the registry
- DNS records for `predict_host` and `taste_os_host`
- an ACM certificate ARN if you want HTTPS on the ALB

## Bootstrap

1. Copy the example vars file:

```bash
cd deploy/terraform/aws
cp terraform.tfvars.example terraform.tfvars
```

For the cheapest balanced starting point, use:

```bash
cp terraform.tfvars.cheap-balanced.example terraform.tfvars
```

2. Fill in:

- `container_image`
- `predict_host`
- `taste_os_host`
- `certificate_arn` if using HTTPS
- `auth_token`
- `jwt_secret`
- `db_password`

3. Initialize and apply:

```bash
terraform init
terraform plan
terraform apply
```

4. Build and push your container image to the emitted ECR repository.

5. Publish a promoted run into the deployment registry, either on shared storage or with remote artifact publishing:

```bash
python -m spotify.deployment_registry \
  --run-dir outputs/runs/<run_id> \
  --registry-root outputs/deployments/registry \
  --channel stable \
  --artifact-base-uri s3://your-bucket/spotify-releases \
  --publish-artifacts
```

6. Mount the shared `outputs/` path for the ECS tasks. This stack provisions EFS, but you still need to sync or materialize the promoted artifacts into that mounted path if you are not publishing them straight into a shared filesystem.

## Cheap And Balanced Profile

If you want to keep cost down without making the first deployment too brittle, use the `terraform.tfvars.cheap-balanced.example` profile:

- `1` ECS task for `predict`
- `1` ECS task for `taste-os`
- `1024 CPU / 2048 MiB` per task
- `db.t4g.micro` for Postgres
- `cache.t4g.micro` for Redis
- single ALB, single EFS filesystem, single RDS instance, single Redis node

Why this is the recommended starting point:

- it avoids paying for idle duplicate app tasks on day one
- it keeps enough memory headroom that the Python serving stack is unlikely to become the first failure mode
- it preserves the same architecture you’ll keep later, so upgrades are mostly resizing and scaling, not replatforming

What it does not protect you from:

- a single task restart will briefly interrupt that service
- there is no app-tier redundancy until you raise `predict_desired_count` and `taste_os_desired_count` to `2`

Once traffic or trust requirements increase, the first upgrade should be:

- `predict_desired_count = 2`
- `taste_os_desired_count = 2`

## Important Notes

- The ECS tasks are assigned public IPs for a lower-friction first deployment. The data stores remain private.
- Host-based routing is used because both APIs expose overlapping paths like `/readyz` and `/metrics`.
- If you already have a production VPC, you may want to adapt this stack to consume existing subnets instead of creating new ones.
