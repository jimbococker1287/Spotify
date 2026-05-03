# ECS / Fargate Task Templates

These task definitions assume:

- the service image is published to ECR
- an EFS volume is mounted at `/app/outputs`
- Redis and Postgres are reachable from the task network
- Secrets Manager or SSM Parameter Store supplies sensitive values

Suggested rollout pattern:

1. Publish a promoted run into the deployment registry channel.
2. Ensure the ECS service mounts the shared EFS volume used by every task.
3. Update the task definition image tag or rollout channel config.
4. Use ALB health checks against `/readyz`.

The JSON files use placeholders for ARNs, image names, and hostnames so you can fill in environment-specific values.
