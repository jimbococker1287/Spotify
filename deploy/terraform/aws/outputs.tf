output "ecr_repository_url" {
  value       = aws_ecr_repository.service.repository_url
  description = "Push the production service image here."
}

output "alb_dns_name" {
  value       = aws_lb.main.dns_name
  description = "Public DNS name of the load balancer."
}

output "predict_host" {
  value       = var.predict_host
  description = "Host header routed to the prediction API."
}

output "taste_os_host" {
  value       = var.taste_os_host
  description = "Host header routed to the Taste OS API."
}

output "postgres_address" {
  value       = aws_db_instance.taste_os.address
  description = "RDS address backing Taste OS state."
}

output "redis_address" {
  value       = aws_elasticache_cluster.redis.cache_nodes[0].address
  description = "ElastiCache Redis endpoint."
}

output "efs_file_system_id" {
  value       = aws_efs_file_system.outputs.id
  description = "EFS filesystem mounted into the services."
}

output "auth_token_secret_arn" {
  value       = aws_secretsmanager_secret.auth_token.arn
  description = "Secrets Manager ARN for the legacy auth token."
}

output "jwt_secret_arn" {
  value       = aws_secretsmanager_secret.jwt_secret.arn
  description = "Secrets Manager ARN for the JWT shared secret."
}

output "taste_os_db_url_secret_arn" {
  value       = aws_secretsmanager_secret.taste_os_db_url.arn
  description = "Secrets Manager ARN for the Taste OS SQLAlchemy database URL."
}
