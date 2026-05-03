variable "project_name" {
  description = "Short project name used as the resource prefix."
  type        = string
  default     = "spotify-prod"
}

variable "environment" {
  description = "Deployment environment name."
  type        = string
  default     = "prod"
}

variable "aws_region" {
  description = "AWS region for the deployment."
  type        = string
  default     = "us-east-1"
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC."
  type        = string
  default     = "10.42.0.0/16"
}

variable "public_subnet_cidrs" {
  description = "Two public subnet CIDRs."
  type        = list(string)
  default     = ["10.42.0.0/24", "10.42.1.0/24"]
}

variable "private_subnet_cidrs" {
  description = "Two private subnet CIDRs."
  type        = list(string)
  default     = ["10.42.10.0/24", "10.42.11.0/24"]
}

variable "allowed_ingress_cidrs" {
  description = "CIDRs allowed to reach the public load balancer."
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "container_image" {
  description = "Full container image URI for the Spotify serving image."
  type        = string
}

variable "predict_host" {
  description = "Hostname routed to the prediction API."
  type        = string
  default     = "predict.example.com"
}

variable "taste_os_host" {
  description = "Hostname routed to the Taste OS API."
  type        = string
  default     = "taste.example.com"
}

variable "certificate_arn" {
  description = "Optional ACM certificate ARN for HTTPS termination on the ALB."
  type        = string
  default     = ""
}

variable "predict_desired_count" {
  description = "Desired number of prediction API tasks."
  type        = number
  default     = 2
}

variable "taste_os_desired_count" {
  description = "Desired number of Taste OS API tasks."
  type        = number
  default     = 2
}

variable "predict_cpu" {
  description = "Fargate CPU units for the prediction API."
  type        = number
  default     = 1024
}

variable "predict_memory" {
  description = "Fargate memory (MiB) for the prediction API."
  type        = number
  default     = 2048
}

variable "taste_os_cpu" {
  description = "Fargate CPU units for the Taste OS API."
  type        = number
  default     = 1024
}

variable "taste_os_memory" {
  description = "Fargate memory (MiB) for the Taste OS API."
  type        = number
  default     = 2048
}

variable "request_rate_limit" {
  description = "Per-instance API request limit per minute."
  type        = number
  default     = 240
}

variable "max_top_k" {
  description = "Maximum top-k allowed by the APIs."
  type        = number
  default     = 20
}

variable "auth_mode" {
  description = "API auth mode."
  type        = string
  default     = "token_or_jwt"
}

variable "auth_scope" {
  description = "Whether auth protects only mutations or all routes."
  type        = string
  default     = "all"
}

variable "jwt_issuer" {
  description = "JWT issuer used by the APIs."
  type        = string
  default     = "https://issuer.example.com/"
}

variable "predict_jwt_audience" {
  description = "JWT audience for the prediction API."
  type        = string
  default     = "spotify-predict"
}

variable "taste_os_jwt_audience" {
  description = "JWT audience for the Taste OS API."
  type        = string
  default     = "spotify-taste-os"
}

variable "predict_jwt_required_scopes" {
  description = "Required JWT scopes for the prediction API."
  type        = string
  default     = "predict:write"
}

variable "taste_os_jwt_required_scopes" {
  description = "Required JWT scopes for the Taste OS API."
  type        = string
  default     = "taste-os:write"
}

variable "auth_token" {
  description = "Legacy API token used when token auth remains enabled."
  type        = string
  sensitive   = true
}

variable "jwt_secret" {
  description = "Shared JWT secret when not using a JWKS URL."
  type        = string
  sensitive   = true
}

variable "db_name" {
  description = "Postgres database name for Taste OS state."
  type        = string
  default     = "spotify"
}

variable "db_username" {
  description = "Postgres master username."
  type        = string
  default     = "spotify_user"
}

variable "db_password" {
  description = "Postgres master password."
  type        = string
  sensitive   = true
}

variable "db_instance_class" {
  description = "RDS instance class."
  type        = string
  default     = "db.t4g.micro"
}

variable "redis_node_type" {
  description = "ElastiCache Redis node type."
  type        = string
  default     = "cache.t4g.micro"
}

variable "efs_throughput_mode" {
  description = "EFS throughput mode."
  type        = string
  default     = "bursting"
}
