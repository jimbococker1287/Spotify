data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  prefix                    = "${var.project_name}-${var.environment}"
  predict_log_group_name    = "/ecs/${local.prefix}-predict"
  taste_os_log_group_name   = "/ecs/${local.prefix}-taste-os"
  outputs_mount_path        = "/app/outputs"
  predict_listener_port     = 8000
  taste_os_listener_port    = 8010
  predict_container_name    = "spotify-predict-api"
  taste_os_container_name   = "spotify-taste-os-api"
  public_azs                = slice(data.aws_availability_zones.available.names, 0, 2)
  active_listener_arn       = var.certificate_arn != "" ? aws_lb_listener.https[0].arn : aws_lb_listener.http[0].arn
  db_connection_url         = "postgresql+psycopg://${var.db_username}:${var.db_password}@${aws_db_instance.taste_os.address}:${aws_db_instance.taste_os.port}/${var.db_name}"
}

resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "${local.prefix}-vpc"
  }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "${local.prefix}-igw"
  }
}

resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.main.id
  cidr_block              = var.public_subnet_cidrs[count.index]
  availability_zone       = local.public_azs[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "${local.prefix}-public-${count.index + 1}"
  }
}

resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = var.private_subnet_cidrs[count.index]
  availability_zone = local.public_azs[count.index]

  tags = {
    Name = "${local.prefix}-private-${count.index + 1}"
  }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name = "${local.prefix}-public-rt"
  }
}

resource "aws_route_table_association" "public" {
  count          = 2
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "${local.prefix}-private-rt"
  }
}

resource "aws_route_table_association" "private" {
  count          = 2
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
}

resource "aws_security_group" "alb" {
  name        = "${local.prefix}-alb-sg"
  description = "Public ALB security group"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = var.allowed_ingress_cidrs
  }

  dynamic "ingress" {
    for_each = var.certificate_arn != "" ? [443] : []
    content {
      from_port   = 443
      to_port     = 443
      protocol    = "tcp"
      cidr_blocks = var.allowed_ingress_cidrs
    }
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "ecs" {
  name        = "${local.prefix}-ecs-sg"
  description = "Service task security group"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port       = local.predict_listener_port
    to_port         = local.predict_listener_port
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  ingress {
    from_port       = local.taste_os_listener_port
    to_port         = local.taste_os_listener_port
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "postgres" {
  name        = "${local.prefix}-postgres-sg"
  description = "Postgres security group"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "redis" {
  name        = "${local.prefix}-redis-sg"
  description = "Redis security group"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "efs" {
  name        = "${local.prefix}-efs-sg"
  description = "EFS security group"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port       = 2049
    to_port         = 2049
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_efs_file_system" "outputs" {
  throughput_mode = var.efs_throughput_mode

  tags = {
    Name = "${local.prefix}-outputs"
  }
}

resource "aws_efs_mount_target" "outputs" {
  count           = 2
  file_system_id  = aws_efs_file_system.outputs.id
  subnet_id       = aws_subnet.private[count.index].id
  security_groups = [aws_security_group.efs.id]
}

resource "aws_db_subnet_group" "main" {
  name       = "${local.prefix}-db-subnets"
  subnet_ids = aws_subnet.private[*].id
}

resource "aws_db_instance" "taste_os" {
  identifier             = "${local.prefix}-postgres"
  engine                 = "postgres"
  engine_version         = "16.3"
  instance_class         = var.db_instance_class
  allocated_storage      = 20
  db_name                = var.db_name
  username               = var.db_username
  password               = var.db_password
  skip_final_snapshot    = true
  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.postgres.id]
  publicly_accessible    = false
  deletion_protection    = false
  multi_az               = false
}

resource "aws_elasticache_subnet_group" "main" {
  name       = "${local.prefix}-redis-subnets"
  subnet_ids = aws_subnet.private[*].id
}

resource "aws_elasticache_cluster" "redis" {
  cluster_id           = "${replace(local.prefix, "_", "-")}-redis"
  engine               = "redis"
  node_type            = var.redis_node_type
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  port                 = 6379
  subnet_group_name    = aws_elasticache_subnet_group.main.name
  security_group_ids   = [aws_security_group.redis.id]
}

resource "aws_secretsmanager_secret" "auth_token" {
  name = "${local.prefix}/auth-token"
}

resource "aws_secretsmanager_secret_version" "auth_token" {
  secret_id     = aws_secretsmanager_secret.auth_token.id
  secret_string = var.auth_token
}

resource "aws_secretsmanager_secret" "jwt_secret" {
  name = "${local.prefix}/jwt-secret"
}

resource "aws_secretsmanager_secret_version" "jwt_secret" {
  secret_id     = aws_secretsmanager_secret.jwt_secret.id
  secret_string = var.jwt_secret
}

resource "aws_secretsmanager_secret" "taste_os_db_url" {
  name = "${local.prefix}/taste-os-db-url"
}

resource "aws_secretsmanager_secret_version" "taste_os_db_url" {
  secret_id     = aws_secretsmanager_secret.taste_os_db_url.id
  secret_string = local.db_connection_url
}

resource "aws_ecr_repository" "service" {
  name                 = local.prefix
  image_tag_mutability = "MUTABLE"
  force_delete         = true
}

resource "aws_iam_role" "task_execution" {
  name = "${local.prefix}-task-execution"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "task_execution" {
  role       = aws_iam_role.task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "task_execution_secrets" {
  name = "${local.prefix}-task-execution-secrets"
  role = aws_iam_role.task_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = [
          aws_secretsmanager_secret.auth_token.arn,
          aws_secretsmanager_secret.jwt_secret.arn,
          aws_secretsmanager_secret.taste_os_db_url.arn
        ]
      }
    ]
  })
}

resource "aws_iam_role" "task" {
  name = "${local.prefix}-task"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_cloudwatch_log_group" "predict" {
  name              = local.predict_log_group_name
  retention_in_days = 14
}

resource "aws_cloudwatch_log_group" "taste_os" {
  name              = local.taste_os_log_group_name
  retention_in_days = 14
}

resource "aws_ecs_cluster" "main" {
  name = "${local.prefix}-cluster"
}

resource "aws_lb" "main" {
  name               = substr(replace("${local.prefix}-alb", "_", "-"), 0, 32)
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id
}

resource "aws_lb_target_group" "predict" {
  name        = substr(replace("${local.prefix}-predict", "_", "-"), 0, 32)
  port        = local.predict_listener_port
  protocol    = "HTTP"
  target_type = "ip"
  vpc_id      = aws_vpc.main.id

  health_check {
    enabled             = true
    path                = "/readyz"
    matcher             = "200"
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }
}

resource "aws_lb_target_group" "taste_os" {
  name        = substr(replace("${local.prefix}-tasteos", "_", "-"), 0, 32)
  port        = local.taste_os_listener_port
  protocol    = "HTTP"
  target_type = "ip"
  vpc_id      = aws_vpc.main.id

  health_check {
    enabled             = true
    path                = "/readyz"
    matcher             = "200"
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }
}

resource "aws_lb_listener" "http" {
  count             = var.certificate_arn == "" ? 1 : 0
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type = "fixed-response"

    fixed_response {
      content_type = "text/plain"
      message_body = "No route configured for this host."
      status_code  = "404"
    }
  }
}

resource "aws_lb_listener" "http_redirect" {
  count             = var.certificate_arn != "" ? 1 : 0
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type = "redirect"

    redirect {
      port        = "443"
      protocol    = "HTTPS"
      status_code = "HTTP_301"
    }
  }
}

resource "aws_lb_listener" "https" {
  count             = var.certificate_arn != "" ? 1 : 0
  load_balancer_arn = aws_lb.main.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = var.certificate_arn

  default_action {
    type = "fixed-response"

    fixed_response {
      content_type = "text/plain"
      message_body = "No route configured for this host."
      status_code  = "404"
    }
  }
}

resource "aws_lb_listener_rule" "predict" {
  listener_arn = local.active_listener_arn
  priority     = 10

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.predict.arn
  }

  condition {
    host_header {
      values = [var.predict_host]
    }
  }
}

resource "aws_lb_listener_rule" "taste_os" {
  listener_arn = local.active_listener_arn
  priority     = 20

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.taste_os.arn
  }

  condition {
    host_header {
      values = [var.taste_os_host]
    }
  }
}

resource "aws_ecs_task_definition" "predict" {
  family                   = "${local.prefix}-predict"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = tostring(var.predict_cpu)
  memory                   = tostring(var.predict_memory)
  execution_role_arn       = aws_iam_role.task_execution.arn
  task_role_arn            = aws_iam_role.task.arn

  volume {
    name = "outputs"

    efs_volume_configuration {
      file_system_id = aws_efs_file_system.outputs.id
      root_directory = "/"
      transit_encryption = "ENABLED"
    }
  }

  container_definitions = jsonencode([
    {
      name      = local.predict_container_name
      image     = var.container_image
      essential = true
      portMappings = [
        {
          containerPort = local.predict_listener_port
          protocol      = "tcp"
        }
      ]
      mountPoints = [
        {
          sourceVolume  = "outputs"
          containerPath = local.outputs_mount_path
          readOnly      = false
        }
      ]
      environment = [
        { name = "SERVICE_MODE", value = "predict" },
        { name = "PORT", value = tostring(local.predict_listener_port) },
        { name = "RUN_DIR", value = "${local.outputs_mount_path}/deployments/registry/channels/stable" },
        { name = "REQUEST_RATE_LIMIT", value = tostring(var.request_rate_limit) },
        { name = "RATE_LIMIT_BACKEND", value = "redis" },
        { name = "REDIS_URL", value = "redis://${aws_elasticache_cluster.redis.cache_nodes[0].address}:6379/0" },
        { name = "AUTH_MODE", value = var.auth_mode },
        { name = "AUTH_SCOPE", value = var.auth_scope },
        { name = "JWT_ALGORITHMS", value = "RS256" },
        { name = "JWT_ISSUER", value = var.jwt_issuer },
        { name = "JWT_AUDIENCE", value = var.predict_jwt_audience },
        { name = "JWT_REQUIRED_SCOPES", value = var.predict_jwt_required_scopes },
        { name = "MAX_TOP_K", value = tostring(var.max_top_k) },
        { name = "OTEL_ENABLED", value = "1" }
      ]
      secrets = [
        { name = "AUTH_TOKEN", valueFrom = aws_secretsmanager_secret.auth_token.arn },
        { name = "JWT_SECRET", valueFrom = aws_secretsmanager_secret.jwt_secret.arn }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.predict.name
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "ecs"
        }
      }
    }
  ])
}

resource "aws_ecs_task_definition" "taste_os" {
  family                   = "${local.prefix}-taste-os"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = tostring(var.taste_os_cpu)
  memory                   = tostring(var.taste_os_memory)
  execution_role_arn       = aws_iam_role.task_execution.arn
  task_role_arn            = aws_iam_role.task.arn

  volume {
    name = "outputs"

    efs_volume_configuration {
      file_system_id = aws_efs_file_system.outputs.id
      root_directory = "/"
      transit_encryption = "ENABLED"
    }
  }

  container_definitions = jsonencode([
    {
      name      = local.taste_os_container_name
      image     = var.container_image
      essential = true
      portMappings = [
        {
          containerPort = local.taste_os_listener_port
          protocol      = "tcp"
        }
      ]
      mountPoints = [
        {
          sourceVolume  = "outputs"
          containerPath = local.outputs_mount_path
          readOnly      = false
        }
      ]
      environment = [
        { name = "SERVICE_MODE", value = "taste-os" },
        { name = "PORT", value = tostring(local.taste_os_listener_port) },
        { name = "RUN_DIR", value = "${local.outputs_mount_path}/deployments/registry/channels/stable" },
        { name = "TASTE_OS_OUTPUT_DIR", value = "${local.outputs_mount_path}/analysis/taste_os_service" },
        { name = "REQUEST_RATE_LIMIT", value = tostring(var.request_rate_limit) },
        { name = "RATE_LIMIT_BACKEND", value = "redis" },
        { name = "REDIS_URL", value = "redis://${aws_elasticache_cluster.redis.cache_nodes[0].address}:6379/0" },
        { name = "AUTH_MODE", value = var.auth_mode },
        { name = "AUTH_SCOPE", value = var.auth_scope },
        { name = "JWT_ALGORITHMS", value = "RS256" },
        { name = "JWT_ISSUER", value = var.jwt_issuer },
        { name = "JWT_AUDIENCE", value = var.taste_os_jwt_audience },
        { name = "JWT_REQUIRED_SCOPES", value = var.taste_os_jwt_required_scopes },
        { name = "MAX_TOP_K", value = tostring(var.max_top_k) },
        { name = "OTEL_ENABLED", value = "1" }
      ]
      secrets = [
        { name = "AUTH_TOKEN", valueFrom = aws_secretsmanager_secret.auth_token.arn },
        { name = "JWT_SECRET", valueFrom = aws_secretsmanager_secret.jwt_secret.arn },
        { name = "TASTE_OS_DATABASE_URL", valueFrom = aws_secretsmanager_secret.taste_os_db_url.arn }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.taste_os.name
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "ecs"
        }
      }
    }
  ])
}

resource "aws_ecs_service" "predict" {
  name            = "${local.prefix}-predict"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.predict.arn
  desired_count   = var.predict_desired_count
  launch_type     = "FARGATE"

  network_configuration {
    assign_public_ip = true
    security_groups  = [aws_security_group.ecs.id]
    subnets          = aws_subnet.public[*].id
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.predict.arn
    container_name   = local.predict_container_name
    container_port   = local.predict_listener_port
  }

  depends_on = [aws_lb_listener_rule.predict]
}

resource "aws_ecs_service" "taste_os" {
  name            = "${local.prefix}-taste-os"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.taste_os.arn
  desired_count   = var.taste_os_desired_count
  launch_type     = "FARGATE"

  network_configuration {
    assign_public_ip = true
    security_groups  = [aws_security_group.ecs.id]
    subnets          = aws_subnet.public[*].id
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.taste_os.arn
    container_name   = local.taste_os_container_name
    container_port   = local.taste_os_listener_port
  }

  depends_on = [aws_lb_listener_rule.taste_os]
}
