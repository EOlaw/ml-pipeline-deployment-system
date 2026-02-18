# Deployment Guide

## Local Development

### Prerequisites
- Python 3.11+
- pip or virtualenv
- Docker + Docker Compose (for containerized deployment)

### Quick Start

```bash
# 1. Clone and enter the project
cd ml-pipeline-deployment-system

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment configuration
cp .env.example .env
# Edit .env if needed (defaults work out-of-the-box)

# 5. Run the full training pipeline
python run_pipeline.py
#    ➜ Generates synthetic data, trains models, saves artifacts

# 6. Start the Flask API
python -m src.api.app
#    ➜ Listening on http://localhost:5000

# 7. Test a prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"income":65000,"age":35,"loan_amount":15000,"credit_score":720,
       "employment_years":5.0,"debt_ratio":0.25,"num_accounts":3,
       "employment_type":"full_time"}'
```

---

## Docker Deployment

### Build and Run (Single Container)

```bash
# Build the image
docker build -t ml-pipeline:latest .

# Run the training pipeline first to create model artifacts
docker run --rm \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  ml-pipeline:latest \
  python run_pipeline.py

# Start the API server
docker run -d \
  --name ml-pipeline-api \
  -p 5000:5000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  ml-pipeline:latest
```

### Docker Compose (Recommended)

```bash
# Start the API service
docker compose up -d ml-api

# Run training as a one-shot service
docker compose --profile training up ml-trainer

# View logs
docker compose logs -f ml-api

# Stop all services
docker compose down
```

---

## AWS Deployment (Production Architecture)

```
┌──────────────────────────────────────────────────────────────────────┐
│                         AWS Architecture                            │
│                                                                      │
│  Internet ──► Route 53 ──► ALB ──► ECS (Fargate)                   │
│                                        │                             │
│                                    Flask API                         │
│                                    (Docker)                          │
│                                        │                             │
│                         ┌─────────────┴──────────────────┐          │
│                         │                                │          │
│                     S3 Bucket                       RDS (Postgres)  │
│                   (model artifacts)                (training data)  │
│                         │                                │          │
│                    CloudWatch                       Secrets Manager  │
│                   (logs + alarms)                  (credentials)    │
└──────────────────────────────────────────────────────────────────────┘
```

### AWS Service Roles

| Service | Role |
|---|---|
| **S3** | Stores model artifacts (`model_v1.pkl`, `preprocessor_v1.pkl`, `metadata.json`). Versioning enabled for rollback. |
| **EC2 / ECS Fargate** | Runs the Dockerized Flask API. ECS preferred (managed, auto-scaling). |
| **RDS (PostgreSQL)** | Stores raw training dataset. App user has read-only access. |
| **CloudWatch Logs** | Receives all application logs via CloudWatch Agent or AWS logging driver. |
| **CloudWatch Alarms** | Alert on latency p95 > 500ms, error rate > 1%, drift detected. |
| **ECR** | Docker image registry. CI/CD pushes new images here. |
| **Secrets Manager** | DB password, API keys — never in code or env files. |
| **IAM Roles** | EC2/ECS task role grants least-privilege access to S3, CloudWatch, Secrets Manager. |

### Step-by-Step AWS Deployment

#### 1. Build and Push Docker Image to ECR

```bash
# Authenticate with ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS \
    --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com

# Build
docker build -t ml-pipeline .

# Tag
docker tag ml-pipeline:latest \
  123456789.dkr.ecr.us-east-1.amazonaws.com/ml-pipeline:v1

# Push
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/ml-pipeline:v1
```

#### 2. Store Model in S3

```bash
# Upload model artifacts
aws s3 cp models/model_v1.pkl \
  s3://ml-pipeline-models/models/model_v1.pkl

aws s3 cp models/preprocessor_v1.pkl \
  s3://ml-pipeline-models/models/preprocessor_v1.pkl

aws s3 cp models/metadata.json \
  s3://ml-pipeline-models/models/metadata.json
```

In production the `ModelRegistry.simulate_s3_upload()` method becomes:

```python
import boto3
s3 = boto3.client("s3")
s3.upload_file(
    str(model_path),
    config.aws.s3_bucket,
    config.aws.s3_model_key
)
```

#### 3. ECS Task Definition (excerpt)

```json
{
  "family": "ml-pipeline-api",
  "cpu": "512",
  "memory": "1024",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "executionRoleArn": "arn:aws:iam::123456789:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::123456789:role/ml-pipeline-task-role",
  "containerDefinitions": [
    {
      "name": "ml-api",
      "image": "123456789.dkr.ecr.us-east-1.amazonaws.com/ml-pipeline:v1",
      "portMappings": [{ "containerPort": 5000 }],
      "environment": [
        { "name": "MODEL_VERSION", "value": "v1" },
        { "name": "LOG_LEVEL", "value": "INFO" }
      ],
      "secrets": [
        { "name": "RDS_PASSWORD", "valueFrom": "arn:aws:secretsmanager:..." }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ml-pipeline/api",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:5000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

#### 4. ALB Target Group Health Check

```
Path: /health
Protocol: HTTP
Port: 5000
Healthy threshold: 2
Unhealthy threshold: 3
Timeout: 5 seconds
Interval: 30 seconds
Success codes: 200
```

#### 5. IAM Role Permissions (Least Privilege)

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": "arn:aws:s3:::ml-pipeline-models/models/*"
    },
    {
      "Effect": "Allow",
      "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
      "Resource": "arn:aws:logs:us-east-1:*:log-group:/ml-pipeline/*"
    },
    {
      "Effect": "Allow",
      "Action": "secretsmanager:GetSecretValue",
      "Resource": "arn:aws:secretsmanager:us-east-1:*:secret:ml-pipeline-*"
    }
  ]
}
```

---

## Secrets Handling Best Practices

| Concern | Solution |
|---|---|
| **Database passwords** | AWS Secrets Manager → ECS task secrets injection |
| **API keys** | Secrets Manager — never in code or `.env` files |
| **Environment-specific config** | ENV vars in ECS Task Definition or SSM Parameter Store |
| **Local development** | `.env` file (gitignored); use `.env.example` as template |
| **CI/CD secrets** | GitHub Actions secrets → `aws configure` with OIDC (no long-lived keys) |

**Golden rule:** If a secret appears in your Git history, rotate it immediately.

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src --cov-report=term-missing

# Single test file
pytest tests/test_api.py -v

# Specific test
pytest tests/test_training.py::TestModelTrainer::test_training_completes -v
```

---

## Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `API_HOST` | `0.0.0.0` | Flask bind address |
| `API_PORT` | `5000` | Flask port |
| `FLASK_DEBUG` | `false` | Enable debug mode |
| `MODEL_VERSION` | `v1` | Active model version label |
| `RANDOM_STATE` | `42` | NumPy/sklearn random seed |
| `TEST_SIZE` | `0.2` | Train/test split ratio |
| `CV_FOLDS` | `5` | Cross-validation folds |
| `SCORING_METRIC` | `f1` | CV scoring metric |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `S3_BUCKET` | `ml-pipeline-models` | AWS S3 bucket name |
| `AWS_REGION` | `us-east-1` | AWS region |
