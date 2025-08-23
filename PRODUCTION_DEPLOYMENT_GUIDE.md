# 🚀 Connectome-GNN-Suite Production Deployment Guide

## 📋 Prerequisites

### System Requirements
- **CPU**: 4+ cores (8+ recommended)
- **RAM**: 8GB minimum (16GB+ recommended)
- **GPU**: CUDA-compatible (optional but recommended)
- **Storage**: 50GB+ available space
- **Python**: 3.9+ (3.11+ recommended)

### Environment Setup
```bash
# Create production environment
python3 -m venv venv-prod
source venv-prod/bin/activate

# Install production dependencies
pip install --no-deps -r requirements-prod.txt

# Verify installation
python -c "import connectome_gnn; print('✅ Installation verified')"
```

## 🏗️ Production Architecture

### Container Deployment (Recommended)
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  connectome-gnn-api:
    build:
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    environment:
      - ENV=production
      - LOG_LEVEL=info
      - WORKERS=4
    volumes:
      - ./data:/app/data:ro
      - ./logs:/app/logs
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
        reservations:
          memory: 4G
          cpus: '2'

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: connectome_gnn
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

### Kubernetes Deployment
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: connectome-gnn-suite
spec:
  replicas: 3
  selector:
    matchLabels:
      app: connectome-gnn-suite
  template:
    metadata:
      labels:
        app: connectome-gnn-suite
    spec:
      containers:
      - name: connectome-gnn-api
        image: connectome-gnn-suite:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENV
          value: "production"
        - name: LOG_LEVEL
          value: "info"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: connectome-gnn-service
spec:
  selector:
    app: connectome-gnn-suite
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## ⚙️ Configuration Management

### Environment Variables
```bash
# Production environment configuration
export CONNECTOME_GNN_ENV=production
export CONNECTOME_GNN_LOG_LEVEL=info
export CONNECTOME_GNN_LOG_FILE=/app/logs/connectome-gnn.log
export CONNECTOME_GNN_DATA_DIR=/app/data
export CONNECTOME_GNN_CACHE_DIR=/app/cache
export CONNECTOME_GNN_MODEL_DIR=/app/models

# Database configuration
export DB_HOST=postgres
export DB_PORT=5432
export DB_NAME=connectome_gnn
export DB_USER=${DB_USER}
export DB_PASSWORD=${DB_PASSWORD}

# Redis configuration  
export REDIS_HOST=redis
export REDIS_PORT=6379
export REDIS_PASSWORD=${REDIS_PASSWORD}

# Security
export SECRET_KEY=${SECRET_KEY}
export JWT_SECRET=${JWT_SECRET}

# Performance
export WORKERS=4
export BATCH_SIZE=32
export MAX_MEMORY_GB=6
export CACHE_SIZE_MB=1000
```

### Production Configuration File
```yaml
# config/production.yaml
application:
  name: "connectome-gnn-suite"
  version: "0.1.0"
  environment: "production"
  debug: false

logging:
  level: "info"
  format: "json"
  file: "/app/logs/connectome-gnn.log"
  max_size_mb: 100
  backup_count: 5

database:
  host: "${DB_HOST}"
  port: "${DB_PORT}"
  name: "${DB_NAME}"
  user: "${DB_USER}"
  password: "${DB_PASSWORD}"
  pool_size: 20
  max_overflow: 30

cache:
  type: "redis"
  host: "${REDIS_HOST}"
  port: "${REDIS_PORT}"
  password: "${REDIS_PASSWORD}"
  db: 0
  ttl: 3600

performance:
  workers: 4
  batch_size: 32
  max_memory_gb: 6
  enable_multiprocessing: true
  enable_gpu: true
  gradient_checkpointing: true

security:
  secret_key: "${SECRET_KEY}"
  jwt_secret: "${JWT_SECRET}"
  cors_origins: ["https://connectome-gnn.example.com"]
  rate_limit: "100/minute"

monitoring:
  enable_metrics: true
  metrics_port: 9090
  health_check_port: 8001
  prometheus_endpoint: "/metrics"
```

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow
```yaml
# .github/workflows/deploy-production.yml
name: Production Deployment

on:
  push:
    branches: [main]
    tags: ['v*']

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        pip install torch-geometric
        pip install -e .
    
    - name: Run tests
      run: |
        python comprehensive_test.py
    
    - name: Security scan
      run: |
        pip install bandit safety
        bandit -r connectome_gnn/
        safety check
  
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: |
        docker build -f Dockerfile.prod -t connectome-gnn-suite:${{ github.sha }} .
        docker tag connectome-gnn-suite:${{ github.sha }} connectome-gnn-suite:latest
    
    - name: Push to registry
      run: |
        echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
        docker push connectome-gnn-suite:${{ github.sha }}
        docker push connectome-gnn-suite:latest
  
  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment: production
    steps:
    - name: Deploy to Kubernetes
      run: |
        kubectl set image deployment/connectome-gnn-suite \
          connectome-gnn-api=connectome-gnn-suite:${{ github.sha }}
        kubectl rollout status deployment/connectome-gnn-suite
```

### Production Dockerfile
```dockerfile
# Dockerfile.prod
FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements-prod.txt .
RUN pip install --no-cache-dir --user -r requirements-prod.txt

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY connectome_gnn/ ./connectome_gnn/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Create non-root user
RUN useradd --create-home --shell /bin/bash connectome
RUN chown -R connectome:connectome /app
USER connectome

# Make sure scripts are on PATH
ENV PATH=/root/.local/bin:$PATH

# Create necessary directories
RUN mkdir -p logs data cache models

EXPOSE 8000 8001 9090

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["python", "-m", "connectome_gnn.api.server", "--host", "0.0.0.0", "--port", "8000"]
```

## 📊 Monitoring & Observability

### Prometheus Metrics
```python
# connectome_gnn/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Application metrics
REQUESTS_TOTAL = Counter('connectome_gnn_requests_total', 
                        'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('connectome_gnn_request_duration_seconds',
                            'Request duration')
ACTIVE_CONNECTIONS = Gauge('connectome_gnn_active_connections',
                          'Active connections')
MEMORY_USAGE = Gauge('connectome_gnn_memory_usage_bytes',
                    'Memory usage in bytes')
GPU_UTILIZATION = Gauge('connectome_gnn_gpu_utilization_percent',
                       'GPU utilization percentage')

# ML-specific metrics
MODEL_INFERENCE_TIME = Histogram('connectome_gnn_model_inference_seconds',
                                 'Model inference time', ['model_type'])
TRAINING_LOSS = Gauge('connectome_gnn_training_loss',
                     'Current training loss', ['model_id'])
BATCH_PROCESSING_TIME = Histogram('connectome_gnn_batch_processing_seconds',
                                 'Batch processing time', ['batch_size'])
```

### Grafana Dashboard
```json
{
  "dashboard": {
    "title": "Connectome-GNN-Suite Production Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(connectome_gnn_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph", 
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(connectome_gnn_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "connectome_gnn_memory_usage_bytes / 1024 / 1024 / 1024",
            "legendFormat": "Memory (GB)"
          }
        ]
      },
      {
        "title": "Model Inference Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(connectome_gnn_model_inference_seconds_bucket[5m]))",
            "legendFormat": "{{model_type}}"
          }
        ]
      }
    ]
  }
}
```

### Logging Configuration
```python
# connectome_gnn/logging_config.py
import logging
import logging.config

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'json': {
            'format': '%(asctime)s %(name)s %(levelname)s %(message)s',
            'class': 'pythonjsonlogger.jsonlogger.JsonFormatter'
        },
        'detailed': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'json',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': '/app/logs/connectome-gnn.log',
            'maxBytes': 104857600,  # 100MB
            'backupCount': 5
        }
    },
    'loggers': {
        'connectome_gnn': {
            'level': 'DEBUG',
            'handlers': ['console', 'file'],
            'propagate': False
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console']
    }
}
```

## 🔒 Security Configuration

### Security Headers
```python
# Security middleware configuration
SECURITY_HEADERS = {
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
    'X-XSS-Protection': '1; mode=block',
    'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
    'Content-Security-Policy': "default-src 'self'",
    'Referrer-Policy': 'strict-origin-when-cross-origin'
}
```

### Rate Limiting
```python
# Rate limiting configuration
RATE_LIMITS = {
    'default': '100/minute',
    'api': '1000/hour',
    'training': '10/hour',
    'inference': '100/minute'
}
```

## 📈 Performance Tuning

### Auto-scaling Configuration
```yaml
# HPA configuration
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: connectome-gnn-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: connectome-gnn-suite
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Load Balancer Configuration
```nginx
# nginx.conf
upstream connectome_gnn_backend {
    least_conn;
    server connectome-gnn-1:8000;
    server connectome-gnn-2:8000;
    server connectome-gnn-3:8000;
}

server {
    listen 80;
    server_name connectome-gnn.example.com;
    
    location / {
        proxy_pass http://connectome_gnn_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 300s;
    }
}
```

## 🚨 Disaster Recovery

### Backup Strategy
```bash
#!/bin/bash
# backup.sh - Daily backup script

BACKUP_DIR="/backups/$(date +%Y-%m-%d)"
mkdir -p $BACKUP_DIR

# Database backup
pg_dump connectome_gnn > $BACKUP_DIR/database.sql

# Model checkpoint backup
cp -r /app/models/checkpoints $BACKUP_DIR/

# Configuration backup
cp -r /app/config $BACKUP_DIR/

# Upload to S3
aws s3 sync $BACKUP_DIR s3://connectome-gnn-backups/$(date +%Y-%m-%d)/
```

### Recovery Procedures
1. **Database Recovery**: Restore from PostgreSQL backup
2. **Model Recovery**: Restore model checkpoints
3. **Configuration Recovery**: Restore configuration files
4. **Container Recovery**: Redeploy from latest stable image

## 📋 Production Checklist

### Pre-Deployment
- [ ] All tests passing (100% success rate)
- [ ] Security scan completed
- [ ] Performance benchmarks met
- [ ] Database migrations applied
- [ ] Configuration validated
- [ ] Secrets properly configured
- [ ] Monitoring dashboard configured
- [ ] Backup strategy implemented

### Post-Deployment
- [ ] Health checks passing
- [ ] Metrics being collected
- [ ] Logs flowing properly
- [ ] Performance within SLA
- [ ] Security headers active
- [ ] Auto-scaling functioning
- [ ] Backup verification

## 🆘 Troubleshooting

### Common Issues
1. **High Memory Usage**: Enable gradient checkpointing, reduce batch size
2. **Slow Inference**: Check GPU utilization, optimize model
3. **Database Connection Issues**: Verify connection pool settings
4. **Cache Issues**: Clear Redis cache, check network connectivity

### Support Contacts
- **Production Issues**: ops@terragon.ai
- **Technical Issues**: dev@terragon.ai
- **Security Issues**: security@terragon.ai

---

**🎯 Production Ready**: The Connectome-GNN-Suite is now optimized and configured for production deployment with enterprise-grade reliability, scalability, and security.