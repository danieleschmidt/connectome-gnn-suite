# Monitoring & Observability

Comprehensive monitoring and observability setup for Connectome-GNN-Suite using Prometheus and Grafana.

## Overview

The monitoring stack provides:

- **Metrics Collection**: Prometheus for time-series metrics
- **Visualization**: Grafana dashboards for system and application monitoring
- **Alerting**: Alert rules for critical system and ML pipeline events
- **Log Aggregation**: Structured logging and analysis
- **Performance Monitoring**: GPU, CPU, memory, and application-specific metrics

## Quick Start

### Start Monitoring Stack

```bash
# Start monitoring services
make docker-monitoring

# Or using docker-compose directly
docker-compose up -d prometheus grafana
```

### Access Dashboards

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

### Available Dashboards

1. **Connectome-GNN-Suite Overview**: System health and application metrics
2. **GPU Monitoring**: CUDA utilization, temperature, memory usage
3. **ML Pipeline**: Training metrics, model performance, data processing
4. **Container Health**: Docker container resource usage and health

## Configuration

### Prometheus Configuration

Prometheus is configured to scrape metrics from:

- Application services (development, production, GPU)
- System metrics (CPU, memory, disk)
- GPU metrics (NVIDIA DCGM exporter)
- Container metrics (cAdvisor)
- Database metrics (PostgreSQL)

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'connectome-gnn-dev'
    static_configs:
      - targets: ['connectome-dev:8888']
```

### Alert Rules

Alert rules are defined for:

#### System Health
- High CPU usage (>85%)
- High memory usage (>90%)
- Low disk space (<10%)

#### GPU Monitoring
- High GPU utilization (>95%)
- High GPU temperature (>85°C)
- High GPU memory usage (>90%)

#### Application Health
- Service downtime
- High error rates (>10%)
- Slow response times (>2s)

#### ML Pipeline
- Training job failures
- Model accuracy drops (<80%)
- Long training times (>2 hours)
- Data pipeline stalls

### Grafana Dashboards

Dashboards are automatically provisioned and include:

#### System Overview
- CPU and memory utilization
- Disk usage and I/O
- Network traffic
- Service availability

#### GPU Monitoring
- GPU utilization per device
- Temperature monitoring
- Memory usage
- Power consumption

#### Application Metrics
- Request rates and response times
- Error rates by service
- Database connections and queries
- Cache hit rates

#### ML Pipeline Metrics
- Training job status and duration
- Model accuracy and loss trends
- Data processing throughput
- Resource utilization during training

## Metrics Exposed

### Application Metrics

The Connectome-GNN-Suite exposes the following metrics:

```python
# Training metrics
ml_training_jobs_total
ml_training_jobs_failed_total
ml_training_duration_seconds
ml_model_accuracy
ml_model_loss

# Data processing metrics
ml_data_processed_total
ml_data_processing_duration_seconds
ml_batch_size
ml_epoch_duration_seconds

# System metrics
http_requests_total
http_request_duration_seconds
cpu_usage_percent
memory_usage_bytes
gpu_utilization_percent
gpu_memory_usage_bytes
```

### Custom Metrics

You can add custom metrics to your application:

```python
from prometheus_client import Counter, Histogram, Gauge

# Example metrics
TRAINING_JOBS = Counter('ml_training_jobs_total', 'Total training jobs')
MODEL_ACCURACY = Gauge('ml_model_accuracy', 'Current model accuracy')
PROCESSING_TIME = Histogram('ml_processing_duration_seconds', 'Processing time')

# In your code
TRAINING_JOBS.inc()
MODEL_ACCURACY.set(0.95)
PROCESSING_TIME.observe(123.45)
```

## GPU Monitoring

For NVIDIA GPU monitoring, the stack includes DCGM exporter:

```bash
# Start GPU monitoring (requires nvidia-docker)
docker run -d --gpus all --rm -p 9400:9400 \
  nvcr.io/nvidia/k8s/dcgm-exporter:3.1.7-3.1.4-ubuntu20.04
```

### GPU Metrics Available

- `DCGM_FI_DEV_GPU_UTIL`: GPU utilization percentage
- `DCGM_FI_DEV_GPU_TEMP`: GPU temperature in Celsius
- `DCGM_FI_DEV_FB_USED`: GPU memory used in bytes
- `DCGM_FI_DEV_FB_TOTAL`: Total GPU memory in bytes
- `DCGM_FI_DEV_POWER_USAGE`: GPU power usage in watts

## Log Management

### Structured Logging

Configure structured logging in your application:

```python
import logging
import json

# Configure structured logging
logging.basicConfig(
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)

# Log structured data
logger.info(json.dumps({
    "event": "training_started",
    "model": "HierarchicalBrainGNN", 
    "batch_size": 32,
    "learning_rate": 0.001
}))
```

### Log Aggregation

Logs can be aggregated using Loki (optional):

```bash
# Add Loki to docker-compose.yml
loki:
  image: grafana/loki:2.9.0
  ports:
    - "3100:3100"
  volumes:
    - ./monitoring/loki-config.yaml:/etc/loki/local-config.yaml
```

## Alerting

### Configure Alert Manager

```bash
# alertmanager.yml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@connectome-gnn-suite.com'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
- name: 'web.hook'
  email_configs:
  - to: 'admin@connectome-gnn-suite.com'
    subject: 'Connectome-GNN-Suite Alert'
    body: |
      {{ range .Alerts }}
      Alert: {{ .Annotations.summary }}
      Description: {{ .Annotations.description }}
      {{ end }}
```

### Slack Integration

```yaml
receivers:
- name: 'slack-notifications'
  slack_configs:
  - api_url: 'YOUR_SLACK_WEBHOOK_URL'
    channel: '#alerts'
    title: 'Connectome-GNN-Suite Alert'
    text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
```

## Performance Optimization

### Monitoring Performance

Monitor the monitoring stack itself:

```bash
# Check Prometheus performance
curl -s http://localhost:9090/api/v1/query?query=prometheus_tsdb_head_samples_appended_total

# Check Grafana performance
curl -s http://admin:admin@localhost:3000/api/health
```

### Resource Limits

Set appropriate resource limits:

```yaml
# docker-compose.yml
prometheus:
  deploy:
    resources:
      limits:
        memory: 2G
        cpus: '1.0'
      reservations:
        memory: 1G
        cpus: '0.5'
```

### Data Retention

Configure data retention policies:

```yaml
# prometheus.yml
storage:
  tsdb:
    retention.time: 30d
    retention.size: 10GB
```

## Troubleshooting

### Common Issues

1. **Metrics not appearing**
   - Check service discovery configuration
   - Verify network connectivity between services
   - Check Prometheus targets page

2. **High memory usage**
   - Reduce scrape frequency
   - Implement metric filtering
   - Adjust retention policies

3. **Missing GPU metrics**
   - Ensure NVIDIA Docker runtime is installed
   - Check DCGM exporter logs
   - Verify GPU accessibility

### Debug Commands

```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Validate Prometheus config
docker exec prometheus promtool check config /etc/prometheus/prometheus.yml

# Check Grafana logs
docker logs connectome-gnn-grafana

# Test alert rules
docker exec prometheus promtool check rules /etc/prometheus/alert_rules.yml
```

## Security Considerations

### Authentication

Enable authentication for production:

```yaml
# grafana.ini
[auth]
disable_login_form = false
[auth.basic]
enabled = true
```

### Network Security

- Use internal networks for metric collection
- Enable HTTPS for external access
- Implement proper firewall rules

### Data Privacy

- Sanitize sensitive information in logs
- Use metric relabeling to remove sensitive labels
- Implement data retention policies

## Best Practices

1. **Metric Naming**: Use consistent naming conventions
2. **Label Usage**: Keep cardinality low, avoid high-cardinality labels
3. **Alert Fatigue**: Set appropriate thresholds to avoid noise
4. **Dashboard Design**: Create focused, actionable dashboards
5. **Documentation**: Document custom metrics and alert meanings

## Integration with CI/CD

Monitor deployment health:

```yaml
# .github/workflows/deploy.yml
- name: Wait for deployment health
  run: |
    curl -f http://localhost:9090/api/v1/query?query=up{job="connectome-gnn-prod"}
```

Monitor test performance:

```python
# In tests
import time
from prometheus_client import push_to_gateway, CollectorRegistry, Histogram

registry = CollectorRegistry()
test_duration = Histogram('test_duration_seconds', 'Test execution time', registry=registry)

with test_duration.time():
    # Run tests
    pass

push_to_gateway('localhost:9091', job='test_metrics', registry=registry)
```