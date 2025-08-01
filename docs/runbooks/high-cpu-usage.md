# High CPU Usage Runbook

## Summary
This runbook addresses situations where CPU utilization is consistently above normal thresholds (typically >85% for extended periods), which can impact system performance and user experience.

## Symptoms
- **Metrics**: CPU utilization >85% for >5 minutes
- **Alerts**: "HighCPUUsage" alert triggered
- **Observable Issues**:
  - Slow response times
  - Increased request latency
  - System unresponsiveness
  - High load averages

## Alert Details
```
Alert: HighCPUUsage
Expression: 100 - (avg by(instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 85
Duration: 5 minutes
Severity: Warning
```

## Immediate Actions

### 1. Acknowledge Alert
```bash
# Check current CPU usage
curl -s "http://localhost:9090/api/v1/query?query=100-avg(irate(node_cpu_seconds_total{mode=\"idle\"}[5m]))*100" | jq '.data.result[0].value[1]'

# Check load average
docker exec connectome-dev cat /proc/loadavg
```

### 2. Identify Affected Services
```bash
# Check which containers are using high CPU
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Check system processes
docker exec connectome-dev top -b -n 1 | head -20
```

### 3. Quick Mitigation
```bash
# Scale horizontally if possible
docker-compose up -d --scale connectome-dev=2

# Restart high-CPU container if necessary (last resort)
docker-compose restart [high-cpu-container]
```

## Investigation Steps

### 1. Analyze CPU Usage Patterns
```bash
# Prometheus queries for investigation

# CPU usage by mode
100 - (avg by(instance,mode) (irate(node_cpu_seconds_total[5m])) * 100)

# Container CPU usage
rate(container_cpu_usage_seconds_total[5m]) * 100

# Process-level CPU usage (if node_exporter with process collector)
topk(10, rate(namedprocess_namegroup_cpu_seconds_total[5m]) * 100)
```

### 2. Check Application Logs
```bash
# Look for performance-related errors
docker-compose logs --since=1h | grep -E "(timeout|slow|performance|cpu)"

# Check for memory leaks causing high CPU
docker-compose logs --since=1h | grep -E "(memory|gc|garbage)"

# Look for infinite loops or stuck processes
docker-compose logs --since=1h | grep -E "(stuck|loop|deadlock)"
```

### 3. Identify Root Cause
Common causes and their signatures:

#### Training Job Issues
```bash
# Check if training jobs are consuming excessive CPU
docker-compose logs connectome-gpu | grep -E "(training|epoch|batch)"

# Look for inefficient data loading
grep -E "(DataLoader|preprocessing)" logs/*
```

#### Memory Pressure
```bash
# High CPU can be caused by garbage collection
docker exec connectome-dev free -h

# Check swap usage
docker exec connectome-dev cat /proc/meminfo | grep Swap
```

#### Inefficient Queries
```bash
# Check database query performance
docker exec connectome-gnn-db psql -U connectome -d connectome_gnn -c "
SELECT query, calls, total_time, mean_time 
FROM pg_stat_statements 
ORDER BY total_time DESC 
LIMIT 10;"
```

### 4. Profile Application Performance
```bash
# Python profiling for ML workloads
docker exec connectome-dev python -m cProfile -o profile.stats your_script.py

# Use py-spy for live profiling
docker exec connectome-dev py-spy record -o profile.svg -d 60 -p $(pgrep python)
```

## Resolution Steps

### 1. Optimize Resource-Intensive Operations

#### For Training Jobs
```python
# Reduce batch size
BATCH_SIZE = 16  # Instead of 32 or 64

# Implement gradient checkpointing
model = torch.utils.checkpoint.checkpoint(model)

# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

#### For Data Processing
```python
# Optimize data loading
dataloader = DataLoader(
    dataset, 
    batch_size=batch_size,
    num_workers=2,  # Reduce if CPU-bound
    pin_memory=True,
    prefetch_factor=2
)
```

### 2. Scale Resources

#### Horizontal Scaling
```bash
# Scale up replicas
docker-compose up -d --scale connectome-dev=3

# Use load balancer if available
# Update nginx.conf or similar
```

#### Vertical Scaling
```yaml
# Update docker-compose.yml
services:
  connectome-dev:
    deploy:
      resources:
        limits:
          cpus: '4.0'  # Increase from 2.0
          memory: 8G
```

### 3. Configuration Optimization

#### Python/PyTorch Settings
```bash
# Set appropriate number of threads
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TORCH_NUM_THREADS=4

# Optimize garbage collection
export PYTHONHASHSEED=0
export MALLOC_ARENA_MAX=2
```

#### Container Settings
```yaml
# docker-compose.yml optimizations
services:
  connectome-dev:
    environment:
      - OMP_NUM_THREADS=4
      - TORCH_NUM_THREADS=4
    ulimits:
      memlock: -1
      stack: 67108864
```

### 4. Code Optimization

#### Identify Bottlenecks
```python
# Use profiling decorators
import time
from functools import wraps

def profile_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.2f}s")
        return result
    return wrapper

@profile_time
def expensive_function():
    # Your code here
    pass
```

#### Optimize Common Patterns
```python
# Use vectorized operations instead of loops
# Bad
for i in range(len(data)):
    result[i] = data[i] * 2

# Good  
result = data * 2

# Use appropriate data structures
# For frequent lookups, use dict instead of list
lookup = {item.id: item for item in items}
```

### 5. Verify Resolution
```bash
# Check CPU usage has decreased
curl -s "http://localhost:9090/api/v1/query?query=100-avg(irate(node_cpu_seconds_total{mode=\"idle\"}[5m]))*100"

# Monitor for sustained improvement
watch -n 5 'docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}"'

# Check application performance
curl -w "@curl-format.txt" -o /dev/null -s "http://localhost:8888/health"
```

## Prevention

### 1. Resource Monitoring
```yaml
# Add CPU limits to prevent resource hogging
services:
  connectome-dev:
    deploy:
      resources:
        limits:
          cpus: '2.0'
        reservations:
          cpus: '1.0'
```

### 2. Performance Testing
```bash
# Regular load testing
make benchmark

# Profile new code changes
pytest tests/benchmarks/ -v --benchmark-json=benchmark-results.json
```

### 3. Alerting Improvements
```yaml
# Add predictive alerts
- alert: CPUTrendIncreasing
  expr: predict_linear(cpu_usage[1h], 3600) > 90
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "CPU usage trending upward"
```

### 4. Automated Scaling
```yaml
# Implement auto-scaling based on CPU usage
# Example with Docker Swarm
services:
  connectome-dev:
    deploy:
      replicas: 1
      update_config:
        parallelism: 1
      restart_policy:
        condition: on-failure
```

## Related Information

### Useful Commands
```bash
# System information
docker exec connectome-dev nproc  # Number of CPUs
docker exec connectome-dev lscpu   # CPU information

# Process analysis
docker exec connectome-dev ps aux --sort=-%cpu | head -10
docker exec connectome-dev htop

# Memory pressure
docker exec connectome-dev vmstat 1 5
docker exec connectome-dev iostat 1 5
```

### Grafana Dashboards
- **System Overview**: CPU usage trends and patterns
- **Container Resources**: Per-container resource utilization
- **Application Performance**: Request latency and throughput

### Related Runbooks
- [Memory Issues](./memory-issues.md)
- [Training Job Failures](./training-job-failures.md)
- [Container Issues](./container-issues.md)

### Performance Tuning Resources
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [Docker Performance Best Practices](https://docs.docker.com/config/containers/resource_constraints/)
- [Linux Performance Tools](http://www.brendangregg.com/linuxperf.html)

## Post-Incident Actions

1. **Document Findings**: Update this runbook with new insights
2. **Code Review**: Review code changes that may have caused the issue
3. **Monitoring**: Add specific metrics for the identified bottleneck
4. **Testing**: Add performance tests to prevent regression
5. **Team Training**: Share learnings with the development team