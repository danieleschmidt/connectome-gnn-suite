# Operational Runbooks

This directory contains operational runbooks for common scenarios and incident response procedures for the Connectome-GNN-Suite.

## Available Runbooks

### System Operations
- [High CPU Usage](./high-cpu-usage.md) - Diagnosing and resolving high CPU utilization
- [Memory Issues](./memory-issues.md) - Handling memory leaks and high memory usage
- [Disk Space Management](./disk-space-management.md) - Managing storage and cleanup procedures
- [Service Recovery](./service-recovery.md) - Restarting and recovering failed services

### GPU Operations
- [GPU Issues](./gpu-issues.md) - Troubleshooting CUDA and GPU-related problems
- [GPU Memory Management](./gpu-memory-management.md) - Handling GPU memory issues
- [GPU Temperature Issues](./gpu-temperature-issues.md) - Managing GPU overheating

### Application Operations
- [Training Job Failures](./training-job-failures.md) - Diagnosing failed ML training jobs
- [Model Performance Issues](./model-performance-issues.md) - Addressing model accuracy problems
- [Data Pipeline Issues](./data-pipeline-issues.md) - Resolving data processing problems

### Database Operations
- [PostgreSQL Issues](./postgresql-issues.md) - Database troubleshooting and maintenance
- [Connection Pool Issues](./connection-pool-issues.md) - Managing database connections
- [Query Performance](./query-performance.md) - Optimizing database queries

### Container Operations
- [Container Issues](./container-issues.md) - Docker container troubleshooting
- [Image Build Problems](./image-build-problems.md) - Resolving Docker build issues
- [Network Issues](./network-issues.md) - Container networking problems

### Security Operations
- [Security Incident Response](./security-incident-response.md) - Handling security alerts
- [Vulnerability Management](./vulnerability-management.md) - Managing security vulnerabilities
- [Access Control Issues](./access-control-issues.md) - Authentication and authorization problems

## Runbook Template

Each runbook follows this standard template:

```markdown
# [Issue Title]

## Summary
Brief description of the issue and its impact.

## Symptoms
- Observable symptoms and indicators
- Relevant metrics and alerts
- Log patterns

## Immediate Actions
1. Steps to take immediately to mitigate impact
2. Safety measures and precautions
3. Communication requirements

## Investigation Steps
1. Diagnostic procedures
2. Log analysis techniques
3. Metric queries to run

## Resolution Steps
1. Step-by-step resolution procedures
2. Verification steps
3. Recovery procedures

## Prevention
- Monitoring improvements
- Configuration changes
- Process improvements

## Related Information
- Links to relevant documentation
- Related runbooks
- Contact information
```

## Emergency Contacts

### Primary On-Call
- **Role**: Primary System Administrator
- **Contact**: +1-XXX-XXX-XXXX
- **Email**: primary-oncall@connectome-gnn-suite.com

### Secondary On-Call
- **Role**: Machine Learning Engineer
- **Contact**: +1-XXX-XXX-XXXX
- **Email**: ml-oncall@connectome-gnn-suite.com

### Escalation Contacts
- **Technical Lead**: tech-lead@connectome-gnn-suite.com
- **Product Manager**: product@connectome-gnn-suite.com
- **Security Team**: security@connectome-gnn-suite.com

## Incident Response Process

### Severity Levels

#### Severity 1 (Critical)
- Complete service outage
- Data corruption or loss
- Security breach
- **Response Time**: Immediate (< 15 minutes)

#### Severity 2 (High)
- Partial service degradation
- Performance issues affecting users
- Failed automated processes
- **Response Time**: < 1 hour

#### Severity 3 (Medium)
- Minor service issues
- Non-critical feature problems
- Monitoring alerts
- **Response Time**: < 4 hours

#### Severity 4 (Low)
- Cosmetic issues
- Enhancement requests
- Documentation updates
- **Response Time**: < 24 hours

### Response Workflow

1. **Detection**: Alert triggered or issue reported
2. **Acknowledgment**: On-call engineer acknowledges within SLA
3. **Assessment**: Determine severity and impact
4. **Communication**: Notify stakeholders based on severity
5. **Investigation**: Follow relevant runbook procedures
6. **Resolution**: Implement fix and verify recovery
7. **Post-Incident**: Conduct review and update procedures

## Common Commands

### Docker Operations
```bash
# Check service status
docker-compose ps

# View logs
docker-compose logs -f [service_name]

# Restart service
docker-compose restart [service_name]

# Scale service
docker-compose up -d --scale [service_name]=3
```

### Monitoring Queries
```bash
# Check service health
curl http://localhost:9090/api/v1/query?query=up

# Check CPU usage
curl http://localhost:9090/api/v1/query?query=100-avg(irate(node_cpu_seconds_total{mode="idle"}[5m]))*100

# Check memory usage
curl http://localhost:9090/api/v1/query?query=(1-(node_memory_MemAvailable_bytes/node_memory_MemTotal_bytes))*100
```

### Database Operations
```bash
# Connect to PostgreSQL
docker exec -it connectome-gnn-db psql -U connectome -d connectome_gnn

# Check connections
SELECT count(*) FROM pg_stat_activity;

# Check locks
SELECT * FROM pg_locks WHERE NOT granted;
```

### Log Analysis
```bash
# Search for errors in logs
docker-compose logs | grep -i error

# Monitor real-time logs
docker-compose logs -f --tail=100

# Search for specific patterns
docker-compose logs | grep "training_job_failed"
```

## Monitoring Integration

Each runbook includes relevant:
- Prometheus queries for investigation
- Grafana dashboard links
- Alert correlation information
- Metric thresholds and baselines

## Version Control

Runbooks are version controlled and should be:
- Updated after each incident
- Reviewed quarterly
- Tested during disaster recovery exercises
- Integrated with monitoring and alerting systems

## Training and Documentation

Regular training should cover:
- Runbook usage and updates
- Tool familiarity
- Escalation procedures
- Communication protocols

## Automation

Where possible, runbooks should be automated:
- Self-healing procedures
- Automated diagnostics
- Chatbot integration
- Workflow automation