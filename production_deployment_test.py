#!/usr/bin/env python3
"""Production deployment readiness validation."""

import sys
sys.path.append('/root/repo')

import asyncio
import time
import subprocess
import os
from pathlib import Path
import yaml
import json
import tempfile

print('🚀 Production Deployment Readiness Validation')
print('=' * 60)

# Test 1: Docker Environment Check
print('\n🐳 Test 1: Docker Environment Check')

try:
    result = subprocess.run(['docker', '--version'], capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        print(f'✅ Docker available: {result.stdout.strip()}')
        docker_available = True
    else:
        print('❌ Docker not available')
        docker_available = False
except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
    print('❌ Docker not available')
    docker_available = False

# Test 2: Docker Compose Validation
print('\n📋 Test 2: Docker Compose Configuration')

try:
    compose_file = Path('/root/repo/docker-compose.yml')
    if compose_file.exists():
        with open(compose_file, 'r') as f:
            compose_data = yaml.safe_load(f)
        
        services = compose_data.get('services', {})
        volumes = compose_data.get('volumes', {})
        networks = compose_data.get('networks', {})
        
        print(f'✅ Services defined: {len(services)}')
        print(f'✅ Volumes defined: {len(volumes)}')
        print(f'✅ Networks defined: {len(networks)}')
        
        # Check key services
        key_services = ['connectome-prod', 'connectome-test', 'connectome-benchmark']
        for service in key_services:
            if service in services:
                print(f'✅ Service {service}: configured')
            else:
                print(f'⚠️  Service {service}: not found')
        
        compose_valid = True
    else:
        print('❌ docker-compose.yml not found')
        compose_valid = False
        
except Exception as e:
    print(f'❌ Docker Compose validation failed: {e}')
    compose_valid = False

# Test 3: Production Container Build Test
print('\n🏗️  Test 3: Production Container Build Test')

if docker_available:
    try:
        dockerfile_content = """
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY pyproject.toml requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir torch torch-geometric numpy scipy scikit-learn matplotlib pandas tqdm pyyaml psutil

# Copy application code
COPY connectome_gnn/ ./connectome_gnn/
COPY scripts/ ./scripts/

# Set Python path
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s \\
    CMD python -c "import connectome_gnn; print('OK')" || exit 1

# Default command
CMD ["python", "-c", "print('Connectome-GNN Production Container Ready')"]
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name
        
        try:
            print('Building test production image...')
            build_result = subprocess.run([
                'docker', 'build',
                '-f', dockerfile_path,
                '-t', 'connectome-gnn:test-prod',
                '/root/repo'
            ], capture_output=True, text=True, timeout=300)
            
            if build_result.returncode == 0:
                print('✅ Production image build: SUCCESS')
                
                # Test run the container
                run_result = subprocess.run([
                    'docker', 'run', '--rm',
                    'connectome-gnn:test-prod',
                    'python', '-c', 'import connectome_gnn; print("Import test passed")'
                ], capture_output=True, text=True, timeout=60)
                
                if run_result.returncode == 0:
                    print('✅ Production container test: SUCCESS')
                    print(f'Container output: {run_result.stdout.strip()}')
                    container_test = True
                else:
                    print(f'❌ Production container test failed: {run_result.stderr}')
                    container_test = False
                
            else:
                print(f'❌ Production image build failed: {build_result.stderr}')
                container_test = False
                
        finally:
            os.unlink(dockerfile_path)
            
    except subprocess.TimeoutExpired:
        print('❌ Container build timed out')
        container_test = False
    except Exception as e:
        print(f'❌ Container build test failed: {e}')
        container_test = False
        
else:
    print('⚠️  Skipping container build (Docker not available)')
    container_test = False

# Test 4: Configuration Management
print('\n⚙️  Test 4: Configuration Management')

try:
    # Check for configuration files
    config_files = [
        'pyproject.toml',
        'requirements-dev.txt',
        'pytest.ini'
    ]
    
    config_score = 0
    for config_file in config_files:
        config_path = Path(f'/root/repo/{config_file}')
        if config_path.exists():
            print(f'✅ Configuration file: {config_file}')
            config_score += 1
        else:
            print(f'⚠️  Missing configuration: {config_file}')
    
    config_management = config_score >= len(config_files) * 0.7
    print(f'✅ Configuration management: {config_score}/{len(config_files)} files present')
    
except Exception as e:
    print(f'❌ Configuration check failed: {e}')
    config_management = False

# Test 5: Health Check System
print('\n💓 Test 5: Health Check System')

try:
    health_check_code = '''
import sys
import time
import psutil
import torch

def health_check():
    """Comprehensive health check."""
    checks = []
    
    # Memory check
    memory = psutil.virtual_memory()
    memory_ok = memory.percent < 90
    checks.append(("Memory", memory_ok, f"{memory.percent:.1f}%"))
    
    # CPU check
    cpu_percent = psutil.cpu_percent(interval=0.1)
    cpu_ok = cpu_percent < 95
    checks.append(("CPU", cpu_ok, f"{cpu_percent:.1f}%"))
    
    # PyTorch check
    torch_ok = True
    try:
        tensor = torch.randn(10, 10)
        torch_ok = tensor.numel() == 100
    except Exception:
        torch_ok = False
    checks.append(("PyTorch", torch_ok, "functional"))
    
    # Import check
    try:
        import connectome_gnn
        import_ok = True
    except Exception:
        import_ok = False
    checks.append(("Imports", import_ok, "available"))
    
    all_ok = all(check[1] for check in checks)
    
    print("Health Check Results:")
    for name, status, detail in checks:
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {name}: {detail}")
    
    return all_ok

if __name__ == "__main__":
    healthy = health_check()
    sys.exit(0 if healthy else 1)
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(health_check_code)
        health_check_path = f.name
    
    try:
        # Add our working environment to path for the health check
        env = os.environ.copy()
        env['PYTHONPATH'] = '/root/repo'
        
        health_result = subprocess.run([
            '/root/repo/terragon_working/bin/python', 
            health_check_path
        ], capture_output=True, text=True, timeout=30, env=env)
        
        if health_result.returncode == 0:
            print('✅ Health check system: WORKING')
            print(health_result.stdout)
            health_system = True
        else:
            print(f'❌ Health check failed: {health_result.stderr}')
            health_system = False
            
    finally:
        os.unlink(health_check_path)
        
except Exception as e:
    print(f'❌ Health check system test failed: {e}')
    health_system = False

# Test 6: Monitoring and Logging
print('\n📊 Test 6: Monitoring and Logging Setup')

try:
    monitoring_files = [
        'monitoring/prometheus.yml',
        'monitoring/alert_rules.yml'
    ]
    
    monitoring_score = 0
    for monitor_file in monitoring_files:
        monitor_path = Path(f'/root/repo/{monitor_file}')
        if monitor_path.exists():
            print(f'✅ Monitoring config: {monitor_file}')
            monitoring_score += 1
        else:
            print(f'⚠️  Missing monitoring config: {monitor_file}')
    
    # Check for Grafana dashboards
    grafana_dir = Path('/root/repo/monitoring/grafana/dashboards')
    if grafana_dir.exists():
        dashboard_count = len(list(grafana_dir.glob('*.json')))
        print(f'✅ Grafana dashboards: {dashboard_count} found')
        monitoring_score += 1
    else:
        print('⚠️  No Grafana dashboards found')
    
    monitoring_setup = monitoring_score >= 2
    print(f'✅ Monitoring setup: {monitoring_score}/3 components ready')
    
except Exception as e:
    print(f'❌ Monitoring setup check failed: {e}')
    monitoring_setup = False

# Test 7: Security Configuration
print('\n🔒 Test 7: Security Configuration')

try:
    security_files = [
        'SECURITY.md',
        '.github/workflows/security.yml' 
    ]
    
    security_score = 0
    for sec_file in security_files:
        sec_path = Path(f'/root/repo/{sec_file}')
        if sec_path.exists():
            print(f'✅ Security file: {sec_file}')
            security_score += 1
        else:
            print(f'⚠️  Missing security file: {sec_file}')
    
    # Check for bandit configuration
    pyproject_path = Path('/root/repo/pyproject.toml')
    if pyproject_path.exists():
        with open(pyproject_path, 'r') as f:
            pyproject_content = f.read()
        
        if '[tool.bandit]' in pyproject_content:
            print('✅ Bandit security scanning configured')
            security_score += 1
        else:
            print('⚠️  Bandit security scanning not configured')
    
    security_config = security_score >= 2
    print(f'✅ Security configuration: {security_score}/3 components ready')
    
except Exception as e:
    print(f'❌ Security configuration check failed: {e}')
    security_config = False

# Test 8: Deployment Scripts
print('\n🚀 Test 8: Deployment Scripts and Documentation')

try:
    deployment_files = [
        'scripts/build.sh',
        'PRODUCTION_DEPLOYMENT_GUIDE.md',
        'deployment_guide.md'
    ]
    
    deployment_score = 0
    for deploy_file in deployment_files:
        deploy_path = Path(f'/root/repo/{deploy_file}')
        if deploy_path.exists():
            print(f'✅ Deployment file: {deploy_file}')
            deployment_score += 1
        else:
            print(f'⚠️  Missing deployment file: {deploy_file}')
    
    deployment_ready = deployment_score >= 2
    print(f'✅ Deployment documentation: {deployment_score}/3 files present')
    
except Exception as e:
    print(f'❌ Deployment scripts check failed: {e}')
    deployment_ready = False

# Overall Production Readiness Assessment
print('\n🎯 PRODUCTION READINESS ASSESSMENT')
print('=' * 60)

readiness_tests = {
    'Docker Environment': docker_available,
    'Docker Compose Config': compose_valid,
    'Container Build': container_test,
    'Configuration Management': config_management,
    'Health Check System': health_system,
    'Monitoring Setup': monitoring_setup,
    'Security Configuration': security_config,
    'Deployment Documentation': deployment_ready
}

passed_tests = sum(1 for result in readiness_tests.values() if result)
total_tests = len(readiness_tests)
readiness_score = (passed_tests / total_tests) * 100

print(f'\n📊 Test Results Summary:')
for test_name, result in readiness_tests.items():
    status = '✅ PASS' if result else '❌ FAIL'
    print(f'{status} {test_name}')

print(f'\n🎯 Overall Readiness Score: {readiness_score:.1f}%')
print(f'📈 Tests Passed: {passed_tests}/{total_tests}')

if readiness_score >= 90:
    print('\n🏆 PRODUCTION READY - All systems go!')
    deployment_status = 'READY'
elif readiness_score >= 75:
    print('\n✅ MOSTLY READY - Minor issues to address')
    deployment_status = 'MOSTLY_READY'
elif readiness_score >= 60:
    print('\n⚠️  NEEDS WORK - Several issues require attention')
    deployment_status = 'NEEDS_WORK'
else:
    print('\n❌ NOT READY - Major issues must be resolved')
    deployment_status = 'NOT_READY'

# Generate deployment report
deployment_report = {
    'timestamp': time.time(),
    'readiness_score': readiness_score,
    'tests_passed': passed_tests,
    'total_tests': total_tests,
    'deployment_status': deployment_status,
    'test_results': readiness_tests,
    'recommendations': []
}

# Add recommendations based on failed tests
if not docker_available:
    deployment_report['recommendations'].append('Install Docker for container deployment')
if not compose_valid:
    deployment_report['recommendations'].append('Fix docker-compose.yml configuration')
if not container_test:
    deployment_report['recommendations'].append('Resolve container build and runtime issues')
if not config_management:
    deployment_report['recommendations'].append('Complete configuration file setup')
if not health_system:
    deployment_report['recommendations'].append('Implement comprehensive health checks')
if not monitoring_setup:
    deployment_report['recommendations'].append('Set up monitoring and alerting infrastructure')
if not security_config:
    deployment_report['recommendations'].append('Configure security scanning and policies')
if not deployment_ready:
    deployment_report['recommendations'].append('Create deployment scripts and documentation')

# Save report
report_path = Path('/root/repo/production_readiness_report.json')
with open(report_path, 'w') as f:
    json.dump(deployment_report, f, indent=2)

print(f'\n📋 Detailed report saved to: {report_path}')
print('\n🚀 Production Deployment Readiness Validation Complete!')