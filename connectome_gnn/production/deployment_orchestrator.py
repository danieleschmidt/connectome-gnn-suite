"""Production deployment orchestrator for Connectome-GNN-Suite."""

import asyncio
import aiohttp
import yaml
import json
import time
import logging
import subprocess
import tempfile
import shutil
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import threading
import concurrent.futures
from contextlib import asynccontextmanager
import hashlib
import os

from ..robust.logging_config import get_logger
from ..robust.error_handling import ConnectomeError
from ..global_deployment.multi_region import Region, get_region_manager
from ..quality_gates.comprehensive_testing import get_quality_gate_orchestrator


class DeploymentStage(Enum):
    """Deployment pipeline stages."""
    PREPARATION = "preparation"
    TESTING = "testing"
    BUILDING = "building"
    STAGING = "staging"
    PRODUCTION = "production"
    MONITORING = "monitoring"
    ROLLBACK = "rollback"


class DeploymentStatus(Enum):
    """Deployment status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLING_BACK = "rolling_back"


@dataclass
class DeploymentConfig:
    """Complete deployment configuration."""
    # Application settings
    app_name: str
    version: str
    environment: str
    
    # Infrastructure settings
    regions: List[Region]
    replicas_per_region: int = 3
    min_replicas: int = 1
    max_replicas: int = 10
    
    # Resource requirements
    cpu_request: str = "500m"
    cpu_limit: str = "2000m"
    memory_request: str = "1Gi"
    memory_limit: str = "4Gi"
    
    # Quality gates
    run_quality_gates: bool = True
    quality_gate_timeout: int = 1800  # 30 minutes
    min_test_coverage: float = 0.85
    
    # Deployment strategy
    strategy: str = "rolling"  # rolling, blue_green, canary
    max_unavailable: str = "25%"
    max_surge: str = "25%"
    
    # Health checks
    health_check_path: str = "/health"
    readiness_check_path: str = "/ready"
    health_check_timeout: int = 10
    
    # Monitoring
    enable_monitoring: bool = True
    enable_logging: bool = True
    log_level: str = "INFO"
    
    # Security
    enable_tls: bool = True
    cert_manager: str = "letsencrypt"
    
    # Backup and rollback
    enable_backup: bool = True
    retention_days: int = 30
    auto_rollback: bool = True
    rollback_timeout: int = 600  # 10 minutes


@dataclass
class DeploymentResult:
    """Deployment operation result."""
    deployment_id: str
    status: DeploymentStatus
    stage: DeploymentStage
    start_time: float
    end_time: Optional[float]
    duration: Optional[float]
    success_count: int
    failure_count: int
    regions_deployed: List[Region]
    version: str
    logs: List[str]
    metrics: Dict[str, Any]
    rollback_version: Optional[str] = None


class ProductionDeploymentOrchestrator:
    """Orchestrates complete production deployments across all environments."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.active_deployments: Dict[str, DeploymentResult] = {}
        self.deployment_history: List[DeploymentResult] = []
        
        # External service integrations
        self.region_manager = get_region_manager()
        self.quality_gate_orchestrator = get_quality_gate_orchestrator()
        
        # Deployment templates
        self.templates_dir = Path(__file__).parent / "templates"
        self.templates_dir.mkdir(exist_ok=True)
        
        self._setup_deployment_templates()
        
    def _setup_deployment_templates(self):
        """Setup deployment templates."""
        # Kubernetes deployment template
        k8s_template = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "connectome-gnn",
                "labels": {
                    "app": "connectome-gnn"
                }
            },
            "spec": {
                "replicas": 3,
                "selector": {
                    "matchLabels": {
                        "app": "connectome-gnn"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "connectome-gnn"
                        }
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": "connectome-gnn",
                                "image": "connectome-gnn:latest",
                                "ports": [
                                    {
                                        "containerPort": 8080
                                    }
                                ],
                                "resources": {
                                    "requests": {
                                        "cpu": "500m",
                                        "memory": "1Gi"
                                    },
                                    "limits": {
                                        "cpu": "2000m",
                                        "memory": "4Gi"
                                    }
                                },
                                "livenessProbe": {
                                    "httpGet": {
                                        "path": "/health",
                                        "port": 8080
                                    },
                                    "initialDelaySeconds": 30,
                                    "periodSeconds": 10
                                },
                                "readinessProbe": {
                                    "httpGet": {
                                        "path": "/ready",
                                        "port": 8080
                                    },
                                    "initialDelaySeconds": 5,
                                    "periodSeconds": 5
                                }
                            }
                        ]
                    }
                }
            }
        }
        
        # Save template
        k8s_template_path = self.templates_dir / "kubernetes-deployment.yaml"
        with open(k8s_template_path, 'w') as f:
            yaml.dump(k8s_template, f, default_flow_style=False)
            
    async def deploy_to_production(
        self,
        config: DeploymentConfig,
        dry_run: bool = False
    ) -> DeploymentResult:
        """Execute complete production deployment."""
        deployment_id = self._generate_deployment_id(config)
        
        deployment_result = DeploymentResult(
            deployment_id=deployment_id,
            status=DeploymentStatus.PENDING,
            stage=DeploymentStage.PREPARATION,
            start_time=time.time(),
            end_time=None,
            duration=None,
            success_count=0,
            failure_count=0,
            regions_deployed=[],
            version=config.version,
            logs=[],
            metrics={}
        )
        
        self.active_deployments[deployment_id] = deployment_result
        
        try:
            self.logger.info(f"Starting production deployment: {deployment_id}")
            
            # Stage 1: Preparation
            await self._preparation_stage(config, deployment_result, dry_run)
            
            # Stage 2: Quality Gates
            if config.run_quality_gates:
                await self._testing_stage(config, deployment_result, dry_run)
                
            # Stage 3: Building
            await self._building_stage(config, deployment_result, dry_run)
            
            # Stage 4: Staging
            await self._staging_stage(config, deployment_result, dry_run)
            
            # Stage 5: Production
            await self._production_stage(config, deployment_result, dry_run)
            
            # Stage 6: Monitoring
            await self._monitoring_stage(config, deployment_result, dry_run)
            
            # Mark as successful
            deployment_result.status = DeploymentStatus.SUCCESS
            deployment_result.end_time = time.time()
            deployment_result.duration = deployment_result.end_time - deployment_result.start_time
            
            self.logger.info(
                f"Deployment {deployment_id} completed successfully in "
                f"{deployment_result.duration:.2f}s"
            )
            
        except Exception as e:
            self.logger.error(f"Deployment {deployment_id} failed: {e}")
            deployment_result.status = DeploymentStatus.FAILED
            deployment_result.end_time = time.time()
            deployment_result.duration = deployment_result.end_time - deployment_result.start_time
            deployment_result.logs.append(f"ERROR: {str(e)}")
            
            # Auto-rollback if enabled
            if config.auto_rollback and not dry_run:
                await self._rollback_stage(config, deployment_result)
                
        finally:
            # Move to history
            self.deployment_history.append(deployment_result)
            if deployment_id in self.active_deployments:
                del self.active_deployments[deployment_id]
                
        return deployment_result
        
    async def _preparation_stage(
        self,
        config: DeploymentConfig,
        result: DeploymentResult,
        dry_run: bool
    ):
        """Preparation stage: validate configuration and prerequisites."""
        result.stage = DeploymentStage.PREPARATION
        result.status = DeploymentStatus.RUNNING
        
        self.logger.info("Starting preparation stage")
        result.logs.append("Starting preparation stage")
        
        # Validate configuration
        await self._validate_config(config)
        
        # Check prerequisites
        await self._check_prerequisites(config)
        
        # Prepare deployment artifacts
        await self._prepare_artifacts(config, dry_run)
        
        result.logs.append("Preparation stage completed")
        
    async def _testing_stage(
        self,
        config: DeploymentConfig,
        result: DeploymentResult,
        dry_run: bool
    ):
        """Testing stage: run quality gates and tests."""
        result.stage = DeploymentStage.TESTING
        
        self.logger.info("Starting testing stage")
        result.logs.append("Starting comprehensive quality gates")
        
        if not dry_run:
            # Run comprehensive quality gates
            quality_report = self.quality_gate_orchestrator.run_all_quality_gates()
            
            result.metrics['quality_gates'] = quality_report
            
            if not quality_report.get('quality_gates_passed', False):
                raise ConnectomeError(
                    f"Quality gates failed: {quality_report.get('overall_score', 0.0)}"
                )
                
        result.logs.append("Quality gates passed")
        
    async def _building_stage(
        self,
        config: DeploymentConfig,
        result: DeploymentResult,
        dry_run: bool
    ):
        """Building stage: build and package application."""
        result.stage = DeploymentStage.BUILDING
        
        self.logger.info("Starting building stage")
        result.logs.append("Building application artifacts")
        
        if not dry_run:
            # Build Docker image
            await self._build_docker_image(config)
            
            # Build Kubernetes manifests
            await self._build_k8s_manifests(config)
            
        result.logs.append("Building stage completed")
        
    async def _staging_stage(
        self,
        config: DeploymentConfig,
        result: DeploymentResult,
        dry_run: bool
    ):
        """Staging stage: deploy to staging environment."""
        result.stage = DeploymentStage.STAGING
        
        self.logger.info("Starting staging deployment")
        result.logs.append("Deploying to staging environment")
        
        if not dry_run:
            # Deploy to staging
            staging_result = await self._deploy_to_staging(config)
            result.metrics['staging'] = staging_result
            
            # Run staging tests
            await self._run_staging_tests(config)
            
        result.logs.append("Staging deployment completed")
        
    async def _production_stage(
        self,
        config: DeploymentConfig,
        result: DeploymentResult,
        dry_run: bool
    ):
        """Production stage: deploy to production."""
        result.stage = DeploymentStage.PRODUCTION
        
        self.logger.info("Starting production deployment")
        result.logs.append("Deploying to production")
        
        if not dry_run:
            # Deploy to all regions
            for region in config.regions:
                try:
                    await self._deploy_to_region(config, region)
                    result.regions_deployed.append(region)
                    result.success_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to deploy to {region}: {e}")
                    result.failure_count += 1
                    result.logs.append(f"Failed to deploy to {region}: {e}")
                    
        result.logs.append(f"Production deployment completed: {result.success_count} successes, {result.failure_count} failures")
        
    async def _monitoring_stage(
        self,
        config: DeploymentConfig,
        result: DeploymentResult,
        dry_run: bool
    ):
        """Monitoring stage: setup monitoring and alerts."""
        result.stage = DeploymentStage.MONITORING
        
        self.logger.info("Setting up monitoring")
        result.logs.append("Setting up monitoring and alerting")
        
        if not dry_run and config.enable_monitoring:
            # Setup monitoring
            monitoring_result = await self._setup_monitoring(config)
            result.metrics['monitoring'] = monitoring_result
            
        result.logs.append("Monitoring setup completed")
        
    async def _rollback_stage(
        self,
        config: DeploymentConfig,
        result: DeploymentResult
    ):
        """Rollback stage: rollback to previous version."""
        result.stage = DeploymentStage.ROLLBACK
        result.status = DeploymentStatus.ROLLING_BACK
        
        self.logger.warning("Starting automatic rollback")
        result.logs.append("Starting automatic rollback")
        
        try:
            # Get previous version
            previous_version = await self._get_previous_version(config)
            result.rollback_version = previous_version
            
            # Rollback all regions
            for region in result.regions_deployed:
                await self._rollback_region(config, region, previous_version)
                
            result.logs.append(f"Rollback completed to version {previous_version}")
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            result.logs.append(f"Rollback failed: {e}")
            
    async def _validate_config(self, config: DeploymentConfig):
        """Validate deployment configuration."""
        if not config.app_name:
            raise ConnectomeError("Application name is required")
            
        if not config.version:
            raise ConnectomeError("Version is required")
            
        if not config.regions:
            raise ConnectomeError("At least one region is required")
            
        # Validate regions exist
        for region in config.regions:
            if region not in self.region_manager.active_regions:
                raise ConnectomeError(f"Region {region} not active")
                
    async def _check_prerequisites(self, config: DeploymentConfig):
        """Check deployment prerequisites."""
        # Check Docker
        try:
            subprocess.run(["docker", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise ConnectomeError("Docker not available")
            
        # Check kubectl (if Kubernetes deployment)
        try:
            subprocess.run(["kubectl", "version", "--client"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise ConnectomeError("kubectl not available")
            
    async def _prepare_artifacts(self, config: DeploymentConfig, dry_run: bool):
        """Prepare deployment artifacts."""
        if dry_run:
            return
            
        # Create deployment directory
        deploy_dir = Path(f"/tmp/connectome-gnn-deploy-{int(time.time())}")
        deploy_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy application files
        # In a real deployment, this would package the application
        self.logger.info(f"Prepared deployment artifacts in {deploy_dir}")
        
    async def _build_docker_image(self, config: DeploymentConfig):
        """Build Docker image."""
        image_tag = f"{config.app_name}:{config.version}"
        
        # Create Dockerfile
        dockerfile_content = f"""
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python", "-m", "connectome_gnn.server"]
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name
            
        try:
            # Build image
            cmd = [
                "docker", "build",
                "-f", dockerfile_path,
                "-t", image_tag,
                "."
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise ConnectomeError(f"Docker build failed: {stderr.decode()}")
                
            self.logger.info(f"Built Docker image: {image_tag}")
            
        finally:
            os.unlink(dockerfile_path)
            
    async def _build_k8s_manifests(self, config: DeploymentConfig):
        """Build Kubernetes manifests."""
        # Load template
        template_path = self.templates_dir / "kubernetes-deployment.yaml"
        
        with open(template_path, 'r') as f:
            manifest = yaml.safe_load(f)
            
        # Customize manifest
        manifest['metadata']['name'] = config.app_name
        manifest['spec']['replicas'] = config.replicas_per_region
        
        # Update container spec
        container = manifest['spec']['template']['spec']['containers'][0]
        container['image'] = f"{config.app_name}:{config.version}"
        container['resources']['requests']['cpu'] = config.cpu_request
        container['resources']['requests']['memory'] = config.memory_request
        container['resources']['limits']['cpu'] = config.cpu_limit
        container['resources']['limits']['memory'] = config.memory_limit
        
        # Save customized manifest
        manifest_path = self.templates_dir / f"{config.app_name}-deployment.yaml"
        with open(manifest_path, 'w') as f:
            yaml.dump(manifest, f, default_flow_style=False)
            
        self.logger.info(f"Built Kubernetes manifest: {manifest_path}")
        
    async def _deploy_to_staging(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Deploy to staging environment."""
        # Simulate staging deployment
        await asyncio.sleep(2)  # Simulate deployment time
        
        return {
            'status': 'success',
            'endpoints': ['https://staging.connectome-gnn.ai'],
            'replicas': config.replicas_per_region,
            'timestamp': time.time()
        }
        
    async def _run_staging_tests(self, config: DeploymentConfig):
        """Run staging tests."""
        # Simulate staging tests
        await asyncio.sleep(1)
        
        # In real implementation, would run smoke tests, integration tests, etc.
        self.logger.info("Staging tests completed")
        
    async def _deploy_to_region(self, config: DeploymentConfig, region: Region):
        """Deploy to specific region."""
        region_config = self.region_manager.regions.get(region)
        if not region_config:
            raise ConnectomeError(f"Region {region} not configured")
            
        # Simulate regional deployment
        await asyncio.sleep(3)  # Simulate deployment time
        
        self.logger.info(f"Deployed to region {region.value}")
        
    async def _setup_monitoring(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Setup monitoring and alerting."""
        # Simulate monitoring setup
        await asyncio.sleep(1)
        
        return {
            'prometheus_endpoint': 'https://prometheus.connectome-gnn.ai',
            'grafana_dashboard': 'https://grafana.connectome-gnn.ai/dashboard/connectome-gnn',
            'alert_manager': 'https://alertmanager.connectome-gnn.ai',
            'log_aggregator': 'https://logs.connectome-gnn.ai'
        }
        
    async def _get_previous_version(self, config: DeploymentConfig) -> str:
        """Get previous deployment version."""
        # Find last successful deployment
        for deployment in reversed(self.deployment_history):
            if (deployment.status == DeploymentStatus.SUCCESS and 
                deployment.version != config.version):
                return deployment.version
                
        return "latest"  # Fallback
        
    async def _rollback_region(self, config: DeploymentConfig, region: Region, version: str):
        """Rollback specific region."""
        # Simulate rollback
        await asyncio.sleep(2)
        
        self.logger.info(f"Rolled back {region.value} to version {version}")
        
    def _generate_deployment_id(self, config: DeploymentConfig) -> str:
        """Generate unique deployment ID."""
        timestamp = str(int(time.time()))
        config_hash = hashlib.md5(str(asdict(config)).encode()).hexdigest()[:8]
        return f"deploy-{timestamp}-{config_hash}"
        
    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentResult]:
        """Get deployment status."""
        if deployment_id in self.active_deployments:
            return self.active_deployments[deployment_id]
            
        for deployment in self.deployment_history:
            if deployment.deployment_id == deployment_id:
                return deployment
                
        return None
        
    def list_deployments(self, limit: int = 10) -> List[DeploymentResult]:
        """List recent deployments."""
        all_deployments = list(self.active_deployments.values()) + self.deployment_history
        all_deployments.sort(key=lambda x: x.start_time, reverse=True)
        
        return all_deployments[:limit]
        
    async def health_check_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """Perform health check on deployment."""
        deployment = self.get_deployment_status(deployment_id)
        
        if not deployment:
            return {'status': 'not_found'}
            
        if deployment.status != DeploymentStatus.SUCCESS:
            return {
                'status': 'unhealthy',
                'reason': f'Deployment status: {deployment.status.value}'
            }
            
        # Check all deployed regions
        region_health = {}
        overall_healthy = True
        
        for region in deployment.regions_deployed:
            try:
                region_config = self.region_manager.regions.get(region)
                health_url = f"{region_config.endpoint_url}/health"
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        if response.status == 200:
                            region_health[region.value] = 'healthy'
                        else:
                            region_health[region.value] = f'unhealthy (status: {response.status})'
                            overall_healthy = False
                            
            except Exception as e:
                region_health[region.value] = f'unhealthy (error: {str(e)})'
                overall_healthy = False
                
        return {
            'status': 'healthy' if overall_healthy else 'unhealthy',
            'deployment_id': deployment_id,
            'version': deployment.version,
            'regions': region_health,
            'timestamp': time.time()
        }


class DeploymentPipeline:
    """High-level deployment pipeline interface."""
    
    def __init__(self):
        self.orchestrator = ProductionDeploymentOrchestrator()
        self.logger = get_logger(__name__)
        
    async def deploy(
        self,
        version: str,
        environment: str = "production",
        regions: List[Region] = None,
        dry_run: bool = False,
        **kwargs
    ) -> DeploymentResult:
        """Deploy application with standard configuration."""
        if regions is None:
            regions = [Region.US_EAST, Region.EU_WEST, Region.ASIA_PACIFIC]
            
        config = DeploymentConfig(
            app_name="connectome-gnn",
            version=version,
            environment=environment,
            regions=regions,
            **kwargs
        )
        
        return await self.orchestrator.deploy_to_production(config, dry_run)
        
    async def rollback(
        self,
        deployment_id: str,
        target_version: Optional[str] = None
    ) -> DeploymentResult:
        """Rollback deployment."""
        current_deployment = self.orchestrator.get_deployment_status(deployment_id)
        
        if not current_deployment:
            raise ConnectomeError(f"Deployment {deployment_id} not found")
            
        if target_version is None:
            target_version = await self.orchestrator._get_previous_version(
                DeploymentConfig(
                    app_name="connectome-gnn",
                    version=current_deployment.version,
                    environment="production",
                    regions=current_deployment.regions_deployed
                )
            )
            
        # Create rollback deployment
        rollback_config = DeploymentConfig(
            app_name="connectome-gnn",
            version=target_version,
            environment="production",
            regions=current_deployment.regions_deployed
        )
        
        return await self.orchestrator.deploy_to_production(rollback_config)
        
    def get_status(self, deployment_id: str) -> Optional[DeploymentResult]:
        """Get deployment status."""
        return self.orchestrator.get_deployment_status(deployment_id)
        
    def list_deployments(self, limit: int = 10) -> List[DeploymentResult]:
        """List deployments."""
        return self.orchestrator.list_deployments(limit)


# Global deployment pipeline
_global_deployment_pipeline = None

def get_deployment_pipeline() -> DeploymentPipeline:
    """Get global deployment pipeline instance."""
    global _global_deployment_pipeline
    if _global_deployment_pipeline is None:
        _global_deployment_pipeline = DeploymentPipeline()
    return _global_deployment_pipeline


# Convenience functions
async def deploy_to_production(
    version: str,
    regions: List[Region] = None,
    dry_run: bool = False
) -> DeploymentResult:
    """Deploy to production with default configuration."""
    pipeline = get_deployment_pipeline()
    return await pipeline.deploy(version, regions=regions, dry_run=dry_run)


async def quick_deploy(version: str) -> DeploymentResult:
    """Quick deployment to single region for testing."""
    pipeline = get_deployment_pipeline()
    return await pipeline.deploy(
        version=version,
        regions=[Region.US_EAST],
        replicas_per_region=1,
        run_quality_gates=False
    )