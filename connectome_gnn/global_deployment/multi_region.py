"""Multi-region deployment and global scaling for Connectome-GNN-Suite."""

import asyncio
import aiohttp
import json
import time
import logging
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
import concurrent.futures
from abc import ABC, abstractmethod
import hashlib
import os
from pathlib import Path

from ..robust.logging_config import get_logger
from ..robust.error_handling import ConnectomeError


class Region(Enum):
    """Global regions for deployment."""
    US_EAST = "us-east-1"
    US_WEST = "us-west-2"
    EU_WEST = "eu-west-1"
    EU_CENTRAL = "eu-central-1"
    ASIA_PACIFIC = "ap-southeast-1"
    ASIA_NORTHEAST = "ap-northeast-1"
    CANADA = "ca-central-1"
    AUSTRALIA = "ap-southeast-2"


@dataclass
class RegionConfig:
    """Configuration for a specific region."""
    region: Region
    endpoint_url: str
    data_center: str
    compliance_requirements: List[str]
    data_residency_required: bool
    encryption_at_rest: bool
    encryption_in_transit: bool
    backup_regions: List[Region]
    latency_sla_ms: int
    availability_sla: float


@dataclass
class GlobalUser:
    """Global user with region-specific data."""
    user_id: str
    primary_region: Region
    allowed_regions: List[Region]
    data_residency_region: Region
    consent_preferences: Dict[str, bool]
    language: str
    timezone: str
    created_at: float
    last_active: float


class LoadBalancer:
    """Global load balancer with region affinity."""
    
    def __init__(self):
        self.region_weights: Dict[Region, float] = {}
        self.region_health: Dict[Region, float] = {}
        self.user_affinity: Dict[str, Region] = {}
        
        self.logger = get_logger(__name__)
        
        # Initialize default weights
        for region in Region:
            self.region_weights[region] = 1.0
            self.region_health[region] = 1.0
            
    def select_region(
        self,
        user_id: Optional[str] = None,
        preferred_regions: List[Region] = None,
        exclude_regions: List[Region] = None
    ) -> Region:
        """Select optimal region for user request."""
        available_regions = list(Region)
        
        # Remove excluded regions
        if exclude_regions:
            available_regions = [r for r in available_regions if r not in exclude_regions]
            
        # Filter by preferred regions
        if preferred_regions:
            available_regions = [r for r in available_regions if r in preferred_regions]
            
        if not available_regions:
            raise ConnectomeError("No available regions for request")
            
        # Check user affinity
        if user_id and user_id in self.user_affinity:
            preferred_region = self.user_affinity[user_id]
            if preferred_region in available_regions:
                return preferred_region
                
        # Weighted selection based on health and load
        weights = []
        for region in available_regions:
            weight = self.region_weights[region] * self.region_health[region]
            weights.append(weight)
            
        if not weights:
            return available_regions[0]
            
        # Select region with highest weight
        max_weight_idx = weights.index(max(weights))
        selected_region = available_regions[max_weight_idx]
        
        # Update user affinity
        if user_id:
            self.user_affinity[user_id] = selected_region
            
        return selected_region
        
    def update_region_health(self, region: Region, health_score: float):
        """Update region health score (0.0 to 1.0)."""
        self.region_health[region] = max(0.0, min(1.0, health_score))
        self.logger.info(f"Updated {region.value} health to {health_score:.3f}")
        
    def update_region_weight(self, region: Region, weight: float):
        """Update region weight."""
        self.region_weights[region] = max(0.0, weight)
        self.logger.info(f"Updated {region.value} weight to {weight:.3f}")


class RegionManager:
    """Manages multi-region deployment and coordination."""
    
    def __init__(self):
        self.regions: Dict[Region, RegionConfig] = {}
        self.load_balancer = LoadBalancer()
        self.active_regions: List[Region] = []
        
        self.logger = get_logger(__name__)
        
        self._setup_default_regions()
        
    def _setup_default_regions(self):
        """Setup default region configurations."""
        default_configs = {
            Region.US_EAST: RegionConfig(
                region=Region.US_EAST,
                endpoint_url="https://us-east.connectome-gnn.ai",
                data_center="Virginia",
                compliance_requirements=["HIPAA", "SOC2"],
                data_residency_required=False,
                encryption_at_rest=True,
                encryption_in_transit=True,
                backup_regions=[Region.US_WEST],
                latency_sla_ms=100,
                availability_sla=0.999
            ),
            Region.EU_WEST: RegionConfig(
                region=Region.EU_WEST,
                endpoint_url="https://eu-west.connectome-gnn.ai",
                data_center="Ireland",
                compliance_requirements=["GDPR", "ISO27001"],
                data_residency_required=True,
                encryption_at_rest=True,
                encryption_in_transit=True,
                backup_regions=[Region.EU_CENTRAL],
                latency_sla_ms=80,
                availability_sla=0.999
            ),
            Region.ASIA_PACIFIC: RegionConfig(
                region=Region.ASIA_PACIFIC,
                endpoint_url="https://apac.connectome-gnn.ai",
                data_center="Singapore",
                compliance_requirements=["PDPA", "ISO27001"],
                data_residency_required=True,
                encryption_at_rest=True,
                encryption_in_transit=True,
                backup_regions=[Region.ASIA_NORTHEAST],
                latency_sla_ms=120,
                availability_sla=0.995
            )
        }
        
        for region, config in default_configs.items():
            self.regions[region] = config
            self.active_regions.append(region)
            
    def add_region(self, config: RegionConfig):
        """Add new region to deployment."""
        self.regions[config.region] = config
        
        if config.region not in self.active_regions:
            self.active_regions.append(config.region)
            
        self.logger.info(f"Added region: {config.region.value}")
        
    def remove_region(self, region: Region):
        """Remove region from deployment."""
        if region in self.regions:
            del self.regions[region]
            
        if region in self.active_regions:
            self.active_regions.remove(region)
            
        self.logger.info(f"Removed region: {region.value}")
        
    def get_region_for_user(self, user: GlobalUser) -> Region:
        """Get optimal region for user based on preferences and compliance."""
        # Check data residency requirements
        if user.data_residency_region:
            region_config = self.regions.get(user.data_residency_region)
            if region_config and region_config.data_residency_required:
                return user.data_residency_region
                
        # Use load balancer for selection
        return self.load_balancer.select_region(
            user_id=user.user_id,
            preferred_regions=user.allowed_regions
        )
        
    async def health_check_all_regions(self) -> Dict[Region, Dict[str, Any]]:
        """Perform health checks on all active regions."""
        health_results = {}
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for region in self.active_regions:
                config = self.regions[region]
                task = self._health_check_region(session, region, config)
                tasks.append(task)
                
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                region = self.active_regions[i]
                
                if isinstance(result, Exception):
                    health_results[region] = {
                        'healthy': False,
                        'error': str(result),
                        'latency_ms': float('inf'),
                        'timestamp': time.time()
                    }
                    self.load_balancer.update_region_health(region, 0.0)
                else:
                    health_results[region] = result
                    health_score = 1.0 if result['healthy'] else 0.5
                    self.load_balancer.update_region_health(region, health_score)
                    
        return health_results
        
    async def _health_check_region(
        self,
        session: aiohttp.ClientSession,
        region: Region,
        config: RegionConfig
    ) -> Dict[str, Any]:
        """Perform health check on specific region."""
        health_endpoint = f"{config.endpoint_url}/health"
        start_time = time.time()
        
        try:
            timeout = aiohttp.ClientTimeout(total=5.0)
            async with session.get(health_endpoint, timeout=timeout) as response:
                latency_ms = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    return {
                        'healthy': True,
                        'status_code': response.status,
                        'latency_ms': latency_ms,
                        'data': data,
                        'timestamp': time.time()
                    }
                else:
                    return {
                        'healthy': False,
                        'status_code': response.status,
                        'latency_ms': latency_ms,
                        'timestamp': time.time()
                    }
                    
        except asyncio.TimeoutError:
            return {
                'healthy': False,
                'error': 'Timeout',
                'latency_ms': (time.time() - start_time) * 1000,
                'timestamp': time.time()
            }
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'latency_ms': (time.time() - start_time) * 1000,
                'timestamp': time.time()
            }
            
    def get_region_stats(self) -> Dict[str, Any]:
        """Get statistics for all regions."""
        return {
            'total_regions': len(self.regions),
            'active_regions': len(self.active_regions),
            'region_health': dict(self.load_balancer.region_health),
            'region_weights': dict(self.load_balancer.region_weights),
            'regions': {
                region.value: {
                    'endpoint': config.endpoint_url,
                    'data_center': config.data_center,
                    'compliance': config.compliance_requirements,
                    'data_residency': config.data_residency_required,
                    'latency_sla_ms': config.latency_sla_ms,
                    'availability_sla': config.availability_sla
                }
                for region, config in self.regions.items()
            }
        }


class GlobalDataReplication:
    """Handles data replication across regions."""
    
    def __init__(self, region_manager: RegionManager):
        self.region_manager = region_manager
        self.replication_tasks: Dict[str, asyncio.Task] = {}
        
        self.logger = get_logger(__name__)
        
    async def replicate_data(
        self,
        data_id: str,
        source_region: Region,
        target_regions: List[Region],
        data_type: str = "model"
    ) -> Dict[Region, bool]:
        """Replicate data from source to target regions."""
        replication_results = {}
        
        source_config = self.region_manager.regions.get(source_region)
        if not source_config:
            raise ConnectomeError(f"Source region {source_region} not configured")
            
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for target_region in target_regions:
                target_config = self.region_manager.regions.get(target_region)
                if target_config:
                    task = self._replicate_to_region(
                        session, data_id, source_config, target_config, data_type
                    )
                    tasks.append((target_region, task))
                    
            for target_region, task in tasks:
                try:
                    success = await task
                    replication_results[target_region] = success
                except Exception as e:
                    self.logger.error(f"Replication to {target_region} failed: {e}")
                    replication_results[target_region] = False
                    
        return replication_results
        
    async def _replicate_to_region(
        self,
        session: aiohttp.ClientSession,
        data_id: str,
        source_config: RegionConfig,
        target_config: RegionConfig,
        data_type: str
    ) -> bool:
        """Replicate data to specific target region."""
        try:
            # Get data from source region
            source_url = f"{source_config.endpoint_url}/data/{data_type}/{data_id}"
            target_url = f"{target_config.endpoint_url}/data/{data_type}/{data_id}"
            
            # Download from source
            async with session.get(source_url) as response:
                if response.status != 200:
                    return False
                    
                data = await response.read()
                
            # Upload to target
            async with session.put(target_url, data=data) as response:
                return response.status == 200
                
        except Exception as e:
            self.logger.error(f"Replication error: {e}")
            return False
            
    def start_continuous_replication(
        self,
        data_id: str,
        source_region: Region,
        target_regions: List[Region],
        interval_minutes: int = 60
    ):
        """Start continuous data replication."""
        task_id = f"{data_id}_{source_region.value}"
        
        if task_id in self.replication_tasks:
            self.replication_tasks[task_id].cancel()
            
        async def replication_loop():
            while True:
                try:
                    await self.replicate_data(data_id, source_region, target_regions)
                    await asyncio.sleep(interval_minutes * 60)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Continuous replication error: {e}")
                    await asyncio.sleep(60)  # Wait before retry
                    
        task = asyncio.create_task(replication_loop())
        self.replication_tasks[task_id] = task
        
        self.logger.info(
            f"Started continuous replication for {data_id} "
            f"from {source_region.value} every {interval_minutes} minutes"
        )
        
    def stop_continuous_replication(self, data_id: str, source_region: Region):
        """Stop continuous data replication."""
        task_id = f"{data_id}_{source_region.value}"
        
        if task_id in self.replication_tasks:
            self.replication_tasks[task_id].cancel()
            del self.replication_tasks[task_id]
            self.logger.info(f"Stopped continuous replication for {data_id}")


class GlobalConfigManager:
    """Manages global configuration across regions."""
    
    def __init__(self, region_manager: RegionManager):
        self.region_manager = region_manager
        self.global_config: Dict[str, Any] = {}
        self.region_specific_configs: Dict[Region, Dict[str, Any]] = {}
        
        self.logger = get_logger(__name__)
        
        self._load_default_config()
        
    def _load_default_config(self):
        """Load default global configuration."""
        self.global_config = {
            'api_version': '1.0.0',
            'max_request_size_mb': 100,
            'default_timeout_seconds': 30,
            'rate_limits': {
                'requests_per_minute': 1000,
                'requests_per_hour': 50000
            },
            'security': {
                'require_https': True,
                'min_tls_version': '1.2',
                'enable_cors': True,
                'allowed_origins': ['*']
            },
            'monitoring': {
                'enable_metrics': True,
                'enable_tracing': True,
                'log_level': 'INFO'
            }
        }
        
        # Region-specific overrides
        for region in Region:
            self.region_specific_configs[region] = {}
            
        # EU specific configurations (GDPR compliance)
        self.region_specific_configs[Region.EU_WEST].update({
            'data_retention_days': 90,
            'require_explicit_consent': True,
            'enable_data_portability': True,
            'enable_right_to_deletion': True
        })
        
        # Asia-Pacific specific configurations
        self.region_specific_configs[Region.ASIA_PACIFIC].update({
            'data_localization_required': True,
            'compliance_frameworks': ['PDPA', 'PIPEDA']
        })
        
    def get_config_for_region(self, region: Region) -> Dict[str, Any]:
        """Get configuration for specific region."""
        config = self.global_config.copy()
        
        # Apply region-specific overrides
        if region in self.region_specific_configs:
            config.update(self.region_specific_configs[region])
            
        return config
        
    def update_global_config(self, updates: Dict[str, Any]):
        """Update global configuration."""
        self.global_config.update(updates)
        self.logger.info("Updated global configuration")
        
    def update_region_config(self, region: Region, updates: Dict[str, Any]):
        """Update region-specific configuration."""
        if region not in self.region_specific_configs:
            self.region_specific_configs[region] = {}
            
        self.region_specific_configs[region].update(updates)
        self.logger.info(f"Updated configuration for region {region.value}")
        
    async def sync_config_to_all_regions(self) -> Dict[Region, bool]:
        """Synchronize configuration to all regions."""
        sync_results = {}
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for region in self.region_manager.active_regions:
                config = self.get_config_for_region(region)
                task = self._sync_config_to_region(session, region, config)
                tasks.append((region, task))
                
            for region, task in tasks:
                try:
                    success = await task
                    sync_results[region] = success
                except Exception as e:
                    self.logger.error(f"Config sync to {region} failed: {e}")
                    sync_results[region] = False
                    
        return sync_results
        
    async def _sync_config_to_region(
        self,
        session: aiohttp.ClientSession,
        region: Region,
        config: Dict[str, Any]
    ) -> bool:
        """Sync configuration to specific region."""
        try:
            region_config = self.region_manager.regions.get(region)
            if not region_config:
                return False
                
            config_url = f"{region_config.endpoint_url}/admin/config"
            
            async with session.post(config_url, json=config) as response:
                return response.status == 200
                
        except Exception as e:
            self.logger.error(f"Config sync error for {region}: {e}")
            return False


# Global instances
_global_region_manager = None
_global_data_replication = None
_global_config_manager = None


def get_region_manager() -> RegionManager:
    """Get global region manager instance."""
    global _global_region_manager
    if _global_region_manager is None:
        _global_region_manager = RegionManager()
    return _global_region_manager


def get_data_replication() -> GlobalDataReplication:
    """Get global data replication instance."""
    global _global_data_replication
    if _global_data_replication is None:
        region_manager = get_region_manager()
        _global_data_replication = GlobalDataReplication(region_manager)
    return _global_data_replication


def get_config_manager() -> GlobalConfigManager:
    """Get global config manager instance."""
    global _global_config_manager
    if _global_config_manager is None:
        region_manager = get_region_manager()
        _global_config_manager = GlobalConfigManager(region_manager)
    return _global_config_manager


async def deploy_to_all_regions(deployment_config: Dict[str, Any]) -> Dict[str, Any]:
    """Deploy application to all configured regions."""
    region_manager = get_region_manager()
    
    deployment_results = {}
    
    for region in region_manager.active_regions:
        try:
            # Deploy to specific region
            # This would integrate with actual deployment infrastructure
            deployment_results[region.value] = {
                'status': 'success',
                'timestamp': time.time(),
                'version': deployment_config.get('version', '1.0.0')
            }
            
        except Exception as e:
            deployment_results[region.value] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': time.time()
            }
            
    return {
        'deployment_id': hashlib.md5(str(time.time()).encode()).hexdigest()[:8],
        'total_regions': len(region_manager.active_regions),
        'successful_deployments': sum(
            1 for r in deployment_results.values() 
            if r['status'] == 'success'
        ),
        'results': deployment_results,
        'timestamp': time.time()
    }