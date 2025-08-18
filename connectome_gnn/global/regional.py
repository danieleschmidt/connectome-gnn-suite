"""Regional deployment management for global-first architecture."""

import os
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

from ..robust.logging_config import get_logger
from .compliance import get_compliance_manager


class Region(Enum):
    """Supported deployment regions."""
    US_EAST_1 = "us-east-1"
    US_WEST_2 = "us-west-2"
    EU_WEST_1 = "eu-west-1"
    EU_CENTRAL_1 = "eu-central-1"
    ASIA_PACIFIC_1 = "ap-southeast-1"
    ASIA_PACIFIC_2 = "ap-northeast-1"
    CANADA_CENTRAL = "ca-central-1"
    AUSTRALIA_SOUTHEAST = "ap-southeast-2"


@dataclass
class RegionConfig:
    """Configuration for a deployment region."""
    region_code: str
    region_name: str
    country_codes: List[str]
    data_residency_required: bool
    compliance_frameworks: List[str]
    timezone: str
    currency: str
    language_codes: List[str]
    cdn_endpoint: Optional[str] = None
    api_endpoint: Optional[str] = None
    storage_class: str = "STANDARD"
    backup_regions: List[str] = None
    
    def __post_init__(self):
        if self.backup_regions is None:
            self.backup_regions = []


class RegionalDeploymentManager:
    """Manages regional deployments and data residency."""
    
    def __init__(self, config_dir: str = "./regional_config",
                 logger: Optional[logging.Logger] = None):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or get_logger("regional_deployment")
        
        # Load region configurations
        self.regions = self._load_region_configs()
        
        # Current deployment region
        self.current_region = self._detect_current_region()
        
        # Data residency mappings
        self.data_residency_rules = self._load_data_residency_rules()
    
    def _load_region_configs(self) -> Dict[str, RegionConfig]:
        """Load regional configurations."""
        regions = {}
        
        # Default region configurations
        default_regions = {
            Region.US_EAST_1.value: RegionConfig(
                region_code="us-east-1",
                region_name="US East (N. Virginia)",
                country_codes=["US"],
                data_residency_required=False,
                compliance_frameworks=["SOC2", "HIPAA", "FedRAMP"],
                timezone="America/New_York",
                currency="USD",
                language_codes=["en_US", "es_US"],
                cdn_endpoint="https://cdn-us-east.connectome-gnn.com",
                api_endpoint="https://api-us-east.connectome-gnn.com"
            ),
            Region.EU_WEST_1.value: RegionConfig(
                region_code="eu-west-1",
                region_name="EU West (Ireland)",
                country_codes=["IE", "GB", "FR", "DE", "ES", "IT", "NL", "BE"],
                data_residency_required=True,
                compliance_frameworks=["GDPR", "ISO27001", "SOC2"],
                timezone="Europe/Dublin",
                currency="EUR",
                language_codes=["en_GB", "fr_FR", "de_DE", "es_ES", "it_IT", "nl_NL"],
                cdn_endpoint="https://cdn-eu-west.connectome-gnn.com",
                api_endpoint="https://api-eu-west.connectome-gnn.com",
                backup_regions=["eu-central-1"]
            ),
            Region.ASIA_PACIFIC_1.value: RegionConfig(
                region_code="ap-southeast-1",
                region_name="Asia Pacific (Singapore)",
                country_codes=["SG", "MY", "TH", "ID", "PH", "VN"],
                data_residency_required=True,
                compliance_frameworks=["PDPA", "ISO27001"],
                timezone="Asia/Singapore",
                currency="USD",
                language_codes=["en_US", "zh_CN", "ja_JP", "ko_KR"],
                cdn_endpoint="https://cdn-ap-southeast.connectome-gnn.com",
                api_endpoint="https://api-ap-southeast.connectome-gnn.com",
                backup_regions=["ap-northeast-1"]
            ),
            Region.CANADA_CENTRAL.value: RegionConfig(
                region_code="ca-central-1",
                region_name="Canada (Central)",
                country_codes=["CA"],
                data_residency_required=True,
                compliance_frameworks=["PIPEDA", "SOC2"],
                timezone="America/Toronto",
                currency="CAD",
                language_codes=["en_CA", "fr_CA"],
                cdn_endpoint="https://cdn-ca-central.connectome-gnn.com",
                api_endpoint="https://api-ca-central.connectome-gnn.com"
            )
        }
        
        # Load from config files if they exist
        for region_code, default_config in default_regions.items():
            config_file = self.config_dir / f"{region_code}.json"
            
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config_data = json.load(f)
                        regions[region_code] = RegionConfig(**config_data)
                except Exception as e:
                    self.logger.error(f"Failed to load config for {region_code}: {e}")
                    regions[region_code] = default_config
            else:
                regions[region_code] = default_config
                self._save_region_config(region_code, default_config)
        
        return regions
    
    def _save_region_config(self, region_code: str, config: RegionConfig):
        """Save region configuration to file."""
        config_file = self.config_dir / f"{region_code}.json"
        
        try:
            config_dict = {
                'region_code': config.region_code,
                'region_name': config.region_name,
                'country_codes': config.country_codes,
                'data_residency_required': config.data_residency_required,
                'compliance_frameworks': config.compliance_frameworks,
                'timezone': config.timezone,
                'currency': config.currency,
                'language_codes': config.language_codes,
                'cdn_endpoint': config.cdn_endpoint,
                'api_endpoint': config.api_endpoint,
                'storage_class': config.storage_class,
                'backup_regions': config.backup_regions
            }
            
            with open(config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save config for {region_code}: {e}")
    
    def _detect_current_region(self) -> str:
        """Detect current deployment region."""
        # Check environment variable first
        region = os.environ.get('CONNECTOME_REGION')
        if region and region in self.regions:
            return region
        
        # Check cloud provider metadata (simplified)
        # In production, this would query cloud provider APIs
        
        # Default to US East
        return Region.US_EAST_1.value
    
    def _load_data_residency_rules(self) -> Dict[str, List[str]]:
        """Load data residency rules."""
        rules_file = self.config_dir / "data_residency_rules.json"
        
        default_rules = {
            # EU countries must store data in EU regions
            "EU": ["eu-west-1", "eu-central-1"],
            "CA": ["ca-central-1"],
            "AU": ["ap-southeast-2"],
            "SG": ["ap-southeast-1"],
            "JP": ["ap-northeast-1"],
            # US data can be stored in US regions
            "US": ["us-east-1", "us-west-2"],
            # Default: can use any region
            "*": list(self.regions.keys())
        }
        
        if rules_file.exists():
            try:
                with open(rules_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load data residency rules: {e}")
        else:
            # Save default rules
            try:
                with open(rules_file, 'w') as f:
                    json.dump(default_rules, f, indent=2)
            except Exception as e:
                self.logger.error(f"Failed to save default rules: {e}")
        
        return default_rules
    
    def get_current_region(self) -> RegionConfig:
        """Get current region configuration."""
        return self.regions[self.current_region]
    
    def get_region_config(self, region_code: str) -> Optional[RegionConfig]:
        """Get configuration for a specific region."""
        return self.regions.get(region_code)
    
    def get_available_regions(self) -> List[str]:
        """Get list of available regions."""
        return list(self.regions.keys())
    
    def get_regions_for_country(self, country_code: str) -> List[str]:
        """Get suitable regions for a country."""
        suitable_regions = []
        
        for region_code, config in self.regions.items():
            if country_code in config.country_codes:
                suitable_regions.append(region_code)
        
        # If no specific region found, check data residency rules
        if not suitable_regions:
            if country_code in self.data_residency_rules:
                suitable_regions = self.data_residency_rules[country_code]
            else:
                suitable_regions = self.data_residency_rules.get("*", [])
        
        return suitable_regions
    
    def validate_data_residency(self, user_country: str, 
                              data_region: str) -> Tuple[bool, str]:
        """Validate if data can be stored in a region for a user."""
        suitable_regions = self.get_regions_for_country(user_country)
        
        if data_region in suitable_regions:
            return True, "Data residency compliant"
        
        region_config = self.regions.get(data_region)
        if region_config and region_config.data_residency_required:
            return False, f"Data residency violation: {user_country} data cannot be stored in {data_region}"
        
        return True, "Data residency check passed (not strictly required)"
    
    def get_compliance_requirements(self, region_code: str) -> List[str]:
        """Get compliance frameworks required for a region."""
        region_config = self.regions.get(region_code)
        if region_config:
            return region_config.compliance_frameworks
        return []
    
    def get_optimal_region(self, user_country: str, 
                          requirements: Dict[str, Any] = None) -> str:
        """Get optimal region for a user based on location and requirements."""
        suitable_regions = self.get_regions_for_country(user_country)
        
        if not suitable_regions:
            return self.current_region
        
        # If only one suitable region, return it
        if len(suitable_regions) == 1:
            return suitable_regions[0]
        
        # Score regions based on requirements
        region_scores = {}
        
        for region_code in suitable_regions:
            region_config = self.regions[region_code]
            score = 0
            
            # Prefer regions in the same country
            if user_country in region_config.country_codes:
                score += 10
            
            # Consider latency (simplified - based on geographic proximity)
            if requirements and 'low_latency' in requirements:
                if user_country in region_config.country_codes:
                    score += 5
            
            # Consider compliance requirements
            if requirements and 'compliance' in requirements:
                required_frameworks = requirements['compliance']
                matching_frameworks = set(region_config.compliance_frameworks) & set(required_frameworks)
                score += len(matching_frameworks) * 2
            
            region_scores[region_code] = score
        
        # Return region with highest score
        return max(region_scores.items(), key=lambda x: x[1])[0]
    
    def configure_region_routing(self, user_country: str) -> Dict[str, str]:
        """Configure routing endpoints for a user's country."""
        optimal_region = self.get_optimal_region(user_country)
        region_config = self.regions[optimal_region]
        
        routing_config = {
            'region': optimal_region,
            'api_endpoint': region_config.api_endpoint,
            'cdn_endpoint': region_config.cdn_endpoint,
            'timezone': region_config.timezone,
            'currency': region_config.currency,
            'language_codes': region_config.language_codes
        }
        
        return routing_config
    
    def get_backup_regions(self, primary_region: str) -> List[str]:
        """Get backup regions for failover."""
        region_config = self.regions.get(primary_region)
        if region_config and region_config.backup_regions:
            return region_config.backup_regions
        
        # Default backup strategy: use regions in the same geographical area
        if primary_region.startswith('us-'):
            return [r for r in self.regions.keys() if r.startswith('us-') and r != primary_region]
        elif primary_region.startswith('eu-'):
            return [r for r in self.regions.keys() if r.startswith('eu-') and r != primary_region]
        elif primary_region.startswith('ap-'):
            return [r for r in self.regions.keys() if r.startswith('ap-') and r != primary_region]
        
        return []
    
    def health_check_region(self, region_code: str) -> Dict[str, Any]:
        """Perform health check for a region."""
        region_config = self.regions.get(region_code)
        if not region_config:
            return {
                'region': region_code,
                'healthy': False,
                'error': 'Region not found'
            }
        
        health_status = {
            'region': region_code,
            'healthy': True,
            'api_endpoint': region_config.api_endpoint,
            'cdn_endpoint': region_config.cdn_endpoint,
            'compliance_frameworks': region_config.compliance_frameworks,
            'data_residency_required': region_config.data_residency_required
        }
        
        # In production, this would perform actual health checks
        # For now, assume all configured regions are healthy
        
        return health_status
    
    def generate_deployment_report(self) -> Dict[str, Any]:
        """Generate regional deployment report."""
        report = {
            'current_region': self.current_region,
            'total_regions': len(self.regions),
            'regions_with_data_residency': sum(1 for r in self.regions.values() if r.data_residency_required),
            'compliance_frameworks': set(),
            'supported_countries': set(),
            'supported_languages': set(),
            'regions': {}
        }
        
        for region_code, config in self.regions.items():
            report['compliance_frameworks'].update(config.compliance_frameworks)
            report['supported_countries'].update(config.country_codes)
            report['supported_languages'].update(config.language_codes)
            
            report['regions'][region_code] = {
                'name': config.region_name,
                'countries': config.country_codes,
                'data_residency_required': config.data_residency_required,
                'compliance': config.compliance_frameworks,
                'healthy': True  # Would be actual health check result
            }
        
        # Convert sets to lists for JSON serialization
        report['compliance_frameworks'] = list(report['compliance_frameworks'])
        report['supported_countries'] = list(report['supported_countries'])
        report['supported_languages'] = list(report['supported_languages'])
        
        return report


# Global regional manager
_global_regional_manager = None

def get_regional_manager() -> RegionalDeploymentManager:
    """Get global regional deployment manager."""
    global _global_regional_manager
    if _global_regional_manager is None:
        _global_regional_manager = RegionalDeploymentManager()
    return _global_regional_manager


def route_user_to_region(user_country: str, user_requirements: Dict[str, Any] = None) -> Dict[str, str]:
    """Route user to optimal region based on location and requirements."""
    manager = get_regional_manager()
    return manager.configure_region_routing(user_country)