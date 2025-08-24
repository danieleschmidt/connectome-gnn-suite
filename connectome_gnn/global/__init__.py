"""Global-first implementation for multi-region deployment and compliance."""

from .i18n import InternationalizationManager, get_i18n_manager
from .compliance import ComplianceManager, GDPRCompliance, CCPACompliance
from .regional import RegionalDeploymentManager, get_regional_manager
from .localization import LocalizationService, CurrencyConverter, TimezoneManager

try:
    from ..global_deployment import (
        Region,
        RegionConfig,
        GlobalUser,
        LoadBalancer,
        RegionManager,
        GlobalDataReplication,
        GlobalConfigManager,
        get_region_manager,
        get_data_replication,
        get_config_manager,
        deploy_to_all_regions
    )
    
    __all__ = [
        'InternationalizationManager',
        'get_i18n_manager',
        'ComplianceManager', 
        'GDPRCompliance',
        'CCPACompliance',
        'RegionalDeploymentManager',
        'get_regional_manager',
        'LocalizationService',
        'CurrencyConverter',
        'TimezoneManager',
        'Region',
        'RegionConfig',
        'GlobalUser',
        'LoadBalancer',
        'RegionManager',
        'GlobalDataReplication',
        'GlobalConfigManager',
        'get_region_manager',
        'get_data_replication',
        'get_config_manager',
        'deploy_to_all_regions'
    ]
except ImportError:
    __all__ = [
        'InternationalizationManager',
        'get_i18n_manager',
        'ComplianceManager', 
        'GDPRCompliance',
        'CCPACompliance',
        'RegionalDeploymentManager',
        'get_regional_manager',
        'LocalizationService',
        'CurrencyConverter',
        'TimezoneManager'
    ]