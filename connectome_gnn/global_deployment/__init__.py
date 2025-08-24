"""Global deployment and multi-region infrastructure for Connectome-GNN-Suite."""

from .multi_region import (
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