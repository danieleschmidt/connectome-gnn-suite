"""Core functionality that doesn't depend on external ML libraries."""

from .base_types import ConnectomeData, ModelConfig, TrainingConfig
from .utils import validate_input, safe_import, get_device_info

__all__ = [
    "ConnectomeData", 
    "ModelConfig", 
    "TrainingConfig",
    "validate_input",
    "safe_import", 
    "get_device_info"
]