"""Connectome-GNN-Suite: Graph Neural Networks for Human Brain Connectivity Analysis.

A comprehensive graph learning benchmark and toolkit built on the Human Connectome
Project's massive brain network data. Features hierarchical message-passing baselines,
sub-graph visualization, and neurologically-informed graph neural network architectures.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@example.com"

# Core framework imports (no external dependencies)
from .core import base_types, utils
from .robust import error_handling, validation, logging_config, security
from .scale import memory_optimization, performance_monitoring, caching
from .testing import test_framework, quality_gates
from .deployment import config_management

# Optional imports with graceful degradation
try:
    from .data import ConnectomeDataset, ConnectomeProcessor, HCPLoader
    _DATA_AVAILABLE = True
except ImportError:
    _DATA_AVAILABLE = False

try:
    from .models import (
        BaseConnectomeModel,
        HierarchicalBrainGNN,
        TemporalConnectomeGNN,
        MultiModalBrainGNN,
        PopulationGraphGNN
    )
    _MODELS_AVAILABLE = True
except ImportError:
    _MODELS_AVAILABLE = False

try:
    from .tasks import (
        BaseConnectomeTask,
        CognitiveScorePrediction,
        SubjectClassification,
        AgeRegression,
        ConnectomeTaskSuite
    )
    _TASKS_AVAILABLE = True
except ImportError:
    _TASKS_AVAILABLE = False

try:
    from .training import ConnectomeTrainer
    _TRAINING_AVAILABLE = True
except ImportError:
    _TRAINING_AVAILABLE = False

# Always available core utilities
from .core.utils import set_random_seed, get_device_info

# Define what's available
__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    # Core framework (always available)
    "base_types",
    "utils", 
    "error_handling",
    "validation",
    "logging_config",
    "security",
    "memory_optimization",
    "performance_monitoring", 
    "caching",
    "test_framework",
    "quality_gates",
    "config_management",
    # Utilities
    "set_random_seed",
    "get_device_info",
]

# Add optional components if available
if _DATA_AVAILABLE:
    __all__.extend([
        "ConnectomeDataset",
        "ConnectomeProcessor", 
        "HCPLoader"
    ])

if _MODELS_AVAILABLE:
    __all__.extend([
        "BaseConnectomeModel",
        "HierarchicalBrainGNN",
        "TemporalConnectomeGNN",
        "MultiModalBrainGNN",
        "PopulationGraphGNN"
    ])

if _TASKS_AVAILABLE:
    __all__.extend([
        "BaseConnectomeTask",
        "CognitiveScorePrediction",
        "SubjectClassification",
        "AgeRegression",
        "ConnectomeTaskSuite"
    ])

if _TRAINING_AVAILABLE:
    __all__.extend([
        "ConnectomeTrainer"
    ])

def get_available_components():
    """Get information about available components."""
    return {
        'core_framework': True,
        'data_loading': _DATA_AVAILABLE,
        'models': _MODELS_AVAILABLE, 
        'tasks': _TASKS_AVAILABLE,
        'training': _TRAINING_AVAILABLE,
        'version': __version__
    }