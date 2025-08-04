"""Connectome-GNN-Suite: Graph Neural Networks for Human Brain Connectivity Analysis.

A comprehensive graph learning benchmark and toolkit built on the Human Connectome
Project's massive brain network data. Features hierarchical message-passing baselines,
sub-graph visualization, and neurologically-informed graph neural network architectures.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@example.com"

# Core imports
from .data import ConnectomeDataset, ConnectomeProcessor, HCPLoader
from .models import (
    BaseConnectomeModel,
    HierarchicalBrainGNN,
    TemporalConnectomeGNN,
    MultiModalBrainGNN,
    PopulationGraphGNN
)
from .tasks import (
    BaseConnectomeTask,
    CognitiveScorePrediction,
    SubjectClassification,
    AgeRegression,
    ConnectomeTaskSuite
)
from .training import ConnectomeTrainer
from .utils import set_random_seed, get_device

__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    # Data
    "ConnectomeDataset",
    "ConnectomeProcessor", 
    "HCPLoader",
    # Models
    "BaseConnectomeModel",
    "HierarchicalBrainGNN",
    "TemporalConnectomeGNN",
    "MultiModalBrainGNN",
    "PopulationGraphGNN",
    # Tasks
    "BaseConnectomeTask",
    "CognitiveScorePrediction",
    "SubjectClassification",
    "AgeRegression",
    "ConnectomeTaskSuite",
    # Training
    "ConnectomeTrainer",
    # Utils
    "set_random_seed",
    "get_device",
]