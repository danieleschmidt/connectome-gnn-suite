"""Task definitions for connectome analysis."""

from .base import BaseConnectomeTask
from .graph_level import (
    CognitiveScorePrediction,
    SubjectClassification, 
    AgeRegression,
    ConnectomeTaskSuite
)
from .node_level import NodeLevelTask
from .edge_level import EdgeLevelTask

__all__ = [
    "BaseConnectomeTask",
    "CognitiveScorePrediction",
    "SubjectClassification",
    "AgeRegression", 
    "ConnectomeTaskSuite",
    "NodeLevelTask",
    "EdgeLevelTask"
]