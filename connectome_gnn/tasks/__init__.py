"""Task definitions for connectome analysis."""

from .base import BaseConnectomeTask
from .node_level import NodeLevelTask, BrainRegionPrediction
from .edge_level import EdgeLevelTask, ConnectionPrediction
from .graph_level import GraphLevelTask, CognitiveScorePrediction, SubjectClassification

__all__ = [
    "BaseConnectomeTask",
    "NodeLevelTask",
    "BrainRegionPrediction",
    "EdgeLevelTask", 
    "ConnectionPrediction",
    "GraphLevelTask",
    "CognitiveScorePrediction",
    "SubjectClassification",
]