"""Comprehensive benchmarking suite for connectome GNNs."""

from .subject_benchmark import SubjectBenchmark
from .edge_prediction import EdgePredictionBenchmark
from .clinical_classification import ClinicalClassification
from .cross_dataset import CrossDatasetBenchmark
from .performance_profiler import PerformanceProfiler
from .statistical_tests import StatisticalValidator

__all__ = [
    "SubjectBenchmark",
    "EdgePredictionBenchmark", 
    "ClinicalClassification",
    "CrossDatasetBenchmark",
    "PerformanceProfiler",
    "StatisticalValidator",
]