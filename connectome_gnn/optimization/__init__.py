"""Optimization utilities for large-scale connectome GNN training and inference."""

try:
    from .distributed import DistributedTrainer, MultiGPUTrainer
    from .memory_efficient import MemoryEfficientModel, GradientCheckpointing, MixedPrecisionTrainer  
    from .inference import InferenceOptimizer, BatchedInference, ModelServer
    
    __all__ = [
        "DistributedTrainer",
        "MultiGPUTrainer", 
        "MemoryEfficientModel",
        "GradientCheckpointing", 
        "MixedPrecisionTrainer",
        "InferenceOptimizer",
        "BatchedInference",
        "ModelServer"
    ]
except ImportError:
    # Fallback if some optimization modules are not available
    __all__ = []