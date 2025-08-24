"""Optimization utilities for large-scale connectome GNN training and inference."""

try:
    from .distributed import DistributedTrainer, MultiGPUTrainer
    from .memory_efficient import MemoryEfficientModel, GradientCheckpointing, MixedPrecisionTrainer  
    from .inference import InferenceOptimizer, BatchedInference, ModelServer
    from .adaptive_optimization import (
        OptimizationConfig,
        AdaptiveLearningRateScheduler,
        AdaptiveOptimizer,
        RAdam,
        Lookahead,
        GradientClippingManager,
        OptimizerFactory,
        OptimizationPipeline
    )
    
    __all__ = [
        "DistributedTrainer",
        "MultiGPUTrainer", 
        "MemoryEfficientModel",
        "GradientCheckpointing", 
        "MixedPrecisionTrainer",
        "InferenceOptimizer",
        "BatchedInference",
        "ModelServer",
        "OptimizationConfig",
        "AdaptiveLearningRateScheduler",
        "AdaptiveOptimizer",
        "RAdam",
        "Lookahead",
        "GradientClippingManager",
        "OptimizerFactory",
        "OptimizationPipeline"
    ]
except ImportError:
    # Fallback if some optimization modules are not available
    try:
        from .adaptive_optimization import (
            OptimizationConfig,
            AdaptiveLearningRateScheduler,
            AdaptiveOptimizer,
            RAdam,
            Lookahead,
            GradientClippingManager,
            OptimizerFactory,
            OptimizationPipeline
        )
        
        __all__ = [
            "OptimizationConfig",
            "AdaptiveLearningRateScheduler", 
            "AdaptiveOptimizer",
            "RAdam",
            "Lookahead",
            "GradientClippingManager",
            "OptimizerFactory",
            "OptimizationPipeline"
        ]
    except ImportError:
        __all__ = []