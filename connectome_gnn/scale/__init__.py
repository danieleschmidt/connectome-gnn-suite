"""Scaling and optimization components for production deployment."""

# Import with graceful degradation
__all__ = []

try:
    from .memory_optimization import MemoryOptimizer, MemoryProfiler, GradientCheckpointing
    __all__.extend(["MemoryOptimizer", "MemoryProfiler", "GradientCheckpointing"])
except ImportError:
    pass

try:
    from .performance_monitoring import PerformanceMonitor, BenchmarkSuite, MetricsCollector
    __all__.extend(["PerformanceMonitor", "BenchmarkSuite", "MetricsCollector"])
except ImportError:
    pass

try:
    from .caching import SmartCache, ResultCache, ModelCache
    __all__.extend(["SmartCache", "ResultCache", "ModelCache"])
except ImportError:
    pass

# Optional components that may not exist
try:
    from .distributed_training import DistributedTrainer, ModelParallelism, DataParallelism
    __all__.extend(["DistributedTrainer", "ModelParallelism", "DataParallelism"])
except ImportError:
    pass

try:
    from .inference_optimization import InferenceOptimizer, ModelQuantization, ONNXExporter
    __all__.extend(["InferenceOptimizer", "ModelQuantization", "ONNXExporter"])
except ImportError:
    pass

try:
    from .auto_scaling import AutoScaler, ResourceMonitor, LoadBalancer
    __all__.extend(["AutoScaler", "ResourceMonitor", "LoadBalancer"])
except ImportError:
    pass