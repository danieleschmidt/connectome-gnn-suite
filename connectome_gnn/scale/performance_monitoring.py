"""Performance monitoring and benchmarking for the connectome GNN framework."""

import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import logging
from pathlib import Path
from contextlib import contextmanager
import statistics

from ..core.utils import safe_import, Timer
from ..robust.logging_config import get_logger


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_percent: float = 0.0
    gpu_utilization: float = 0.0
    gpu_memory_mb: float = 0.0
    throughput: float = 0.0  # samples/sec or operations/sec
    latency_p50: float = 0.0  # 50th percentile latency
    latency_p95: float = 0.0  # 95th percentile latency
    latency_p99: float = 0.0  # 99th percentile latency
    custom_metrics: Dict[str, float] = field(default_factory=dict)


class MetricsCollector:
    """Collects and aggregates performance metrics."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger("metrics_collector")
        self.torch = safe_import('torch')
        self.process = psutil.Process()
        
        # Metric storage
        self.metrics_history = defaultdict(list)
        self.latency_buffer = deque(maxlen=1000)  # Store recent latencies
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread = None
        self.monitor_interval = 1.0  # seconds
        
    def collect_system_metrics(self) -> Dict[str, float]:
        """Collect current system metrics."""
        metrics = {}
        
        # CPU and memory
        metrics['cpu_percent'] = self.process.cpu_percent()
        memory_info = self.process.memory_info()
        metrics['memory_mb'] = memory_info.rss / 1024 / 1024
        
        # System-wide metrics
        system_memory = psutil.virtual_memory()
        metrics['system_memory_percent'] = system_memory.percent
        metrics['system_cpu_percent'] = psutil.cpu_percent()
        
        # GPU metrics if available
        if self.torch and self.torch.cuda.is_available():
            try:
                metrics['gpu_memory_allocated_mb'] = self.torch.cuda.memory_allocated() / 1024 / 1024
                metrics['gpu_memory_reserved_mb'] = self.torch.cuda.memory_reserved() / 1024 / 1024
                
                # Try to get GPU utilization (requires nvidia-ml-py)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    metrics['gpu_utilization_percent'] = utilization.gpu
                    pynvml.nvmlShutdown()
                except:
                    metrics['gpu_utilization_percent'] = 0.0
                    
            except Exception as e:
                self.logger.debug(f"Error collecting GPU metrics: {e}")
        
        return metrics
    
    def record_latency(self, latency: float):
        """Record a latency measurement."""
        self.latency_buffer.append(latency)
    
    def calculate_latency_percentiles(self) -> Dict[str, float]:
        """Calculate latency percentiles from recent measurements."""
        if not self.latency_buffer:
            return {'p50': 0.0, 'p95': 0.0, 'p99': 0.0}
        
        latencies = sorted(self.latency_buffer)
        n = len(latencies)
        
        return {
            'p50': latencies[int(n * 0.5)],
            'p95': latencies[int(n * 0.95)],
            'p99': latencies[int(n * 0.99)]
        }
    
    def start_monitoring(self, interval: float = 1.0):
        """Start continuous monitoring in background thread."""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitor_interval = interval
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    metrics = self.collect_system_metrics()
                    timestamp = time.time()
                    
                    for key, value in metrics.items():
                        self.metrics_history[key].append((timestamp, value))
                    
                    time.sleep(self.monitor_interval)
                    
                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {e}")
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info(f"Started performance monitoring (interval: {interval}s)")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        self.logger.info("Stopped performance monitoring")
    
    def get_metrics_summary(self, metric_name: str, 
                          time_window: Optional[float] = None) -> Dict[str, float]:
        """Get summary statistics for a metric."""
        if metric_name not in self.metrics_history:
            return {}
        
        data = self.metrics_history[metric_name]
        
        # Filter by time window if specified
        if time_window:
            cutoff_time = time.time() - time_window
            data = [(t, v) for t, v in data if t >= cutoff_time]
        
        if not data:
            return {}
        
        values = [v for t, v in data]
        
        return {
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
            'min': min(values),
            'max': max(values),
            'count': len(values)
        }
    
    def export_metrics(self, filepath: str, time_window: Optional[float] = None):
        """Export metrics to JSON file."""
        export_data = {
            'timestamp': time.time(),
            'time_window': time_window,
            'metrics': {}
        }
        
        for metric_name in self.metrics_history:
            summary = self.get_metrics_summary(metric_name, time_window)
            if summary:
                export_data['metrics'][metric_name] = summary
        
        # Add latency percentiles
        export_data['latency_percentiles'] = self.calculate_latency_percentiles()
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Exported metrics to {filepath}")


class PerformanceMonitor:
    """High-level performance monitoring with profiling capabilities."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger("performance_monitor")
        self.metrics_collector = MetricsCollector(logger)
        self.torch = safe_import('torch')
        
        # Performance profiles
        self.operation_profiles = defaultdict(list)
        self.active_operations = {}
        
    def start_monitoring(self):
        """Start comprehensive monitoring."""
        self.metrics_collector.start_monitoring()
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring and generate report."""
        self.metrics_collector.stop_monitoring()
        self.logger.info("Performance monitoring stopped")
    
    @contextmanager
    def profile_operation(self, operation_name: str, 
                         track_memory: bool = True,
                         track_gpu: bool = True):
        """Context manager for profiling operations."""
        start_time = time.time()
        
        # Collect initial metrics
        initial_metrics = {}
        if track_memory:
            initial_metrics.update(self.metrics_collector.collect_system_metrics())
        
        # PyTorch profiler if available
        torch_profiler = None
        if self.torch and track_gpu and self.torch.cuda.is_available():
            try:
                torch_profiler = self.torch.profiler.profile(
                    activities=[
                        self.torch.profiler.ProfilerActivity.CPU,
                        self.torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=True,
                    with_stack=True
                )
                torch_profiler.start()
            except Exception as e:
                self.logger.debug(f"Could not start torch profiler: {e}")
        
        try:
            yield
        finally:
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Stop torch profiler
            if torch_profiler:
                try:
                    torch_profiler.stop()
                except Exception as e:
                    self.logger.debug(f"Error stopping torch profiler: {e}")
            
            # Collect final metrics
            final_metrics = {}
            if track_memory:
                final_metrics.update(self.metrics_collector.collect_system_metrics())
            
            # Create performance metrics
            metrics = PerformanceMetrics(execution_time=execution_time)
            
            if track_memory and initial_metrics and final_metrics:
                metrics.memory_usage_mb = final_metrics.get('memory_mb', 0) - initial_metrics.get('memory_mb', 0)
                metrics.cpu_percent = final_metrics.get('cpu_percent', 0)
                
                if track_gpu:
                    metrics.gpu_memory_mb = final_metrics.get('gpu_memory_allocated_mb', 0)
            
            # Record latency
            self.metrics_collector.record_latency(execution_time)
            
            # Store profile
            self.operation_profiles[operation_name].append(metrics)
            
            self.logger.debug(f"Operation '{operation_name}' completed in {execution_time:.4f}s")
    
    def benchmark_function(self, func: Callable, *args, 
                          iterations: int = 10,
                          warmup_iterations: int = 2,
                          **kwargs) -> PerformanceMetrics:
        """Benchmark a function with multiple iterations."""
        self.logger.info(f"Benchmarking function '{func.__name__}' for {iterations} iterations")
        
        # Warmup iterations
        for _ in range(warmup_iterations):
            try:
                func(*args, **kwargs)
            except Exception as e:
                self.logger.warning(f"Error during warmup: {e}")
        
        # Clear caches
        if self.torch and self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
            self.torch.cuda.empty_cache()
        
        # Benchmark iterations
        execution_times = []
        memory_usages = []
        
        for i in range(iterations):
            with self.profile_operation(f"benchmark_{func.__name__}_iter_{i}") as _:
                initial_memory = self.metrics_collector.collect_system_metrics().get('memory_mb', 0)
                
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    
                    if self.torch and self.torch.cuda.is_available():
                        self.torch.cuda.synchronize()
                    
                    end_time = time.time()
                    execution_time = end_time - start_time
                    execution_times.append(execution_time)
                    
                    final_memory = self.metrics_collector.collect_system_metrics().get('memory_mb', 0)
                    memory_usages.append(final_memory - initial_memory)
                    
                except Exception as e:
                    self.logger.error(f"Error in benchmark iteration {i}: {e}")
        
        # Calculate statistics
        if not execution_times:
            return PerformanceMetrics()
        
        metrics = PerformanceMetrics(
            execution_time=statistics.mean(execution_times),
            memory_usage_mb=statistics.mean(memory_usages) if memory_usages else 0.0,
            throughput=1.0 / statistics.mean(execution_times) if execution_times else 0.0
        )
        
        # Calculate percentiles
        if len(execution_times) >= 3:
            sorted_times = sorted(execution_times)
            n = len(sorted_times)
            metrics.latency_p50 = sorted_times[int(n * 0.5)]
            metrics.latency_p95 = sorted_times[int(n * 0.95)]
            metrics.latency_p99 = sorted_times[int(n * 0.99)]
        
        self.logger.info(f"Benchmark completed: {metrics.execution_time:.4f}s avg, "
                        f"{metrics.throughput:.2f} ops/sec")
        
        return metrics
    
    def get_operation_profile(self, operation_name: str) -> Dict[str, Any]:
        """Get performance profile for an operation."""
        if operation_name not in self.operation_profiles:
            return {}
        
        profiles = self.operation_profiles[operation_name]
        
        execution_times = [p.execution_time for p in profiles]
        memory_usages = [p.memory_usage_mb for p in profiles]
        
        return {
            'operation_name': operation_name,
            'total_calls': len(profiles),
            'avg_execution_time': statistics.mean(execution_times),
            'total_execution_time': sum(execution_times),
            'avg_memory_usage': statistics.mean(memory_usages) if memory_usages else 0.0,
            'min_execution_time': min(execution_times),
            'max_execution_time': max(execution_times),
            'std_execution_time': statistics.stdev(execution_times) if len(execution_times) > 1 else 0.0
        }
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            'timestamp': time.time(),
            'system_metrics': self.metrics_collector.collect_system_metrics(),
            'latency_percentiles': self.metrics_collector.calculate_latency_percentiles(),
            'operation_profiles': {}
        }
        
        for operation_name in self.operation_profiles:
            report['operation_profiles'][operation_name] = self.get_operation_profile(operation_name)
        
        return report


class BenchmarkSuite:
    """Comprehensive benchmark suite for the connectome GNN framework."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger("benchmark_suite")
        self.monitor = PerformanceMonitor(logger)
        self.torch = safe_import('torch')
        
        # Benchmark configurations
        self.benchmark_configs = {
            'small': {'batch_size': 8, 'hidden_dim': 128, 'num_layers': 2},
            'medium': {'batch_size': 32, 'hidden_dim': 256, 'num_layers': 3},
            'large': {'batch_size': 64, 'hidden_dim': 512, 'num_layers': 4}
        }
    
    def benchmark_data_loading(self, dataset: Any, 
                             batch_sizes: List[int] = [8, 16, 32]) -> Dict[str, Any]:
        """Benchmark data loading performance."""
        results = {}
        
        for batch_size in batch_sizes:
            self.logger.info(f"Benchmarking data loading with batch size {batch_size}")
            
            def load_batch():
                # Simulate loading a batch
                batch = []
                for i in range(batch_size):
                    if i < len(dataset):
                        batch.append(dataset[i])
                return batch
            
            metrics = self.monitor.benchmark_function(
                load_batch, iterations=20, warmup_iterations=5
            )
            
            results[f'batch_size_{batch_size}'] = {
                'avg_time': metrics.execution_time,
                'throughput': metrics.throughput,
                'samples_per_sec': batch_size * metrics.throughput
            }
        
        return results
    
    def benchmark_model_inference(self, model: Any, input_data: Any,
                                device: str = 'cpu') -> Dict[str, Any]:
        """Benchmark model inference performance."""
        if not self.torch:
            self.logger.error("PyTorch required for model benchmarking")
            return {}
        
        self.logger.info(f"Benchmarking model inference on {device}")
        
        # Move model and data to device
        if hasattr(model, 'to'):
            model = model.to(device)
        
        # Warm up
        for _ in range(5):
            try:
                with self.torch.no_grad():
                    _ = model(input_data)
                if device.startswith('cuda'):
                    self.torch.cuda.synchronize()
            except Exception as e:
                self.logger.warning(f"Warmup iteration failed: {e}")
        
        # Benchmark
        def inference_step():
            with self.torch.no_grad():
                output = model(input_data)
                if device.startswith('cuda'):
                    self.torch.cuda.synchronize()
                return output
        
        metrics = self.monitor.benchmark_function(
            inference_step, iterations=50, warmup_iterations=10
        )
        
        return {
            'device': device,
            'avg_inference_time': metrics.execution_time,
            'throughput': metrics.throughput,
            'latency_p50': metrics.latency_p50,
            'latency_p95': metrics.latency_p95,
            'memory_usage': metrics.memory_usage_mb
        }
    
    def benchmark_memory_scaling(self, model_factory: Callable,
                                input_factory: Callable) -> Dict[str, Any]:
        """Benchmark memory scaling with different model sizes."""
        results = {}
        
        for config_name, config in self.benchmark_configs.items():
            self.logger.info(f"Benchmarking memory scaling: {config_name}")
            
            try:
                # Create model and input
                model = model_factory(**config)
                input_data = input_factory(**config)
                
                # Measure memory before
                initial_memory = self.monitor.metrics_collector.collect_system_metrics()
                
                # Run inference
                with self.monitor.profile_operation(f"memory_scaling_{config_name}"):
                    if self.torch:
                        with self.torch.no_grad():
                            _ = model(input_data)
                
                # Measure memory after
                final_memory = self.monitor.metrics_collector.collect_system_metrics()
                
                results[config_name] = {
                    'config': config,
                    'memory_increase_mb': final_memory.get('memory_mb', 0) - initial_memory.get('memory_mb', 0),
                    'gpu_memory_mb': final_memory.get('gpu_memory_allocated_mb', 0),
                    'parameters': sum(p.numel() for p in model.parameters()) if hasattr(model, 'parameters') else 0
                }
                
            except Exception as e:
                self.logger.error(f"Error in memory scaling benchmark for {config_name}: {e}")
                results[config_name] = {'error': str(e)}
        
        return results
    
    def run_comprehensive_benchmark(self, model: Any, dataset: Any) -> Dict[str, Any]:
        """Run comprehensive benchmark suite."""
        self.logger.info("Starting comprehensive benchmark suite")
        
        self.monitor.start_monitoring()
        
        benchmark_results = {
            'timestamp': time.time(),
            'system_info': self.monitor.metrics_collector.collect_system_metrics(),
            'benchmarks': {}
        }
        
        try:
            # Data loading benchmark
            benchmark_results['benchmarks']['data_loading'] = \
                self.benchmark_data_loading(dataset)
            
            # Model inference benchmark
            if dataset and len(dataset) > 0:
                sample_input = dataset[0]
                benchmark_results['benchmarks']['inference_cpu'] = \
                    self.benchmark_model_inference(model, sample_input, 'cpu')
                
                if self.torch and self.torch.cuda.is_available():
                    benchmark_results['benchmarks']['inference_gpu'] = \
                        self.benchmark_model_inference(model, sample_input, 'cuda')
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive benchmark: {e}")
            benchmark_results['error'] = str(e)
        finally:
            self.monitor.stop_monitoring()
        
        # Add performance report
        benchmark_results['performance_report'] = self.monitor.generate_performance_report()
        
        self.logger.info("Comprehensive benchmark completed")
        return benchmark_results
    
    def save_benchmark_results(self, results: Dict[str, Any], filepath: str):
        """Save benchmark results to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Benchmark results saved to {filepath}")


def create_performance_dashboard(monitor: PerformanceMonitor, 
                               output_dir: str) -> str:
    """Create a simple HTML performance dashboard."""
    report = monitor.generate_performance_report()
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Connectome GNN Performance Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .metric {{ margin: 10px 0; padding: 10px; background: #f5f5f5; border-radius: 5px; }}
            .header {{ background: #007bff; color: white; padding: 15px; border-radius: 5px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Connectome GNN Performance Dashboard</h1>
            <p>Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <h2>System Metrics</h2>
        <div class="metric">
            <strong>CPU Usage:</strong> {report['system_metrics'].get('cpu_percent', 0):.1f}%<br>
            <strong>Memory Usage:</strong> {report['system_metrics'].get('memory_mb', 0):.1f} MB<br>
            <strong>GPU Memory:</strong> {report['system_metrics'].get('gpu_memory_allocated_mb', 0):.1f} MB
        </div>
        
        <h2>Latency Percentiles</h2>
        <table>
            <tr><th>Percentile</th><th>Latency (seconds)</th></tr>
            <tr><td>50th</td><td>{report['latency_percentiles'].get('p50', 0):.4f}</td></tr>
            <tr><td>95th</td><td>{report['latency_percentiles'].get('p95', 0):.4f}</td></tr>
            <tr><td>99th</td><td>{report['latency_percentiles'].get('p99', 0):.4f}</td></tr>
        </table>
        
        <h2>Operation Profiles</h2>
        <table>
            <tr><th>Operation</th><th>Calls</th><th>Avg Time</th><th>Total Time</th></tr>
    """
    
    for op_name, profile in report['operation_profiles'].items():
        html_content += f"""
            <tr>
                <td>{op_name}</td>
                <td>{profile.get('total_calls', 0)}</td>
                <td>{profile.get('avg_execution_time', 0):.4f}s</td>
                <td>{profile.get('total_execution_time', 0):.4f}s</td>
            </tr>
        """
    
    html_content += """
        </table>
    </body>
    </html>
    """
    
    output_path = Path(output_dir) / "performance_dashboard.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    return str(output_path)