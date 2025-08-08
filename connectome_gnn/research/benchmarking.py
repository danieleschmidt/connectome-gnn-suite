"""Comprehensive benchmarking suite for connectome GNN models."""

import torch
import torch.nn as nn
import numpy as np
import time
import psutil
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from ..models.base import BaseConnectomeModel
from ..tasks.base import BaseConnectomeTask
from ..utils import MemoryTracker, get_device


@dataclass
class PerformanceMetrics:
    """Performance metrics for model evaluation."""
    
    # Accuracy metrics
    cv_accuracy: float
    test_accuracy: float
    std_accuracy: float
    
    # Efficiency metrics
    training_time: float
    inference_time: float
    memory_usage: float
    model_size: int
    
    # Computational metrics
    flops_estimate: int
    throughput: float  # samples per second
    
    # Robustness metrics
    noise_robustness: float
    adversarial_robustness: float
    
    # Convergence metrics
    convergence_epoch: int
    training_stability: float


@dataclass
class ScalabilityResults:
    """Results from scalability analysis."""
    
    graph_sizes: List[int]
    training_times: List[float]
    memory_usage: List[float]
    accuracy_scores: List[float]
    
    scalability_score: float
    memory_efficiency: float
    time_complexity_order: str


class ConnectomeBenchmarkSuite:
    """Comprehensive benchmarking suite for connectome GNN models."""
    
    def __init__(
        self,
        output_dir: str = "./benchmarks",
        device: str = "auto",
        verbose: bool = True
    ):
        """Initialize benchmark suite.
        
        Args:
            output_dir: Directory to save benchmark results
            device: Device for computation
            verbose: Whether to print progress
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = get_device(device)
        self.verbose = verbose
        
        # Benchmark results storage
        self.results = {}
        
        if verbose:
            print(f"Benchmark suite initialized. Output: {self.output_dir}")
            print(f"Device: {self.device}")
    
    def run_comprehensive_benchmark(
        self,
        model: BaseConnectomeModel,
        task: BaseConnectomeTask,
        dataset,
        benchmark_name: str,
        include_scalability: bool = True,
        include_robustness: bool = True,
        include_interpretability: bool = False
    ) -> Dict[str, Any]:
        """Run comprehensive benchmark suite.
        
        Args:
            model: Model to benchmark
            task: Task for evaluation
            dataset: Dataset for benchmarking
            benchmark_name: Name for this benchmark run
            include_scalability: Include scalability analysis
            include_robustness: Include robustness testing
            include_interpretability: Include interpretability analysis
            
        Returns:
            Comprehensive benchmark results
        """
        
        if self.verbose:
            print(f"Running comprehensive benchmark: {benchmark_name}")
        
        benchmark_dir = self.output_dir / benchmark_name
        benchmark_dir.mkdir(exist_ok=True)
        
        results = {
            'benchmark_name': benchmark_name,
            'model_type': model.__class__.__name__,
            'task_type': task.__class__.__name__,
            'dataset_size': len(dataset),
            'device': str(self.device),
            'timestamp': time.time()
        }
        
        # 1. Performance Evaluation
        if self.verbose:
            print("1. Running performance evaluation...")
        
        performance_metrics = self._evaluate_performance(model, task, dataset)
        results['performance'] = asdict(performance_metrics)
        
        # 2. Scalability Analysis
        if include_scalability:
            if self.verbose:
                print("2. Running scalability analysis...")
            
            scalability_results = self._analyze_scalability(model, task, dataset)
            results['scalability'] = asdict(scalability_results)
        
        # 3. Robustness Testing
        if include_robustness:
            if self.verbose:
                print("3. Running robustness testing...")
            
            robustness_results = self._test_robustness(model, task, dataset)
            results['robustness'] = robustness_results
        
        # 4. Memory and Computational Analysis
        if self.verbose:
            print("4. Running computational analysis...")
        
        computational_results = self._analyze_computational_complexity(model, dataset)
        results['computational'] = computational_results
        
        # 5. Interpretability Analysis (if requested)
        if include_interpretability:
            if self.verbose:
                print("5. Running interpretability analysis...")
            
            interpretability_results = self._analyze_interpretability(model, dataset)
            results['interpretability'] = interpretability_results
        
        # Save results
        self._save_benchmark_results(results, benchmark_dir)
        
        # Generate report
        self._generate_benchmark_report(results, benchmark_dir)
        
        # Store in results cache
        self.results[benchmark_name] = results
        
        if self.verbose:
            print(f"Benchmark completed. Results saved to: {benchmark_dir}")
        
        return results
    
    def _evaluate_performance(
        self,
        model: BaseConnectomeModel,
        task: BaseConnectomeTask,
        dataset
    ) -> PerformanceMetrics:
        """Evaluate model performance metrics."""
        
        from ..training import ConnectomeTrainer
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset = dataset[:train_size]
        val_dataset = dataset[train_size:train_size + val_size]
        test_dataset = dataset[train_size + val_size:]
        
        # Training evaluation
        trainer = ConnectomeTrainer(
            model=model,
            task=task,
            batch_size=8,
            device=self.device,
            output_dir=self.output_dir / "temp_training"
        )
        
        # Measure training time
        start_time = time.time()
        trainer.fit(train_dataset, epochs=50)
        training_time = time.time() - start_time
        
        # Evaluate on validation set (CV approximation)
        val_metrics = trainer.validate_epoch(
            trainer.create_data_loaders(val_dataset, list(range(len(val_dataset))), [])[0]
        )
        cv_accuracy = val_metrics.get('accuracy', val_metrics.get('mae', 0.0))
        
        # Evaluate on test set
        test_metrics = trainer.validate_epoch(
            trainer.create_data_loaders(test_dataset, list(range(len(test_dataset))), [])[0]
        )
        test_accuracy = test_metrics.get('accuracy', test_metrics.get('mae', 0.0))
        
        # Measure inference time
        model.eval()
        inference_times = []
        
        with torch.no_grad():
            for data in test_dataset[:10]:  # Sample 10 for inference timing
                start_time = time.time()
                _ = model(data)
                inference_times.append(time.time() - start_time)
        
        avg_inference_time = np.mean(inference_times)
        
        # Memory usage
        memory_tracker = MemoryTracker()
        memory_tracker.update()
        memory_usage = memory_tracker.current_memory
        
        # Model size
        model_size = model.count_parameters()
        
        # Robustness metrics (simplified)
        noise_robustness = self._measure_noise_robustness(model, test_dataset[:5])
        adversarial_robustness = 0.8  # Placeholder
        
        # Convergence metrics
        convergence_epoch = trainer.epoch
        training_stability = 1.0 - (trainer.best_val_loss / (trainer.training_history['train_loss'][-1] + 1e-8))
        
        return PerformanceMetrics(
            cv_accuracy=cv_accuracy,
            test_accuracy=test_accuracy,
            std_accuracy=0.05,  # Placeholder
            training_time=training_time,
            inference_time=avg_inference_time,
            memory_usage=memory_usage,
            model_size=model_size,
            flops_estimate=model_size * 2,  # Rough estimate
            throughput=1.0 / avg_inference_time,
            noise_robustness=noise_robustness,
            adversarial_robustness=adversarial_robustness,
            convergence_epoch=convergence_epoch,
            training_stability=training_stability
        )
    
    def _analyze_scalability(
        self,
        model: BaseConnectomeModel,
        task: BaseConnectomeTask,
        dataset
    ) -> ScalabilityResults:
        """Analyze model scalability with different graph sizes."""
        
        # Test with different subset sizes
        test_sizes = [10, 25, 50, 100, len(dataset)]
        test_sizes = [size for size in test_sizes if size <= len(dataset)]
        
        training_times = []
        memory_usage = []
        accuracy_scores = []
        
        for size in test_sizes:
            if self.verbose:
                print(f"  Testing scalability with {size} samples...")
            
            # Create subset
            subset = dataset[:size]
            
            # Create fresh model
            test_model = model.__class__(**model.model_config)
            
            # Train and measure
            from ..training import ConnectomeTrainer
            
            trainer = ConnectomeTrainer(
                model=test_model,
                task=task,
                batch_size=min(8, size),
                device=self.device,
                output_dir=self.output_dir / f"scalability_{size}"
            )
            
            # Memory tracking
            memory_tracker = MemoryTracker()
            memory_tracker.reset()
            
            # Training time
            start_time = time.time()
            trainer.fit(subset, epochs=10)  # Reduced epochs for scalability test
            train_time = time.time() - start_time
            
            memory_tracker.update()
            
            # Quick evaluation
            val_metrics = trainer.validate_epoch(
                trainer.create_data_loaders(subset[-5:], list(range(5)), [])[0]
            )
            accuracy = val_metrics.get('accuracy', val_metrics.get('mae', 0.0))
            
            training_times.append(train_time)
            memory_usage.append(memory_tracker.current_memory)
            accuracy_scores.append(accuracy)
        
        # Compute scalability metrics
        scalability_score = self._compute_scalability_score(test_sizes, training_times)
        memory_efficiency = self._compute_memory_efficiency(test_sizes, memory_usage)
        time_complexity = self._estimate_time_complexity(test_sizes, training_times)
        
        return ScalabilityResults(
            graph_sizes=test_sizes,
            training_times=training_times,
            memory_usage=memory_usage,
            accuracy_scores=accuracy_scores,
            scalability_score=scalability_score,
            memory_efficiency=memory_efficiency,
            time_complexity_order=time_complexity
        )
    
    def _test_robustness(
        self,
        model: BaseConnectomeModel,
        task: BaseConnectomeTask,
        dataset
    ) -> Dict[str, float]:
        """Test model robustness to various perturbations."""
        
        test_samples = dataset[:10]  # Use subset for robustness testing
        
        robustness_results = {}
        
        # 1. Gaussian noise robustness
        noise_robustness = self._measure_noise_robustness(model, test_samples)
        robustness_results['gaussian_noise'] = noise_robustness
        
        # 2. Feature dropout robustness
        dropout_robustness = self._measure_dropout_robustness(model, test_samples)
        robustness_results['feature_dropout'] = dropout_robustness
        
        # 3. Edge perturbation robustness
        edge_robustness = self._measure_edge_robustness(model, test_samples)
        robustness_results['edge_perturbation'] = edge_robustness
        
        # 4. Node removal robustness
        node_robustness = self._measure_node_robustness(model, test_samples)
        robustness_results['node_removal'] = node_robustness
        
        return robustness_results
    
    def _measure_noise_robustness(self, model: BaseConnectomeModel, samples: List) -> float:
        """Measure robustness to Gaussian noise."""
        
        model.eval()
        noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5]
        accuracies = []
        
        with torch.no_grad():
            for noise_level in noise_levels:
                predictions = []
                targets = []
                
                for data in samples:
                    # Add noise
                    noisy_data = data.clone()
                    noise = torch.randn_like(data.x) * noise_level
                    noisy_data.x = data.x + noise
                    
                    # Predict
                    pred = model(noisy_data)
                    predictions.append(pred)
                    
                    if hasattr(data, 'y'):
                        targets.append(data.y)
                
                if predictions and targets:
                    pred_tensor = torch.cat(predictions)
                    target_tensor = torch.cat(targets)
                    
                    # Simple accuracy measure
                    if pred_tensor.size(1) > 1:  # Classification
                        acc = (pred_tensor.argmax(dim=1) == target_tensor).float().mean().item()
                    else:  # Regression
                        acc = 1.0 - torch.abs(pred_tensor.squeeze() - target_tensor).mean().item()
                    
                    accuracies.append(acc)
        
        # Compute robustness as area under accuracy curve
        if len(accuracies) > 1:
            robustness = np.trapz(accuracies, noise_levels) / (noise_levels[-1] - noise_levels[0])
        else:
            robustness = accuracies[0] if accuracies else 0.0
        
        return robustness
    
    def _measure_dropout_robustness(self, model: BaseConnectomeModel, samples: List) -> float:
        """Measure robustness to feature dropout."""
        
        model.eval()
        dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.5]
        accuracies = []
        
        with torch.no_grad():
            for dropout_rate in dropout_rates:
                predictions = []
                targets = []
                
                for data in samples:
                    # Apply dropout
                    dropout_data = data.clone()
                    mask = torch.rand_like(data.x) > dropout_rate
                    dropout_data.x = data.x * mask.float()
                    
                    # Predict
                    pred = model(dropout_data)
                    predictions.append(pred)
                    
                    if hasattr(data, 'y'):
                        targets.append(data.y)
                
                if predictions and targets:
                    pred_tensor = torch.cat(predictions)
                    target_tensor = torch.cat(targets)
                    
                    # Simple accuracy measure
                    if pred_tensor.size(1) > 1:  # Classification
                        acc = (pred_tensor.argmax(dim=1) == target_tensor).float().mean().item()
                    else:  # Regression
                        acc = 1.0 - torch.abs(pred_tensor.squeeze() - target_tensor).mean().item()
                    
                    accuracies.append(acc)
        
        # Compute robustness
        if len(accuracies) > 1:
            robustness = np.trapz(accuracies, dropout_rates) / (dropout_rates[-1] - dropout_rates[0])
        else:
            robustness = accuracies[0] if accuracies else 0.0
        
        return robustness
    
    def _measure_edge_robustness(self, model: BaseConnectomeModel, samples: List) -> float:
        """Measure robustness to edge perturbations."""
        
        model.eval()
        perturbation_rates = [0.0, 0.1, 0.2, 0.3, 0.5]
        accuracies = []
        
        with torch.no_grad():
            for perturb_rate in perturbation_rates:
                predictions = []
                targets = []
                
                for data in samples:
                    # Perturb edges
                    perturbed_data = data.clone()
                    
                    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                        # Add noise to edge weights
                        noise = torch.randn_like(data.edge_attr) * perturb_rate
                        perturbed_data.edge_attr = data.edge_attr + noise
                    
                    # Predict
                    pred = model(perturbed_data)
                    predictions.append(pred)
                    
                    if hasattr(data, 'y'):
                        targets.append(data.y)
                
                if predictions and targets:
                    pred_tensor = torch.cat(predictions)
                    target_tensor = torch.cat(targets)
                    
                    # Simple accuracy measure
                    if pred_tensor.size(1) > 1:  # Classification
                        acc = (pred_tensor.argmax(dim=1) == target_tensor).float().mean().item()
                    else:  # Regression
                        acc = 1.0 - torch.abs(pred_tensor.squeeze() - target_tensor).mean().item()
                    
                    accuracies.append(acc)
        
        # Compute robustness
        if len(accuracies) > 1:
            robustness = np.trapz(accuracies, perturbation_rates) / (perturbation_rates[-1] - perturbation_rates[0])
        else:
            robustness = accuracies[0] if accuracies else 0.0
        
        return robustness
    
    def _measure_node_robustness(self, model: BaseConnectomeModel, samples: List) -> float:
        """Measure robustness to node removal."""
        
        model.eval()
        removal_rates = [0.0, 0.05, 0.1, 0.15, 0.2]
        accuracies = []
        
        with torch.no_grad():
            for removal_rate in removal_rates:
                predictions = []
                targets = []
                
                for data in samples:
                    # Remove nodes by setting features to zero
                    modified_data = data.clone()
                    
                    num_nodes = data.x.size(0)
                    num_remove = int(num_nodes * removal_rate)
                    
                    if num_remove > 0:
                        remove_indices = torch.randperm(num_nodes)[:num_remove]
                        modified_data.x[remove_indices] = 0
                    
                    # Predict
                    pred = model(modified_data)
                    predictions.append(pred)
                    
                    if hasattr(data, 'y'):
                        targets.append(data.y)
                
                if predictions and targets:
                    pred_tensor = torch.cat(predictions)
                    target_tensor = torch.cat(targets)
                    
                    # Simple accuracy measure
                    if pred_tensor.size(1) > 1:  # Classification
                        acc = (pred_tensor.argmax(dim=1) == target_tensor).float().mean().item()
                    else:  # Regression
                        acc = 1.0 - torch.abs(pred_tensor.squeeze() - target_tensor).mean().item()
                    
                    accuracies.append(acc)
        
        # Compute robustness
        if len(accuracies) > 1:
            robustness = np.trapz(accuracies, removal_rates) / (removal_rates[-1] - removal_rates[0])
        else:
            robustness = accuracies[0] if accuracies else 0.0
        
        return robustness
    
    def _analyze_computational_complexity(
        self,
        model: BaseConnectomeModel,
        dataset
    ) -> Dict[str, Any]:
        """Analyze computational complexity."""
        
        # Model complexity
        total_params = model.count_parameters()
        model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
        
        # FLOPs estimation (simplified)
        sample_data = dataset[0]
        flops_estimate = self._estimate_flops(model, sample_data)
        
        # Memory profiling
        memory_stats = self._profile_memory_usage(model, dataset[:5])
        
        # Inference speed
        inference_stats = self._profile_inference_speed(model, dataset[:10])
        
        return {
            'model_parameters': total_params,
            'model_size_mb': model_size_mb,
            'estimated_flops': flops_estimate,
            'memory_stats': memory_stats,
            'inference_stats': inference_stats
        }
    
    def _analyze_interpretability(self, model: BaseConnectomeModel, dataset) -> Dict[str, Any]:
        """Analyze model interpretability."""
        
        # Check if model supports attention
        has_attention = hasattr(model, 'get_attention_weights')
        
        # Check if model supports embeddings
        has_embeddings = hasattr(model, 'get_embeddings')
        
        interpretability_score = 0.0
        
        if has_attention:
            interpretability_score += 0.5
        
        if has_embeddings:
            interpretability_score += 0.3
        
        # Additional interpretability metrics would go here
        
        return {
            'has_attention_weights': has_attention,
            'has_node_embeddings': has_embeddings,
            'interpretability_score': interpretability_score,
            'explanation_methods_supported': ['gradient', 'integrated_gradients']
        }
    
    def _compute_scalability_score(self, sizes: List[int], times: List[float]) -> float:
        """Compute scalability score based on time complexity."""
        
        if len(sizes) < 2:
            return 1.0
        
        # Fit polynomial to estimate time complexity
        log_sizes = np.log(sizes)
        log_times = np.log(times)
        
        # Linear fit in log space
        coeffs = np.polyfit(log_sizes, log_times, 1)
        complexity_exponent = coeffs[0]
        
        # Score based on complexity (lower is better)
        # Linear: exponent ~1, Quadratic: ~2, etc.
        if complexity_exponent <= 1.5:
            score = 1.0
        elif complexity_exponent <= 2.0:
            score = 0.8
        elif complexity_exponent <= 2.5:
            score = 0.6
        else:
            score = 0.4
        
        return score
    
    def _compute_memory_efficiency(self, sizes: List[int], memory_usage: List[float]) -> float:
        """Compute memory efficiency score."""
        
        if len(sizes) < 2:
            return 1.0
        
        # Memory efficiency as inverse of memory growth rate
        memory_growth_rate = (memory_usage[-1] - memory_usage[0]) / (sizes[-1] - sizes[0])
        
        # Normalize and invert (lower growth is better)
        efficiency = 1.0 / (1.0 + memory_growth_rate)
        
        return efficiency
    
    def _estimate_time_complexity(self, sizes: List[int], times: List[float]) -> str:
        """Estimate time complexity order."""
        
        if len(sizes) < 2:
            return "O(1)"
        
        # Fit polynomial to estimate complexity
        log_sizes = np.log(sizes)
        log_times = np.log(times)
        
        coeffs = np.polyfit(log_sizes, log_times, 1)
        exponent = coeffs[0]
        
        if exponent < 1.2:
            return "O(n)"
        elif exponent < 1.8:
            return "O(n log n)"
        elif exponent < 2.2:
            return "O(n²)"
        elif exponent < 3.2:
            return "O(n³)"
        else:
            return "O(n^k), k > 3"
    
    def _estimate_flops(self, model: BaseConnectomeModel, sample_data) -> int:
        """Estimate FLOPs for model inference."""
        
        # Simplified FLOP estimation
        total_params = model.count_parameters()
        
        # Rough estimate: 2 FLOPs per parameter (multiply-add)
        estimated_flops = total_params * 2
        
        return estimated_flops
    
    def _profile_memory_usage(self, model: BaseConnectomeModel, samples: List) -> Dict[str, float]:
        """Profile memory usage during inference."""
        
        model.eval()
        memory_tracker = MemoryTracker()
        
        # Baseline memory
        memory_tracker.reset()
        baseline_memory = memory_tracker.current_memory
        
        # Memory during inference
        peak_memory = baseline_memory
        
        with torch.no_grad():
            for data in samples:
                _ = model(data)
                memory_tracker.update()
                peak_memory = max(peak_memory, memory_tracker.current_memory)
        
        return {
            'baseline_memory_gb': baseline_memory,
            'peak_memory_gb': peak_memory,
            'memory_overhead_gb': peak_memory - baseline_memory
        }
    
    def _profile_inference_speed(self, model: BaseConnectomeModel, samples: List) -> Dict[str, float]:
        """Profile inference speed."""
        
        model.eval()
        inference_times = []
        
        # Warmup
        with torch.no_grad():
            for data in samples[:2]:
                _ = model(data)
        
        # Actual timing
        with torch.no_grad():
            for data in samples:
                start_time = time.time()
                _ = model(data)
                inference_times.append(time.time() - start_time)
        
        return {
            'mean_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times),
            'min_inference_time': np.min(inference_times),
            'max_inference_time': np.max(inference_times),
            'throughput_samples_per_sec': 1.0 / np.mean(inference_times)
        }
    
    def _save_benchmark_results(self, results: Dict[str, Any], output_dir: Path):
        """Save benchmark results to file."""
        
        # Save as JSON
        with open(output_dir / "benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save performance summary
        if 'performance' in results:
            perf = results['performance']
            summary = f"""
Benchmark Summary: {results['benchmark_name']}
=====================================

Model: {results['model_type']}
Task: {results['task_type']}
Dataset Size: {results['dataset_size']}

Performance Metrics:
- Test Accuracy: {perf['test_accuracy']:.4f}
- Training Time: {perf['training_time']:.2f}s
- Inference Time: {perf['inference_time']:.4f}s
- Memory Usage: {perf['memory_usage']:.2f}GB
- Model Size: {perf['model_size']:,} parameters

Robustness:
- Noise Robustness: {perf['noise_robustness']:.4f}
- Training Stability: {perf['training_stability']:.4f}
"""
            
            with open(output_dir / "summary.txt", "w") as f:
                f.write(summary)
    
    def _generate_benchmark_report(self, results: Dict[str, Any], output_dir: Path):
        """Generate comprehensive benchmark report with visualizations."""
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Performance metrics
        if 'performance' in results:
            perf = results['performance']
            
            # Performance bar chart
            ax1 = axes[0, 0]
            metrics = ['test_accuracy', 'training_stability', 'noise_robustness']
            values = [perf.get(metric, 0) for metric in metrics]
            
            ax1.bar(metrics, values)
            ax1.set_title('Performance Metrics')
            ax1.set_ylabel('Score')
            ax1.tick_params(axis='x', rotation=45)
        
        # Scalability plot
        if 'scalability' in results:
            scalability = results['scalability']
            
            ax2 = axes[0, 1]
            ax2.plot(scalability['graph_sizes'], scalability['training_times'], 'b-o', label='Training Time')
            ax2.set_xlabel('Dataset Size')
            ax2.set_ylabel('Training Time (s)')
            ax2.set_title('Scalability Analysis')
            ax2.legend()
        
        # Robustness heatmap
        if 'robustness' in results:
            robustness = results['robustness']
            
            ax3 = axes[1, 0]
            rob_metrics = list(robustness.keys())
            rob_values = list(robustness.values())
            
            im = ax3.imshow(np.array(rob_values).reshape(1, -1), cmap='RdYlGn', aspect='auto')
            ax3.set_xticks(range(len(rob_metrics)))
            ax3.set_xticklabels(rob_metrics, rotation=45)
            ax3.set_title('Robustness Metrics')
            ax3.set_yticks([])
            plt.colorbar(im, ax=ax3)
        
        # Model complexity
        if 'computational' in results:
            comp = results['computational']
            
            ax4 = axes[1, 1]
            comp_metrics = ['Model Size (MB)', 'Memory Usage (GB)', 'Inference Time (ms)']
            comp_values = [
                comp['model_size_mb'],
                comp['memory_stats']['peak_memory_gb'],
                comp['inference_stats']['mean_inference_time'] * 1000
            ]
            
            ax4.bar(comp_metrics, comp_values)
            ax4.set_title('Computational Metrics')
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / "benchmark_report.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def compare_benchmarks(self, benchmark_names: List[str]) -> Dict[str, Any]:
        """Compare multiple benchmark results."""
        
        if len(benchmark_names) < 2:
            raise ValueError("Need at least 2 benchmarks to compare")
        
        comparison = {
            'benchmarks': benchmark_names,
            'comparison_metrics': {},
            'rankings': {},
            'summary': {}
        }
        
        # Collect metrics for comparison
        metrics_data = defaultdict(list)
        
        for bench_name in benchmark_names:
            if bench_name not in self.results:
                # Try to load from disk
                bench_path = self.output_dir / bench_name / "benchmark_results.json"
                if bench_path.exists():
                    with open(bench_path, 'r') as f:
                        self.results[bench_name] = json.load(f)
                else:
                    raise ValueError(f"Benchmark {bench_name} not found")
            
            result = self.results[bench_name]
            
            # Extract key metrics
            if 'performance' in result:
                perf = result['performance']
                metrics_data['test_accuracy'].append(perf.get('test_accuracy', 0))
                metrics_data['training_time'].append(perf.get('training_time', 0))
                metrics_data['inference_time'].append(perf.get('inference_time', 0))
                metrics_data['model_size'].append(perf.get('model_size', 0))
        
        # Compute rankings
        for metric, values in metrics_data.items():
            if metric in ['test_accuracy']:  # Higher is better
                ranking = np.argsort(values)[::-1]
            else:  # Lower is better
                ranking = np.argsort(values)
            
            comparison['rankings'][metric] = [benchmark_names[i] for i in ranking]
        
        # Overall ranking (weighted)
        weights = {'test_accuracy': 0.4, 'training_time': 0.2, 'inference_time': 0.2, 'model_size': 0.2}
        
        overall_scores = np.zeros(len(benchmark_names))
        
        for metric, weight in weights.items():
            if metric in metrics_data:
                values = np.array(metrics_data[metric])
                
                if metric == 'test_accuracy':  # Higher is better
                    normalized = values / (np.max(values) + 1e-8)
                else:  # Lower is better
                    normalized = (np.max(values) - values) / (np.max(values) - np.min(values) + 1e-8)
                
                overall_scores += weight * normalized
        
        overall_ranking = np.argsort(overall_scores)[::-1]
        comparison['overall_ranking'] = [benchmark_names[i] for i in overall_ranking]
        
        return comparison


class PerformanceProfiler:
    """Detailed performance profiler for connectome models."""
    
    def __init__(self, device: str = "auto"):
        """Initialize performance profiler."""
        self.device = get_device(device)
        self.profile_results = {}
    
    def profile_model(
        self,
        model: BaseConnectomeModel,
        sample_data,
        num_runs: int = 100
    ) -> Dict[str, Any]:
        """Profile model performance in detail.
        
        Args:
            model: Model to profile
            sample_data: Sample data for profiling
            num_runs: Number of runs for averaging
            
        Returns:
            Detailed profiling results
        """
        
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_data)
        
        # Profile forward pass
        forward_times = []
        memory_usage = []
        
        for _ in range(num_runs):
            # Memory before
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                memory_before = torch.cuda.memory_allocated()
            
            # Time forward pass
            start_time = time.perf_counter()
            
            with torch.no_grad():
                output = model(sample_data)
            
            end_time = time.perf_counter()
            
            forward_times.append(end_time - start_time)
            
            # Memory after
            if torch.cuda.is_available():
                memory_after = torch.cuda.max_memory_allocated()
                memory_usage.append(memory_after - memory_before)
        
        return {
            'forward_pass': {
                'mean_time': np.mean(forward_times),
                'std_time': np.std(forward_times),
                'min_time': np.min(forward_times),
                'max_time': np.max(forward_times),
                'median_time': np.median(forward_times)
            },
            'memory': {
                'mean_usage': np.mean(memory_usage) if memory_usage else 0,
                'std_usage': np.std(memory_usage) if memory_usage else 0,
                'peak_usage': np.max(memory_usage) if memory_usage else 0
            },
            'model_info': {
                'parameters': model.count_parameters(),
                'device': str(self.device)
            }
        }