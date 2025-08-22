"""Advanced Benchmarking Suite for Connectome-GNN Research.

Comprehensive benchmarking framework including novel evaluation metrics,
statistical validation, cross-domain evaluation, and research publication tools.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from typing import Optional, Dict, Any, Tuple, List, Union, Callable
import math
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import time
from collections import defaultdict
import json

try:
    from ..models.base import BaseConnectomeModel
    from .quantum_gnn import QuantumEnhancedGNN, QuantumMetrics
    from .neuromorphic_gnn import NeuromorphicGNN, NeuromorphicMetrics
    from .quantum_neuromorphic_fusion import QuantumNeuromorphicGNN, QuantumNeuromorphicMetrics
    from .meta_learning_gnn import MetaLearningTrainer, FewShotConnectomeDataset
    from ..optimization.edge_computing import EdgeOptimizedGNN, EdgeInferenceEngine
except ImportError:
    # Fallback for standalone operation
    class BaseConnectomeModel(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__()
    class QuantumEnhancedGNN(BaseConnectomeModel): pass
    class NeuromorphicGNN(BaseConnectomeModel): pass
    class QuantumNeuromorphicGNN(BaseConnectomeModel): pass
    class EdgeOptimizedGNN(BaseConnectomeModel): pass
    class QuantumMetrics: pass
    class NeuromorphicMetrics: pass
    class QuantumNeuromorphicMetrics: pass
    class MetaLearningTrainer: pass
    class FewShotConnectomeDataset: pass
    class EdgeInferenceEngine: pass


class AdvancedMetrics:
    """Advanced evaluation metrics for connectome analysis."""
    
    @staticmethod
    def connectome_similarity_index(pred_graphs: List[torch.Tensor], 
                                   true_graphs: List[torch.Tensor]) -> Dict[str, float]:
        """Measure similarity between predicted and true connectomes."""
        similarities = []
        
        for pred, true in zip(pred_graphs, true_graphs):
            # Normalize adjacency matrices
            pred_norm = F.normalize(pred, p=1, dim=1)
            true_norm = F.normalize(true, p=1, dim=1)
            
            # Compute similarity metrics
            cosine_sim = F.cosine_similarity(pred_norm.flatten(), true_norm.flatten(), dim=0)
            frobenius_sim = 1.0 / (1.0 + torch.norm(pred_norm - true_norm, p='fro'))
            
            # Topological similarity (simplified)
            pred_triu = torch.triu(pred_norm, diagonal=1)
            true_triu = torch.triu(true_norm, diagonal=1)
            topological_sim = F.cosine_similarity(pred_triu.flatten(), true_triu.flatten(), dim=0)
            
            similarities.append({
                'cosine': cosine_sim.item(),
                'frobenius': frobenius_sim.item(),
                'topological': topological_sim.item()
            })
        
        # Aggregate results
        return {
            'mean_cosine_similarity': np.mean([s['cosine'] for s in similarities]),
            'mean_frobenius_similarity': np.mean([s['frobenius'] for s in similarities]),
            'mean_topological_similarity': np.mean([s['topological'] for s in similarities]),
            'std_cosine_similarity': np.std([s['cosine'] for s in similarities]),
            'std_frobenius_similarity': np.std([s['frobenius'] for s in similarities]),
            'std_topological_similarity': np.std([s['topological'] for s in similarities])
        }
    
    @staticmethod
    def neurological_validity_score(predictions: torch.Tensor, 
                                   brain_atlas: Dict[str, Any]) -> Dict[str, float]:
        """Assess neurological validity of predictions."""
        # Simplified neurological validity assessment
        scores = {}
        
        # Hemispheric symmetry (brain networks should be somewhat symmetric)
        if 'hemisphere_mapping' in brain_atlas:
            left_regions = brain_atlas['hemisphere_mapping']['left']
            right_regions = brain_atlas['hemisphere_mapping']['right']
            
            left_activity = predictions[:, left_regions].mean(dim=1)
            right_activity = predictions[:, right_regions].mean(dim=1)
            symmetry_score = 1.0 - torch.abs(left_activity - right_activity).mean().item()
            scores['hemispheric_symmetry'] = max(0.0, symmetry_score)
        
        # Anatomical plausibility (closer regions should have higher connectivity)
        if 'distance_matrix' in brain_atlas:
            distance_matrix = torch.tensor(brain_atlas['distance_matrix'])
            
            # Inverse relationship between distance and connectivity
            expected_connectivity = 1.0 / (1.0 + distance_matrix)
            predicted_connectivity = torch.softmax(predictions, dim=1)
            
            plausibility = F.cosine_similarity(
                expected_connectivity.flatten(), 
                predicted_connectivity.mean(dim=0).flatten(), 
                dim=0
            ).item()
            scores['anatomical_plausibility'] = max(0.0, plausibility)
        
        # Default-mode network coherence (simplified)
        if 'default_mode_regions' in brain_atlas:
            dmn_regions = brain_atlas['default_mode_regions']
            dmn_activity = predictions[:, dmn_regions]
            
            # DMN regions should be coherent
            coherence = torch.corrcoef(dmn_activity.T).mean().item()
            scores['dmn_coherence'] = max(0.0, coherence)
        
        # Overall validity score
        scores['overall_validity'] = np.mean(list(scores.values())) if scores else 0.0
        
        return scores
    
    @staticmethod
    def cognitive_prediction_accuracy(model_predictions: torch.Tensor,
                                    cognitive_scores: torch.Tensor,
                                    cognitive_domains: List[str]) -> Dict[str, float]:
        """Evaluate cognitive score prediction accuracy."""
        results = {}
        
        for i, domain in enumerate(cognitive_domains):
            if i < cognitive_scores.size(1):
                pred_scores = model_predictions[:, i] if model_predictions.size(1) > i else model_predictions[:, 0]
                true_scores = cognitive_scores[:, i]
                
                # Correlation
                correlation = torch.corrcoef(torch.stack([pred_scores, true_scores]))[0, 1].item()
                
                # Mean absolute error
                mae = torch.abs(pred_scores - true_scores).mean().item()
                
                # R-squared
                ss_res = torch.sum((true_scores - pred_scores) ** 2).item()
                ss_tot = torch.sum((true_scores - true_scores.mean()) ** 2).item()
                r2 = 1 - (ss_res / (ss_tot + 1e-8))
                
                results[f'{domain}_correlation'] = correlation
                results[f'{domain}_mae'] = mae
                results[f'{domain}_r2'] = max(0.0, r2)
        
        # Overall cognitive prediction score
        correlations = [v for k, v in results.items() if k.endswith('_correlation')]
        results['overall_cognitive_accuracy'] = np.mean(correlations) if correlations else 0.0
        
        return results
    
    @staticmethod
    def network_topology_preservation(original_graphs: List[torch.Tensor],
                                    reconstructed_graphs: List[torch.Tensor]) -> Dict[str, float]:
        """Measure how well network topology is preserved."""
        preservation_scores = []
        
        for orig, recon in zip(original_graphs, reconstructed_graphs):
            # Convert to binary adjacency matrices
            orig_binary = (orig > orig.mean()).float()
            recon_binary = (recon > recon.mean()).float()
            
            # Calculate graph metrics
            orig_degree = orig_binary.sum(dim=1)
            recon_degree = recon_binary.sum(dim=1)
            
            # Degree distribution preservation
            degree_corr = torch.corrcoef(torch.stack([orig_degree, recon_degree]))[0, 1].item()
            
            # Clustering coefficient preservation (simplified)
            orig_clustering = torch.diag(torch.mm(torch.mm(orig_binary, orig_binary), orig_binary)) / (orig_degree * (orig_degree - 1) + 1e-8)
            recon_clustering = torch.diag(torch.mm(torch.mm(recon_binary, recon_binary), recon_binary)) / (recon_degree * (recon_degree - 1) + 1e-8)
            clustering_corr = torch.corrcoef(torch.stack([orig_clustering, recon_clustering]))[0, 1].item()
            
            # Edge overlap
            edge_overlap = torch.sum(orig_binary * recon_binary) / torch.sum(torch.clamp(orig_binary + recon_binary, max=1.0))
            
            preservation_scores.append({
                'degree_preservation': degree_corr,
                'clustering_preservation': clustering_corr,
                'edge_overlap': edge_overlap.item()
            })
        
        return {
            'mean_degree_preservation': np.mean([s['degree_preservation'] for s in preservation_scores]),
            'mean_clustering_preservation': np.mean([s['clustering_preservation'] for s in preservation_scores]),
            'mean_edge_overlap': np.mean([s['edge_overlap'] for s in preservation_scores]),
            'std_degree_preservation': np.std([s['degree_preservation'] for s in preservation_scores]),
            'std_clustering_preservation': np.std([s['clustering_preservation'] for s in preservation_scores]),
            'std_edge_overlap': np.std([s['edge_overlap'] for s in preservation_scores])
        }


class StatisticalValidator:
    """Statistical validation framework for research claims."""
    
    def __init__(self, alpha: float = 0.05, num_bootstrap: int = 1000):
        self.alpha = alpha
        self.num_bootstrap = num_bootstrap
    
    def paired_t_test(self, model_a_scores: np.ndarray, model_b_scores: np.ndarray) -> Dict[str, Any]:
        """Perform paired t-test for model comparison."""
        t_stat, p_value = stats.ttest_rel(model_a_scores, model_b_scores)
        
        # Effect size (Cohen's d)
        diff = model_a_scores - model_b_scores
        pooled_std = np.sqrt((np.var(model_a_scores, ddof=1) + np.var(model_b_scores, ddof=1)) / 2)
        cohens_d = np.mean(diff) / pooled_std
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'cohens_d': cohens_d,
            'effect_size_interpretation': self._interpret_effect_size(abs(cohens_d)),
            'confidence_interval': stats.t.interval(1 - self.alpha, len(diff) - 1, 
                                                   np.mean(diff), stats.sem(diff))
        }
    
    def bootstrap_confidence_interval(self, scores: np.ndarray, 
                                    statistic: Callable = np.mean) -> Dict[str, float]:
        """Calculate bootstrap confidence intervals."""
        bootstrap_stats = []
        
        for _ in range(self.num_bootstrap):
            bootstrap_sample = np.random.choice(scores, size=len(scores), replace=True)
            bootstrap_stats.append(statistic(bootstrap_sample))
        
        ci_lower = np.percentile(bootstrap_stats, (self.alpha / 2) * 100)
        ci_upper = np.percentile(bootstrap_stats, (1 - self.alpha / 2) * 100)
        
        return {
            'statistic': statistic(scores),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'bootstrap_std': np.std(bootstrap_stats),
            'bootstrap_samples': len(bootstrap_stats)
        }
    
    def cross_validation_analysis(self, model: nn.Module, dataset: List[Data], 
                                labels: torch.Tensor, k_folds: int = 5) -> Dict[str, Any]:
        """Comprehensive cross-validation analysis."""
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        fold_scores = []
        fold_predictions = []
        fold_true_labels = []
        
        # Convert labels to numpy for stratification
        labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(dataset, labels_np)):
            # Create fold datasets
            train_data = [dataset[i] for i in train_idx]
            val_data = [dataset[i] for i in val_idx]
            val_labels = labels[val_idx]
            
            # Train model (simplified - would need actual training loop)
            model.train()
            # ... training code would go here ...
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                fold_preds = []
                for data in val_data:
                    pred = model(data)
                    fold_preds.append(pred)
                
                fold_preds = torch.stack(fold_preds).squeeze()
                
                # Calculate metrics
                if len(torch.unique(val_labels)) > 2:  # Regression
                    score = -F.mse_loss(fold_preds, val_labels).item()
                else:  # Classification
                    probs = torch.softmax(fold_preds, dim=1) if fold_preds.dim() > 1 else torch.sigmoid(fold_preds)
                    score = roc_auc_score(val_labels.numpy(), probs.numpy())
                
                fold_scores.append(score)
                fold_predictions.append(fold_preds)
                fold_true_labels.append(val_labels)
        
        # Statistical analysis of fold scores
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        # Confidence interval for mean performance
        ci = stats.t.interval(1 - self.alpha, len(fold_scores) - 1, 
                             mean_score, stats.sem(fold_scores))
        
        return {
            'fold_scores': fold_scores,
            'mean_score': mean_score,
            'std_score': std_score,
            'confidence_interval': ci,
            'coefficient_of_variation': std_score / abs(mean_score) if mean_score != 0 else float('inf'),
            'stability_score': 1.0 / (1.0 + std_score)  # Higher = more stable
        }
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        if cohens_d < 0.2:
            return "negligible"
        elif cohens_d < 0.5:
            return "small"
        elif cohens_d < 0.8:
            return "medium"
        else:
            return "large"


class ComprehensiveBenchmark:
    """Comprehensive benchmarking suite for connectome-GNN research."""
    
    def __init__(self, models: Dict[str, nn.Module], datasets: Dict[str, List[Data]]):
        self.models = models
        self.datasets = datasets
        self.results = defaultdict(dict)
        self.statistical_validator = StatisticalValidator()
    
    def run_performance_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmarks."""
        print("Running performance benchmarks...")
        
        for model_name, model in self.models.items():
            model_results = {}
            
            for dataset_name, dataset in self.datasets.items():
                print(f"  Evaluating {model_name} on {dataset_name}...")
                
                # Basic performance metrics
                start_time = time.time()
                
                model.eval()
                with torch.no_grad():
                    predictions = []
                    for data in dataset[:100]:  # Sample for speed
                        pred = model(data)
                        predictions.append(pred)
                
                inference_time = time.time() - start_time
                
                # Store results
                model_results[dataset_name] = {
                    'inference_time_ms': (inference_time / len(predictions)) * 1000,
                    'throughput_samples_per_sec': len(predictions) / inference_time,
                    'memory_usage_mb': self._estimate_memory_usage(model),
                    'model_parameters': sum(p.numel() for p in model.parameters()),
                    'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
                }
            
            self.results['performance'][model_name] = model_results
        
        return self.results['performance']
    
    def run_accuracy_benchmark(self, test_labels: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Run accuracy benchmarks across datasets."""
        print("Running accuracy benchmarks...")
        
        for model_name, model in self.models.items():
            model_results = {}
            
            for dataset_name, dataset in self.datasets.items():
                if dataset_name in test_labels:
                    labels = test_labels[dataset_name]
                    
                    # Get predictions
                    model.eval()
                    with torch.no_grad():
                        predictions = []
                        for data in dataset:
                            pred = model(data)
                            predictions.append(pred)
                    
                    predictions = torch.stack(predictions).squeeze()
                    
                    # Calculate metrics
                    if len(torch.unique(labels)) > 2:  # Regression
                        mse = F.mse_loss(predictions, labels).item()
                        mae = F.l1_loss(predictions, labels).item()
                        r2 = 1 - F.mse_loss(predictions, labels) / torch.var(labels)
                        
                        metrics = {
                            'mse': mse,
                            'mae': mae,
                            'rmse': math.sqrt(mse),
                            'r2': r2.item()
                        }
                    else:  # Classification
                        probs = torch.softmax(predictions, dim=1) if predictions.dim() > 1 else torch.sigmoid(predictions)
                        
                        # Convert to binary predictions
                        binary_preds = (probs > 0.5).float()
                        accuracy = (binary_preds == labels).float().mean().item()
                        
                        # AUC if probabilities available
                        try:
                            auc_score = roc_auc_score(labels.numpy(), probs.numpy())
                        except:
                            auc_score = None
                        
                        metrics = {
                            'accuracy': accuracy,
                            'auc': auc_score
                        }
                    
                    model_results[dataset_name] = metrics
            
            self.results['accuracy'][model_name] = model_results
        
        return self.results['accuracy']
    
    def run_specialized_benchmarks(self) -> Dict[str, Any]:
        """Run specialized benchmarks for different model types."""
        print("Running specialized benchmarks...")
        
        for model_name, model in self.models.items():
            specialized_results = {}
            
            # Quantum-specific metrics
            if isinstance(model, QuantumEnhancedGNN):
                for dataset_name, dataset in self.datasets.items():
                    if dataset:
                        quantum_fidelity = QuantumMetrics.quantum_fidelity(
                            model(dataset[0]), torch.randn_like(model(dataset[0]))
                        )
                        entanglement = QuantumMetrics.entanglement_measure(model, dataset[0])
                        
                        specialized_results[f'{dataset_name}_quantum'] = {
                            'quantum_fidelity': quantum_fidelity,
                            'entanglement_measure': entanglement
                        }
            
            # Neuromorphic-specific metrics
            elif isinstance(model, NeuromorphicGNN):
                for dataset_name, dataset in self.datasets.items():
                    if dataset:
                        spike_stats = model.get_spike_statistics()
                        temporal_precision = NeuromorphicMetrics.temporal_precision(model)
                        
                        specialized_results[f'{dataset_name}_neuromorphic'] = {
                            'spike_rate': spike_stats.get('spike_rate', 0),
                            'temporal_precision': temporal_precision,
                            'spike_sparsity': spike_stats.get('spike_sparsity', 0)
                        }
            
            # Quantum-neuromorphic fusion metrics
            elif isinstance(model, QuantumNeuromorphicGNN):
                for dataset_name, dataset in self.datasets.items():
                    if dataset:
                        efficiency = QuantumNeuromorphicMetrics.quantum_neuromorphic_efficiency(model, dataset[0])
                        temporal_dynamics = QuantumNeuromorphicMetrics.neuromorphic_temporal_dynamics(model, dataset[0])
                        entanglement = QuantumNeuromorphicMetrics.quantum_entanglement_measure(model, dataset[0])
                        
                        specialized_results[f'{dataset_name}_hybrid'] = {
                            'efficiency': efficiency,
                            'temporal_dynamics': temporal_dynamics,
                            'quantum_entanglement': entanglement
                        }
            
            # Edge computing metrics
            elif isinstance(model, EdgeOptimizedGNN):
                for dataset_name, dataset in self.datasets.items():
                    if dataset:
                        # Create edge inference engine
                        engine = EdgeInferenceEngine(model)
                        _, edge_metrics = engine.predict(dataset[0])
                        
                        specialized_results[f'{dataset_name}_edge'] = edge_metrics
            
            if specialized_results:
                self.results['specialized'][model_name] = specialized_results
        
        return self.results['specialized']
    
    def run_statistical_validation(self) -> Dict[str, Any]:
        """Run statistical validation of results."""
        print("Running statistical validation...")
        
        validation_results = {}
        
        # Compare models pairwise
        model_names = list(self.models.keys())
        for i, model_a in enumerate(model_names):
            for model_b in model_names[i+1:]:
                comparison_key = f"{model_a}_vs_{model_b}"
                
                # Extract performance scores for comparison
                if 'accuracy' in self.results and model_a in self.results['accuracy'] and model_b in self.results['accuracy']:
                    scores_a = []
                    scores_b = []
                    
                    for dataset_name in self.datasets.keys():
                        if (dataset_name in self.results['accuracy'][model_a] and 
                            dataset_name in self.results['accuracy'][model_b]):
                            
                            # Use primary metric (accuracy for classification, r2 for regression)
                            metrics_a = self.results['accuracy'][model_a][dataset_name]
                            metrics_b = self.results['accuracy'][model_b][dataset_name]
                            
                            score_a = metrics_a.get('accuracy', metrics_a.get('r2', 0))
                            score_b = metrics_b.get('accuracy', metrics_b.get('r2', 0))
                            
                            scores_a.append(score_a)
                            scores_b.append(score_b)
                    
                    if len(scores_a) > 1:  # Need multiple scores for statistical test
                        test_result = self.statistical_validator.paired_t_test(
                            np.array(scores_a), np.array(scores_b)
                        )
                        validation_results[comparison_key] = test_result
        
        self.results['statistical_validation'] = validation_results
        return validation_results
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        print("Generating performance report...")
        
        report = {
            'summary': {},
            'detailed_results': self.results,
            'recommendations': [],
            'statistical_significance': {}
        }
        
        # Performance summary
        if 'performance' in self.results:
            best_throughput = max(
                (model_name, max(datasets.values(), key=lambda x: x['throughput_samples_per_sec'])['throughput_samples_per_sec'])
                for model_name, datasets in self.results['performance'].items()
            )
            
            smallest_model = min(
                (model_name, min(datasets.values(), key=lambda x: x['model_size_mb'])['model_size_mb'])
                for model_name, datasets in self.results['performance'].items()
            )
            
            report['summary']['best_throughput'] = best_throughput
            report['summary']['smallest_model'] = smallest_model
        
        # Accuracy summary
        if 'accuracy' in self.results:
            accuracy_scores = {}
            for model_name, datasets in self.results['accuracy'].items():
                scores = []
                for dataset_metrics in datasets.values():
                    score = dataset_metrics.get('accuracy', dataset_metrics.get('r2', 0))
                    scores.append(score)
                accuracy_scores[model_name] = np.mean(scores) if scores else 0
            
            best_accuracy = max(accuracy_scores.items(), key=lambda x: x[1])
            report['summary']['best_accuracy'] = best_accuracy
        
        # Statistical significance summary
        if 'statistical_validation' in self.results:
            significant_differences = [
                comparison for comparison, result in self.results['statistical_validation'].items()
                if result['significant']
            ]
            report['statistical_significance']['significant_comparisons'] = significant_differences
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations()
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on benchmark results."""
        recommendations = []
        
        # Performance recommendations
        if 'performance' in self.results:
            # Find models with best trade-offs
            efficiency_scores = {}
            for model_name, datasets in self.results['performance'].items():
                avg_throughput = np.mean([d['throughput_samples_per_sec'] for d in datasets.values()])
                avg_size = np.mean([d['model_size_mb'] for d in datasets.values()])
                efficiency_scores[model_name] = avg_throughput / (avg_size + 1)  # Avoid division by zero
            
            best_efficiency = max(efficiency_scores.items(), key=lambda x: x[1])
            recommendations.append(f"For best efficiency, consider {best_efficiency[0]} (efficiency score: {best_efficiency[1]:.2f})")
        
        # Accuracy recommendations
        if 'accuracy' in self.results:
            # Find consistently good performers
            consistency_scores = {}
            for model_name, datasets in self.results['accuracy'].items():
                scores = [d.get('accuracy', d.get('r2', 0)) for d in datasets.values()]
                consistency_scores[model_name] = np.std(scores) if len(scores) > 1 else 0
            
            most_consistent = min(consistency_scores.items(), key=lambda x: x[1])
            recommendations.append(f"For most consistent performance, consider {most_consistent[0]} (std: {most_consistent[1]:.3f})")
        
        # Statistical recommendations
        if 'statistical_validation' in self.results:
            significant_improvements = [
                comp for comp, result in self.results['statistical_validation'].items()
                if result['significant'] and result['cohens_d'] > 0.5
            ]
            
            if significant_improvements:
                recommendations.append(f"Statistically significant improvements found in: {', '.join(significant_improvements)}")
        
        return recommendations
    
    def export_results(self, filepath: str = 'benchmark_results.json'):
        """Export benchmark results to file."""
        # Convert any non-serializable objects to serializable format
        serializable_results = self._make_serializable(dict(self.results))
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"Results exported to {filepath}")
    
    def _make_serializable(self, obj):
        """Convert non-serializable objects to serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.tolist() if hasattr(obj, 'tolist') else str(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        else:
            return obj
    
    def _estimate_memory_usage(self, model: nn.Module) -> float:
        """Estimate memory usage in MB."""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / 1024 / 1024


class VisualizationSuite:
    """Comprehensive visualization suite for benchmark results."""
    
    def __init__(self, benchmark_results: Dict[str, Any]):
        self.results = benchmark_results
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def plot_performance_comparison(self, save_path: str = 'performance_comparison.png'):
        """Plot performance comparison across models."""
        if 'performance' not in self.results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        # Extract data
        models = list(self.results['performance'].keys())
        datasets = list(next(iter(self.results['performance'].values())).keys())
        
        # Throughput comparison
        throughput_data = []
        for model in models:
            model_throughput = [self.results['performance'][model][dataset]['throughput_samples_per_sec'] 
                              for dataset in datasets]
            throughput_data.append(model_throughput)
        
        axes[0, 0].boxplot(throughput_data, labels=models)
        axes[0, 0].set_title('Throughput (samples/sec)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Model size comparison
        size_data = []
        for model in models:
            model_sizes = [self.results['performance'][model][dataset]['model_size_mb'] 
                          for dataset in datasets]
            size_data.append(model_sizes)
        
        axes[0, 1].boxplot(size_data, labels=models)
        axes[0, 1].set_title('Model Size (MB)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Inference time comparison
        time_data = []
        for model in models:
            model_times = [self.results['performance'][model][dataset]['inference_time_ms'] 
                          for dataset in datasets]
            time_data.append(model_times)
        
        axes[1, 0].boxplot(time_data, labels=models)
        axes[1, 0].set_title('Inference Time (ms)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Efficiency scatter plot (throughput vs size)
        for i, model in enumerate(models):
            throughput = np.mean(throughput_data[i])
            size = np.mean(size_data[i])
            axes[1, 1].scatter(size, throughput, label=model, s=100)
        
        axes[1, 1].set_xlabel('Model Size (MB)')
        axes[1, 1].set_ylabel('Throughput (samples/sec)')
        axes[1, 1].set_title('Efficiency (Throughput vs Size)')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance comparison plot saved to {save_path}")
    
    def plot_accuracy_heatmap(self, save_path: str = 'accuracy_heatmap.png'):
        """Plot accuracy heatmap across models and datasets."""
        if 'accuracy' not in self.results:
            return
        
        # Prepare data for heatmap
        models = list(self.results['accuracy'].keys())
        datasets = set()
        for model_results in self.results['accuracy'].values():
            datasets.update(model_results.keys())
        datasets = sorted(list(datasets))
        
        # Create accuracy matrix
        accuracy_matrix = np.zeros((len(models), len(datasets)))
        for i, model in enumerate(models):
            for j, dataset in enumerate(datasets):
                if dataset in self.results['accuracy'][model]:
                    metrics = self.results['accuracy'][model][dataset]
                    accuracy = metrics.get('accuracy', metrics.get('r2', 0))
                    accuracy_matrix[i, j] = accuracy
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(accuracy_matrix, 
                   xticklabels=datasets, 
                   yticklabels=models,
                   annot=True, 
                   fmt='.3f', 
                   cmap='viridis',
                   cbar_kws={'label': 'Accuracy/R²'})
        
        plt.title('Model Accuracy Across Datasets')
        plt.xlabel('Datasets')
        plt.ylabel('Models')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Accuracy heatmap saved to {save_path}")
    
    def plot_statistical_significance(self, save_path: str = 'statistical_significance.png'):
        """Plot statistical significance results."""
        if 'statistical_validation' not in self.results:
            return
        
        comparisons = list(self.results['statistical_validation'].keys())
        p_values = [self.results['statistical_validation'][comp]['p_value'] 
                   for comp in comparisons]
        effect_sizes = [abs(self.results['statistical_validation'][comp]['cohens_d']) 
                       for comp in comparisons]
        significant = [self.results['statistical_validation'][comp]['significant'] 
                      for comp in comparisons]
        
        # Create volcano plot style visualization
        plt.figure(figsize=(12, 8))
        
        colors = ['red' if sig else 'blue' for sig in significant]
        plt.scatter(effect_sizes, [-np.log10(p) for p in p_values], c=colors, alpha=0.7, s=100)
        
        # Add significance threshold line
        plt.axhline(y=-np.log10(0.05), color='gray', linestyle='--', alpha=0.5, label='p=0.05')
        
        # Add effect size threshold line
        plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Medium effect')
        
        # Annotate points
        for i, comp in enumerate(comparisons):
            if significant[i] and effect_sizes[i] > 0.5:
                plt.annotate(comp, (effect_sizes[i], -np.log10(p_values[i])), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('Effect Size (|Cohen\'s d|)')
        plt.ylabel('-log₁₀(p-value)')
        plt.title('Statistical Significance of Model Comparisons')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Statistical significance plot saved to {save_path}")


def create_comprehensive_benchmark(models: Dict[str, nn.Module], 
                                 datasets: Dict[str, List[Data]],
                                 test_labels: Optional[Dict[str, torch.Tensor]] = None) -> ComprehensiveBenchmark:
    """Factory function for creating comprehensive benchmarks."""
    benchmark = ComprehensiveBenchmark(models, datasets)
    
    # Run all benchmarks
    benchmark.run_performance_benchmark()
    
    if test_labels:
        benchmark.run_accuracy_benchmark(test_labels)
    
    benchmark.run_specialized_benchmarks()
    benchmark.run_statistical_validation()
    
    return benchmark


# Example usage utilities
class BenchmarkUtils:
    """Utilities for setting up and running benchmarks."""
    
    @staticmethod
    def create_synthetic_brain_atlas() -> Dict[str, Any]:
        """Create synthetic brain atlas for testing."""
        num_regions = 100
        
        return {
            'hemisphere_mapping': {
                'left': list(range(0, num_regions // 2)),
                'right': list(range(num_regions // 2, num_regions))
            },
            'distance_matrix': np.random.exponential(1.0, (num_regions, num_regions)),
            'default_mode_regions': [10, 15, 20, 25, 30, 35],  # Example DMN regions
            'region_names': [f'Region_{i}' for i in range(num_regions)]
        }
    
    @staticmethod
    def create_synthetic_datasets(num_samples: int = 100, 
                                num_nodes: int = 100, 
                                num_features: int = 50) -> Dict[str, List[Data]]:
        """Create synthetic datasets for benchmarking."""
        datasets = {}
        
        # Create different types of datasets
        dataset_configs = {
            'small_dense': {'num_edges_factor': 0.3, 'noise_level': 0.1},
            'large_sparse': {'num_edges_factor': 0.1, 'noise_level': 0.05},
            'high_noise': {'num_edges_factor': 0.2, 'noise_level': 0.3}
        }
        
        for dataset_name, config in dataset_configs.items():
            dataset = []
            
            for _ in range(num_samples):
                # Generate node features
                x = torch.randn(num_nodes, num_features)
                
                # Generate edges
                num_edges = int(num_nodes * (num_nodes - 1) * config['num_edges_factor'] / 2)
                edge_index = torch.randint(0, num_nodes, (2, num_edges))
                
                # Add noise
                noise = torch.randn_like(x) * config['noise_level']
                x = x + noise
                
                data = Data(x=x, edge_index=edge_index)
                dataset.append(data)
            
            datasets[dataset_name] = dataset
        
        return datasets