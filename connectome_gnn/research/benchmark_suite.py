"""Research Benchmarking Suite for Novel GNN Architectures.

Comprehensive benchmarking framework for evaluating novel graph neural network
architectures on connectome data with reproducible experimental protocols.
"""

import time
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings
import hashlib

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, average_precision_score
)
from sklearn.model_selection import StratifiedKFold, KFold

# Import our novel architectures
try:
    from .quantum_gnn import QuantumEnhancedGNN, QuantumMetrics
    from .neuromorphic_gnn import NeuromorphicGNN, NeuromorphicMetrics
    from .advanced_validation import AdvancedStatisticalValidator, ValidationConfig, ModelComparator
except ImportError as e:
    warnings.warn(f"Could not import novel architectures: {e}")


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking experiments."""
    
    # Experiment metadata
    experiment_name: str
    description: str
    random_seed: int = 42
    
    # Model configurations
    models_to_test: List[Dict[str, Any]]
    
    # Dataset configuration  
    dataset_name: str
    dataset_config: Dict[str, Any]
    
    # Training configuration
    max_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    patience: int = 20
    
    # Evaluation configuration
    cv_folds: int = 5
    test_split: float = 0.2
    metrics: List[str] = None
    
    # Hardware configuration
    device: str = "auto"
    num_workers: int = 4
    memory_limit_gb: Optional[float] = None
    
    # Output configuration
    save_models: bool = True
    save_predictions: bool = True
    save_embeddings: bool = False
    results_dir: str = "./benchmark_results"
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']


@dataclass
class ModelResult:
    """Results for a single model."""
    
    model_name: str
    model_config: Dict[str, Any]
    
    # Performance metrics
    train_metrics: Dict[str, List[float]]
    val_metrics: Dict[str, List[float]]
    test_metrics: Dict[str, float]
    
    # Training dynamics
    training_time: float
    inference_time: float
    memory_usage: float
    convergence_epoch: int
    
    # Model characteristics
    num_parameters: int
    model_size_mb: float
    
    # Cross-validation results
    cv_scores: Dict[str, List[float]]
    cv_mean: Dict[str, float]
    cv_std: Dict[str, float]
    
    # Novel architecture specific metrics
    novel_metrics: Optional[Dict[str, Any]] = None


@dataclass  
class BenchmarkReport:
    """Comprehensive benchmark report."""
    
    config: BenchmarkConfig
    results: List[ModelResult]
    statistical_analysis: Dict[str, Any]
    
    # Summary statistics
    best_model: str
    performance_ranking: List[Tuple[str, float]]
    significant_improvements: List[str]
    
    # Computational analysis
    efficiency_analysis: Dict[str, Any]
    scalability_analysis: Dict[str, Any]
    
    # Reproducibility information
    system_info: Dict[str, Any]
    git_hash: Optional[str]
    timestamp: str
    
    # Publication readiness
    latex_table: str
    figures_data: Dict[str, Any]


class BenchmarkingSuite:
    """Comprehensive benchmarking suite for novel GNN architectures."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = self._setup_device()
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Set random seeds for reproducibility
        self._set_random_seeds(config.random_seed)
        
        # Initialize statistical validator
        validation_config = ValidationConfig()
        self.statistical_validator = AdvancedStatisticalValidator(validation_config)
        self.model_comparator = ModelComparator(validation_config)
        
        self.logger.info(f"Initialized benchmarking suite: {config.experiment_name}")
    
    def _setup_device(self) -> torch.device:
        """Setup compute device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                self.logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                device = torch.device("cpu")
                self.logger.info("Using CPU")
        else:
            device = torch.device(self.config.device)
        
        return device
    
    def _setup_logging(self):
        """Setup experiment logging."""
        log_file = self.results_dir / f"{self.config.experiment_name}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(f"BenchmarkSuite.{self.config.experiment_name}")
    
    def _set_random_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def run_comprehensive_benchmark(self, dataset) -> BenchmarkReport:
        """Run comprehensive benchmarking experiment.
        
        Args:
            dataset: PyTorch Geometric dataset
            
        Returns:
            BenchmarkReport with complete results
        """
        
        self.logger.info("Starting comprehensive benchmark...")
        start_time = time.time()
        
        # Initialize results storage
        all_results = []
        performance_data = {}
        
        # Test each model configuration
        for i, model_config in enumerate(self.config.models_to_test):
            self.logger.info(f"Testing model {i+1}/{len(self.config.models_to_test)}: {model_config['name']}")
            
            try:
                # Run model benchmark
                model_result = self._benchmark_single_model(model_config, dataset)
                all_results.append(model_result)
                
                # Store performance data for statistical analysis
                performance_data[model_result.model_name] = np.array([
                    model_result.cv_mean[metric] for metric in self.config.metrics 
                    if metric in model_result.cv_mean
                ])
                
                self.logger.info(f"Completed {model_config['name']} - Test accuracy: {model_result.test_metrics.get('accuracy', 'N/A'):.4f}")
                
            except Exception as e:
                self.logger.error(f"Failed to benchmark {model_config['name']}: {str(e)}")
                continue
        
        # Statistical analysis
        self.logger.info("Performing statistical analysis...")
        statistical_analysis = self._perform_statistical_analysis(all_results)
        
        # Performance ranking
        ranking = self._rank_models(all_results)
        best_model = ranking[0][0] if ranking else "None"
        
        # Efficiency analysis
        efficiency_analysis = self._analyze_efficiency(all_results)
        
        # Scalability analysis  
        scalability_analysis = self._analyze_scalability(all_results)
        
        # System information
        system_info = self._collect_system_info()
        
        # Generate publication materials
        latex_table = self._generate_latex_table(all_results)
        figures_data = self._prepare_figures_data(all_results)
        
        # Create comprehensive report
        report = BenchmarkReport(
            config=self.config,
            results=all_results,
            statistical_analysis=statistical_analysis,
            best_model=best_model,
            performance_ranking=ranking,
            significant_improvements=statistical_analysis.get('significant_improvements', []),
            efficiency_analysis=efficiency_analysis,
            scalability_analysis=scalability_analysis,
            system_info=system_info,
            git_hash=self._get_git_hash(),
            timestamp=datetime.now().isoformat(),
            latex_table=latex_table,
            figures_data=figures_data
        )
        
        # Save report
        self._save_benchmark_report(report)
        
        total_time = time.time() - start_time
        self.logger.info(f"Benchmark completed in {total_time:.2f} seconds")
        
        return report
    
    def _benchmark_single_model(self, model_config: Dict[str, Any], dataset) -> ModelResult:
        """Benchmark a single model configuration."""
        
        model_name = model_config['name']
        
        # Create model
        model = self._create_model(model_config)
        model.to(self.device)
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        model_size_mb = num_params * 4 / (1024 * 1024)  # Assuming float32
        
        # Cross-validation evaluation
        cv_results = self._cross_validate_model(model, dataset, model_config)
        
        # Final training on full dataset
        train_metrics, val_metrics, test_metrics, training_time, convergence_epoch = \
            self._train_and_evaluate_model(model, dataset, model_config)
        
        # Inference time measurement
        inference_time = self._measure_inference_time(model, dataset)
        
        # Memory usage measurement
        memory_usage = self._measure_memory_usage(model, dataset)
        
        # Novel architecture specific metrics
        novel_metrics = self._compute_novel_metrics(model, dataset)
        
        return ModelResult(
            model_name=model_name,
            model_config=model_config,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            training_time=training_time,
            inference_time=inference_time,
            memory_usage=memory_usage,
            convergence_epoch=convergence_epoch,
            num_parameters=num_params,
            model_size_mb=model_size_mb,
            cv_scores=cv_results['scores'],
            cv_mean=cv_results['mean'],
            cv_std=cv_results['std'],
            novel_metrics=novel_metrics
        )
    
    def _create_model(self, model_config: Dict[str, Any]) -> nn.Module:
        """Create model from configuration."""
        
        model_type = model_config.get('type', 'standard')
        model_params = model_config.get('params', {})
        
        if model_type == 'quantum_enhanced':
            return QuantumEnhancedGNN(**model_params)
        elif model_type == 'neuromorphic':
            return NeuromorphicGNN(**model_params)
        else:
            # Import standard architectures
            try:
                from ..models.architectures import HierarchicalBrainGNN
                return HierarchicalBrainGNN(**model_params)
            except ImportError:
                # Fallback to simple model
                return self._create_simple_gnn(model_params)
    
    def _create_simple_gnn(self, params: Dict[str, Any]) -> nn.Module:
        """Create simple GNN as fallback."""
        from torch_geometric.nn import GCNConv, global_mean_pool
        
        class SimpleGNN(nn.Module):
            def __init__(self, input_dim=100, hidden_dim=64, output_dim=1, num_layers=2):
                super().__init__()
                self.convs = nn.ModuleList()
                self.convs.append(GCNConv(input_dim, hidden_dim))
                for _ in range(num_layers - 1):
                    self.convs.append(GCNConv(hidden_dim, hidden_dim))
                self.classifier = nn.Linear(hidden_dim, output_dim)
                
            def forward(self, data):
                x, edge_index, batch = data.x, data.edge_index, data.batch
                for conv in self.convs:
                    x = torch.relu(conv(x, edge_index))
                x = global_mean_pool(x, batch)
                return self.classifier(x)
        
        return SimpleGNN(**params)
    
    def _cross_validate_model(self, model: nn.Module, dataset, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform cross-validation evaluation."""
        
        kfold = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, 
                               random_state=self.config.random_seed)
        
        cv_scores = {metric: [] for metric in self.config.metrics}
        
        # Get labels for stratification
        labels = [data.y.item() if hasattr(data, 'y') else 0 for data in dataset]
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(dataset)), labels)):
            self.logger.info(f"CV Fold {fold + 1}/{self.config.cv_folds}")
            
            # Create fold datasets
            train_data = [dataset[i] for i in train_idx]
            val_data = [dataset[i] for i in val_idx]
            
            # Reset model for each fold
            fold_model = self._create_model(model_config)
            fold_model.to(self.device)
            
            # Train model on fold
            fold_metrics = self._train_model_fold(fold_model, train_data, val_data)
            
            # Store fold results
            for metric in self.config.metrics:
                if metric in fold_metrics:
                    cv_scores[metric].append(fold_metrics[metric])
        
        # Compute mean and std
        cv_mean = {metric: np.mean(scores) for metric, scores in cv_scores.items()}
        cv_std = {metric: np.std(scores) for metric, scores in cv_scores.items()}
        
        return {
            'scores': cv_scores,
            'mean': cv_mean,
            'std': cv_std
        }
    
    def _train_model_fold(self, model: nn.Module, train_data: List, val_data: List) -> Dict[str, float]:
        """Train model on a single fold."""
        
        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=self.config.batch_size, 
                                 shuffle=True, collate_fn=self._custom_collate)
        val_loader = DataLoader(val_data, batch_size=self.config.batch_size,
                               shuffle=False, collate_fn=self._custom_collate)
        
        # Setup training
        optimizer = torch.optim.Adam(model.parameters(), 
                                   lr=self.config.learning_rate,
                                   weight_decay=self.config.weight_decay)
        
        criterion = self._get_loss_function()
        
        # Training loop
        model.train()
        for epoch in range(self.config.max_epochs):
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                output = model(batch)
                loss = criterion(output, batch.y)
                
                loss.backward()
                optimizer.step()
        
        # Evaluation
        model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                output = model(batch)
                
                predictions.extend(output.cpu().numpy())
                targets.extend(batch.y.cpu().numpy())
        
        # Compute metrics
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        return self._compute_metrics(predictions, targets)
    
    def _custom_collate(self, batch):
        """Custom collate function for data loading."""
        return Batch.from_data_list(batch)
    
    def _get_loss_function(self):
        """Get appropriate loss function."""
        # This should be configurable based on task type
        return nn.CrossEntropyLoss()
    
    def _train_and_evaluate_model(self, model: nn.Module, dataset, model_config: Dict[str, Any]) -> Tuple:
        """Train and evaluate model on full dataset."""
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_data, val_data, test_data = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.config.random_seed)
        )
        
        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=self.config.batch_size,
                                 shuffle=True, collate_fn=self._custom_collate)
        val_loader = DataLoader(val_data, batch_size=self.config.batch_size,
                               shuffle=False, collate_fn=self._custom_collate)
        test_loader = DataLoader(test_data, batch_size=self.config.batch_size,
                                shuffle=False, collate_fn=self._custom_collate)
        
        # Training setup
        optimizer = torch.optim.Adam(model.parameters(),
                                   lr=self.config.learning_rate,
                                   weight_decay=self.config.weight_decay)
        criterion = self._get_loss_function()
        
        # Training tracking
        train_metrics_history = {metric: [] for metric in self.config.metrics}
        val_metrics_history = {metric: [] for metric in self.config.metrics}
        best_val_metric = -np.inf
        patience_counter = 0
        convergence_epoch = self.config.max_epochs
        
        start_time = time.time()
        
        # Training loop
        for epoch in range(self.config.max_epochs):
            # Training
            model.train()
            train_preds, train_targets = [], []
            
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                output = model(batch)
                loss = criterion(output, batch.y)
                
                loss.backward()
                optimizer.step()
                
                train_preds.extend(output.detach().cpu().numpy())
                train_targets.extend(batch.y.cpu().numpy())
            
            # Validation
            model.eval()
            val_preds, val_targets = [], []
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    output = model(batch)
                    
                    val_preds.extend(output.cpu().numpy())
                    val_targets.extend(batch.y.cpu().numpy())
            
            # Compute metrics
            train_metrics = self._compute_metrics(np.array(train_preds), np.array(train_targets))
            val_metrics = self._compute_metrics(np.array(val_preds), np.array(val_targets))
            
            # Store metrics
            for metric in self.config.metrics:
                if metric in train_metrics:
                    train_metrics_history[metric].append(train_metrics[metric])
                if metric in val_metrics:
                    val_metrics_history[metric].append(val_metrics[metric])
            
            # Early stopping
            current_val_metric = val_metrics.get('accuracy', val_metrics.get('f1', 0))
            if current_val_metric > best_val_metric:
                best_val_metric = current_val_metric
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.patience:
                convergence_epoch = epoch + 1
                break
        
        training_time = time.time() - start_time
        
        # Final test evaluation
        model.eval()
        test_preds, test_targets = [], []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                output = model(batch)
                
                test_preds.extend(output.cpu().numpy())
                test_targets.extend(batch.y.cpu().numpy())
        
        test_metrics = self._compute_metrics(np.array(test_preds), np.array(test_targets))
        
        return train_metrics_history, val_metrics_history, test_metrics, training_time, convergence_epoch
    
    def _compute_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Compute evaluation metrics."""
        
        metrics = {}
        
        # Handle different prediction formats
        if predictions.ndim > 1 and predictions.shape[1] > 1:
            # Multi-class classification
            pred_classes = np.argmax(predictions, axis=1)
        else:
            # Binary classification or regression
            pred_classes = (predictions > 0.5).astype(int).flatten()
        
        targets = targets.flatten()
        
        try:
            if 'accuracy' in self.config.metrics:
                metrics['accuracy'] = accuracy_score(targets, pred_classes)
            
            if any(m in self.config.metrics for m in ['precision', 'recall', 'f1']):
                precision, recall, f1, _ = precision_recall_fscore_support(
                    targets, pred_classes, average='weighted', zero_division=0
                )
                if 'precision' in self.config.metrics:
                    metrics['precision'] = precision
                if 'recall' in self.config.metrics:
                    metrics['recall'] = recall
                if 'f1' in self.config.metrics:
                    metrics['f1'] = f1
            
            if 'auc' in self.config.metrics and predictions.ndim > 1:
                try:
                    metrics['auc'] = roc_auc_score(targets, predictions[:, 1])
                except (ValueError, IndexError):
                    metrics['auc'] = 0.5  # Random performance
            
            if 'mse' in self.config.metrics:
                metrics['mse'] = mean_squared_error(targets, predictions.flatten())
            
            if 'mae' in self.config.metrics:
                metrics['mae'] = mean_absolute_error(targets, predictions.flatten())
            
            if 'r2' in self.config.metrics:
                metrics['r2'] = r2_score(targets, predictions.flatten())
                
        except Exception as e:
            self.logger.warning(f"Error computing metrics: {e}")
            
        return metrics
    
    def _measure_inference_time(self, model: nn.Module, dataset) -> float:
        """Measure inference time per sample."""
        
        model.eval()
        
        # Use a subset for timing
        test_data = dataset[:min(100, len(dataset))]
        test_loader = DataLoader(test_data, batch_size=1, collate_fn=self._custom_collate)
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                _ = model(batch)
        
        total_time = time.time() - start_time
        return total_time / len(test_data)
    
    def _measure_memory_usage(self, model: nn.Module, dataset) -> float:
        """Measure memory usage in MB."""
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
            # Forward pass with representative batch
            test_batch = Batch.from_data_list(dataset[:self.config.batch_size])
            test_batch = test_batch.to(self.device)
            
            model.eval()
            with torch.no_grad():
                _ = model(test_batch)
            
            memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            torch.cuda.empty_cache()
            
            return memory_mb
        else:
            # Rough estimation for CPU
            return sum(p.numel() * 4 for p in model.parameters()) / (1024 * 1024)
    
    def _compute_novel_metrics(self, model: nn.Module, dataset) -> Optional[Dict[str, Any]]:
        """Compute novel architecture specific metrics."""
        
        novel_metrics = {}
        
        if isinstance(model, QuantumEnhancedGNN):
            # Quantum-specific metrics
            test_data = dataset[0]
            test_data = test_data.to(self.device)
            
            try:
                entanglement = QuantumMetrics.entanglement_measure(model, test_data)
                novel_metrics['quantum_entanglement'] = entanglement
                
                # Add more quantum metrics as needed
            except Exception as e:
                self.logger.warning(f"Error computing quantum metrics: {e}")
        
        elif isinstance(model, NeuromorphicGNN):
            # Neuromorphic-specific metrics
            try:
                spike_stats = model.get_spike_statistics()
                novel_metrics['neuromorphic_stats'] = spike_stats
                
                # Add more neuromorphic metrics
                novel_metrics['spike_rate'] = spike_stats.get('spike_rate', 0.0)
                novel_metrics['spike_sparsity'] = spike_stats.get('spike_sparsity', 0.0)
                
            except Exception as e:
                self.logger.warning(f"Error computing neuromorphic metrics: {e}")
        
        return novel_metrics if novel_metrics else None
    
    def _perform_statistical_analysis(self, results: List[ModelResult]) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis."""
        
        if len(results) < 2:
            return {'message': 'Need at least 2 models for statistical analysis'}
        
        # Extract performance data
        performance_data = {}
        for result in results:
            for metric in self.config.metrics:
                if metric in result.cv_mean:
                    key = f"{result.model_name}_{metric}"
                    performance_data[key] = np.array(result.cv_scores[metric])
        
        # Model comparison
        comparisons = self.model_comparator.compare_models(performance_data)
        
        # Identify significant improvements
        significant_improvements = []
        for comparison_name, validation_result in comparisons.items():
            if validation_result.significant_after_correction:
                significant_improvements.append(comparison_name)
        
        return {
            'comparisons': {k: asdict(v) for k, v in comparisons.items()},
            'significant_improvements': significant_improvements,
            'model_rankings': self.model_comparator.rank_models(performance_data)
        }
    
    def _rank_models(self, results: List[ModelResult]) -> List[Tuple[str, float]]:
        """Rank models by primary performance metric."""
        
        primary_metric = 'accuracy' if 'accuracy' in self.config.metrics else self.config.metrics[0]
        
        rankings = []
        for result in results:
            if primary_metric in result.cv_mean:
                rankings.append((result.model_name, result.cv_mean[primary_metric]))
        
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
    
    def _analyze_efficiency(self, results: List[ModelResult]) -> Dict[str, Any]:
        """Analyze computational efficiency."""
        
        efficiency_data = {}
        
        for result in results:
            efficiency_data[result.model_name] = {
                'training_time': result.training_time,
                'inference_time': result.inference_time,
                'memory_usage': result.memory_usage,
                'parameters': result.num_parameters,
                'model_size_mb': result.model_size_mb,
                'performance': result.cv_mean.get('accuracy', 0.0),
                'efficiency_score': result.cv_mean.get('accuracy', 0.0) / (result.training_time + 1e-6)
            }
        
        return efficiency_data
    
    def _analyze_scalability(self, results: List[ModelResult]) -> Dict[str, Any]:
        """Analyze model scalability."""
        
        scalability_analysis = {}
        
        for result in results:
            scalability_analysis[result.model_name] = {
                'parameter_efficiency': result.cv_mean.get('accuracy', 0.0) / result.num_parameters,
                'memory_efficiency': result.cv_mean.get('accuracy', 0.0) / result.memory_usage,
                'convergence_speed': result.convergence_epoch,
                'scalability_score': (
                    result.cv_mean.get('accuracy', 0.0) / 
                    (result.num_parameters / 1e6 + result.memory_usage / 1000 + 1e-6)
                )
            }
        
        return scalability_analysis
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information for reproducibility."""
        
        import platform
        import sys
        
        system_info = {
            'platform': platform.platform(),
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'device_name': str(self.device)
        }
        
        if torch.cuda.is_available():
            system_info['gpu_name'] = torch.cuda.get_device_name()
            system_info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory
        
        return system_info
    
    def _get_git_hash(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, cwd=Path(__file__).parent)
            return result.stdout.strip() if result.returncode == 0 else None
        except:
            return None
    
    def _generate_latex_table(self, results: List[ModelResult]) -> str:
        """Generate LaTeX table for publication."""
        
        latex_lines = []
        latex_lines.append("\\begin{table}[htbp]")
        latex_lines.append("\\centering")
        latex_lines.append("\\caption{Model Performance Comparison}")
        latex_lines.append("\\label{tab:model_comparison}")
        
        # Table header
        metrics_str = " & ".join([m.capitalize() for m in self.config.metrics[:4]])  # Limit columns
        latex_lines.append(f"\\begin{{tabular}}{{l{('c' * len(self.config.metrics[:4]))}c}}")
        latex_lines.append("\\toprule")
        latex_lines.append(f"Model & {metrics_str} & Parameters \\\\")
        latex_lines.append("\\midrule")
        
        # Table rows
        for result in results:
            row_data = [result.model_name.replace('_', '\\_')]
            
            for metric in self.config.metrics[:4]:
                if metric in result.cv_mean:
                    value = result.cv_mean[metric]
                    std = result.cv_std.get(metric, 0)
                    row_data.append(f"{value:.3f} ± {std:.3f}")
                else:
                    row_data.append("N/A")
            
            row_data.append(f"{result.num_parameters / 1e6:.1f}M")
            
            latex_lines.append(" & ".join(row_data) + " \\\\")
        
        latex_lines.append("\\bottomrule")
        latex_lines.append("\\end{tabular}")
        latex_lines.append("\\end{table}")
        
        return "\n".join(latex_lines)
    
    def _prepare_figures_data(self, results: List[ModelResult]) -> Dict[str, Any]:
        """Prepare data for generating figures."""
        
        figures_data = {
            'performance_comparison': {},
            'efficiency_analysis': {},
            'training_curves': {}
        }
        
        for result in results:
            # Performance data
            figures_data['performance_comparison'][result.model_name] = {
                metric: result.cv_mean.get(metric, 0) for metric in self.config.metrics
            }
            
            # Efficiency data
            figures_data['efficiency_analysis'][result.model_name] = {
                'training_time': result.training_time,
                'inference_time': result.inference_time,
                'memory_usage': result.memory_usage,
                'parameters': result.num_parameters
            }
            
            # Training curves (if available)
            figures_data['training_curves'][result.model_name] = {
                'train_metrics': result.train_metrics,
                'val_metrics': result.val_metrics
            }
        
        return figures_data
    
    def _save_benchmark_report(self, report: BenchmarkReport):
        """Save comprehensive benchmark report."""
        
        # Save JSON report
        json_file = self.results_dir / f"{self.config.experiment_name}_report.json"
        
        # Convert to serializable format
        serializable_report = asdict(report)
        
        with open(json_file, 'w') as f:
            json.dump(serializable_report, f, indent=2, default=str)
        
        # Save LaTeX table
        latex_file = self.results_dir / f"{self.config.experiment_name}_table.tex"
        latex_file.write_text(report.latex_table)
        
        # Save individual model results
        for result in report.results:
            result_file = self.results_dir / f"{result.model_name}_detailed.json"
            with open(result_file, 'w') as f:
                json.dump(asdict(result), f, indent=2, default=str)
        
        self.logger.info(f"Benchmark report saved to {json_file}")


# Factory functions for easy model configuration
def create_quantum_gnn_config(node_features=100, hidden_dim=256, **kwargs):
    """Create configuration for Quantum-Enhanced GNN."""
    return {
        'name': 'quantum_enhanced_gnn',
        'type': 'quantum_enhanced',
        'params': {
            'node_features': node_features,
            'hidden_dim': hidden_dim,
            **kwargs
        }
    }


def create_neuromorphic_gnn_config(node_features=100, hidden_dim=256, **kwargs):
    """Create configuration for Neuromorphic GNN."""
    return {
        'name': 'neuromorphic_gnn', 
        'type': 'neuromorphic',
        'params': {
            'node_features': node_features,
            'hidden_dim': hidden_dim,
            **kwargs
        }
    }


def create_baseline_gnn_config(node_features=100, hidden_dim=256, **kwargs):
    """Create configuration for baseline GNN."""
    return {
        'name': 'baseline_gnn',
        'type': 'standard',
        'params': {
            'input_dim': node_features,
            'hidden_dim': hidden_dim,
            **kwargs
        }
    }