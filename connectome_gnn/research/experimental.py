"""Experimental framework for connectome GNN research."""

import torch
import torch.nn as nn
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
import pickle
from sklearn.model_selection import KFold, StratifiedKFold
from scipy.stats import ttest_ind, wilcoxon

from ..models.base import BaseConnectomeModel
from ..tasks.base import BaseConnectomeTask
from ..training import ConnectomeTrainer
from ..utils import set_random_seed


@dataclass
class ExperimentConfig:
    """Configuration for experimental runs."""
    
    # Experiment metadata
    experiment_name: str
    description: str
    random_seed: int = 42
    
    # Model configuration
    model_class: str
    model_params: Dict[str, Any]
    
    # Task configuration
    task_class: str
    task_params: Dict[str, Any]
    
    # Training configuration
    training_params: Dict[str, Any]
    
    # Evaluation configuration
    cv_folds: int = 5
    test_size: float = 0.2
    evaluation_metrics: List[str]
    
    # Computational configuration
    device: str = "auto"
    max_epochs: int = 100
    early_stopping_patience: int = 20
    
    # Output configuration
    save_models: bool = True
    save_predictions: bool = True
    save_attention_weights: bool = False


@dataclass
class ExperimentResult:
    """Results from an experimental run."""
    
    config: ExperimentConfig
    cv_results: Dict[str, List[float]]
    test_results: Dict[str, float]
    training_time: float
    inference_time: float
    model_size: int
    convergence_epoch: int
    
    # Statistical analysis
    mean_cv_score: float
    std_cv_score: float
    confidence_interval: Tuple[float, float]
    
    # Additional metrics
    best_fold: int
    worst_fold: int
    fold_consistency: float


class ExperimentalFramework:
    """Framework for conducting reproducible connectome GNN experiments."""
    
    def __init__(
        self,
        output_dir: str = "./experiments",
        device: str = "auto",
        verbose: bool = True
    ):
        """Initialize experimental framework.
        
        Args:
            output_dir: Directory to save experiment results
            device: Device for computation
            verbose: Whether to print progress
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device
        self.verbose = verbose
        
        # Track all experiments
        self.experiment_registry = {}
        self.results_cache = {}
        
        if verbose:
            print(f"Experimental framework initialized. Output: {self.output_dir}")
    
    def run_experiment(
        self,
        config: ExperimentConfig,
        dataset,
        force_rerun: bool = False
    ) -> ExperimentResult:
        """Run a complete experiment with cross-validation.
        
        Args:
            config: Experiment configuration
            dataset: Dataset for training and evaluation
            force_rerun: Whether to force rerun if results exist
            
        Returns:
            Experiment results
        """
        experiment_id = self._generate_experiment_id(config)
        experiment_dir = self.output_dir / experiment_id
        
        # Check if experiment already exists
        if not force_rerun and experiment_dir.exists():
            if self.verbose:
                print(f"Loading existing experiment: {experiment_id}")
            return self._load_experiment_result(experiment_dir)
        
        if self.verbose:
            print(f"Running experiment: {config.experiment_name}")
            print(f"Experiment ID: {experiment_id}")
        
        # Create experiment directory
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(experiment_dir / "config.json", "w") as f:
            json.dump(asdict(config), f, indent=2, default=str)
        
        # Set random seed
        set_random_seed(config.random_seed)
        
        # Run cross-validation
        start_time = time.time()
        cv_results = self._run_cross_validation(config, dataset, experiment_dir)
        training_time = time.time() - start_time
        
        # Run final test evaluation
        test_results, inference_time, model_info = self._run_test_evaluation(
            config, dataset, experiment_dir
        )
        
        # Statistical analysis
        primary_metric = config.evaluation_metrics[0]
        cv_scores = cv_results[primary_metric]
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        
        # 95% confidence interval
        n = len(cv_scores)
        confidence_interval = (
            mean_score - 1.96 * std_score / np.sqrt(n),
            mean_score + 1.96 * std_score / np.sqrt(n)
        )
        
        # Fold analysis
        best_fold = int(np.argmax(cv_scores))
        worst_fold = int(np.argmin(cv_scores))
        fold_consistency = 1.0 - (std_score / (mean_score + 1e-8))
        
        # Create result object
        result = ExperimentResult(
            config=config,
            cv_results=cv_results,
            test_results=test_results,
            training_time=training_time,
            inference_time=inference_time,
            model_size=model_info['parameters'],
            convergence_epoch=model_info['convergence_epoch'],
            mean_cv_score=mean_score,
            std_cv_score=std_score,
            confidence_interval=confidence_interval,
            best_fold=best_fold,
            worst_fold=worst_fold,
            fold_consistency=fold_consistency
        )
        
        # Save results
        with open(experiment_dir / "results.pkl", "wb") as f:
            pickle.dump(result, f)
        
        # Save summary
        self._save_experiment_summary(result, experiment_dir)
        
        # Register experiment
        self.experiment_registry[experiment_id] = result
        
        if self.verbose:
            print(f"Experiment completed. Results saved to: {experiment_dir}")
            print(f"CV Score: {mean_score:.4f} ± {std_score:.4f}")
            print(f"Test Score: {test_results[primary_metric]:.4f}")
        
        return result
    
    def _run_cross_validation(
        self,
        config: ExperimentConfig,
        dataset,
        experiment_dir: Path
    ) -> Dict[str, List[float]]:
        """Run cross-validation experiment."""
        
        # Initialize CV splitter
        if hasattr(dataset, 'get_targets'):
            targets = dataset.get_targets()
            if config.task_params.get('task_type') == 'classification':
                cv_splitter = StratifiedKFold(
                    n_splits=config.cv_folds,
                    shuffle=True,
                    random_state=config.random_seed
                )
                splits = list(cv_splitter.split(range(len(dataset)), targets))
            else:
                cv_splitter = KFold(
                    n_splits=config.cv_folds,
                    shuffle=True,
                    random_state=config.random_seed
                )
                splits = list(cv_splitter.split(range(len(dataset))))
        else:
            # Simple random splits
            n_samples = len(dataset)
            indices = np.random.permutation(n_samples)
            fold_size = n_samples // config.cv_folds
            
            splits = []
            for i in range(config.cv_folds):
                start_idx = i * fold_size
                end_idx = (i + 1) * fold_size if i < config.cv_folds - 1 else n_samples
                
                test_indices = indices[start_idx:end_idx]
                train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
                
                splits.append((train_indices, test_indices))
        
        # Results storage
        cv_results = {metric: [] for metric in config.evaluation_metrics}
        
        # Run each fold
        for fold_idx, (train_indices, val_indices) in enumerate(splits):
            if self.verbose:
                print(f"Running fold {fold_idx + 1}/{config.cv_folds}")
            
            fold_dir = experiment_dir / f"fold_{fold_idx}"
            fold_dir.mkdir(exist_ok=True)
            
            # Create fold datasets
            train_dataset = [dataset[i] for i in train_indices]
            val_dataset = [dataset[i] for i in val_indices]
            
            # Initialize model and task
            model = self._create_model(config)
            task = self._create_task(config)
            
            # Train model
            trainer = ConnectomeTrainer(
                model=model,
                task=task,
                output_dir=fold_dir,
                **config.training_params
            )
            
            # Prepare data splits for trainer
            trainer.fit(
                train_dataset,
                epochs=config.max_epochs,
                val_ratio=0.0  # We're providing validation set explicitly
            )
            
            # Evaluate on validation set
            val_metrics = self._evaluate_model(model, task, val_dataset)
            
            # Store results
            for metric in config.evaluation_metrics:
                if metric in val_metrics:
                    cv_results[metric].append(val_metrics[metric])
            
            # Save fold results
            with open(fold_dir / "metrics.json", "w") as f:
                json.dump(val_metrics, f, indent=2)
            
            if config.save_models:
                torch.save(model.state_dict(), fold_dir / "model.pth")
        
        return cv_results
    
    def _run_test_evaluation(
        self,
        config: ExperimentConfig,
        dataset,
        experiment_dir: Path
    ) -> Tuple[Dict[str, float], float, Dict[str, Any]]:
        """Run final test evaluation."""
        
        # Split dataset
        n_samples = len(dataset)
        test_size = int(n_samples * config.test_size)
        
        indices = np.random.permutation(n_samples)
        train_indices = indices[:-test_size]
        test_indices = indices[-test_size:]
        
        train_dataset = [dataset[i] for i in train_indices]
        test_dataset = [dataset[i] for i in test_indices]
        
        # Train final model
        if self.verbose:
            print("Training final model on full training set...")
        
        model = self._create_model(config)
        task = self._create_task(config)
        
        final_dir = experiment_dir / "final_model"
        final_dir.mkdir(exist_ok=True)
        
        trainer = ConnectomeTrainer(
            model=model,
            task=task,
            output_dir=final_dir,
            **config.training_params
        )
        
        trainer.fit(train_dataset, epochs=config.max_epochs)
        
        # Evaluate on test set
        start_time = time.time()
        test_metrics = self._evaluate_model(model, task, test_dataset)
        inference_time = time.time() - start_time
        
        # Model info
        model_info = {
            'parameters': model.count_parameters(),
            'convergence_epoch': trainer.epoch
        }
        
        # Save final model
        if config.save_models:
            torch.save(model.state_dict(), final_dir / "final_model.pth")
        
        return test_metrics, inference_time, model_info
    
    def _create_model(self, config: ExperimentConfig) -> BaseConnectomeModel:
        """Create model from configuration."""
        # Import model class dynamically
        if config.model_class == "HierarchicalBrainGNN":
            from ..models import HierarchicalBrainGNN
            model_class = HierarchicalBrainGNN
        elif config.model_class == "TemporalConnectomeGNN":
            from ..models import TemporalConnectomeGNN
            model_class = TemporalConnectomeGNN
        elif config.model_class == "MultiModalBrainGNN":
            from ..models import MultiModalBrainGNN
            model_class = MultiModalBrainGNN
        elif config.model_class == "PopulationGraphGNN":
            from ..models import PopulationGraphGNN
            model_class = PopulationGraphGNN
        else:
            raise ValueError(f"Unknown model class: {config.model_class}")
        
        return model_class(**config.model_params)
    
    def _create_task(self, config: ExperimentConfig) -> BaseConnectomeTask:
        """Create task from configuration."""
        from ..tasks import TaskFactory
        
        task_config = {
            'type': config.task_class,
            **config.task_params
        }
        
        return TaskFactory.create_task(task_config)
    
    def _evaluate_model(
        self,
        model: BaseConnectomeModel,
        task: BaseConnectomeTask,
        dataset: List
    ) -> Dict[str, float]:
        """Evaluate model on dataset."""
        model.eval()
        
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data in dataset:
                pred = model(data)
                target = task.prepare_targets([data])
                
                predictions.append(pred)
                targets.append(target)
        
        if predictions:
            predictions = torch.cat(predictions, dim=0)
            targets = torch.cat(targets, dim=0)
            
            # Compute metrics
            metrics = task.compute_metrics(predictions, targets)
        else:
            metrics = {}
        
        return metrics
    
    def _generate_experiment_id(self, config: ExperimentConfig) -> str:
        """Generate unique experiment ID."""
        import hashlib
        
        # Create hash from configuration
        config_str = json.dumps(asdict(config), sort_keys=True, default=str)
        hash_obj = hashlib.md5(config_str.encode())
        hash_hex = hash_obj.hexdigest()[:8]
        
        # Format: experiment_name_timestamp_hash
        timestamp = int(time.time())
        experiment_id = f"{config.experiment_name}_{timestamp}_{hash_hex}"
        
        return experiment_id
    
    def _save_experiment_summary(self, result: ExperimentResult, experiment_dir: Path):
        """Save human-readable experiment summary."""
        
        summary = f"""
# Experiment Summary

## Configuration
- **Experiment**: {result.config.experiment_name}
- **Description**: {result.config.description}
- **Model**: {result.config.model_class}
- **Task**: {result.config.task_class}
- **Random Seed**: {result.config.random_seed}

## Results
- **CV Score**: {result.mean_cv_score:.4f} ± {result.std_cv_score:.4f}
- **Test Score**: {result.test_results.get(result.config.evaluation_metrics[0], 'N/A')}
- **Confidence Interval**: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]
- **Fold Consistency**: {result.fold_consistency:.4f}

## Performance
- **Training Time**: {result.training_time:.2f} seconds
- **Inference Time**: {result.inference_time:.4f} seconds
- **Model Size**: {result.model_size:,} parameters
- **Convergence Epoch**: {result.convergence_epoch}

## Cross-Validation Details
- **Best Fold**: {result.best_fold}
- **Worst Fold**: {result.worst_fold}
- **CV Folds**: {result.config.cv_folds}

## Detailed CV Results
"""
        
        for metric, scores in result.cv_results.items():
            summary += f"- **{metric}**: {scores}\n"
        
        with open(experiment_dir / "summary.md", "w") as f:
            f.write(summary)
    
    def _load_experiment_result(self, experiment_dir: Path) -> ExperimentResult:
        """Load existing experiment result."""
        with open(experiment_dir / "results.pkl", "rb") as f:
            return pickle.load(f)
    
    def compare_experiments(
        self,
        experiment_ids: List[str],
        metric: str = "accuracy",
        statistical_test: str = "ttest"
    ) -> Dict[str, Any]:
        """Compare multiple experiments statistically.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            metric: Metric to compare
            statistical_test: Statistical test to use (ttest, wilcoxon)
            
        Returns:
            Comparison results
        """
        if len(experiment_ids) < 2:
            raise ValueError("Need at least 2 experiments to compare")
        
        # Collect results
        results = []
        for exp_id in experiment_ids:
            if exp_id not in self.experiment_registry:
                # Try to load from disk
                exp_dir = self.output_dir / exp_id
                if exp_dir.exists():
                    result = self._load_experiment_result(exp_dir)
                    self.experiment_registry[exp_id] = result
                else:
                    raise ValueError(f"Experiment {exp_id} not found")
            
            result = self.experiment_registry[exp_id]
            if metric in result.cv_results:
                results.append(result.cv_results[metric])
            else:
                raise ValueError(f"Metric {metric} not found in experiment {exp_id}")
        
        # Statistical comparison
        comparison_results = {
            'experiment_ids': experiment_ids,
            'metric': metric,
            'means': [np.mean(scores) for scores in results],
            'stds': [np.std(scores) for scores in results],
            'pairwise_comparisons': []
        }
        
        # Pairwise statistical tests
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                scores_i = results[i]
                scores_j = results[j]
                
                if statistical_test == "ttest":
                    statistic, p_value = ttest_ind(scores_i, scores_j)
                elif statistical_test == "wilcoxon":
                    statistic, p_value = wilcoxon(scores_i, scores_j)
                else:
                    raise ValueError(f"Unknown statistical test: {statistical_test}")
                
                comparison_results['pairwise_comparisons'].append({
                    'experiment_pair': (experiment_ids[i], experiment_ids[j]),
                    'statistic': statistic,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'effect_size': abs(comparison_results['means'][i] - comparison_results['means'][j])
                })
        
        return comparison_results
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of all experiments."""
        return {
            'total_experiments': len(self.experiment_registry),
            'experiments': list(self.experiment_registry.keys()),
            'output_directory': str(self.output_dir)
        }


class BaselineComparison:
    """Framework for comparing novel architectures against baselines."""
    
    def __init__(self, experimental_framework: ExperimentalFramework):
        """Initialize baseline comparison.
        
        Args:
            experimental_framework: Experimental framework instance
        """
        self.framework = experimental_framework
        
        # Standard baseline models
        self.baseline_configs = {
            'mlp_baseline': {
                'model_class': 'ConnectomeMLPBaseline',
                'model_params': {'hidden_dims': [512, 256, 128]}
            },
            'simple_gcn': {
                'model_class': 'HierarchicalBrainGNN',
                'model_params': {'num_levels': 1, 'hidden_dim': 256}
            },
            'standard_gat': {
                'model_class': 'HierarchicalBrainGNN',
                'model_params': {'attention_type': 'dense', 'num_levels': 2}
            }
        }
    
    def run_baseline_comparison(
        self,
        novel_model_config: Dict[str, Any],
        task_config: Dict[str, Any],
        dataset,
        experiment_name: str = "baseline_comparison"
    ) -> Dict[str, Any]:
        """Run comparison between novel model and baselines.
        
        Args:
            novel_model_config: Configuration for novel model
            task_config: Task configuration
            dataset: Dataset for evaluation
            experiment_name: Name for this comparison
            
        Returns:
            Comparison results
        """
        experiments_to_run = {
            'novel_model': novel_model_config,
            **self.baseline_configs
        }
        
        results = {}
        experiment_ids = []
        
        # Run experiments
        for model_name, model_config in experiments_to_run.items():
            config = ExperimentConfig(
                experiment_name=f"{experiment_name}_{model_name}",
                description=f"Baseline comparison: {model_name}",
                model_class=model_config['model_class'],
                model_params=model_config['model_params'],
                task_class=task_config['task_class'],
                task_params=task_config['task_params'],
                training_params={'batch_size': 16, 'learning_rate': 1e-3},
                evaluation_metrics=['accuracy', 'f1', 'auc'] if task_config.get('task_type') == 'classification' else ['mae', 'rmse', 'r2']
            )
            
            result = self.framework.run_experiment(config, dataset)
            results[model_name] = result
            experiment_ids.append(self.framework._generate_experiment_id(config))
        
        # Statistical comparison
        primary_metric = config.evaluation_metrics[0]
        comparison = self.framework.compare_experiments(experiment_ids, primary_metric)
        
        # Format results
        formatted_results = {
            'experiment_name': experiment_name,
            'models_compared': list(experiments_to_run.keys()),
            'primary_metric': primary_metric,
            'results_summary': {},
            'statistical_comparison': comparison,
            'best_model': None,
            'performance_gains': {}
        }
        
        # Results summary
        best_score = float('-inf') if primary_metric in ['accuracy', 'f1', 'auc', 'r2'] else float('inf')
        best_model = None
        
        for model_name, result in results.items():
            score = result.mean_cv_score
            formatted_results['results_summary'][model_name] = {
                'cv_score': f"{score:.4f} ± {result.std_cv_score:.4f}",
                'test_score': result.test_results.get(primary_metric, 'N/A'),
                'training_time': f"{result.training_time:.2f}s",
                'model_size': f"{result.model_size:,} params"
            }
            
            # Determine best model
            is_better = (score > best_score) if primary_metric in ['accuracy', 'f1', 'auc', 'r2'] else (score < best_score)
            if is_better:
                best_score = score
                best_model = model_name
        
        formatted_results['best_model'] = best_model
        
        # Calculate performance gains relative to baselines
        if best_model == 'novel_model':
            for baseline_name in self.baseline_configs.keys():
                if baseline_name in results:
                    baseline_score = results[baseline_name].mean_cv_score
                    if primary_metric in ['accuracy', 'f1', 'auc', 'r2']:
                        gain = ((best_score - baseline_score) / baseline_score) * 100
                    else:
                        gain = ((baseline_score - best_score) / baseline_score) * 100
                    
                    formatted_results['performance_gains'][baseline_name] = f"{gain:.2f}%"
        
        return formatted_results