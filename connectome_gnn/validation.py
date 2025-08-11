"""Validation utilities for connectome GNN data and models."""

import torch
import numpy as np
from torch_geometric.data import Data
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
from pathlib import Path
import logging

# Configure logging
logger = logging.getLogger(__name__)


class ConnectomeDataValidator:
    """Validator for connectome data integrity and quality."""
    
    def __init__(
        self,
        min_nodes: int = 50,
        max_nodes: int = 5000,
        min_edges: int = 100,
        connectivity_threshold: float = 0.01,
        feature_nan_threshold: float = 0.1
    ):
        """Initialize data validator.
        
        Args:
            min_nodes: Minimum number of nodes required
            max_nodes: Maximum number of nodes allowed
            min_edges: Minimum number of edges required
            connectivity_threshold: Minimum connectivity strength
            feature_nan_threshold: Maximum fraction of NaN features allowed
        """
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.min_edges = min_edges
        self.connectivity_threshold = connectivity_threshold
        self.feature_nan_threshold = feature_nan_threshold
        
    def validate_data(self, data: Data) -> Dict[str, Any]:
        """Validate a single connectome data object.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Validation results dictionary
        """
        results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'metrics': {}
        }
        
        try:
            # Basic structure validation
            self._validate_structure(data, results)
            
            # Node validation
            self._validate_nodes(data, results)
            
            # Edge validation  
            self._validate_edges(data, results)
            
            # Feature validation
            self._validate_features(data, results)
            
            # Connectivity validation
            self._validate_connectivity(data, results)
            
            # Target validation
            self._validate_targets(data, results)
            
            # Overall validity
            results['is_valid'] = len(results['errors']) == 0
            
        except Exception as e:
            results['errors'].append(f"Validation failed with exception: {str(e)}")
            results['is_valid'] = False
            logger.error(f"Data validation error: {e}")
        
        return results
    
    def _validate_structure(self, data: Data, results: Dict) -> None:
        """Validate basic data structure."""
        # Check required attributes
        required_attrs = ['x', 'edge_index']
        for attr in required_attrs:
            if not hasattr(data, attr) or getattr(data, attr) is None:
                results['errors'].append(f"Missing required attribute: {attr}")
        
        # Check tensor types
        if hasattr(data, 'x') and data.x is not None:
            if not isinstance(data.x, torch.Tensor):
                results['errors'].append("Node features (x) must be a torch.Tensor")
        
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            if not isinstance(data.edge_index, torch.Tensor):
                results['errors'].append("Edge index must be a torch.Tensor")
    
    def _validate_nodes(self, data: Data, results: Dict) -> None:
        """Validate node-related data."""
        if not hasattr(data, 'x') or data.x is None:
            return
            
        num_nodes = data.x.size(0)
        
        # Check node count
        if num_nodes < self.min_nodes:
            results['errors'].append(
                f"Too few nodes: {num_nodes} < {self.min_nodes}"
            )
        elif num_nodes > self.max_nodes:
            results['warnings'].append(
                f"Many nodes: {num_nodes} > {self.max_nodes}, may impact performance"
            )
        
        results['metrics']['num_nodes'] = num_nodes
    
    def _validate_edges(self, data: Data, results: Dict) -> None:
        """Validate edge-related data."""
        if not hasattr(data, 'edge_index') or data.edge_index is None:
            return
        
        edge_index = data.edge_index
        num_edges = edge_index.size(1)
        
        # Check edge count
        if num_edges < self.min_edges:
            results['warnings'].append(
                f"Few edges: {num_edges} < {self.min_edges}, graph may be sparse"
            )
        
        # Check edge index validity
        if hasattr(data, 'x') and data.x is not None:
            max_node_idx = data.x.size(0) - 1
            if edge_index.max() > max_node_idx:
                results['errors'].append(
                    f"Edge index contains invalid node indices (max: {edge_index.max()}, "
                    f"should be <= {max_node_idx})"
                )
            if edge_index.min() < 0:
                results['errors'].append("Edge index contains negative indices")
        
        # Check for self-loops
        self_loops = (edge_index[0] == edge_index[1]).sum().item()
        if self_loops > 0:
            results['warnings'].append(f"Found {self_loops} self-loops")
        
        results['metrics']['num_edges'] = num_edges
        results['metrics']['self_loops'] = self_loops
    
    def _validate_features(self, data: Data, results: Dict) -> None:
        """Validate feature data quality."""
        if not hasattr(data, 'x') or data.x is None:
            return
        
        features = data.x
        
        # Check for NaN values
        nan_mask = torch.isnan(features)
        nan_fraction = nan_mask.float().mean().item()
        
        if nan_fraction > self.feature_nan_threshold:
            results['errors'].append(
                f"Too many NaN features: {nan_fraction:.3f} > {self.feature_nan_threshold}"
            )
        elif nan_fraction > 0:
            results['warnings'].append(f"Found {nan_fraction:.3f} fraction of NaN features")
        
        # Check for infinite values
        inf_mask = torch.isinf(features)
        if inf_mask.any():
            results['warnings'].append("Found infinite feature values")
        
        # Feature statistics
        if nan_fraction < 1.0:  # If not all NaN
            valid_features = features[~nan_mask]
            results['metrics'].update({
                'feature_mean': valid_features.mean().item(),
                'feature_std': valid_features.std().item(),
                'feature_min': valid_features.min().item(),
                'feature_max': valid_features.max().item(),
                'nan_fraction': nan_fraction
            })
    
    def _validate_connectivity(self, data: Data, results: Dict) -> None:
        """Validate connectivity patterns."""
        if not hasattr(data, 'edge_index') or not hasattr(data, 'x'):
            return
        
        if data.edge_index is None or data.x is None:
            return
        
        num_nodes = data.x.size(0)
        num_edges = data.edge_index.size(1)
        
        # Compute connectivity density
        max_possible_edges = num_nodes * (num_nodes - 1)  # Directed graph
        density = num_edges / max_possible_edges if max_possible_edges > 0 else 0
        
        if density < self.connectivity_threshold:
            results['warnings'].append(
                f"Low connectivity density: {density:.4f} < {self.connectivity_threshold}"
            )
        
        results['metrics']['connectivity_density'] = density
        
        # Check edge weights if available
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_weights = data.edge_attr
            
            # Check for negative weights (might be unexpected)
            negative_weights = (edge_weights < 0).sum().item()
            if negative_weights > 0:
                results['warnings'].append(f"Found {negative_weights} negative edge weights")
            
            # Weight statistics
            results['metrics'].update({
                'edge_weight_mean': edge_weights.mean().item(),
                'edge_weight_std': edge_weights.std().item(),
                'negative_weights': negative_weights
            })
    
    def _validate_targets(self, data: Data, results: Dict) -> None:
        """Validate target variables."""
        target_attrs = ['y', 'y_age', 'y_sex', 'y_cognitive', 'age', 'sex', 'cognitive_score']
        
        found_targets = []
        for attr in target_attrs:
            if hasattr(data, attr) and getattr(data, attr) is not None:
                target = getattr(data, attr)
                if isinstance(target, torch.Tensor):
                    # Check for NaN targets
                    if torch.isnan(target).any():
                        results['warnings'].append(f"Target {attr} contains NaN values")
                    # Check for infinite targets
                    if torch.isinf(target).any():
                        results['warnings'].append(f"Target {attr} contains infinite values")
                    
                    found_targets.append(attr)
        
        if not found_targets:
            results['warnings'].append("No target variables found")
        
        results['metrics']['available_targets'] = found_targets


class ModelValidator:
    """Validator for connectome GNN model configurations and states."""
    
    def __init__(self):
        self.validation_history = []
    
    def validate_model(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Validate model architecture and state.
        
        Args:
            model: PyTorch model to validate
            
        Returns:
            Validation results
        """
        results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'metrics': {}
        }
        
        try:
            # Parameter validation
            self._validate_parameters(model, results)
            
            # Architecture validation
            self._validate_architecture(model, results)
            
            # Memory usage estimation
            self._estimate_memory_usage(model, results)
            
            results['is_valid'] = len(results['errors']) == 0
            
        except Exception as e:
            results['errors'].append(f"Model validation failed: {str(e)}")
            results['is_valid'] = False
            logger.error(f"Model validation error: {e}")
        
        self.validation_history.append(results)
        return results
    
    def _validate_parameters(self, model: torch.nn.Module, results: Dict) -> None:
        """Validate model parameters."""
        total_params = 0
        trainable_params = 0
        
        nan_params = 0
        inf_params = 0
        zero_params = 0
        
        for name, param in model.named_parameters():
            if param is None:
                results['warnings'].append(f"Parameter {name} is None")
                continue
                
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
            
            # Check for problematic values
            if torch.isnan(param).any():
                nan_params += torch.isnan(param).sum().item()
                results['warnings'].append(f"Parameter {name} contains NaN values")
            
            if torch.isinf(param).any():
                inf_params += torch.isinf(param).sum().item()
                results['warnings'].append(f"Parameter {name} contains infinite values")
            
            if (param == 0).all():
                zero_params += param.numel()
                results['warnings'].append(f"Parameter {name} is all zeros")
        
        results['metrics'].update({
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'nan_parameters': nan_params,
            'inf_parameters': inf_params,
            'zero_parameters': zero_params
        })
        
        # Parameter count warnings
        if total_params > 10_000_000:  # 10M parameters
            results['warnings'].append(f"Large model: {total_params:,} parameters")
        elif total_params < 1000:  # Very small model
            results['warnings'].append(f"Very small model: {total_params:,} parameters")
    
    def _validate_architecture(self, model: torch.nn.Module, results: Dict) -> None:
        """Validate model architecture."""
        # Check for common architectural issues
        module_counts = {}
        for name, module in model.named_modules():
            module_type = type(module).__name__
            module_counts[module_type] = module_counts.get(module_type, 0) + 1
        
        results['metrics']['module_counts'] = module_counts
        
        # Check for missing activation functions
        if 'ReLU' not in module_counts and 'ELU' not in module_counts and 'GELU' not in module_counts:
            results['warnings'].append("No common activation functions found")
        
        # Check for missing normalization
        if 'BatchNorm1d' not in module_counts and 'LayerNorm' not in module_counts:
            results['warnings'].append("No normalization layers found")
        
        # Check for missing dropout
        if 'Dropout' not in module_counts:
            results['warnings'].append("No dropout layers found (may lead to overfitting)")
    
    def _estimate_memory_usage(self, model: torch.nn.Module, results: Dict) -> None:
        """Estimate model memory usage."""
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers())
        
        total_memory_bytes = param_memory + buffer_memory
        total_memory_mb = total_memory_bytes / (1024 * 1024)
        
        results['metrics'].update({
            'memory_usage_mb': total_memory_mb,
            'parameter_memory_mb': param_memory / (1024 * 1024),
            'buffer_memory_mb': buffer_memory / (1024 * 1024)
        })
        
        if total_memory_mb > 1000:  # 1GB
            results['warnings'].append(f"High memory usage: {total_memory_mb:.1f} MB")


def validate_batch_data(batch: Union[Data, List[Data]]) -> Dict[str, Any]:
    """Validate a batch of connectome data.
    
    Args:
        batch: Single Data object or list of Data objects
        
    Returns:
        Batch validation results
    """
    validator = ConnectomeDataValidator()
    
    if isinstance(batch, Data):
        data_list = [batch]
    else:
        data_list = batch
    
    results = {
        'batch_size': len(data_list),
        'valid_samples': 0,
        'invalid_samples': 0,
        'warnings': [],
        'errors': [],
        'sample_results': []
    }
    
    for i, data in enumerate(data_list):
        sample_result = validator.validate_data(data)
        sample_result['sample_idx'] = i
        
        if sample_result['is_valid']:
            results['valid_samples'] += 1
        else:
            results['invalid_samples'] += 1
            results['errors'].extend([f"Sample {i}: {err}" for err in sample_result['errors']])
        
        results['warnings'].extend([f"Sample {i}: {warn}" for warn in sample_result['warnings']])
        results['sample_results'].append(sample_result)
    
    results['is_valid'] = results['invalid_samples'] == 0
    
    return results


def sanitize_file_path(path: Union[str, Path]) -> Path:
    """Sanitize file paths to prevent directory traversal attacks.
    
    Args:
        path: Input file path
        
    Returns:
        Sanitized Path object
        
    Raises:
        ValueError: If path contains suspicious patterns
    """
    path = Path(path).resolve()
    
    # Check for suspicious patterns
    suspicious_patterns = ['..', '~', '$']
    path_str = str(path)
    
    for pattern in suspicious_patterns:
        if pattern in path_str:
            raise ValueError(f"Suspicious path pattern '{pattern}' found in: {path}")
    
    # Ensure path doesn't go outside allowed directories
    cwd = Path.cwd().resolve()
    try:
        path.relative_to(cwd)
    except ValueError:
        raise ValueError(f"Path outside working directory: {path}")
    
    return path


def validate_input_ranges(
    values: torch.Tensor,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    name: str = "input"
) -> None:
    """Validate input tensor ranges.
    
    Args:
        values: Input tensor to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value  
        name: Name for error messages
        
    Raises:
        ValueError: If values are outside allowed ranges
    """
    if min_val is not None and values.min() < min_val:
        raise ValueError(f"{name} contains values below minimum {min_val}: {values.min()}")
    
    if max_val is not None and values.max() > max_val:
        raise ValueError(f"{name} contains values above maximum {max_val}: {values.max()}")
    
    if torch.isnan(values).any():
        raise ValueError(f"{name} contains NaN values")
    
    if torch.isinf(values).any():
        raise ValueError(f"{name} contains infinite values")


class SecurityValidator:
    """Security validation for connectome analysis pipeline."""
    
    @staticmethod
    def validate_model_checkpoint(checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """Validate model checkpoint for security issues.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Security validation results
        """
        results = {
            'is_safe': True,
            'warnings': [],
            'errors': [],
            'file_info': {}
        }
        
        try:
            checkpoint_path = sanitize_file_path(checkpoint_path)
            
            if not checkpoint_path.exists():
                results['errors'].append("Checkpoint file does not exist")
                results['is_safe'] = False
                return results
            
            # File size check
            file_size = checkpoint_path.stat().st_size
            results['file_info']['size_mb'] = file_size / (1024 * 1024)
            
            if file_size > 1024 * 1024 * 1024:  # 1GB
                results['warnings'].append(f"Large checkpoint file: {file_size / (1024**3):.1f} GB")
            
            # Try to load checkpoint metadata safely
            try:
                # Load only metadata, not the actual tensors
                checkpoint = torch.load(
                    checkpoint_path, 
                    map_location='cpu',
                    weights_only=True  # Security: only load tensors, not arbitrary code
                )
                results['file_info']['keys'] = list(checkpoint.keys()) if isinstance(checkpoint, dict) else []
            except Exception as e:
                results['warnings'].append(f"Could not safely load checkpoint metadata: {e}")
            
        except Exception as e:
            results['errors'].append(f"Security validation failed: {e}")
            results['is_safe'] = False
        
        return results