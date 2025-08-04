"""Utility functions for connectome analysis."""

import os
import random
import numpy as np
import torch
from typing import Optional, Union, Dict, Any, List
from pathlib import Path


def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior in CUDA operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: Union[str, torch.device] = "auto") -> torch.device:
    """Get appropriate device for computation.
    
    Args:
        device: Device specification ("auto", "cpu", "cuda", or specific device)
        
    Returns:
        PyTorch device object
    """
    if isinstance(device, torch.device):
        return device
    
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    elif device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("CUDA not available, falling back to CPU")
            return torch.device("cpu")
    else:
        return torch.device(device)


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Estimate model size in megabytes.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def format_time(seconds: float) -> str:
    """Format time duration in human-readable format.
    
    Args:
        seconds: Time duration in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def save_config(config: Dict[str, Any], output_path: Path):
    """Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save configuration
    """
    import json
    
    # Convert non-serializable objects to strings
    serializable_config = {}
    for key, value in config.items():
        if isinstance(value, (str, int, float, bool, list, dict)):
            serializable_config[key] = value
        else:
            serializable_config[key] = str(value)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_config, f, indent=2)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    import json
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def create_output_directory(base_dir: str, experiment_name: str) -> Path:
    """Create output directory with timestamp.
    
    Args:
        base_dir: Base directory for outputs
        experiment_name: Name of the experiment
        
    Returns:
        Path to created output directory
    """
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir


def check_memory_usage():
    """Check current memory usage (GPU and CPU)."""
    import psutil
    
    # CPU memory
    cpu_memory = psutil.virtual_memory()
    print(f"CPU Memory: {cpu_memory.percent}% used "
          f"({cpu_memory.used / 1024**3:.1f}GB / {cpu_memory.total / 1024**3:.1f}GB)")
    
    # GPU memory
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_cached = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i} Memory: {memory_allocated:.1f}GB allocated, {memory_cached:.1f}GB cached")


def get_brain_region_names(parcellation: str = "AAL") -> List[str]:
    """Get brain region names for a given parcellation.
    
    Args:
        parcellation: Brain atlas name
        
    Returns:
        List of region names
    """
    if parcellation == "AAL":
        # Simplified AAL region names
        regions = [
            # Frontal regions
            "Precentral_L", "Precentral_R",
            "Frontal_Sup_L", "Frontal_Sup_R", 
            "Frontal_Sup_Orb_L", "Frontal_Sup_Orb_R",
            "Frontal_Mid_L", "Frontal_Mid_R",
            "Frontal_Mid_Orb_L", "Frontal_Mid_Orb_R",
            "Frontal_Inf_Oper_L", "Frontal_Inf_Oper_R",
            "Frontal_Inf_Tri_L", "Frontal_Inf_Tri_R",
            "Frontal_Inf_Orb_L", "Frontal_Inf_Orb_R",
            "Rolandic_Oper_L", "Rolandic_Oper_R",
            "Supp_Motor_Area_L", "Supp_Motor_Area_R",
            "Olfactory_L", "Olfactory_R",
            "Frontal_Sup_Medial_L", "Frontal_Sup_Medial_R",
            # Parietal regions  
            "Postcentral_L", "Postcentral_R",
            "Parietal_Sup_L", "Parietal_Sup_R",
            "Parietal_Inf_L", "Parietal_Inf_R",
            "SupraMarginal_L", "SupraMarginal_R",
            "Angular_L", "Angular_R",
            "Precuneus_L", "Precuneus_R",
            "Paracentral_Lobule_L", "Paracentral_Lobule_R",
            # Temporal regions
            "Heschl_L", "Heschl_R",
            "Temporal_Sup_L", "Temporal_Sup_R",
            "Temporal_Pole_Sup_L", "Temporal_Pole_Sup_R",
            "Temporal_Mid_L", "Temporal_Mid_R", 
            "Temporal_Pole_Mid_L", "Temporal_Pole_Mid_R",
            "Temporal_Inf_L", "Temporal_Inf_R",
            # Occipital regions
            "Calcarine_L", "Calcarine_R",
            "Cuneus_L", "Cuneus_R",
            "Lingual_L", "Lingual_R",
            "Occipital_Sup_L", "Occipital_Sup_R",
            "Occipital_Mid_L", "Occipital_Mid_R",
            "Occipital_Inf_L", "Occipital_Inf_R",
            "Fusiform_L", "Fusiform_R",
            # Subcortical regions
            "Hippocampus_L", "Hippocampus_R",
            "ParaHippocampal_L", "ParaHippocampal_R", 
            "Amygdala_L", "Amygdala_R",
            "Caudate_L", "Caudate_R",
            "Putamen_L", "Putamen_R",
            "Pallidum_L", "Pallidum_R",
            "Thalamus_L", "Thalamus_R",
            "Insula_L", "Insula_R",
            "Cingulum_Ant_L", "Cingulum_Ant_R",
            "Cingulum_Mid_L", "Cingulum_Mid_R",
            "Cingulum_Post_L", "Cingulum_Post_R",
            # Cerebellar regions
            "Cerebelum_Crus1_L", "Cerebelum_Crus1_R",
            "Cerebelum_Crus2_L", "Cerebelum_Crus2_R",
            "Cerebelum_3_L", "Cerebelum_3_R",
            "Cerebelum_4_5_L", "Cerebelum_4_5_R",
            "Cerebelum_6_L", "Cerebelum_6_R",
            "Cerebelum_7b_L", "Cerebelum_7b_R",
            "Cerebelum_8_L", "Cerebelum_8_R",
            "Cerebelum_9_L", "Cerebelum_9_R",
            "Cerebelum_10_L", "Cerebelum_10_R",
            "Vermis_1_2", "Vermis_3", "Vermis_4_5",
            "Vermis_6", "Vermis_7", "Vermis_8",
            "Vermis_9", "Vermis_10"
        ]
        return regions[:90]  # Return first 90 for standard AAL
    
    elif parcellation == "Schaefer400":
        # Generate Schaefer 400 region names
        networks = [
            "Vis", "SomMot", "DorsAttn", "SalVentAttn", 
            "Limbic", "Cont", "Default"
        ]
        regions = []
        for i in range(400):
            network = networks[i % len(networks)]
            hemisphere = "LH" if i < 200 else "RH"
            region_num = (i % 200) + 1
            regions.append(f"{network}_{hemisphere}_{region_num}")
        return regions
    
    else:
        # Generic region names
        return [f"Region_{i+1}" for i in range(100)]


def validate_connectivity_matrix(connectivity: np.ndarray) -> Dict[str, Any]:
    """Validate properties of a connectivity matrix.
    
    Args:
        connectivity: Connectivity matrix [N, N]
        
    Returns:
        Dictionary with validation results
    """
    validation = {
        'valid': True,
        'issues': [],
        'properties': {}
    }
    
    # Check shape
    if connectivity.ndim != 2:
        validation['valid'] = False
        validation['issues'].append(f"Matrix is not 2D (shape: {connectivity.shape})")
        return validation
    
    if connectivity.shape[0] != connectivity.shape[1]:
        validation['valid'] = False
        validation['issues'].append(f"Matrix is not square (shape: {connectivity.shape})")
        return validation
    
    # Check for NaN/Inf values
    if np.isnan(connectivity).any():
        validation['valid'] = False
        validation['issues'].append("Matrix contains NaN values")
    
    if np.isinf(connectivity).any():
        validation['valid'] = False
        validation['issues'].append("Matrix contains infinite values")
    
    # Check symmetry
    is_symmetric = np.allclose(connectivity, connectivity.T, rtol=1e-5)
    if not is_symmetric:
        validation['issues'].append("Matrix is not symmetric")
    
    # Compute properties
    validation['properties'] = {
        'shape': connectivity.shape,
        'symmetric': is_symmetric,
        'diagonal_zero': np.allclose(np.diag(connectivity), 0),
        'min_value': float(np.min(connectivity)),
        'max_value': float(np.max(connectivity)),
        'mean_value': float(np.mean(connectivity)),
        'std_value': float(np.std(connectivity)),
        'sparsity': float(np.mean(connectivity == 0)),
        'density': float(np.mean(connectivity != 0))
    }
    
    return validation


def normalize_connectivity_matrix(
    connectivity: np.ndarray,
    method: str = "minmax"
) -> np.ndarray:
    """Normalize connectivity matrix.
    
    Args:
        connectivity: Input connectivity matrix
        method: Normalization method ("minmax", "zscore", "log")
        
    Returns:
        Normalized connectivity matrix
    """
    if method == "minmax":
        min_val = np.min(connectivity)
        max_val = np.max(connectivity)
        if max_val > min_val:
            return (connectivity - min_val) / (max_val - min_val)
        else:
            return connectivity
    
    elif method == "zscore":
        mean_val = np.mean(connectivity)
        std_val = np.std(connectivity)
        if std_val > 0:
            return (connectivity - mean_val) / std_val
        else:
            return connectivity - mean_val
    
    elif method == "log":
        # Log transform for positive values
        positive_mask = connectivity > 0
        normalized = connectivity.copy()
        normalized[positive_mask] = np.log1p(normalized[positive_mask])
        return normalized
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_graph_metrics(adjacency_matrix: np.ndarray) -> Dict[str, float]:
    """Compute basic graph theory metrics.
    
    Args:
        adjacency_matrix: Binary or weighted adjacency matrix
        
    Returns:
        Dictionary of graph metrics
    """
    # Binarize for topology metrics
    binary_adj = (adjacency_matrix > 0).astype(int)
    n_nodes = binary_adj.shape[0]
    
    # Basic metrics
    num_edges = np.sum(binary_adj) // 2  # Undirected
    density = num_edges / (n_nodes * (n_nodes - 1) / 2)
    
    # Degree statistics
    degrees = np.sum(binary_adj, axis=1)
    
    metrics = {
        'num_nodes': n_nodes,
        'num_edges': num_edges,
        'density': density,
        'mean_degree': float(np.mean(degrees)),
        'std_degree': float(np.std(degrees)),
        'max_degree': int(np.max(degrees)),
        'min_degree': int(np.min(degrees))
    }
    
    # Add weighted metrics if matrix has weights
    if not np.array_equal(adjacency_matrix, binary_adj):
        weights = adjacency_matrix[adjacency_matrix > 0]
        metrics.update({
            'mean_weight': float(np.mean(weights)),
            'std_weight': float(np.std(weights)),
            'max_weight': float(np.max(weights)),
            'min_weight': float(np.min(weights))
        })
    
    return metrics


class MemoryTracker:
    """Track memory usage during training."""
    
    def __init__(self):
        self.peak_memory = 0
        self.current_memory = 0
        
    def update(self):
        """Update memory tracking."""
        if torch.cuda.is_available():
            self.current_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            self.peak_memory = max(self.peak_memory, self.current_memory)
    
    def reset(self):
        """Reset memory tracking."""
        self.peak_memory = 0
        self.current_memory = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def get_stats(self) -> Dict[str, float]:
        """Get memory statistics."""
        return {
            'current_memory_gb': self.current_memory,
            'peak_memory_gb': self.peak_memory
        }


def print_model_info(model: torch.nn.Module, input_size: Optional[tuple] = None):
    """Print detailed model information.
    
    Args:
        model: PyTorch model
        input_size: Optional input size for computation estimation
    """
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Model size: {get_model_size_mb(model):.2f} MB")
    
    if input_size:
        # Estimate FLOPs (simplified)
        total_params = count_parameters(model)
        input_elements = np.prod(input_size)
        estimated_flops = total_params * input_elements * 2  # Rough estimate
        print(f"Estimated FLOPs: {estimated_flops:,}")
    
    print("\nModel architecture:")
    print(model)