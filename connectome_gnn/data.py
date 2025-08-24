"""Data loading and processing components for Connectome-GNN-Suite."""

import torch
import numpy as np
from torch_geometric.data import Data, Dataset
from typing import List, Optional, Tuple, Union
import os
from pathlib import Path

from .core.base_types import ConnectomeData
from .robust.validation import validate_connectome_data
from .robust.error_handling import ConnectomeError


class ConnectomeDataset(Dataset):
    """Dataset class for Human Connectome Project data.
    
    Handles loading and preprocessing of structural and functional
    connectivity matrices from HCP data.
    """
    
    def __init__(
        self,
        root: str,
        resolution: str = "7mm",
        modality: str = "structural", 
        download: bool = False,
        transform=None,
        pre_transform=None
    ):
        self.resolution = resolution
        self.modality = modality
        self.download = download
        
        super().__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self) -> List[str]:
        return ["subjects.txt"]
        
    @property 
    def processed_file_names(self) -> List[str]:
        return [f"data_{i}.pt" for i in range(50)]  # 50 synthetic subjects
        
    def download(self):
        """Download HCP data (synthetic for demo)."""
        raw_dir = Path(self.root) / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Create synthetic subject list
        with open(raw_dir / "subjects.txt", "w") as f:
            for i in range(50):
                f.write(f"subject_{i:03d}\n")
                
    def process(self):
        """Process raw HCP data into PyTorch Geometric format."""
        # Generate synthetic connectome data
        for idx in range(50):
            data = self._generate_synthetic_connectome(idx)
            torch.save(data, self.processed_paths[idx])
            
    def _generate_synthetic_connectome(self, subject_id: int) -> Data:
        """Generate synthetic connectome data for demonstration."""
        np.random.seed(subject_id)
        torch.manual_seed(subject_id)
        
        # Brain parcellation (e.g., AAL atlas with ~100 regions)
        num_nodes = 90 if self.resolution == "7mm" else 116
        
        # Node features (regional measurements)
        node_features = torch.randn(num_nodes, 10)  # 10 features per region
        
        # Connectivity matrix (structural or functional)
        if self.modality == "structural":
            # Structural connectivity (DTI-based)
            connectivity = np.random.exponential(0.1, (num_nodes, num_nodes))
            connectivity = (connectivity + connectivity.T) / 2  # Symmetric
            np.fill_diagonal(connectivity, 0)  # No self-connections
        else:
            # Functional connectivity (fMRI-based)  
            connectivity = np.random.normal(0, 0.3, (num_nodes, num_nodes))
            connectivity = (connectivity + connectivity.T) / 2  # Symmetric
            np.fill_diagonal(connectivity, 1)  # Self-correlation = 1
            
        # Threshold to create sparse graph
        threshold = 0.1 if self.modality == "structural" else 0.3
        adjacency = (connectivity > threshold).astype(float)
        
        # Convert to edge format
        edge_indices = np.where(adjacency)
        edge_index = torch.tensor([edge_indices[0], edge_indices[1]], dtype=torch.long)
        edge_attr = torch.tensor(connectivity[edge_indices], dtype=torch.float).unsqueeze(1)
        
        # Subject-level labels (for tasks)
        age = 20 + np.random.exponential(10)  # Age distribution
        sex = np.random.randint(0, 2)  # Binary sex
        fluid_intelligence = np.random.normal(100, 15)  # Cognitive score
        
        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([age, sex, fluid_intelligence], dtype=torch.float),
            num_nodes=num_nodes,
            subject_id=subject_id
        )
        
    def len(self) -> int:
        return len(self.processed_file_names)
        
    def get(self, idx: int) -> Data:
        data = torch.load(self.processed_paths[idx])
        return data
        

class ConnectomeProcessor:
    """Preprocessing pipeline for connectome data."""
    
    def __init__(
        self,
        parcellation: str = "AAL",
        edge_threshold: float = 0.01,
        normalization: str = "log_transform"
    ):
        self.parcellation = parcellation
        self.edge_threshold = edge_threshold
        self.normalization = normalization
        
    def process(
        self,
        connectivity_matrix: np.ndarray,
        node_timeseries: Optional[np.ndarray] = None,
        confounds: Optional[np.ndarray] = None
    ) -> Data:
        """Process connectivity matrix into graph format."""
        # Validate inputs
        validate_connectome_data(connectivity_matrix)
        
        # Apply normalization
        if self.normalization == "log_transform":
            connectivity_matrix = np.log1p(connectivity_matrix)
        elif self.normalization == "z_score":
            connectivity_matrix = (connectivity_matrix - np.mean(connectivity_matrix)) / np.std(connectivity_matrix)
            
        # Threshold edges
        adjacency = (connectivity_matrix > self.edge_threshold).astype(float)
        
        # Convert to PyTorch Geometric format
        edge_indices = np.where(adjacency)
        edge_index = torch.tensor([edge_indices[0], edge_indices[1]], dtype=torch.long)
        edge_attr = torch.tensor(connectivity_matrix[edge_indices], dtype=torch.float).unsqueeze(1)
        
        # Default node features if timeseries not provided
        if node_timeseries is None:
            node_features = torch.randn(connectivity_matrix.shape[0], 10)
        else:
            node_features = torch.tensor(node_timeseries, dtype=torch.float)
            
        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=connectivity_matrix.shape[0]
        )


class HCPLoader:
    """Data loader for Human Connectome Project datasets."""
    
    def __init__(self, data_path: str = "data/hcp"):
        self.data_path = Path(data_path)
        
    def load_subject(self, subject_id: str) -> ConnectomeData:
        """Load data for a specific subject."""
        # This would load actual HCP data files
        # For now, return synthetic data
        return self._create_synthetic_subject(subject_id)
        
    def _create_synthetic_subject(self, subject_id: str) -> ConnectomeData:
        """Create synthetic subject data."""
        subject_hash = hash(subject_id) % 1000
        np.random.seed(subject_hash)
        
        num_nodes = 100
        connectivity = np.random.exponential(0.1, (num_nodes, num_nodes))
        connectivity = (connectivity + connectivity.T) / 2
        np.fill_diagonal(connectivity, 0)
        
        return ConnectomeData(
            connectivity_matrix=connectivity,
            node_features=np.random.randn(num_nodes, 10),
            demographics={
                'age': 20 + np.random.exponential(10),
                'sex': np.random.choice(['M', 'F']),
                'subject_id': subject_id
            }
        )
        
    def get_available_subjects(self) -> List[str]:
        """Get list of available subject IDs."""
        return [f"HCP_{i:06d}" for i in range(100, 150)]  # 50 synthetic subjects


# Export main classes
__all__ = [
    "ConnectomeDataset",
    "ConnectomeProcessor", 
    "HCPLoader"
]