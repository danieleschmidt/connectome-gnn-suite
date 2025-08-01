"""Data generators for creating realistic test data."""

import numpy as np
import torch
from torch_geometric.data import Data
from typing import List, Dict, Tuple, Optional
import networkx as nx


class ConnectomeDataGenerator:
    """Generate realistic connectome data for testing."""
    
    def __init__(self, random_seed: int = 42):
        """Initialize the data generator.
        
        Args:
            random_seed: Random seed for reproducible generation
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    
    def generate_connectivity_matrix(
        self,
        num_nodes: int,
        density: float = 0.1,
        noise_level: float = 0.1,
        spatial_structure: bool = True
    ) -> np.ndarray:
        """Generate a realistic connectivity matrix.
        
        Args:
            num_nodes: Number of brain regions/nodes
            density: Connection density (fraction of possible connections)  
            noise_level: Amount of noise to add
            spatial_structure: Whether to add spatial structure
            
        Returns:
            Symmetric connectivity matrix
        """
        # Start with random connectivity
        matrix = np.random.exponential(0.1, (num_nodes, num_nodes))
        
        # Make symmetric
        matrix = (matrix + matrix.T) / 2
        
        # Remove self-connections
        np.fill_diagonal(matrix, 0)
        
        # Apply density threshold
        threshold = np.percentile(matrix, (1 - density) * 100)
        matrix = matrix * (matrix > threshold)
        
        # Add spatial structure if requested
        if spatial_structure:
            for i in range(num_nodes):
                # Strengthen connections to nearby regions
                nearby_range = min(10, num_nodes // 10)
                for j in range(max(0, i - nearby_range), 
                             min(num_nodes, i + nearby_range + 1)):
                    if i != j:
                        distance_factor = 1.0 / (1.0 + abs(i - j))
                        matrix[i, j] *= (1 + distance_factor)
        
        # Add noise
        noise = np.random.normal(0, noise_level, matrix.shape)
        noise = (noise + noise.T) / 2  # Keep symmetric
        np.fill_diagonal(noise, 0)
        matrix += noise
        
        # Ensure non-negative
        matrix = np.maximum(matrix, 0)
        
        return matrix
    
    def generate_subject_data(
        self,
        num_subjects: int,
        num_nodes: int = 400,
        include_demographics: bool = True,
        include_cognitive_scores: bool = True
    ) -> Dict:
        """Generate data for multiple subjects.
        
        Args:
            num_subjects: Number of subjects to generate
            num_nodes: Number of brain regions per subject
            include_demographics: Whether to include demographic data
            include_cognitive_scores: Whether to include cognitive scores
            
        Returns:
            Dictionary containing subject data
        """
        data = {
            "connectivity_matrices": [],
            "subject_ids": [f"SUB_{i:04d}" for i in range(num_subjects)]
        }
        
        # Generate connectivity matrices
        for i in range(num_subjects):
            # Add some inter-subject variability
            subject_density = np.random.uniform(0.05, 0.15)
            subject_noise = np.random.uniform(0.05, 0.15)
            
            matrix = self.generate_connectivity_matrix(
                num_nodes, subject_density, subject_noise
            )
            data["connectivity_matrices"].append(matrix)
        
        data["connectivity_matrices"] = np.array(data["connectivity_matrices"])
        
        # Add demographics if requested
        if include_demographics:
            data["demographics"] = {
                "age": np.random.uniform(18, 80, num_subjects),
                "sex": np.random.choice([0, 1], num_subjects),  # 0: F, 1: M
                "handedness": np.random.choice([0, 1], num_subjects, p=[0.1, 0.9]),  # 0: L, 1: R
                "education": np.random.uniform(8, 20, num_subjects)
            }
        
        # Add cognitive scores if requested
        if include_cognitive_scores:
            data["cognitive_scores"] = {
                "fluid_intelligence": np.random.normal(100, 15, num_subjects),
                "processing_speed": np.random.normal(100, 15, num_subjects),
                "working_memory": np.random.normal(100, 15, num_subjects),
                "episodic_memory": np.random.normal(100, 15, num_subjects)
            }
        
        return data
    
    def generate_hierarchical_connectome(
        self,
        level_sizes: List[int] = [64, 16, 4],
        connection_prob: float = 0.8
    ) -> Data:
        """Generate a hierarchical connectome graph.
        
        Args:
            level_sizes: Number of nodes at each hierarchy level
            connection_prob: Probability of connections between levels
            
        Returns:
            PyTorch Geometric Data object with hierarchy information
        """
        total_nodes = sum(level_sizes)
        
        # Create hierarchy level labels
        hierarchy_labels = []
        node_offset = 0
        level_offsets = [0]
        
        for level, size in enumerate(level_sizes):
            hierarchy_labels.extend([level] * size)
            node_offset += size
            level_offsets.append(node_offset)
        
        # Generate edges between hierarchy levels
        edges = []
        
        # Connect each level to the next
        for level in range(len(level_sizes) - 1):
            current_start = level_offsets[level]
            current_end = level_offsets[level + 1]
            next_start = level_offsets[level + 1]
            next_end = level_offsets[level + 2]
            
            # Group nodes at current level
            nodes_per_group = level_sizes[level] // level_sizes[level + 1]
            
            for group in range(level_sizes[level + 1]):
                group_start = current_start + group * nodes_per_group
                group_end = min(current_start + (group + 1) * nodes_per_group, current_end)
                target_node = next_start + group
                
                # Connect all nodes in group to target
                for node in range(group_start, group_end):
                    if np.random.random() < connection_prob:
                        edges.append([node, target_node])
                        edges.append([target_node, node])  # Bidirectional
        
        # Add some within-level connections
        for level in range(len(level_sizes)):
            start_idx = level_offsets[level]
            end_idx = level_offsets[level + 1]
            
            for i in range(start_idx, end_idx):
                for j in range(i + 1, end_idx):
                    if np.random.random() < 0.1:  # Lower probability for within-level
                        edges.append([i, j])
                        edges.append([j, i])
        
        # Convert to tensor
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Generate node features
        x = torch.randn(total_nodes, 32)
        
        # Generate edge attributes
        if edge_index.size(1) > 0:
            edge_attr = torch.randn(edge_index.size(1), 8)
        else:
            edge_attr = torch.empty((0, 8))
        
        # Create data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.hierarchy_level = torch.tensor(hierarchy_labels, dtype=torch.long)
        data.level_sizes = level_sizes
        
        return data
    
    def generate_temporal_connectome(
        self,
        num_nodes: int = 100,
        num_timepoints: int = 200,
        window_size: int = 50,
        overlap: int = 25
    ) -> List[Data]:
        """Generate temporal connectome data.
        
        Args:
            num_nodes: Number of brain regions
            num_timepoints: Total number of time points
            window_size: Size of sliding window
            overlap: Overlap between windows
            
        Returns:
            List of Data objects representing temporal graphs
        """
        # Generate time series data
        timeseries = np.random.randn(num_timepoints, num_nodes)
        
        # Add some temporal structure
        for t in range(1, num_timepoints):
            # AR(1) process with noise
            timeseries[t] = 0.7 * timeseries[t-1] + 0.3 * np.random.randn(num_nodes)
        
        # Create sliding windows
        step_size = window_size - overlap
        num_windows = (num_timepoints - window_size) // step_size + 1
        
        temporal_graphs = []
        
        for w in range(num_windows):
            start_idx = w * step_size
            end_idx = start_idx + window_size
            
            # Extract window data
            window_data = timeseries[start_idx:end_idx]
            
            # Compute connectivity (correlation)
            connectivity = np.corrcoef(window_data.T)
            np.fill_diagonal(connectivity, 0)
            
            # Threshold to create edges
            threshold = np.percentile(np.abs(connectivity), 80)
            edge_mask = np.abs(connectivity) > threshold
            
            # Create edge index
            edge_indices = np.where(edge_mask)
            edge_index = torch.tensor(np.array(edge_indices), dtype=torch.long)
            
            # Edge weights
            edge_weights = torch.tensor(
                connectivity[edge_indices], dtype=torch.float
            ).unsqueeze(1)
            
            # Node features (mean activity in window)
            x = torch.tensor(
                np.mean(window_data, axis=0), dtype=torch.float
            ).unsqueeze(1)
            
            # Create data object
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_weights)
            data.timepoint = w
            temporal_graphs.append(data)
        
        return temporal_graphs
    
    def generate_multi_modal_data(
        self,
        num_subjects: int = 50,
        num_nodes: int = 200,
        modalities: List[str] = ["structural", "functional"]
    ) -> Dict[str, np.ndarray]:
        """Generate multi-modal connectome data.
        
        Args:
            num_subjects: Number of subjects
            num_nodes: Number of brain regions
            modalities: List of modality names
            
        Returns:
            Dictionary with modality data
        """
        data = {}
        
        for modality in modalities:
            matrices = []
            
            for subject in range(num_subjects):
                if modality == "structural":
                    # Structural: sparser, more stable
                    matrix = self.generate_connectivity_matrix(
                        num_nodes, density=0.05, noise_level=0.05
                    )
                elif modality == "functional":
                    # Functional: denser, more variable
                    matrix = self.generate_connectivity_matrix(
                        num_nodes, density=0.15, noise_level=0.1
                    )
                else:
                    # Default case
                    matrix = self.generate_connectivity_matrix(num_nodes)
                
                matrices.append(matrix)
            
            data[modality] = np.array(matrices)
        
        return data


class BrainAtlasGenerator:
    """Generate realistic brain atlas data for testing."""
    
    def __init__(self, random_seed: int = 42):
        """Initialize atlas generator."""
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def generate_parcellation(
        self,
        num_regions: int = 400,
        coordinate_system: str = "MNI"
    ) -> Dict:
        """Generate brain parcellation data.
        
        Args:
            num_regions: Number of brain regions
            coordinate_system: Coordinate system (MNI, Talairach, etc.)
            
        Returns:
            Dictionary with parcellation information
        """
        # Generate 3D coordinates
        if coordinate_system == "MNI":
            # MNI space roughly: x[-90, 90], y[-126, 72], z[-72, 108]
            x_coords = np.random.uniform(-90, 90, num_regions)
            y_coords = np.random.uniform(-126, 72, num_regions)
            z_coords = np.random.uniform(-72, 108, num_regions)
        else:
            # Generic coordinate system
            x_coords = np.random.uniform(-100, 100, num_regions)
            y_coords = np.random.uniform(-100, 100, num_regions)  
            z_coords = np.random.uniform(-100, 100, num_regions)
        
        coordinates = np.column_stack([x_coords, y_coords, z_coords])
        
        # Generate region names
        region_names = [f"Region_{i:03d}" for i in range(num_regions)]
        
        # Generate network assignments
        network_names = [
            "Visual", "Somatomotor", "DorsalAttention",
            "VentralAttention", "Limbic", "Frontoparietal", "Default"
        ]
        
        regions_per_network = num_regions // len(network_names)
        networks = {}
        
        for i, network in enumerate(network_names):
            start_idx = i * regions_per_network
            if i == len(network_names) - 1:  # Last network gets remaining regions
                end_idx = num_regions
            else:
                end_idx = (i + 1) * regions_per_network
            
            networks[network] = list(range(start_idx, end_idx))
        
        return {
            "name": f"TestAtlas_{num_regions}",
            "num_regions": num_regions,
            "regions": region_names,
            "coordinates": coordinates.astype(np.float32),
            "networks": networks,
            "coordinate_system": coordinate_system
        }
    
    def generate_distance_matrix(self, coordinates: np.ndarray) -> np.ndarray:
        """Generate Euclidean distance matrix from coordinates.
        
        Args:
            coordinates: Array of 3D coordinates [N, 3]
            
        Returns:
            Distance matrix [N, N]
        """
        num_regions = coordinates.shape[0]
        distances = np.zeros((num_regions, num_regions))
        
        for i in range(num_regions):
            for j in range(i+1, num_regions):
                dist = np.linalg.norm(coordinates[i] - coordinates[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances


class ClinicalDataGenerator:
    """Generate realistic clinical data for testing."""
    
    def __init__(self, random_seed: int = 42):
        """Initialize clinical data generator."""
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def generate_patient_cohort(
        self,
        num_patients: int = 100,
        num_controls: int = 100,
        condition: str = "autism"
    ) -> Dict:
        """Generate clinical cohort data.
        
        Args:
            num_patients: Number of patients
            num_controls: Number of healthy controls
            condition: Clinical condition
            
        Returns:
            Dictionary with clinical data
        """
        total_subjects = num_patients + num_controls
        
        # Generate demographics
        if condition == "autism":
            # Autism: more males, younger age
            patient_ages = np.random.normal(12, 4, num_patients)
            control_ages = np.random.normal(12, 4, num_controls)
            patient_sex = np.random.choice([0, 1], num_patients, p=[0.2, 0.8])  # More males
            control_sex = np.random.choice([0, 1], num_controls, p=[0.5, 0.5])
        elif condition == "alzheimers":
            # Alzheimer's: older age
            patient_ages = np.random.normal(75, 8, num_patients)
            control_ages = np.random.normal(73, 8, num_controls)
            patient_sex = np.random.choice([0, 1], num_patients, p=[0.6, 0.4])  # More females
            control_sex = np.random.choice([0, 1], num_controls, p=[0.5, 0.5])
        else:
            # Generic condition
            patient_ages = np.random.normal(40, 15, num_patients)
            control_ages = np.random.normal(40, 15, num_controls)
            patient_sex = np.random.choice([0, 1], num_patients)
            control_sex = np.random.choice([0, 1], num_controls)
        
        # Combine data
        ages = np.concatenate([patient_ages, control_ages])
        sexes = np.concatenate([patient_sex, control_sex])
        labels = np.concatenate([np.ones(num_patients), np.zeros(num_controls)])
        
        # Generate clinical scores
        if condition == "autism":
            # ADOS scores (higher = more severe)
            patient_scores = np.random.uniform(8, 22, num_patients)
            control_scores = np.random.uniform(0, 3, num_controls)
        elif condition == "alzheimers":
            # MMSE scores (lower = more impaired)
            patient_scores = np.random.uniform(10, 24, num_patients)
            control_scores = np.random.uniform(26, 30, num_controls)
        else:
            # Generic severity scores
            patient_scores = np.random.uniform(5, 15, num_patients)
            control_scores = np.random.uniform(0, 3, num_controls)
        
        clinical_scores = np.concatenate([patient_scores, control_scores])
        
        return {
            "subject_ids": [f"{condition.upper()}_{i:04d}" for i in range(total_subjects)],
            "ages": ages,
            "sexes": sexes,
            "labels": labels.astype(int),
            "clinical_scores": clinical_scores,
            "condition": condition,
            "num_patients": num_patients,
            "num_controls": num_controls
        }
    
    def generate_medication_data(self, num_subjects: int) -> Dict:
        """Generate medication data.
        
        Args:
            num_subjects: Number of subjects
            
        Returns:
            Dictionary with medication information
        """
        medications = [
            "none", "ssri", "antipsychotic", "stimulant", 
            "mood_stabilizer", "anxiolytic"
        ]
        
        # Most subjects on no medication
        med_assignments = np.random.choice(
            medications, 
            num_subjects, 
            p=[0.6, 0.15, 0.1, 0.05, 0.05, 0.05]
        )
        
        return {
            "medications": med_assignments,
            "medication_types": medications
        }