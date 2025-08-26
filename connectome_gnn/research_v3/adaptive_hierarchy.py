"""Adaptive Hierarchical GNN with Dynamic Structure Learning.

Implements adaptive hierarchical graph neural networks that can learn
and adapt their hierarchical structure based on data patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import math
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class HierarchyStructure:
    """Container for learned hierarchical structure."""
    levels: List[List[int]]  # Node indices at each level
    connections: Dict[int, List[int]]  # Inter-level connections
    importance_scores: Dict[int, float]  # Node importance scores
    cluster_assignments: Dict[int, int]  # Node to cluster mapping
    adaptation_history: List[Dict[str, Any]]  # History of adaptations


class AdaptiveClusteringLayer(nn.Module):
    """Adaptive clustering layer for hierarchical structure learning."""
    
    def __init__(self, input_dim: int, num_clusters: int, temperature: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.num_clusters = num_clusters
        self.temperature = temperature
        
        # Cluster prototypes
        self.cluster_centers = nn.Parameter(torch.randn(num_clusters, input_dim))
        
        # Attention mechanism for cluster assignment
        self.assignment_network = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_clusters)
        )
        
        # Cluster update mechanism
        self.update_network = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through adaptive clustering layer."""
        batch_size, num_nodes, _ = x.shape
        
        # Compute cluster assignment probabilities
        assignment_logits = self.assignment_network(x)  # [batch, nodes, clusters]
        assignment_probs = F.softmax(assignment_logits / self.temperature, dim=-1)
        
        # Soft clustering
        clustered_features = []
        cluster_assignments = []
        
        for batch_idx in range(batch_size):
            batch_x = x[batch_idx]  # [nodes, features]
            batch_probs = assignment_probs[batch_idx]  # [nodes, clusters]
            
            # Weighted cluster features
            cluster_features = torch.matmul(batch_probs.T, batch_x)  # [clusters, features]
            
            # Update cluster centers using learned updates
            for cluster_idx in range(self.num_clusters):
                old_center = self.cluster_centers[cluster_idx]
                cluster_feature = cluster_features[cluster_idx]
                
                # Combine old center with new cluster feature
                combined = torch.cat([old_center, cluster_feature], dim=0)
                updated_center = self.update_network(combined)
                
                # Soft update of cluster center
                momentum = 0.9
                self.cluster_centers.data[cluster_idx] = (
                    momentum * old_center + (1 - momentum) * updated_center
                )
            
            clustered_features.append(cluster_features)
            cluster_assignments.append(batch_probs)
        
        # Stack results
        clustered_features = torch.stack(clustered_features, dim=0)  # [batch, clusters, features]
        cluster_assignments = torch.stack(cluster_assignments, dim=0)  # [batch, nodes, clusters]
        
        return clustered_features, cluster_assignments, assignment_logits
    
    def get_hard_assignments(self, assignment_probs: torch.Tensor) -> torch.Tensor:
        """Convert soft assignments to hard assignments."""
        return torch.argmax(assignment_probs, dim=-1)


class HierarchicalStructureLearner(nn.Module):
    """Learns hierarchical structure adaptively."""
    
    def __init__(
        self,
        input_dim: int,
        max_levels: int = 4,
        base_clusters: int = 8,
        adaptation_rate: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.max_levels = max_levels
        self.base_clusters = base_clusters
        self.adaptation_rate = adaptation_rate
        
        # Clustering layers for each level
        self.clustering_layers = nn.ModuleList()
        current_clusters = base_clusters
        
        for level in range(max_levels):
            self.clustering_layers.append(
                AdaptiveClusteringLayer(input_dim, current_clusters)
            )
            current_clusters = max(2, current_clusters // 2)  # Reduce clusters at higher levels
        
        # Structure evaluation network
        self.structure_evaluator = nn.Sequential(
            nn.Linear(input_dim + max_levels, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Learned hierarchy structure
        self.current_structure = HierarchyStructure(
            levels=[],
            connections={},
            importance_scores={},
            cluster_assignments={},
            adaptation_history=[]
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], HierarchyStructure]:
        """Learn hierarchical structure and return representations at each level."""
        level_representations = []
        level_assignments = []
        
        current_x = x
        
        # Process through each hierarchical level
        for level, clustering_layer in enumerate(self.clustering_layers):
            # Apply clustering
            clustered_x, assignments, _ = clustering_layer(current_x)
            
            level_representations.append(clustered_x)
            level_assignments.append(assignments)
            
            # Prepare input for next level
            current_x = clustered_x
        
        # Update hierarchy structure
        self._update_hierarchy_structure(level_assignments, x)
        
        return level_representations, self.current_structure
    
    def _update_hierarchy_structure(
        self,
        level_assignments: List[torch.Tensor],
        original_features: torch.Tensor
    ):
        """Update the learned hierarchical structure."""
        # Extract structure information from assignments
        batch_size = original_features.shape[0]
        
        # For simplicity, work with first batch
        if batch_size > 0:
            # Update cluster assignments
            for level, assignments in enumerate(level_assignments):
                hard_assignments = self.clustering_layers[level].get_hard_assignments(assignments[0])
                
                # Update structure
                if level < len(self.current_structure.levels):
                    # Update existing level
                    for node_idx, cluster_idx in enumerate(hard_assignments):
                        self.current_structure.cluster_assignments[f"{level}_{node_idx}"] = cluster_idx.item()
                else:
                    # Add new level
                    self.current_structure.levels.append([])
            
            # Calculate importance scores based on attention weights
            self._calculate_importance_scores(level_assignments, original_features[0])
    
    def _calculate_importance_scores(
        self,
        level_assignments: List[torch.Tensor],
        features: torch.Tensor
    ):
        """Calculate node importance scores."""
        num_nodes = features.shape[0]
        
        for node_idx in range(min(num_nodes, features.shape[0])):
            # Base importance on feature magnitude and clustering consistency
            feature_importance = torch.norm(features[node_idx]).item()
            
            # Clustering consistency across levels
            consistency_score = 0.0
            for level, assignments in enumerate(level_assignments):
                # Check bounds before accessing
                if node_idx < assignments.shape[1]:
                    assignment_entropy = -torch.sum(
                        assignments[0, node_idx] * torch.log(assignments[0, node_idx] + 1e-8)
                    ).item()
                    consistency_score += (1.0 - assignment_entropy / math.log(assignments.shape[-1]))
            
            if len(level_assignments) > 0:
                consistency_score /= len(level_assignments)
            else:
                consistency_score = 1.0
            
            # Combined importance score
            importance = feature_importance * consistency_score
            self.current_structure.importance_scores[node_idx] = importance
    
    def adapt_structure(self, performance_feedback: Dict[str, float]):
        """Adapt hierarchical structure based on performance feedback."""
        # Record adaptation
        adaptation_record = {
            'timestamp': torch.tensor(0.0),  # Would be actual timestamp in practice
            'feedback': performance_feedback.copy(),
            'structure_changes': []
        }
        
        # Adapt clustering temperatures based on feedback
        performance_score = performance_feedback.get('accuracy', 0.0)
        
        if performance_score < 0.7:  # Poor performance
            # Increase exploration by lowering temperature
            for layer in self.clustering_layers:
                layer.temperature = max(0.1, layer.temperature * 0.9)
                adaptation_record['structure_changes'].append(
                    f"Decreased temperature to {layer.temperature:.3f}"
                )
        elif performance_score > 0.9:  # Good performance
            # Increase exploitation by raising temperature
            for layer in self.clustering_layers:
                layer.temperature = min(2.0, layer.temperature * 1.1)
                adaptation_record['structure_changes'].append(
                    f"Increased temperature to {layer.temperature:.3f}"
                )
        
        # Record adaptation
        self.current_structure.adaptation_history.append(adaptation_record)
        
        # Keep history manageable
        if len(self.current_structure.adaptation_history) > 100:
            self.current_structure.adaptation_history = (
                self.current_structure.adaptation_history[-100:]
            )


class AdaptiveHierarchicalConvolution(MessagePassing):
    """Hierarchical graph convolution with adaptive structure."""
    
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4):
        super().__init__(aggr='add')
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        
        # Multi-level attention
        self.level_attention = nn.MultiheadAttention(
            in_dim, num_heads, batch_first=True
        )
        
        # Message functions for different hierarchy levels
        self.intra_level_message = nn.Linear(in_dim * 2, out_dim)
        self.inter_level_message = nn.Linear(in_dim * 2, out_dim)
        
        # Update function
        self.update_function = nn.Sequential(
            nn.Linear(in_dim + out_dim * 2, out_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Hierarchy-aware normalization
        self.hierarchy_norm = nn.LayerNorm(out_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        hierarchy_structure: HierarchyStructure,
        level: int = 0
    ) -> torch.Tensor:
        """Forward pass with hierarchy-aware message passing."""
        
        # Intra-level message passing
        intra_messages = self.propagate(edge_index, x=x, message_type='intra')
        
        # Inter-level message passing (if hierarchy structure available)
        inter_messages = self._inter_level_messaging(x, hierarchy_structure, level)
        
        # Combine messages
        combined_messages = intra_messages + inter_messages
        
        # Update node features
        updated_x = self.update_function(torch.cat([x, combined_messages], dim=-1))
        
        # Hierarchy-aware normalization
        updated_x = self.hierarchy_norm(updated_x)
        
        return updated_x
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, message_type: str = 'intra') -> torch.Tensor:
        """Compute messages between nodes."""
        edge_features = torch.cat([x_i, x_j], dim=-1)
        
        if message_type == 'intra':
            return self.intra_level_message(edge_features)
        else:
            return self.inter_level_message(edge_features)
    
    def _inter_level_messaging(
        self,
        x: torch.Tensor,
        hierarchy_structure: HierarchyStructure,
        current_level: int
    ) -> torch.Tensor:
        """Compute inter-level messages based on hierarchy structure."""
        num_nodes = x.shape[0]
        inter_messages = torch.zeros(num_nodes, self.out_dim, device=x.device)
        
        # Simple inter-level messaging based on importance scores
        for node_idx in range(num_nodes):
            if node_idx in hierarchy_structure.importance_scores:
                importance = hierarchy_structure.importance_scores[node_idx]
                
                # Scale messages by importance
                if importance > 0.5:  # High importance nodes
                    # Aggregate from multiple levels
                    inter_message = torch.mean(x, dim=0) * importance
                    inter_messages[node_idx] = self.inter_level_message(
                        torch.cat([x[node_idx], inter_message], dim=0)
                    )
        
        return inter_messages


class AdaptiveHierarchyGNN(nn.Module):
    """Adaptive hierarchical GNN with dynamic structure learning."""
    
    def __init__(
        self,
        node_features: int = 100,
        hidden_dim: int = 256,
        num_layers: int = 4,
        max_hierarchy_levels: int = 3,
        num_classes: int = 1,
        adaptation_enabled: bool = True
    ):
        super().__init__()
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_hierarchy_levels = max_hierarchy_levels
        self.num_classes = num_classes
        self.adaptation_enabled = adaptation_enabled
        
        # Input projection
        self.input_projection = nn.Linear(node_features, hidden_dim)
        
        # Hierarchical structure learner
        if adaptation_enabled:
            self.structure_learner = HierarchicalStructureLearner(
                input_dim=hidden_dim,
                max_levels=max_hierarchy_levels,
                base_clusters=8
            )
        
        # Adaptive hierarchical convolution layers
        self.conv_layers = nn.ModuleList([
            AdaptiveHierarchicalConvolution(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Inter-level fusion
        self.level_fusion = nn.MultiheadAttention(
            hidden_dim, num_heads=8, batch_first=True
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Adaptation tracking
        self.performance_history = []
        
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass through adaptive hierarchical GNN."""
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, 'batch', None)
        
        # Handle feature dimension mismatch
        if x.size(1) != self.node_features:
            if x.size(1) < self.node_features:
                padding = torch.zeros(x.size(0), self.node_features - x.size(1), device=x.device)
                x = torch.cat([x, padding], dim=1)
            else:
                x = x[:, :self.node_features]
        
        # Input projection
        x = self.input_projection(x)
        x = F.relu(x)
        
        # Learn hierarchical structure
        hierarchy_structure = None
        level_representations = []
        
        if self.adaptation_enabled:
            level_reprs, hierarchy_structure = self.structure_learner(x.unsqueeze(0))
            level_representations = [repr.squeeze(0) for repr in level_reprs]
        
        # Process through hierarchical convolution layers
        for layer_idx, conv_layer in enumerate(self.conv_layers):
            level = min(layer_idx, self.max_hierarchy_levels - 1)
            
            x = conv_layer(
                x, edge_index, 
                hierarchy_structure or self._default_hierarchy_structure(x.shape[0]),
                level=level
            )
            x = F.relu(x)
        
        # Fuse information from different hierarchy levels
        if level_representations:
            # Prepare for attention fusion
            all_representations = [x] + level_representations
            
            # Pad representations to same size
            max_size = max(repr.shape[0] for repr in all_representations)
            padded_representations = []
            
            for repr in all_representations:
                if repr.shape[0] < max_size:
                    padding = torch.zeros(
                        max_size - repr.shape[0], 
                        repr.shape[1], 
                        device=repr.device
                    )
                    padded_repr = torch.cat([repr, padding], dim=0)
                else:
                    padded_repr = repr[:max_size]
                
                padded_representations.append(padded_repr)
            
            # Stack and fuse
            if padded_representations:
                stacked_reprs = torch.stack(padded_representations, dim=0)
                fused_repr, _ = self.level_fusion(
                    stacked_reprs, stacked_reprs, stacked_reprs
                )
                x = fused_repr[0]  # Use first (original) level as base
        
        # Global pooling
        if batch is None:
            graph_embedding = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        else:
            graph_embedding = global_mean_pool(x, batch)
        
        # Final prediction
        output = self.output_layers(graph_embedding)
        
        return output.squeeze(-1) if self.num_classes == 1 else output
    
    def _default_hierarchy_structure(self, num_nodes: int) -> HierarchyStructure:
        """Create default hierarchy structure when adaptation is disabled."""
        return HierarchyStructure(
            levels=[[i for i in range(num_nodes)]],
            connections={},
            importance_scores={i: 1.0 for i in range(num_nodes)},
            cluster_assignments={},
            adaptation_history=[]
        )
    
    def adapt_to_performance(self, performance_metrics: Dict[str, float]):
        """Adapt hierarchical structure based on performance feedback."""
        if not self.adaptation_enabled:
            return
        
        self.performance_history.append(performance_metrics)
        
        # Adapt structure learner
        if hasattr(self, 'structure_learner'):
            self.structure_learner.adapt_structure(performance_metrics)
        
        # Keep performance history manageable
        if len(self.performance_history) > 50:
            self.performance_history = self.performance_history[-50:]
    
    def get_hierarchy_info(self) -> Dict[str, Any]:
        """Get information about current hierarchical structure."""
        if not self.adaptation_enabled or not hasattr(self, 'structure_learner'):
            return {'adaptation_enabled': False}
        
        structure = self.structure_learner.current_structure
        
        return {
            'adaptation_enabled': True,
            'num_levels': len(structure.levels),
            'total_adaptations': len(structure.adaptation_history),
            'cluster_assignments_count': len(structure.cluster_assignments),
            'importance_scores_count': len(structure.importance_scores),
            'recent_adaptations': structure.adaptation_history[-5:],  # Last 5 adaptations
            'performance_history': self.performance_history[-10:],  # Last 10 performance records
            'clustering_temperatures': [
                layer.temperature for layer in self.structure_learner.clustering_layers
            ]
        }


def create_adaptive_hierarchy_model(config: Dict[str, Any]) -> AdaptiveHierarchyGNN:
    """Factory function for creating adaptive hierarchical GNN models."""
    return AdaptiveHierarchyGNN(
        node_features=config.get('node_features', 100),
        hidden_dim=config.get('hidden_dim', 256),
        num_layers=config.get('num_layers', 4),
        max_hierarchy_levels=config.get('max_hierarchy_levels', 3),
        num_classes=config.get('num_classes', 1),
        adaptation_enabled=config.get('adaptation_enabled', True)
    )