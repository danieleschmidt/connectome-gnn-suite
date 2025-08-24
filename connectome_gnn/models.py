"""Neural network models for connectome analysis."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
from typing import Optional, List, Union
import math


class BaseConnectomeModel(nn.Module):
    """Base class for all connectome models."""
    
    def __init__(self):
        super().__init__()
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize model parameters."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass - to be implemented by subclasses."""
        raise NotImplementedError
        
    def get_node_embeddings(self, data: Data) -> torch.Tensor:
        """Get node-level embeddings."""
        raise NotImplementedError


class HierarchicalBrainGNN(BaseConnectomeModel):
    """Hierarchical Graph Neural Network respecting brain organization."""
    
    def __init__(
        self,
        node_features: int = 100,
        hidden_dim: int = 256,
        num_levels: int = 4,
        parcellation: str = "AAL",
        attention_type: str = "dense"
    ):
        super().__init__()
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.parcellation = parcellation
        self.attention_type = attention_type
        
        # Input projection
        self.input_proj = nn.Linear(node_features, hidden_dim)
        
        # Hierarchical levels
        self.levels = nn.ModuleList()
        for i in range(num_levels):
            level_dim = hidden_dim // (2 ** i)
            self.levels.append(
                HierarchicalLevel(
                    in_dim=hidden_dim if i == 0 else hidden_dim // (2 ** (i-1)),
                    out_dim=level_dim,
                    attention_type=attention_type
                )
            )
            
        # Output projection
        final_dim = hidden_dim // (2 ** (num_levels - 1))
        self.output_proj = nn.Linear(final_dim, 1)  # For regression tasks
        
    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch if hasattr(data, 'batch') else None
        
        # Input projection
        if x.size(1) != self.node_features:
            # Pad or truncate features to expected size
            if x.size(1) < self.node_features:
                padding = torch.zeros(x.size(0), self.node_features - x.size(1), device=x.device)
                x = torch.cat([x, padding], dim=1)
            else:
                x = x[:, :self.node_features]
                
        x = self.input_proj(x)
        x = F.relu(x)
        
        # Hierarchical processing
        for level in self.levels:
            x = level(x, edge_index)
            
        # Global pooling
        if batch is None:
            # Single graph
            out = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        else:
            # Batch of graphs
            out = global_mean_pool(x, batch)
            
        # Output projection
        out = self.output_proj(out)
        return out.squeeze(-1)
        
    def get_node_embeddings(self, data: Data) -> torch.Tensor:
        """Get node embeddings without global pooling."""
        x, edge_index = data.x, data.edge_index
        
        if x.size(1) != self.node_features:
            if x.size(1) < self.node_features:
                padding = torch.zeros(x.size(0), self.node_features - x.size(1), device=x.device)
                x = torch.cat([x, padding], dim=1)
            else:
                x = x[:, :self.node_features]
                
        x = self.input_proj(x)
        x = F.relu(x)
        
        for level in self.levels:
            x = level(x, edge_index)
            
        return x


class HierarchicalLevel(nn.Module):
    """Single hierarchical level with attention mechanism."""
    
    def __init__(self, in_dim: int, out_dim: int, attention_type: str = "dense"):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attention_type = attention_type
        
        if attention_type == "dense":
            self.conv = GATConv(in_dim, out_dim, heads=4, concat=False)
        else:
            self.conv = GCNConv(in_dim, out_dim)
            
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Graph convolution
        out = self.conv(x, edge_index)
        out = F.relu(out)
        
        # Normalization and dropout
        out = self.norm(out)
        out = self.dropout(out)
        
        # Residual connection if dimensions match
        if x.size(1) == out.size(1):
            out = out + x
            
        return out


class TemporalConnectomeGNN(BaseConnectomeModel):
    """GNN for temporal/dynamic functional connectivity."""
    
    def __init__(
        self,
        node_features: int = 100,
        time_steps: int = 200,
        lstm_hidden: int = 128,
        gnn_hidden: int = 256,
        temporal_attention: bool = True
    ):
        super().__init__()
        self.node_features = node_features
        self.time_steps = time_steps
        self.lstm_hidden = lstm_hidden
        self.gnn_hidden = gnn_hidden
        self.temporal_attention = temporal_attention
        
        # Temporal processing
        self.lstm = nn.LSTM(node_features, lstm_hidden, batch_first=True)
        
        # Spatial processing  
        self.gnn_layers = nn.ModuleList([
            GCNConv(lstm_hidden, gnn_hidden),
            GCNConv(gnn_hidden, gnn_hidden // 2),
        ])
        
        # Output
        self.output = nn.Linear(gnn_hidden // 2, 1)
        
    def forward(self, data: Data) -> torch.Tensor:
        # For now, simulate temporal data from static connectome
        x, edge_index = data.x, data.edge_index
        batch_size = 1
        
        # Simulate temporal dynamics
        temporal_x = x.unsqueeze(0).repeat(1, self.time_steps, 1)
        temporal_x += torch.randn_like(temporal_x) * 0.1
        
        # LSTM processing
        lstm_out, _ = self.lstm(temporal_x)
        x_temporal = lstm_out[:, -1, :]  # Use final timestep
        
        # GNN processing
        for layer in self.gnn_layers:
            x_temporal = layer(x_temporal.squeeze(0), edge_index)
            x_temporal = F.relu(x_temporal)
            
        # Global pooling and output
        out = global_mean_pool(x_temporal, torch.zeros(x_temporal.size(0), dtype=torch.long))
        return self.output(out).squeeze(-1)
        
    def get_node_embeddings(self, data: Data) -> torch.Tensor:
        return self.forward(data)  # Simplified for now


class MultiModalBrainGNN(BaseConnectomeModel):
    """Multi-modal GNN combining structural and functional connectivity."""
    
    def __init__(
        self,
        structural_features: int = 100,
        functional_features: int = 200,
        fusion_method: str = "adaptive",
        shared_encoder: bool = False
    ):
        super().__init__()
        self.structural_features = structural_features
        self.functional_features = functional_features
        self.fusion_method = fusion_method
        self.shared_encoder = shared_encoder
        
        # Separate encoders for each modality
        if not shared_encoder:
            self.structural_encoder = GCNConv(structural_features, 256)
            self.functional_encoder = GCNConv(functional_features, 256)
        else:
            self.shared_encoder_layer = GCNConv(max(structural_features, functional_features), 256)
            
        # Fusion layer
        if fusion_method == "adaptive":
            self.fusion = nn.MultiheadAttention(256, 8, batch_first=True)
        else:
            self.fusion = nn.Linear(512, 256)
            
        self.output = nn.Linear(256, 1)
        
    def forward(self, data: Data) -> torch.Tensor:
        # Split features into structural and functional
        x, edge_index = data.x, data.edge_index
        
        structural_x = x[:, :self.structural_features]
        functional_x = x[:, self.structural_features:self.structural_features + self.functional_features]
        
        # Encode each modality
        if not self.shared_encoder:
            struct_emb = F.relu(self.structural_encoder(structural_x, edge_index))
            func_emb = F.relu(self.functional_encoder(functional_x, edge_index))
        else:
            struct_emb = F.relu(self.shared_encoder_layer(structural_x, edge_index))
            func_emb = F.relu(self.shared_encoder_layer(functional_x, edge_index))
            
        # Fusion
        if self.fusion_method == "adaptive":
            # Use attention for adaptive fusion
            combined = torch.stack([struct_emb, func_emb], dim=1)
            fused, _ = self.fusion(combined, combined, combined)
            fused = fused.mean(dim=1)
        else:
            # Simple concatenation
            fused = torch.cat([struct_emb, func_emb], dim=1)
            fused = self.fusion(fused)
            
        # Global pooling and output
        out = global_mean_pool(fused, torch.zeros(fused.size(0), dtype=torch.long))
        return self.output(out).squeeze(-1)
        
    def get_node_embeddings(self, data: Data) -> torch.Tensor:
        return self.forward(data)


class PopulationGraphGNN(BaseConnectomeModel):
    """GNN for population-level analysis across subjects."""
    
    def __init__(
        self,
        subject_features: int = 1000,
        demographic_features: int = 10,
        similarity_metric: str = "learned",
        k_neighbors: int = 50
    ):
        super().__init__()
        self.subject_features = subject_features
        self.demographic_features = demographic_features
        self.similarity_metric = similarity_metric
        self.k_neighbors = k_neighbors
        
        # Subject encoder
        self.subject_encoder = nn.Sequential(
            nn.Linear(subject_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # Population GNN
        self.population_gnn = GCNConv(256 + demographic_features, 128)
        
        # Output
        self.output = nn.Linear(128, 1)
        
    def forward(self, data: Data) -> torch.Tensor:
        # Encode subject-level features
        x = data.x[:, :self.subject_features]
        demographics = data.x[:, self.subject_features:self.subject_features + self.demographic_features]
        
        subject_emb = self.subject_encoder(x)
        combined_features = torch.cat([subject_emb, demographics], dim=1)
        
        # Population-level processing
        pop_emb = F.relu(self.population_gnn(combined_features, data.edge_index))
        
        # Output
        out = global_mean_pool(pop_emb, torch.zeros(pop_emb.size(0), dtype=torch.long))
        return self.output(out).squeeze(-1)
        
    def get_node_embeddings(self, data: Data) -> torch.Tensor:
        return self.forward(data)


# Export all models
__all__ = [
    "BaseConnectomeModel",
    "HierarchicalBrainGNN",
    "TemporalConnectomeGNN", 
    "MultiModalBrainGNN",
    "PopulationGraphGNN"
]