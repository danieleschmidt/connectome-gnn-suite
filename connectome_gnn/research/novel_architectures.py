"""Novel GNN architectures for connectome analysis."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
from torch_geometric.data import Data
from typing import Optional, List, Dict, Any, Tuple
import math
import numpy as np

from ..models.base import BaseConnectomeModel


class GraphWaveletGNN(BaseConnectomeModel):
    """Graph Wavelet Neural Network for multi-scale connectome analysis.
    
    Implements graph wavelet transforms to capture multi-scale connectivity
    patterns, inspired by spectral graph theory and wavelets.
    """
    
    def __init__(
        self,
        node_features: int = 100,
        hidden_dim: int = 256,
        output_dim: int = 1,
        num_scales: int = 4,
        wavelet_type: str = "mexican_hat",
        spectral_features: int = 64,
        dropout: float = 0.1
    ):
        """Initialize Graph Wavelet GNN.
        
        Args:
            node_features: Number of input node features
            hidden_dim: Hidden dimension size
            output_dim: Output dimension
            num_scales: Number of wavelet scales
            wavelet_type: Type of wavelet (mexican_hat, gaussian, laplacian)
            spectral_features: Number of spectral features to extract
            dropout: Dropout probability
        """
        super().__init__(node_features, hidden_dim, output_dim, dropout)
        
        self.num_scales = num_scales
        self.wavelet_type = wavelet_type
        self.spectral_features = spectral_features
        
        # Input projection
        self.input_projection = nn.Linear(node_features, hidden_dim)
        
        # Graph wavelet layers for each scale
        self.wavelet_layers = nn.ModuleList([
            GraphWaveletLayer(hidden_dim, hidden_dim, scale=i+1, wavelet_type=wavelet_type)
            for i in range(num_scales)
        ])
        
        # Spectral feature extractor
        self.spectral_processor = SpectralFeatureExtractor(hidden_dim, spectral_features)
        
        # Multi-scale fusion
        self.scale_fusion = MultiScaleFusion(hidden_dim, num_scales)
        
        # Final prediction layers
        final_dim = hidden_dim + spectral_features
        self.prediction_head = nn.Sequential(
            nn.Linear(final_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, output_dim)
        )
        
        # Update model config
        self.model_config.update({
            'num_scales': num_scales,
            'wavelet_type': wavelet_type,
            'spectral_features': spectral_features
        })
    
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass of Graph Wavelet GNN."""
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = getattr(data, 'batch', None)
        
        # Input projection
        h = self.input_projection(x)
        h = F.relu(h)
        h = self.dropout_layer(h)
        
        # Multi-scale wavelet processing
        scale_representations = []
        
        for scale_idx, wavelet_layer in enumerate(self.wavelet_layers):
            scale_h = wavelet_layer(h, edge_index, edge_attr)
            scale_representations.append(scale_h)
        
        # Fuse multi-scale representations
        fused_h = self.scale_fusion(scale_representations)
        
        # Extract spectral features
        spectral_features = self.spectral_processor(fused_h, edge_index)
        
        # Global pooling for graph-level prediction
        if batch is not None:
            # Batch processing
            graph_features = global_mean_pool(fused_h, batch)
            spectral_global = global_mean_pool(spectral_features, batch)
        else:
            # Single graph
            graph_features = torch.mean(fused_h, dim=0, keepdim=True)
            spectral_global = torch.mean(spectral_features, dim=0, keepdim=True)
        
        # Combine spatial and spectral features
        combined_features = torch.cat([graph_features, spectral_global], dim=1)
        
        # Final prediction
        predictions = self.prediction_head(combined_features)
        
        return predictions


class GraphWaveletLayer(MessagePassing):
    """Graph wavelet convolution layer."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: int = 1,
        wavelet_type: str = "mexican_hat"
    ):
        super().__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = scale
        self.wavelet_type = wavelet_type
        
        # Learnable wavelet parameters
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels))
        self.scale_param = nn.Parameter(torch.tensor(float(scale)))
        
        # Wavelet-specific parameters
        if wavelet_type == "mexican_hat":
            self.wavelet_param = nn.Parameter(torch.tensor(1.0))
        elif wavelet_type == "gaussian":
            self.wavelet_param = nn.Parameter(torch.tensor(1.0))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x, edge_index, edge_attr=None):
        """Forward pass of wavelet layer."""
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_j, edge_attr=None):
        """Compute wavelet-based messages."""
        # Apply wavelet transformation
        wavelet_weight = self._compute_wavelet_weight(edge_attr)
        
        # Transform features
        transformed_features = torch.matmul(x_j, self.weight)
        
        # Apply wavelet weighting
        return wavelet_weight.unsqueeze(-1) * transformed_features
    
    def _compute_wavelet_weight(self, edge_attr):
        """Compute wavelet weights based on edge attributes."""
        if edge_attr is None:
            # Default uniform weights
            return torch.ones(edge_attr.size(0) if edge_attr is not None else 1, device=self.weight.device)
        
        # Normalize edge weights for wavelet computation
        edge_weights = edge_attr.squeeze()
        
        if self.wavelet_type == "mexican_hat":
            # Mexican hat wavelet: (1 - x²) * exp(-x²/2)
            x = edge_weights * self.scale_param
            wavelet_values = (1 - x**2) * torch.exp(-x**2 / 2) * self.wavelet_param
            
        elif self.wavelet_type == "gaussian":
            # Gaussian wavelet: exp(-x²/2σ²)
            x = edge_weights * self.scale_param
            wavelet_values = torch.exp(-x**2 / (2 * self.wavelet_param**2))
            
        elif self.wavelet_type == "laplacian":
            # Laplacian-based wavelet
            x = edge_weights * self.scale_param
            wavelet_values = torch.exp(-torch.abs(x) * self.wavelet_param)
            
        else:
            # Default: identity
            wavelet_values = edge_weights
        
        return wavelet_values


class SpectralFeatureExtractor(nn.Module):
    """Extract spectral features from graph Laplacian."""
    
    def __init__(self, node_dim: int, spectral_dim: int):
        super().__init__()
        
        self.node_dim = node_dim
        self.spectral_dim = spectral_dim
        
        # Spectral projection
        self.spectral_projection = nn.Linear(node_dim, spectral_dim)
    
    def forward(self, x, edge_index):
        """Extract spectral features."""
        num_nodes = x.size(0)
        
        # Simplified spectral analysis (in practice, would use eigendecomposition)
        # This is a learnable approximation to spectral features
        
        # Global spectral representation
        spectral_features = self.spectral_projection(x)
        
        # Add positional encoding based on node indices
        position_encoding = self._get_positional_encoding(num_nodes, self.spectral_dim)
        position_encoding = position_encoding.to(x.device)
        
        spectral_features = spectral_features + position_encoding
        
        return spectral_features
    
    def _get_positional_encoding(self, num_nodes, dim):
        """Generate positional encoding for nodes."""
        position = torch.arange(0, num_nodes, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        
        pe = torch.zeros(num_nodes, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe


class MultiScaleFusion(nn.Module):
    """Fuse multi-scale wavelet representations."""
    
    def __init__(self, hidden_dim: int, num_scales: int):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        
        # Learnable scale weights
        self.scale_weights = nn.Parameter(torch.ones(num_scales))
        
        # Scale-specific transformations
        self.scale_transforms = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_scales)
        ])
        
        # Final fusion layer
        self.fusion_layer = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, scale_representations):
        """Fuse representations from different scales."""
        # Apply scale-specific transformations
        transformed_scales = []
        for i, (scale_repr, transform) in enumerate(zip(scale_representations, self.scale_transforms)):
            transformed = F.relu(transform(scale_repr))
            transformed_scales.append(transformed)
        
        # Weighted fusion
        weights = F.softmax(self.scale_weights, dim=0)
        fused = sum(w * scale for w, scale in zip(weights, transformed_scales))
        
        # Final transformation
        output = self.fusion_layer(fused)
        
        return output


class AttentionPoolingGNN(BaseConnectomeModel):
    """GNN with learnable attention-based pooling for hierarchical analysis."""
    
    def __init__(
        self,
        node_features: int = 100,
        hidden_dim: int = 256,
        output_dim: int = 1,
        num_attention_heads: int = 8,
        num_pooling_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__(node_features, hidden_dim, output_dim, dropout)
        
        self.num_attention_heads = num_attention_heads
        self.num_pooling_layers = num_pooling_layers
        
        # Input projection
        self.input_projection = nn.Linear(node_features, hidden_dim)
        
        # Attention pooling layers
        self.attention_pools = nn.ModuleList([
            AttentionPoolingLayer(hidden_dim, num_attention_heads)
            for _ in range(num_pooling_layers)
        ])
        
        # Final prediction
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.model_config.update({
            'num_attention_heads': num_attention_heads,
            'num_pooling_layers': num_pooling_layers
        })
    
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass with attention pooling."""
        x, edge_index = data.x, data.edge_index
        
        # Input projection
        h = F.relu(self.input_projection(x))
        
        # Hierarchical attention pooling
        for pool_layer in self.attention_pools:
            h = pool_layer(h, edge_index)
        
        # Global representation
        graph_repr = torch.mean(h, dim=0, keepdim=True)
        
        # Prediction
        output = self.prediction_head(graph_repr)
        
        return output


class AttentionPoolingLayer(nn.Module):
    """Attention-based pooling layer."""
    
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=False
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
    
    def forward(self, x, edge_index):
        """Apply attention pooling."""
        # Self-attention
        attended, _ = self.attention(x, x, x)
        x = self.layer_norm(x + attended)
        
        # Feedforward
        ff_output = self.feedforward(x)
        x = self.layer_norm(x + ff_output)
        
        return x


class NeuroGraphTransformer(BaseConnectomeModel):
    """Transformer architecture adapted for brain graphs."""
    
    def __init__(
        self,
        node_features: int = 100,
        hidden_dim: int = 256,
        output_dim: int = 1,
        num_transformer_layers: int = 6,
        num_attention_heads: int = 8,
        feedforward_dim: int = 512,
        dropout: float = 0.1,
        use_positional_encoding: bool = True
    ):
        super().__init__(node_features, hidden_dim, output_dim, dropout)
        
        self.num_transformer_layers = num_transformer_layers
        self.num_attention_heads = num_attention_heads
        self.use_positional_encoding = use_positional_encoding
        
        # Input projection
        self.input_projection = nn.Linear(node_features, hidden_dim)
        
        # Positional encoding for brain regions
        if use_positional_encoding:
            self.positional_encoding = PositionalEncoding(hidden_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_attention_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )
        
        # Graph structure attention
        self.graph_attention = GraphStructureAttention(hidden_dim)
        
        # Final prediction
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.model_config.update({
            'num_transformer_layers': num_transformer_layers,
            'num_attention_heads': num_attention_heads,
            'feedforward_dim': feedforward_dim,
            'use_positional_encoding': use_positional_encoding
        })
    
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass of NeuroGraph Transformer."""
        x, edge_index = data.x, data.edge_index
        
        # Input projection
        h = self.input_projection(x)
        
        # Add positional encoding
        if self.use_positional_encoding:
            h = self.positional_encoding(h)
        
        # Prepare for transformer (add batch dimension)
        h = h.unsqueeze(0)  # [1, num_nodes, hidden_dim]
        
        # Apply transformer
        h = self.transformer(h)
        h = h.squeeze(0)  # [num_nodes, hidden_dim]
        
        # Apply graph structure attention
        h = self.graph_attention(h, edge_index)
        
        # Global pooling
        graph_repr = torch.mean(h, dim=0, keepdim=True)
        
        # Prediction
        output = self.prediction_head(graph_repr)
        
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for brain regions."""
    
    def __init__(self, hidden_dim: int, max_regions: int = 1000):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Create positional encoding table
        pe = torch.zeros(max_regions, hidden_dim)
        position = torch.arange(0, max_regions, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * 
                           (-math.log(10000.0) / hidden_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Add positional encoding to input."""
        num_nodes = x.size(0)
        return x + self.pe[:num_nodes]


class GraphStructureAttention(nn.Module):
    """Attention mechanism that incorporates graph structure."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Edge-aware attention
        self.edge_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, edge_index):
        """Apply graph structure-aware attention."""
        num_nodes = x.size(0)
        
        # Compute edge-based attention weights
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]
        
        edge_features = torch.cat([x[source_nodes], x[target_nodes]], dim=1)
        edge_weights = self.edge_attention(edge_features).squeeze()
        
        # Apply attention to node features
        attended_x = x.clone()
        
        for i in range(num_nodes):
            # Find edges involving this node
            incoming_mask = target_nodes == i
            if incoming_mask.any():
                incoming_sources = source_nodes[incoming_mask]
                incoming_weights = edge_weights[incoming_mask]
                
                # Weighted aggregation
                weighted_features = incoming_weights.unsqueeze(1) * x[incoming_sources]
                aggregated = torch.sum(weighted_features, dim=0)
                
                attended_x[i] = attended_x[i] + aggregated
        
        # Layer normalization
        output = self.layer_norm(attended_x)
        
        return output