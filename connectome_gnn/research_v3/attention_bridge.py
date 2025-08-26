"""Attention Bridge GNN for Multi-Modal Brain Network Integration.

Implements advanced attention mechanisms for bridging different brain network
modalities and scales, with focus on cross-modal information fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from torch_geometric.data import Data
import math
from typing import Optional, Tuple, List, Dict, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class AttentionMap:
    """Container for attention visualization and analysis."""
    attention_weights: torch.Tensor
    source_modality: str
    target_modality: str
    layer_index: int
    head_index: int


class MultiModalCrossAttention(nn.Module):
    """Cross-attention mechanism for multi-modal brain network fusion."""
    
    def __init__(
        self, 
        structural_dim: int,
        functional_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        temperature: float = 1.0
    ):
        super().__init__()
        self.structural_dim = structural_dim
        self.functional_dim = functional_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.temperature = temperature
        self.dropout = dropout
        
        # Projection layers for each modality
        self.struct_proj_q = nn.Linear(structural_dim, hidden_dim)
        self.struct_proj_k = nn.Linear(structural_dim, hidden_dim) 
        self.struct_proj_v = nn.Linear(structural_dim, hidden_dim)
        
        self.func_proj_q = nn.Linear(functional_dim, hidden_dim)
        self.func_proj_k = nn.Linear(functional_dim, hidden_dim)
        self.func_proj_v = nn.Linear(functional_dim, hidden_dim)
        
        # Cross-modal attention projections
        self.cross_struct_to_func = nn.Linear(hidden_dim, hidden_dim)
        self.cross_func_to_struct = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projections
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Learnable modal importance weights
        self.modal_weights = nn.Parameter(torch.ones(2))
        
    def forward(
        self, 
        structural_features: torch.Tensor,
        functional_features: torch.Tensor,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[AttentionMap]]]:
        """Forward pass through multi-modal cross-attention."""
        
        batch_size, num_nodes_s = structural_features.shape[:2]
        num_nodes_f = functional_features.shape[1]
        
        # Project to common embedding space
        struct_q = self.struct_proj_q(structural_features)  # [batch, nodes, hidden]
        struct_k = self.struct_proj_k(structural_features)
        struct_v = self.struct_proj_v(structural_features)
        
        func_q = self.func_proj_q(functional_features)
        func_k = self.func_proj_k(functional_features)
        func_v = self.func_proj_v(functional_features)
        
        # Reshape for multi-head attention
        struct_q = struct_q.view(batch_size, num_nodes_s, self.num_heads, self.head_dim).transpose(1, 2)
        struct_k = struct_k.view(batch_size, num_nodes_s, self.num_heads, self.head_dim).transpose(1, 2)
        struct_v = struct_v.view(batch_size, num_nodes_s, self.num_heads, self.head_dim).transpose(1, 2)
        
        func_q = func_q.view(batch_size, num_nodes_f, self.num_heads, self.head_dim).transpose(1, 2)
        func_k = func_k.view(batch_size, num_nodes_f, self.num_heads, self.head_dim).transpose(1, 2)
        func_v = func_v.view(batch_size, num_nodes_f, self.num_heads, self.head_dim).transpose(1, 2)
        
        attention_maps = []
        
        # Cross-modal attention: Structural -> Functional
        struct_to_func_scores = torch.matmul(struct_q, func_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        struct_to_func_attn = F.softmax(struct_to_func_scores / self.temperature, dim=-1)
        struct_to_func_out = torch.matmul(struct_to_func_attn, func_v)
        
        if return_attention:
            for head in range(self.num_heads):
                attention_maps.append(AttentionMap(
                    attention_weights=struct_to_func_attn[:, head].detach(),
                    source_modality='structural',
                    target_modality='functional',
                    layer_index=0,
                    head_index=head
                ))
        
        # Cross-modal attention: Functional -> Structural
        func_to_struct_scores = torch.matmul(func_q, struct_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        func_to_struct_attn = F.softmax(func_to_struct_scores / self.temperature, dim=-1)
        func_to_struct_out = torch.matmul(func_to_struct_attn, struct_v)
        
        if return_attention:
            for head in range(self.num_heads):
                attention_maps.append(AttentionMap(
                    attention_weights=func_to_struct_attn[:, head].detach(),
                    source_modality='functional',
                    target_modality='structural',
                    layer_index=0,
                    head_index=head
                ))
        
        # Reshape back and combine
        struct_to_func_out = struct_to_func_out.transpose(1, 2).contiguous().view(
            batch_size, num_nodes_s, self.hidden_dim
        )
        func_to_struct_out = func_to_struct_out.transpose(1, 2).contiguous().view(
            batch_size, num_nodes_f, self.hidden_dim
        )
        
        # Apply modal importance weighting
        modal_weights = F.softmax(self.modal_weights, dim=0)
        
        # Since structural and functional may have different node counts, 
        # we need to handle fusion carefully
        if num_nodes_s == num_nodes_f:
            # Direct fusion when node counts match
            combined_features = torch.cat([
                modal_weights[0] * struct_to_func_out,
                modal_weights[1] * func_to_struct_out
            ], dim=-1)
        else:
            # Adaptive pooling to match dimensions
            if num_nodes_s > num_nodes_f:
                # Pool structural to match functional
                pooled_struct = F.adaptive_avg_pool1d(
                    struct_to_func_out.transpose(1, 2), 
                    num_nodes_f
                ).transpose(1, 2)
                combined_features = torch.cat([
                    modal_weights[0] * pooled_struct,
                    modal_weights[1] * func_to_struct_out
                ], dim=-1)
            else:
                # Pool functional to match structural  
                pooled_func = F.adaptive_avg_pool1d(
                    func_to_struct_out.transpose(1, 2),
                    num_nodes_s
                ).transpose(1, 2)
                combined_features = torch.cat([
                    modal_weights[0] * struct_to_func_out,
                    modal_weights[1] * pooled_func
                ], dim=-1)
        
        # Final projection and normalization
        output = self.output_proj(combined_features)
        output = self.layer_norm(output)
        output = self.dropout_layer(output)
        
        if return_attention:
            return output, attention_maps
        return output


class HierarchicalScaleAttention(nn.Module):
    """Hierarchical attention across multiple spatial/temporal scales."""
    
    def __init__(
        self, 
        hidden_dim: int,
        scales: List[int] = [1, 2, 4, 8],
        num_heads: int = 8
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scales = scales
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Scale-specific attention modules
        self.scale_attentions = nn.ModuleList()
        for scale in scales:
            scale_attention = nn.MultiheadAttention(
                hidden_dim, 
                num_heads // len(scales), 
                batch_first=True
            )
            self.scale_attentions.append(scale_attention)
        
        # Scale fusion mechanism
        self.scale_fusion = nn.Sequential(
            nn.Linear(hidden_dim * len(scales), hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Learnable scale importance
        self.scale_importance = nn.Parameter(torch.ones(len(scales)))
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Apply hierarchical scale attention."""
        batch_size, num_nodes, feature_dim = x.shape
        
        scale_outputs = []
        
        for i, (scale, attention_module) in enumerate(zip(self.scales, self.scale_attentions)):
            # Create scale-specific representations
            if scale == 1:
                # Node-level (finest scale)
                scale_input = x
            else:
                # Coarser scales through pooling
                pooled_size = max(1, num_nodes // scale)
                scale_input = F.adaptive_avg_pool1d(
                    x.transpose(1, 2), 
                    pooled_size
                ).transpose(1, 2)
            
            # Apply attention at this scale
            scale_output, _ = attention_module(scale_input, scale_input, scale_input)
            
            # Upsample back to original resolution if needed
            if scale_output.shape[1] != num_nodes:
                scale_output = F.interpolate(
                    scale_output.transpose(1, 2),
                    size=num_nodes,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            
            scale_outputs.append(scale_output)
        
        # Apply learned scale importance
        scale_weights = F.softmax(self.scale_importance, dim=0)
        weighted_scales = [w * output for w, output in zip(scale_weights, scale_outputs)]
        
        # Fuse across scales
        concatenated = torch.cat(weighted_scales, dim=-1)
        fused_output = self.scale_fusion(concatenated)
        
        return fused_output + x  # Residual connection


class AdaptiveRoutingAttention(MessagePassing):
    """Attention-based adaptive routing for graph message passing."""
    
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int,
        num_routing_heads: int = 4,
        routing_iterations: int = 3,
        temperature: float = 1.0
    ):
        super().__init__(aggr='add')
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_routing_heads = num_routing_heads
        self.routing_iterations = routing_iterations
        self.temperature = temperature
        
        # Message transformation
        self.message_transform = nn.Linear(in_dim * 2, out_dim)
        
        # Routing attention
        self.routing_attention = nn.MultiheadAttention(
            out_dim, 
            num_routing_heads, 
            batch_first=True
        )
        
        # Dynamic routing weights
        self.routing_weights = nn.Parameter(torch.randn(num_routing_heads, out_dim))
        
        # Update transformation
        self.update_transform = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptive routing attention."""
        # Initial message passing
        messages = self.propagate(edge_index, x=x)
        
        # Iterative routing refinement
        for iteration in range(self.routing_iterations):
            # Apply routing attention
            attended_messages, routing_weights = self.routing_attention(
                messages.unsqueeze(0), 
                messages.unsqueeze(0), 
                messages.unsqueeze(0)
            )
            messages = attended_messages.squeeze(0)
            
            # Update routing based on agreement
            if iteration < self.routing_iterations - 1:
                # Compute routing agreement (simplified)
                agreement = torch.matmul(messages, messages.T).diagonal()
                routing_boost = F.softmax(agreement / self.temperature, dim=0)
                messages = messages * routing_boost.unsqueeze(1)
        
        # Final update
        output = self.update_transform(torch.cat([x, messages], dim=1))
        
        return output
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        """Compute messages with attention-based routing."""
        edge_features = torch.cat([x_i, x_j], dim=1)
        return self.message_transform(edge_features)


class AttentionBridgeGNN(nn.Module):
    """Attention Bridge GNN for multi-modal brain network analysis."""
    
    def __init__(
        self,
        structural_features: int = 100,
        functional_features: int = 200,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_attention_heads: int = 8,
        scales: List[int] = [1, 2, 4, 8],
        use_cross_modal: bool = True,
        use_hierarchical: bool = True,
        use_adaptive_routing: bool = True,
        output_dim: int = 1
    ):
        super().__init__()
        self.structural_features = structural_features
        self.functional_features = functional_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_cross_modal = use_cross_modal
        self.use_hierarchical = use_hierarchical
        self.use_adaptive_routing = use_adaptive_routing
        
        # Input projections
        self.struct_input_proj = nn.Linear(structural_features, hidden_dim)
        self.func_input_proj = nn.Linear(functional_features, hidden_dim)
        
        # Multi-modal cross-attention layers
        if use_cross_modal:
            self.cross_attention_layers = nn.ModuleList()
            for _ in range(num_layers):
                cross_attention = MultiModalCrossAttention(
                    structural_dim=hidden_dim,
                    functional_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_attention_heads
                )
                self.cross_attention_layers.append(cross_attention)
        
        # Hierarchical scale attention layers
        if use_hierarchical:
            self.hierarchical_layers = nn.ModuleList()
            for _ in range(num_layers):
                hierarchical = HierarchicalScaleAttention(
                    hidden_dim=hidden_dim,
                    scales=scales,
                    num_heads=num_attention_heads
                )
                self.hierarchical_layers.append(hierarchical)
        
        # Adaptive routing layers
        if use_adaptive_routing:
            self.routing_layers = nn.ModuleList()
            for _ in range(num_layers):
                routing = AdaptiveRoutingAttention(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    num_routing_heads=num_attention_heads // 2
                )
                self.routing_layers.append(routing)
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Attention bridge fusion
        self.bridge_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Final output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Attention visualization storage
        self.attention_maps = []
        
    def forward(
        self, 
        data: Data, 
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[AttentionMap]]]:
        """Forward pass through attention bridge GNN."""
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, 'batch', None)
        
        # Split features into structural and functional
        struct_features = x[:, :self.structural_features]
        func_features = x[:, self.structural_features:self.structural_features + self.functional_features]
        
        # Handle dimension mismatches
        if func_features.shape[1] < self.functional_features:
            padding = torch.zeros(
                func_features.shape[0], 
                self.functional_features - func_features.shape[1],
                device=x.device
            )
            func_features = torch.cat([func_features, padding], dim=1)
        
        # Project to hidden dimensions
        struct_h = self.struct_input_proj(struct_features)
        func_h = self.func_input_proj(func_features)
        
        attention_maps = []
        
        # Process through attention layers
        for layer_idx in range(self.num_layers):
            # Store residual connections
            struct_residual = struct_h
            func_residual = func_h
            
            # Multi-modal cross-attention
            if self.use_cross_modal:
                if struct_h.dim() == 2:
                    struct_h = struct_h.unsqueeze(0)
                    func_h = func_h.unsqueeze(0)
                    squeeze_output = True
                else:
                    squeeze_output = False
                
                if return_attention:
                    fused_h, layer_attention_maps = self.cross_attention_layers[layer_idx](
                        struct_h, func_h, return_attention=True
                    )
                    attention_maps.extend(layer_attention_maps)
                else:
                    fused_h = self.cross_attention_layers[layer_idx](struct_h, func_h)
                
                if squeeze_output:
                    fused_h = fused_h.squeeze(0)
            else:
                # Simple concatenation if no cross-attention
                fused_h = torch.cat([struct_h, func_h], dim=-1)
                fused_h = self.bridge_fusion(fused_h)
            
            # Hierarchical scale attention
            if self.use_hierarchical and fused_h.dim() == 2:
                fused_h = fused_h.unsqueeze(0)
                fused_h = self.hierarchical_layers[layer_idx](fused_h, edge_index)
                fused_h = fused_h.squeeze(0)
            
            # Adaptive routing attention
            if self.use_adaptive_routing:
                fused_h = self.routing_layers[layer_idx](fused_h, edge_index)
            
            # Layer normalization and residual connection
            if self.use_cross_modal:
                # For cross-modal, we average the residuals
                combined_residual = (struct_residual + func_residual) / 2
            else:
                combined_residual = fused_h
                
            fused_h = self.layer_norms[layer_idx](fused_h + combined_residual)
            
            # Update both modality representations
            struct_h = fused_h
            func_h = fused_h
        
        # Global pooling
        if batch is None:
            graph_embedding = global_mean_pool(fused_h, torch.zeros(fused_h.size(0), dtype=torch.long, device=x.device))
        else:
            graph_embedding = global_mean_pool(fused_h, batch)
        
        # Final prediction
        output = self.output_layers(graph_embedding)
        
        if return_attention:
            return output.squeeze(-1) if output.size(1) == 1 else output, attention_maps
        return output.squeeze(-1) if output.size(1) == 1 else output
    
    def visualize_attention(self, attention_maps: List[AttentionMap]) -> Dict[str, torch.Tensor]:
        """Create attention visualizations for analysis."""
        visualizations = {}
        
        # Aggregate attention by modality pairs
        for attention_map in attention_maps:
            key = f"{attention_map.source_modality}_to_{attention_map.target_modality}"
            if key not in visualizations:
                visualizations[key] = []
            visualizations[key].append(attention_map.attention_weights)
        
        # Average attention maps
        for key, maps in visualizations.items():
            visualizations[key] = torch.stack(maps, dim=0).mean(dim=0)
        
        return visualizations


def create_attention_bridge_model(config: Dict) -> AttentionBridgeGNN:
    """Factory function for creating attention bridge models."""
    return AttentionBridgeGNN(
        structural_features=config.get('structural_features', 100),
        functional_features=config.get('functional_features', 200),
        hidden_dim=config.get('hidden_dim', 256),
        num_layers=config.get('num_layers', 4),
        num_attention_heads=config.get('num_attention_heads', 8),
        scales=config.get('scales', [1, 2, 4, 8]),
        use_cross_modal=config.get('use_cross_modal', True),
        use_hierarchical=config.get('use_hierarchical', True),
        use_adaptive_routing=config.get('use_adaptive_routing', True),
        output_dim=config.get('output_dim', 1)
    )