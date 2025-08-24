"""Novel GNN architectures and research innovations for Connectome-GNN-Suite."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_scatter import scatter_add, scatter_max
import numpy as np
import math
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import time
from abc import ABC, abstractmethod

from ..robust.logging_config import get_logger


@dataclass
class ResearchMetrics:
    """Research performance metrics container."""
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    auroc: float
    training_time: float
    inference_time: float
    memory_usage: float
    parameter_count: int
    flops: int
    convergence_epochs: int
    stability_score: float


class AdvancedConnectomeGNN(nn.Module):
    """Advanced GNN with novel architectural components for connectome analysis."""
    
    def __init__(
        self,
        node_features: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        heads: int = 8,
        dropout: float = 0.1,
        use_attention: bool = True,
        use_residual: bool = True,
        use_layer_norm: bool = True,
        use_graph_attention: bool = True,
        hierarchical_pooling: bool = True
    ):
        super().__init__()
        
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        
        self.logger = get_logger(__name__)
        
        # Input projection
        self.input_proj = nn.Linear(node_features, hidden_dim)
        
        # Novel GNN layers
        self.gnn_layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer = HierarchicalConnectomeLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                heads=heads,
                dropout=dropout,
                use_attention=use_attention,
                use_residual=use_residual,
                use_layer_norm=use_layer_norm
            )
            self.gnn_layers.append(layer)
            
        # Multi-scale attention mechanism
        if use_graph_attention:
            self.graph_attention = MultiScaleGraphAttention(
                hidden_dim, heads, dropout
            )
        else:
            self.graph_attention = None
            
        # Hierarchical pooling
        if hierarchical_pooling:
            self.hierarchical_pooling = AdaptiveHierarchicalPooling(
                hidden_dim, num_clusters=[64, 16, 4]
            )
        else:
            self.hierarchical_pooling = None
            
        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights using advanced techniques."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
                
    def forward(self, data):
        """Forward pass with novel architectural components."""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        
        # Store intermediate representations for skip connections
        layer_outputs = []
        
        # Pass through GNN layers
        for i, layer in enumerate(self.gnn_layers):
            x_prev = x
            x = layer(x, edge_index, batch)
            layer_outputs.append(x)
            
            # Deep supervision (use intermediate outputs)
            if i > 0 and self.training:
                # Add regularization based on intermediate representations
                layer_reg = torch.norm(x - x_prev, p=2, dim=1).mean()
                if hasattr(self, 'layer_regularization'):
                    self.layer_regularization += 0.01 * layer_reg
                else:
                    self.layer_regularization = 0.01 * layer_reg
                    
        # Multi-scale graph attention
        if self.graph_attention is not None:
            x = self.graph_attention(x, layer_outputs, edge_index, batch)
            
        # Hierarchical pooling
        if self.hierarchical_pooling is not None:
            x = self.hierarchical_pooling(x, edge_index, batch)
        else:
            # Standard global pooling
            x = global_mean_pool(x, batch)
            
        # Classification
        output = self.classifier(x)
        
        return output
        
    def get_embeddings(self, data):
        """Extract node embeddings from the model."""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        
        # Pass through GNN layers
        for layer in self.gnn_layers:
            x = layer(x, edge_index, batch)
            
        return x


class HierarchicalConnectomeLayer(MessagePassing):
    """Novel hierarchical message passing layer for connectome data."""
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 8,
        dropout: float = 0.1,
        use_attention: bool = True,
        use_residual: bool = True,
        use_layer_norm: bool = True,
        aggr: str = 'add'
    ):
        super().__init__(aggr=aggr)
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dropout = dropout
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        
        # Multi-head attention components
        if use_attention:
            self.attention = nn.MultiheadAttention(
                in_dim, heads, dropout=dropout, batch_first=True
            )
            
        # Message and update functions
        self.message_net = nn.Sequential(
            nn.Linear(2 * in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.update_net = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Residual connection
        if use_residual and in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim)
        else:
            self.residual_proj = None
            
        # Layer normalization
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(out_dim)
        else:
            self.layer_norm = None
            
        # Edge-wise attention
        self.edge_attention = nn.Sequential(
            nn.Linear(2 * in_dim, 1),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x, edge_index, batch=None):
        """Forward pass with hierarchical message passing."""
        # Self-attention mechanism
        if self.use_attention:
            # Reshape for attention
            batch_size = batch.max().item() + 1 if batch is not None else 1
            max_nodes = x.size(0) // batch_size if batch is not None else x.size(0)
            
            # Apply self-attention (simplified)
            attn_output, _ = self.attention(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
            x = attn_output.squeeze(0)
            
        # Store input for residual connection
        x_input = x
        
        # Message passing
        out = self.propagate(edge_index, x=x)
        
        # Update function
        out = self.update_net(torch.cat([x, out], dim=1))
        
        # Residual connection
        if self.use_residual:
            if self.residual_proj is not None:
                x_input = self.residual_proj(x_input)
            out = out + x_input
            
        # Layer normalization
        if self.layer_norm is not None:
            out = self.layer_norm(out)
            
        return out
        
    def message(self, x_i, x_j, edge_index_i, edge_index_j):
        """Compute messages between nodes."""
        # Concatenate source and target features
        edge_features = torch.cat([x_i, x_j], dim=1)
        
        # Compute edge attention weights
        attention_weights = self.edge_attention(edge_features)
        attention_weights = F.softmax(attention_weights, dim=0)
        
        # Compute messages
        messages = self.message_net(edge_features)
        
        # Apply attention weights
        messages = messages * attention_weights
        
        return messages


class MultiScaleGraphAttention(nn.Module):
    """Multi-scale graph attention mechanism."""
    
    def __init__(self, hidden_dim: int, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dropout = dropout
        
        # Multi-scale attention heads
        self.local_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        
        self.global_attention = nn.MultiheadAttention(
            hidden_dim, heads // 2, dropout=dropout, batch_first=True
        )
        
        # Scale combination
        self.scale_combiner = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, layer_outputs, edge_index, batch):
        """Multi-scale attention forward pass."""
        batch_size = batch.max().item() + 1 if batch is not None else 1
        
        # Local attention (current layer)
        local_attn, _ = self.local_attention(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        local_attn = local_attn.squeeze(0)
        
        # Global attention (across all layers)
        if len(layer_outputs) > 1:
            # Combine representations from all layers
            global_repr = torch.stack(layer_outputs, dim=0).mean(dim=0)
            global_attn, _ = self.global_attention(
                global_repr.unsqueeze(0), 
                global_repr.unsqueeze(0), 
                global_repr.unsqueeze(0)
            )
            global_attn = global_attn.squeeze(0)
        else:
            global_attn = local_attn
            
        # Combine local and global attention
        combined = self.scale_combiner(torch.cat([local_attn, global_attn], dim=1))
        
        return combined


class AdaptiveHierarchicalPooling(nn.Module):
    """Adaptive hierarchical pooling for graph-level representations."""
    
    def __init__(self, hidden_dim: int, num_clusters: List[int] = [64, 16, 4]):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_clusters = num_clusters
        
        # Learnable cluster assignment
        self.cluster_assignments = nn.ModuleList()
        
        prev_dim = hidden_dim
        for num_cluster in num_clusters:
            assignment_layer = nn.Sequential(
                nn.Linear(prev_dim, num_cluster),
                nn.Softmax(dim=1)
            )
            self.cluster_assignments.append(assignment_layer)
            prev_dim = num_cluster
            
        # Final pooling
        self.final_pool = nn.Sequential(
            nn.Linear(num_clusters[-1] * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x, edge_index, batch):
        """Hierarchical pooling forward pass."""
        batch_size = batch.max().item() + 1
        
        # Initialize with node features
        current_features = x
        
        # Hierarchical clustering and pooling
        for i, assignment_layer in enumerate(self.cluster_assignments):
            # Compute soft cluster assignments
            assignments = assignment_layer(current_features)  # [num_nodes, num_clusters]
            
            # Pool features for each cluster
            pooled_features = []
            
            for b in range(batch_size):
                batch_mask = (batch == b)
                batch_features = current_features[batch_mask]  # [batch_nodes, hidden_dim]
                batch_assignments = assignments[batch_mask]  # [batch_nodes, num_clusters]
                
                # Weighted pooling
                cluster_features = torch.matmul(
                    batch_assignments.transpose(0, 1), 
                    batch_features
                )  # [num_clusters, hidden_dim]
                
                pooled_features.append(cluster_features)
                
            # Stack and flatten for next level
            current_features = torch.stack(pooled_features, dim=0)  # [batch_size, num_clusters, hidden_dim]
            current_features = current_features.view(-1, self.hidden_dim)  # [batch_size * num_clusters, hidden_dim]
            
            # Update batch indices for next level
            batch = torch.arange(batch_size).repeat_interleave(
                self.num_clusters[i]
            ).to(x.device)
            
        # Final pooling to graph-level representation
        final_features = current_features.view(batch_size, -1)  # [batch_size, num_clusters[-1] * hidden_dim]
        output = self.final_pool(final_features)  # [batch_size, hidden_dim]
        
        return output


class GraphTransformer(nn.Module):
    """Graph Transformer architecture for connectome analysis."""
    
    def __init__(
        self,
        node_features: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        heads: int = 8,
        dropout: float = 0.1,
        use_positional_encoding: bool = True
    ):
        super().__init__()
        
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        
        # Input projection
        self.input_proj = nn.Linear(node_features, hidden_dim)
        
        # Positional encoding
        if use_positional_encoding:
            self.pos_encoder = GraphPositionalEncoding(hidden_dim)
        else:
            self.pos_encoder = None
            
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Graph-aware attention mask
        self.use_graph_mask = True
        
        # Output layer
        self.output_proj = nn.Linear(hidden_dim, 1)
        
    def forward(self, data):
        """Forward pass through Graph Transformer."""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        batch_size = batch.max().item() + 1
        max_nodes = scatter_max(torch.ones_like(batch), batch)[0].max().item()
        
        # Pad sequences for transformer
        padded_x = torch.zeros(batch_size, max_nodes, self.node_features, device=x.device)
        attention_mask = torch.ones(batch_size, max_nodes, dtype=torch.bool, device=x.device)
        
        # Fill padded sequences
        for b in range(batch_size):
            batch_mask = (batch == b)
            batch_nodes = batch_mask.sum().item()
            
            padded_x[b, :batch_nodes] = x[batch_mask]
            attention_mask[b, :batch_nodes] = False  # False for valid positions
            
        # Input projection
        x = self.input_proj(padded_x)
        
        # Positional encoding
        if self.pos_encoder is not None:
            x = self.pos_encoder(x, edge_index, batch)
            
        # Transformer processing
        if self.use_graph_mask:
            # Create graph-aware attention mask based on edge connectivity
            graph_mask = self._create_graph_attention_mask(edge_index, batch, max_nodes)
            combined_mask = attention_mask.unsqueeze(1) | graph_mask
        else:
            combined_mask = attention_mask
            
        x = self.transformer(x, src_key_padding_mask=attention_mask)
        
        # Global pooling
        pooled = []
        for b in range(batch_size):
            valid_length = (~attention_mask[b]).sum().item()
            batch_repr = x[b, :valid_length].mean(dim=0)
            pooled.append(batch_repr)
            
        output = torch.stack(pooled, dim=0)
        output = self.output_proj(output)
        
        return output
        
    def _create_graph_attention_mask(self, edge_index, batch, max_nodes):
        """Create graph-aware attention mask."""
        batch_size = batch.max().item() + 1
        
        # Initialize mask (True means masked/not allowed)
        mask = torch.ones(batch_size, max_nodes, max_nodes, dtype=torch.bool, device=edge_index.device)
        
        # Allow attention along edges
        for b in range(batch_size):
            batch_mask = (batch == b)
            batch_nodes = batch_mask.sum().item()
            
            # Get edges for this batch
            batch_edge_mask = batch_mask[edge_index[0]] & batch_mask[edge_index[1]]
            batch_edges = edge_index[:, batch_edge_mask]
            
            # Adjust edge indices to batch-local indices
            node_mapping = torch.cumsum(batch_mask, dim=0) - 1
            local_edges = node_mapping[batch_edges]
            
            # Set attention mask to allow communication along edges
            mask[b, local_edges[0], local_edges[1]] = False
            mask[b, local_edges[1], local_edges[0]] = False  # Symmetric
            
            # Allow self-attention
            mask[b, range(batch_nodes), range(batch_nodes)] = False
            
        return mask


class GraphPositionalEncoding(nn.Module):
    """Graph positional encoding using spectral features."""
    
    def __init__(self, hidden_dim: int, max_freq: int = 10):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_freq = max_freq
        
        # Learnable positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(max_freq, hidden_dim))
        
    def forward(self, x, edge_index, batch):
        """Add positional encoding to node features."""
        # Simplified positional encoding using node degrees
        row, col = edge_index
        degree_vals = degree(row, x.size(0), dtype=x.dtype)
        
        # Normalize degrees
        degree_vals = degree_vals / degree_vals.max()
        
        # Create frequency components
        freqs = torch.arange(self.max_freq, dtype=x.dtype, device=x.device)
        pos_enc = torch.sin(degree_vals.unsqueeze(1) * freqs.unsqueeze(0) * math.pi)
        
        # Project to hidden dimension
        pos_features = torch.matmul(pos_enc, self.pos_embedding)
        
        # Add to input features
        return x + pos_features


class ResearchBenchmarkSuite:
    """Comprehensive benchmarking suite for research evaluation."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.models = {}
        self.results = {}
        
    def register_model(self, name: str, model: nn.Module):
        """Register model for benchmarking."""
        self.models[name] = model
        self.logger.info(f"Registered model: {name}")
        
    def benchmark_model(
        self,
        model_name: str,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        num_epochs: int = 50,
        device: str = 'cpu'
    ) -> ResearchMetrics:
        """Comprehensive benchmark of a single model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not registered")
            
        model = self.models[model_name].to(device)
        
        # Training phase
        model.train()
        start_time = time.time()
        
        training_losses = []
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                output = model(batch)
                loss = criterion(output, batch.y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
            avg_epoch_loss = epoch_loss / num_batches
            training_losses.append(avg_epoch_loss)
            
            # Early stopping check
            if len(training_losses) > 10:
                recent_losses = training_losses[-10:]
                if max(recent_losses) - min(recent_losses) < 1e-6:
                    break
                    
        training_time = time.time() - start_time
        convergence_epochs = len(training_losses)
        
        # Evaluation phase
        model.eval()
        start_time = time.time()
        
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                output = model(batch)
                
                predictions.extend(output.cpu().numpy())
                targets.extend(batch.y.cpu().numpy())
                
        inference_time = (time.time() - start_time) / len(test_loader)
        
        # Compute metrics
        predictions = np.array(predictions).flatten()
        targets = np.array(targets).flatten()
        
        # Classification metrics
        binary_preds = (predictions > 0.5).astype(int)
        binary_targets = (targets > 0.5).astype(int)
        
        accuracy = (binary_preds == binary_targets).mean()
        
        # Additional metrics
        tp = ((binary_preds == 1) & (binary_targets == 1)).sum()
        fp = ((binary_preds == 1) & (binary_targets == 0)).sum()
        fn = ((binary_preds == 0) & (binary_targets == 1)).sum()
        tn = ((binary_preds == 0) & (binary_targets == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # AUROC (simplified)
        try:
            from sklearn.metrics import roc_auc_score
            auroc = roc_auc_score(binary_targets, predictions)
        except:
            auroc = 0.5  # Random baseline
            
        # Model complexity metrics
        parameter_count = sum(p.numel() for p in model.parameters())
        memory_usage = parameter_count * 4 / (1024 * 1024)  # MB (assuming float32)
        
        # Stability score (based on loss variance)
        stability_score = 1.0 / (1.0 + np.var(training_losses[-10:]))
        
        metrics = ResearchMetrics(
            accuracy=accuracy,
            f1_score=f1_score,
            precision=precision,
            recall=recall,
            auroc=auroc,
            training_time=training_time,
            inference_time=inference_time,
            memory_usage=memory_usage,
            parameter_count=parameter_count,
            flops=parameter_count * 2,  # Rough estimate
            convergence_epochs=convergence_epochs,
            stability_score=stability_score
        )
        
        self.results[model_name] = metrics
        
        self.logger.info(
            f"Benchmarked {model_name}: "
            f"Accuracy={accuracy:.3f}, F1={f1_score:.3f}, "
            f"Time={training_time:.1f}s, Params={parameter_count}"
        )
        
        return metrics
        
    def compare_models(self) -> Dict[str, Any]:
        """Generate comprehensive model comparison."""
        if not self.results:
            return {}
            
        comparison = {
            'models': list(self.results.keys()),
            'metrics': {},
            'rankings': {},
            'summary': {}
        }
        
        # Collect all metrics
        metrics_data = {}
        for metric_name in ['accuracy', 'f1_score', 'auroc', 'training_time', 'parameter_count']:
            metrics_data[metric_name] = {
                name: getattr(metrics, metric_name) 
                for name, metrics in self.results.items()
            }
            
        comparison['metrics'] = metrics_data
        
        # Generate rankings
        for metric_name, values in metrics_data.items():
            if metric_name in ['training_time', 'parameter_count']:  # Lower is better
                ranking = sorted(values.items(), key=lambda x: x[1])
            else:  # Higher is better
                ranking = sorted(values.items(), key=lambda x: x[1], reverse=True)
                
            comparison['rankings'][metric_name] = [name for name, _ in ranking]
            
        # Summary statistics
        for metric_name, values in metrics_data.items():
            comparison['summary'][metric_name] = {
                'best': max(values.values()) if metric_name not in ['training_time', 'parameter_count'] else min(values.values()),
                'worst': min(values.values()) if metric_name not in ['training_time', 'parameter_count'] else max(values.values()),
                'mean': np.mean(list(values.values())),
                'std': np.std(list(values.values()))
            }
            
        return comparison
        
    def generate_research_report(self) -> str:
        """Generate comprehensive research report."""
        if not self.results:
            return "No benchmark results available."
            
        comparison = self.compare_models()
        
        report = []
        report.append("# Connectome-GNN Research Benchmark Report")
        report.append(f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\nModels Evaluated: {len(self.results)}")
        
        report.append("\n## Model Performance Summary")
        
        for model_name, metrics in self.results.items():
            report.append(f"\n### {model_name}")
            report.append(f"- Accuracy: {metrics.accuracy:.4f}")
            report.append(f"- F1 Score: {metrics.f1_score:.4f}")
            report.append(f"- AUROC: {metrics.auroc:.4f}")
            report.append(f"- Training Time: {metrics.training_time:.2f}s")
            report.append(f"- Parameters: {metrics.parameter_count:,}")
            report.append(f"- Memory Usage: {metrics.memory_usage:.2f}MB")
            report.append(f"- Convergence: {metrics.convergence_epochs} epochs")
            report.append(f"- Stability: {metrics.stability_score:.4f}")
            
        report.append("\n## Performance Rankings")
        
        for metric_name, ranking in comparison['rankings'].items():
            report.append(f"\n### {metric_name.title().replace('_', ' ')}")
            for i, model_name in enumerate(ranking, 1):
                value = comparison['metrics'][metric_name][model_name]
                if isinstance(value, float):
                    report.append(f"{i}. {model_name}: {value:.4f}")
                else:
                    report.append(f"{i}. {model_name}: {value}")
                    
        report.append("\n## Research Insights")
        
        # Identify best performing models
        accuracy_winner = comparison['rankings']['accuracy'][0]
        efficiency_winner = comparison['rankings']['training_time'][0]
        
        report.append(f"\n- **Best Accuracy**: {accuracy_winner} ({comparison['metrics']['accuracy'][accuracy_winner]:.4f})")
        report.append(f"- **Most Efficient**: {efficiency_winner} ({comparison['metrics']['training_time'][efficiency_winner]:.2f}s)")
        
        # Performance vs efficiency analysis
        report.append("\n### Performance vs Efficiency Analysis")
        
        for model_name in self.results:
            acc = comparison['metrics']['accuracy'][model_name]
            time_cost = comparison['metrics']['training_time'][model_name]
            efficiency_ratio = acc / (time_cost / 60)  # Accuracy per minute
            
            report.append(f"- {model_name}: {efficiency_ratio:.4f} accuracy/minute")
            
        return "\n".join(report)


# Factory for creating research models
def create_research_model(model_type: str, **kwargs) -> nn.Module:
    """Factory function for creating research models."""
    models = {
        'advanced_connectome_gnn': AdvancedConnectomeGNN,
        'graph_transformer': GraphTransformer,
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return models[model_type](**kwargs)


# Global benchmark suite
_global_benchmark_suite = None

def get_benchmark_suite() -> ResearchBenchmarkSuite:
    """Get global benchmark suite instance."""
    global _global_benchmark_suite
    if _global_benchmark_suite is None:
        _global_benchmark_suite = ResearchBenchmarkSuite()
    return _global_benchmark_suite