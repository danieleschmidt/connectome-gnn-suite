"""Meta-Learning Graph Neural Networks for Few-Shot Brain Connectivity Analysis.

Advanced meta-learning framework enabling rapid adaptation to new brain connectivity
tasks with minimal training data. Implements MAML, Prototypical Networks, and
novel gradient-based meta-learning specifically designed for connectome analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import Optional, Dict, Any, Tuple, List
import math
import numpy as np
from collections import OrderedDict
import copy

try:
    from ..models.base import BaseConnectomeModel
except ImportError:
    class BaseConnectomeModel(nn.Module):
        def __init__(self, node_features, hidden_dim, output_dim, dropout=0.1):
            super().__init__()
            self.node_features = node_features
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.dropout_layer = nn.Dropout(dropout)
            self.model_config = {}


class MetaGraphConv(MessagePassing):
    """Meta-learnable graph convolution layer with fast adaptation."""
    
    def __init__(self, in_dim: int, out_dim: int, meta_dim: int = 64):
        super().__init__(aggr='add')
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.meta_dim = meta_dim
        
        # Base parameters (slow weights)
        self.base_linear = nn.Linear(in_dim, out_dim)
        self.base_self_loop = nn.Linear(in_dim, out_dim)
        
        # Meta-learning parameters (fast weights)
        self.meta_context = nn.Parameter(torch.randn(meta_dim))
        self.meta_projection = nn.Linear(meta_dim, out_dim * in_dim)
        self.meta_bias = nn.Linear(meta_dim, out_dim)
        
        # Attention for context adaptation
        self.context_attention = nn.MultiheadAttention(meta_dim, num_heads=4, batch_first=True)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                context_features: Optional[torch.Tensor] = None,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with meta-learning adaptation."""
        
        # Base message passing
        base_output = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        base_output += self.base_self_loop(x)
        
        # Meta adaptation if context is provided
        if context_features is not None:
            # Adapt meta context based on task context
            adapted_context, _ = self.context_attention(
                self.meta_context.unsqueeze(0).unsqueeze(0),
                context_features.unsqueeze(0),
                context_features.unsqueeze(0)
            )
            adapted_context = adapted_context.squeeze(0).squeeze(0)
            
            # Generate fast weights
            fast_weights = self.meta_projection(adapted_context).view(self.out_dim, self.in_dim)
            fast_bias = self.meta_bias(adapted_context)
            
            # Apply fast adaptation
            meta_output = F.linear(x, fast_weights, fast_bias)
            
            # Combine base and meta outputs
            output = 0.7 * base_output + 0.3 * meta_output
        else:
            output = base_output
        
        return F.relu(output)
    
    def message(self, x_j: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Create messages between nodes."""
        message = self.base_linear(x_j)
        if edge_attr is not None:
            message = message * edge_attr.unsqueeze(-1)
        return message


class TaskEmbeddingNetwork(nn.Module):
    """Network to encode task context from support set."""
    
    def __init__(self, node_features: int, hidden_dim: int, embedding_dim: int = 128):
        super().__init__()
        self.node_features = node_features
        self.embedding_dim = embedding_dim
        
        # Graph encoder for support examples
        self.graph_encoder = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, embedding_dim)
        )
        
        # Set aggregation (Deep Sets approach)
        self.set_encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Task representation
        self.task_projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim)
        )
    
    def forward(self, support_graphs: List[Data], support_labels: torch.Tensor) -> torch.Tensor:
        """Encode task context from support set."""
        graph_embeddings = []
        
        for graph in support_graphs:
            # Encode individual graph
            node_embeddings = self.graph_encoder(graph.x)
            graph_embedding = global_mean_pool(node_embeddings, graph.batch if hasattr(graph, 'batch') else torch.zeros(graph.x.size(0), dtype=torch.long))
            graph_embeddings.append(graph_embedding)
        
        # Stack graph embeddings
        graph_embeddings = torch.stack(graph_embeddings, dim=0)
        
        # Apply set encoder to each graph embedding
        encoded_graphs = self.set_encoder(graph_embeddings)
        
        # Aggregate across support set
        task_embedding = torch.mean(encoded_graphs, dim=0)
        
        # Project to final task representation
        task_context = self.task_projector(task_embedding)
        
        return task_context


class MAMLGraphNetwork(BaseConnectomeModel):
    """Model-Agnostic Meta-Learning for Graph Neural Networks."""
    
    def __init__(
        self,
        node_features: int = 100,
        hidden_dim: int = 256,
        output_dim: int = 1,
        num_layers: int = 3,
        meta_dim: int = 128,
        inner_lr: float = 0.01,
        dropout: float = 0.1
    ):
        super().__init__(node_features, hidden_dim, output_dim, dropout)
        
        self.num_layers = num_layers
        self.meta_dim = meta_dim
        self.inner_lr = inner_lr
        
        # Task embedding network
        self.task_encoder = TaskEmbeddingNetwork(node_features, hidden_dim, meta_dim)
        
        # Meta-learnable graph layers
        self.meta_layers = nn.ModuleList([
            MetaGraphConv(
                node_features if i == 0 else hidden_dim,
                hidden_dim,
                meta_dim
            )
            for i in range(num_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Meta parameters for fast adaptation
        self.meta_parameters = list(self.parameters())
        
        self.model_config.update({
            'architecture': 'maml_graph_network',
            'num_layers': num_layers,
            'meta_dim': meta_dim,
            'inner_lr': inner_lr
        })
    
    def forward(self, data: Data, task_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with optional task context."""
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)
        batch = getattr(data, 'batch', None)
        
        h = x
        for layer in self.meta_layers:
            h = layer(h, edge_index, task_context, edge_attr)
            h = self.dropout_layer(h)
        
        # Global pooling
        if batch is not None:
            h = global_mean_pool(h, batch)
        else:
            h = h.mean(dim=0, keepdim=True)
        
        return self.output_layer(h)
    
    def clone_parameters(self) -> OrderedDict:
        """Clone current parameters for MAML inner loop."""
        return OrderedDict({name: param.clone() for name, param in self.named_parameters()})
    
    def update_parameters(self, gradients: Dict[str, torch.Tensor]) -> 'MAMLGraphNetwork':
        """Update parameters using gradients (inner loop update)."""
        updated_model = copy.deepcopy(self)
        
        for name, param in updated_model.named_parameters():
            if name in gradients:
                param.data = param.data - self.inner_lr * gradients[name]
        
        return updated_model


class PrototypicalGraphNetwork(BaseConnectomeModel):
    """Prototypical Networks for Few-Shot Graph Classification."""
    
    def __init__(
        self,
        node_features: int = 100,
        hidden_dim: int = 256,
        output_dim: int = 128,  # Embedding dimension
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__(node_features, hidden_dim, output_dim, dropout)
        
        self.num_layers = num_layers
        
        # Graph encoder
        self.encoder_layers = nn.ModuleList([
            nn.Linear(node_features if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        
        # Final embedding layer
        self.embedding_layer = nn.Linear(hidden_dim, output_dim)
        
        # Distance metric learner
        self.distance_metric = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.model_config.update({
            'architecture': 'prototypical_graph_network',
            'num_layers': num_layers,
            'embedding_dim': output_dim
        })
    
    def encode_graph(self, data: Data) -> torch.Tensor:
        """Encode graph to embedding space."""
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, 'batch', None)
        
        h = x
        for layer in self.encoder_layers:
            h = F.relu(layer(h))
            h = self.dropout_layer(h)
        
        # Global pooling
        if batch is not None:
            h = global_mean_pool(h, batch)
        else:
            h = h.mean(dim=0, keepdim=True)
        
        # Final embedding
        embedding = self.embedding_layer(h)
        return F.normalize(embedding, p=2, dim=1)
    
    def compute_prototypes(self, support_embeddings: torch.Tensor, support_labels: torch.Tensor) -> torch.Tensor:
        """Compute class prototypes from support set."""
        unique_labels = torch.unique(support_labels)
        prototypes = []
        
        for label in unique_labels:
            mask = (support_labels == label)
            prototype = support_embeddings[mask].mean(dim=0)
            prototypes.append(prototype)
        
        return torch.stack(prototypes, dim=0)
    
    def forward(self, query_data: Data, support_data: List[Data], support_labels: torch.Tensor) -> torch.Tensor:
        """Forward pass for prototypical classification."""
        # Encode query
        query_embedding = self.encode_graph(query_data)
        
        # Encode support set
        support_embeddings = torch.stack([self.encode_graph(data) for data in support_data], dim=0)
        
        # Compute prototypes
        prototypes = self.compute_prototypes(support_embeddings, support_labels)
        
        # Compute distances to prototypes
        distances = []
        for prototype in prototypes:
            # Concatenate query and prototype for distance learning
            combined = torch.cat([query_embedding, prototype.unsqueeze(0)], dim=1)
            distance = self.distance_metric(combined)
            distances.append(distance)
        
        distances = torch.stack(distances, dim=1).squeeze(-1)
        
        # Convert distances to probabilities (negative distance = higher similarity)
        probabilities = F.softmax(-distances, dim=1)
        
        return probabilities


class GradientBasedMetaLearner(nn.Module):
    """Gradient-based meta-learner for brain connectivity tasks."""
    
    def __init__(
        self,
        base_model: nn.Module,
        meta_lr: float = 0.001,
        inner_lr: float = 0.01,
        inner_steps: int = 5
    ):
        super().__init__()
        self.base_model = base_model
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        
        # Meta-optimizer
        self.meta_optimizer = torch.optim.Adam(self.base_model.parameters(), lr=meta_lr)
        
    def inner_loop(self, support_data: List[Data], support_labels: torch.Tensor,
                   task_context: torch.Tensor) -> nn.Module:
        """Perform inner loop adaptation."""
        # Clone model for adaptation
        adapted_model = copy.deepcopy(self.base_model)
        inner_optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        for step in range(self.inner_steps):
            inner_optimizer.zero_grad()
            
            # Compute loss on support set
            total_loss = 0
            for data, label in zip(support_data, support_labels):
                pred = adapted_model(data, task_context)
                loss = F.mse_loss(pred, label.unsqueeze(0))
                total_loss += loss
            
            total_loss.backward()
            inner_optimizer.step()
        
        return adapted_model
    
    def meta_update(self, task_batch: List[Tuple[List[Data], torch.Tensor, List[Data], torch.Tensor]]):
        """Perform meta-update across multiple tasks."""
        self.meta_optimizer.zero_grad()
        
        meta_loss = 0
        
        for support_data, support_labels, query_data, query_labels in task_batch:
            # Encode task context
            if hasattr(self.base_model, 'task_encoder'):
                task_context = self.base_model.task_encoder(support_data, support_labels)
            else:
                task_context = None
            
            # Inner loop adaptation
            adapted_model = self.inner_loop(support_data, support_labels, task_context)
            
            # Compute loss on query set
            for data, label in zip(query_data, query_labels):
                pred = adapted_model(data, task_context)
                loss = F.mse_loss(pred, label.unsqueeze(0))
                meta_loss += loss
        
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()


class FewShotConnectomeDataset:
    """Dataset generator for few-shot connectome tasks."""
    
    def __init__(self, base_graphs: List[Data], base_labels: torch.Tensor):
        self.base_graphs = base_graphs
        self.base_labels = base_labels
        self.unique_labels = torch.unique(base_labels)
    
    def sample_task(self, n_way: int = 5, k_shot: int = 5, q_query: int = 15) -> Tuple[List[Data], torch.Tensor, List[Data], torch.Tensor]:
        """Sample a few-shot learning task."""
        # Sample classes for this task
        task_classes = torch.randperm(len(self.unique_labels))[:n_way]
        
        support_data = []
        support_labels = []
        query_data = []
        query_labels = []
        
        for new_label, orig_label in enumerate(task_classes):
            # Get all examples for this class
            class_mask = (self.base_labels == self.unique_labels[orig_label])
            class_indices = torch.nonzero(class_mask, as_tuple=True)[0]
            
            # Sample support and query examples
            perm = torch.randperm(len(class_indices))
            support_indices = class_indices[perm[:k_shot]]
            query_indices = class_indices[perm[k_shot:k_shot + q_query]]
            
            # Add to support set
            for idx in support_indices:
                support_data.append(self.base_graphs[idx])
                support_labels.append(new_label)
            
            # Add to query set
            for idx in query_indices:
                query_data.append(self.base_graphs[idx])
                query_labels.append(new_label)
        
        return (support_data, torch.tensor(support_labels), 
                query_data, torch.tensor(query_labels))


class MetaLearningTrainer:
    """Trainer for meta-learning graph networks."""
    
    def __init__(self, model: nn.Module, method: str = 'maml'):
        self.model = model
        self.method = method
        
        if method == 'maml':
            self.meta_learner = GradientBasedMetaLearner(model)
        elif method == 'prototypical':
            self.meta_learner = None  # Prototypical networks don't need explicit meta-learner
        else:
            raise ValueError(f"Unknown meta-learning method: {method}")
    
    def train_episode(self, support_data: List[Data], support_labels: torch.Tensor,
                     query_data: List[Data], query_labels: torch.Tensor) -> float:
        """Train on a single episode."""
        if self.method == 'maml':
            # MAML training
            task_batch = [(support_data, support_labels, query_data, query_labels)]
            loss = self.meta_learner.meta_update(task_batch)
            return loss
        
        elif self.method == 'prototypical':
            # Prototypical network training
            self.model.train()
            total_loss = 0
            
            for query_graph, true_label in zip(query_data, query_labels):
                pred_probs = self.model(query_graph, support_data, support_labels)
                loss = F.cross_entropy(pred_probs, true_label.unsqueeze(0))
                total_loss += loss
            
            # Backward pass
            total_loss.backward()
            return total_loss.item()
    
    def evaluate_episode(self, support_data: List[Data], support_labels: torch.Tensor,
                        query_data: List[Data], query_labels: torch.Tensor) -> float:
        """Evaluate on a single episode."""
        self.model.eval()
        
        with torch.no_grad():
            if self.method == 'maml':
                # Fast adaptation for MAML
                if hasattr(self.model, 'task_encoder'):
                    task_context = self.model.task_encoder(support_data, support_labels)
                else:
                    task_context = None
                
                adapted_model = self.meta_learner.inner_loop(support_data, support_labels, task_context)
                
                correct = 0
                total = 0
                for query_graph, true_label in zip(query_data, query_labels):
                    pred = adapted_model(query_graph, task_context)
                    predicted_label = torch.round(pred).long()
                    correct += (predicted_label == true_label).sum().item()
                    total += 1
                
                accuracy = correct / total
                
            elif self.method == 'prototypical':
                correct = 0
                total = 0
                
                for query_graph, true_label in zip(query_data, query_labels):
                    pred_probs = self.model(query_graph, support_data, support_labels)
                    predicted_label = torch.argmax(pred_probs, dim=1)
                    correct += (predicted_label == true_label).sum().item()
                    total += 1
                
                accuracy = correct / total
        
        return accuracy


def create_meta_learning_model(config: Dict[str, Any]) -> nn.Module:
    """Factory function for creating meta-learning models."""
    method = config.get('method', 'maml')
    
    if method == 'maml':
        return MAMLGraphNetwork(**{k: v for k, v in config.items() if k != 'method'})
    elif method == 'prototypical':
        return PrototypicalGraphNetwork(**{k: v for k, v in config.items() if k != 'method'})
    else:
        raise ValueError(f"Unknown meta-learning method: {method}")


# Example usage and testing utilities
class MetaLearningMetrics:
    """Metrics for evaluating meta-learning performance."""
    
    @staticmethod
    def few_shot_accuracy(trainer: MetaLearningTrainer, dataset: FewShotConnectomeDataset,
                         n_episodes: int = 100, n_way: int = 5, k_shot: int = 5) -> Dict[str, float]:
        """Evaluate few-shot learning accuracy."""
        accuracies = []
        
        for _ in range(n_episodes):
            support_data, support_labels, query_data, query_labels = dataset.sample_task(n_way, k_shot)
            accuracy = trainer.evaluate_episode(support_data, support_labels, query_data, query_labels)
            accuracies.append(accuracy)
        
        return {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'confidence_interval': np.percentile(accuracies, [5, 95])
        }
    
    @staticmethod
    def adaptation_speed(trainer: MetaLearningTrainer, dataset: FewShotConnectomeDataset,
                        max_shots: int = 20) -> Dict[str, List[float]]:
        """Measure adaptation speed with varying number of shots."""
        shot_range = range(1, max_shots + 1)
        accuracies_by_shots = {shots: [] for shots in shot_range}
        
        for shots in shot_range:
            for _ in range(50):  # 50 episodes per shot count
                support_data, support_labels, query_data, query_labels = dataset.sample_task(k_shot=shots)
                accuracy = trainer.evaluate_episode(support_data, support_labels, query_data, query_labels)
                accuracies_by_shots[shots].append(accuracy)
        
        return {
            'shot_counts': list(shot_range),
            'mean_accuracies': [np.mean(accuracies_by_shots[shots]) for shots in shot_range],
            'std_accuracies': [np.std(accuracies_by_shots[shots]) for shots in shot_range]
        }