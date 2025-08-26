"""Meta-Learning GNN for Few-Shot Brain Network Analysis.

Implements Model-Agnostic Meta-Learning (MAML) and other meta-learning
approaches for rapid adaptation to new brain network analysis tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import Dict, List, Tuple, Optional, Callable, Union
import numpy as np
import copy
from collections import OrderedDict
import math


class MAMLConnectomeGNN(nn.Module):
    """MAML-based meta-learning GNN for connectome analysis."""
    
    def __init__(
        self,
        node_features: int = 100,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_classes: int = 2,
        meta_lr: float = 0.001,
        inner_lr: float = 0.01,
        inner_steps: int = 5,
        first_order: bool = False
    ):
        super().__init__()
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.first_order = first_order
        
        # Base network architecture
        self.input_projection = nn.Linear(node_features, hidden_dim)
        
        # Graph convolution layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = AdaptiveGraphConv(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                num_relations=4  # Multiple relation types for meta-learning
            )
            self.gnn_layers.append(layer)
        
        # Task-specific adaptation modules
        self.task_embeddings = nn.ModuleDict()
        self.adaptation_modules = nn.ModuleDict()
        
        # Output layer
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes if num_classes > 1 else 1)
        )
        
        # Meta-learning components
        self.meta_parameters = list(self.parameters())
        
    def forward(self, data: Data, task_id: Optional[str] = None) -> torch.Tensor:
        """Forward pass with optional task-specific adaptation."""
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
        
        # Apply task-specific adaptation if available
        if task_id is not None and task_id in self.adaptation_modules:
            x = self.adaptation_modules[task_id](x)
        
        # Graph convolution layers
        for layer in self.gnn_layers:
            x = layer(x, edge_index)
            x = F.relu(x)
        
        # Global pooling
        if batch is None:
            graph_embedding = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        else:
            graph_embedding = global_mean_pool(x, batch)
        
        # Classification
        output = self.classifier(graph_embedding)
        
        return output.squeeze(-1) if self.num_classes == 1 else output
    
    def clone_parameters(self) -> OrderedDict:
        """Clone current parameters for meta-learning updates."""
        return OrderedDict((name, param.clone()) for name, param in self.named_parameters())
    
    def set_parameters(self, params: OrderedDict):
        """Set parameters from OrderedDict."""
        for name, param in self.named_parameters():
            if name in params:
                param.data.copy_(params[name].data)
    
    def inner_loop_update(
        self, 
        support_data: List[Data], 
        support_labels: torch.Tensor,
        task_id: str
    ) -> OrderedDict:
        """Perform inner loop adaptation for a specific task."""
        # Clone current parameters
        adapted_params = self.clone_parameters()
        
        # Create task-specific adaptation module if not exists
        if task_id not in self.adaptation_modules:
            self.adaptation_modules[task_id] = TaskAdaptationModule(self.hidden_dim)
        
        # Inner loop gradient updates
        for step in range(self.inner_steps):
            # Forward pass with current adapted parameters
            total_loss = 0.0
            
            for data, label in zip(support_data, support_labels):
                output = self.forward(data, task_id)
                
                if self.num_classes == 1:
                    loss = F.mse_loss(output, label.float())
                else:
                    loss = F.cross_entropy(output, label.long())
                
                total_loss += loss
            
            # Compute gradients
            gradients = torch.autograd.grad(
                total_loss,
                self.parameters(),
                create_graph=not self.first_order,
                allow_unused=True
            )
            
            # Update adapted parameters
            for (name, param), grad in zip(self.named_parameters(), gradients):
                if grad is not None:
                    adapted_params[name] = param - self.inner_lr * grad
        
        return adapted_params
    
    def meta_update(
        self,
        tasks_data: List[Tuple[List[Data], torch.Tensor, List[Data], torch.Tensor]],
        task_ids: List[str]
    ):
        """Perform meta-learning update across multiple tasks."""
        meta_gradients = []
        
        for (support_data, support_labels, query_data, query_labels), task_id in zip(tasks_data, task_ids):
            # Inner loop adaptation
            adapted_params = self.inner_loop_update(support_data, support_labels, task_id)
            
            # Temporarily set adapted parameters
            original_params = self.clone_parameters()
            self.set_parameters(adapted_params)
            
            # Compute query loss with adapted parameters
            query_loss = 0.0
            for data, label in zip(query_data, query_labels):
                output = self.forward(data, task_id)
                
                if self.num_classes == 1:
                    loss = F.mse_loss(output, label.float())
                else:
                    loss = F.cross_entropy(output, label.long())
                
                query_loss += loss
            
            # Compute meta-gradients
            task_gradients = torch.autograd.grad(
                query_loss,
                self.parameters(),
                retain_graph=False,
                allow_unused=True
            )
            
            meta_gradients.append(task_gradients)
            
            # Restore original parameters
            self.set_parameters(original_params)
        
        # Average meta-gradients and update
        averaged_gradients = []
        for grad_lists in zip(*meta_gradients):
            valid_grads = [g for g in grad_lists if g is not None]
            if valid_grads:
                avg_grad = torch.stack(valid_grads).mean(dim=0)
                averaged_gradients.append(avg_grad)
            else:
                averaged_gradients.append(None)
        
        # Apply meta-learning update
        for param, meta_grad in zip(self.parameters(), averaged_gradients):
            if meta_grad is not None:
                param.data -= self.meta_lr * meta_grad


class AdaptiveGraphConv(MessagePassing):
    """Adaptive graph convolution for meta-learning."""
    
    def __init__(self, in_dim: int, out_dim: int, num_relations: int = 4):
        super().__init__(aggr='add')
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        
        # Relation-specific transformations
        self.relation_transforms = nn.ModuleList([
            nn.Linear(in_dim, out_dim) for _ in range(num_relations)
        ])
        
        # Adaptive relation weighting
        self.relation_attention = nn.Sequential(
            nn.Linear(in_dim * 2, num_relations),
            nn.Softmax(dim=-1)
        )
        
        # Self-loop transformation
        self.self_transform = nn.Linear(in_dim, out_dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptive relation selection."""
        # Self-transformation
        self_output = self.self_transform(x)
        
        # Message passing with adaptive relations
        message_output = self.propagate(edge_index, x=x)
        
        return self_output + message_output
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        """Compute adaptive messages based on node features."""
        # Compute relation weights
        edge_features = torch.cat([x_i, x_j], dim=-1)
        relation_weights = self.relation_attention(edge_features)
        
        # Apply relation-specific transformations
        relation_outputs = []
        for i, transform in enumerate(self.relation_transforms):
            relation_output = transform(x_j)
            relation_outputs.append(relation_output)
        
        # Weighted combination of relations
        stacked_relations = torch.stack(relation_outputs, dim=-1)  # [num_edges, out_dim, num_relations]
        weighted_output = torch.sum(
            stacked_relations * relation_weights.unsqueeze(-2), 
            dim=-1
        )
        
        return weighted_output


class TaskAdaptationModule(nn.Module):
    """Task-specific adaptation module for meta-learning."""
    
    def __init__(self, hidden_dim: int, adaptation_strength: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.adaptation_strength = adaptation_strength
        
        # Learnable task-specific parameters
        self.task_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.task_scale = nn.Parameter(torch.ones(hidden_dim))
        
        # Feature transformation
        self.feature_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply task-specific adaptation."""
        # Feature transformation
        transformed = self.feature_transform(x)
        
        # Task-specific scaling and bias
        adapted = transformed * self.task_scale + self.task_bias
        
        # Weighted combination with original features
        output = (1 - self.adaptation_strength) * x + self.adaptation_strength * adapted
        
        return output


class PrototypicalNetworkGNN(nn.Module):
    """Prototypical networks for few-shot brain network classification."""
    
    def __init__(
        self,
        node_features: int = 100,
        hidden_dim: int = 256,
        num_layers: int = 4,
        distance_metric: str = 'euclidean'
    ):
        super().__init__()
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.distance_metric = distance_metric
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Graph neural network layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = AdaptiveGraphConv(hidden_dim, hidden_dim)
            self.gnn_layers.append(layer)
        
        # Prototype attention
        self.prototype_attention = nn.MultiheadAttention(
            hidden_dim, 
            num_heads=8, 
            batch_first=True
        )
        
    def forward(self, data: Data) -> torch.Tensor:
        """Extract features for prototypical learning."""
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, 'batch', None)
        
        # Handle feature dimension mismatch
        if x.size(1) != self.node_features:
            if x.size(1) < self.node_features:
                padding = torch.zeros(x.size(0), self.node_features - x.size(1), device=x.device)
                x = torch.cat([x, padding], dim=1)
            else:
                x = x[:, :self.node_features]
        
        # Feature extraction
        x = self.feature_extractor(x)
        
        # Graph convolution layers
        for layer in self.gnn_layers:
            x = layer(x, edge_index)
            x = F.relu(x)
        
        # Global pooling to get graph-level representation
        if batch is None:
            graph_embedding = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        else:
            graph_embedding = global_mean_pool(x, batch)
        
        return graph_embedding
    
    def compute_prototypes(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute class prototypes from support set."""
        unique_labels = torch.unique(labels)
        prototypes = []
        
        for label in unique_labels:
            class_mask = (labels == label)
            class_embeddings = embeddings[class_mask]
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)
        
        return torch.stack(prototypes, dim=0)
    
    def compute_distances(self, query_embeddings: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """Compute distances between query embeddings and prototypes."""
        if self.distance_metric == 'euclidean':
            # Euclidean distance
            distances = torch.cdist(query_embeddings, prototypes, p=2)
        elif self.distance_metric == 'cosine':
            # Cosine similarity (converted to distance)
            query_norm = F.normalize(query_embeddings, p=2, dim=1)
            prototype_norm = F.normalize(prototypes, p=2, dim=1)
            similarities = torch.mm(query_norm, prototype_norm.T)
            distances = 1 - similarities
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        return distances
    
    def predict(
        self, 
        support_data: List[Data],
        support_labels: torch.Tensor,
        query_data: List[Data]
    ) -> torch.Tensor:
        """Make predictions using prototypical learning."""
        # Extract support embeddings
        support_embeddings = []
        for data in support_data:
            embedding = self.forward(data)
            support_embeddings.append(embedding)
        support_embeddings = torch.cat(support_embeddings, dim=0)
        
        # Compute prototypes
        prototypes = self.compute_prototypes(support_embeddings, support_labels)
        
        # Extract query embeddings
        query_embeddings = []
        for data in query_data:
            embedding = self.forward(data)
            query_embeddings.append(embedding)
        query_embeddings = torch.cat(query_embeddings, dim=0)
        
        # Compute distances and predict
        distances = self.compute_distances(query_embeddings, prototypes)
        predictions = F.softmax(-distances, dim=1)  # Convert distances to probabilities
        
        return predictions


class MetaLearningConnectomeGNN(nn.Module):
    """Meta-learning framework combining MAML and Prototypical Networks."""
    
    def __init__(
        self,
        node_features: int = 100,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_classes: int = 2,
        meta_learning_type: str = 'maml',
        **kwargs
    ):
        super().__init__()
        self.meta_learning_type = meta_learning_type
        
        if meta_learning_type == 'maml':
            self.meta_learner = MAMLConnectomeGNN(
                node_features=node_features,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_classes=num_classes,
                **kwargs
            )
        elif meta_learning_type == 'prototypical':
            self.meta_learner = PrototypicalNetworkGNN(
                node_features=node_features,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown meta-learning type: {meta_learning_type}")
    
    def forward(self, data: Data, **kwargs) -> torch.Tensor:
        """Forward pass through meta-learner."""
        return self.meta_learner(data, **kwargs)
    
    def meta_train(
        self,
        tasks_data: List[Tuple[List[Data], torch.Tensor, List[Data], torch.Tensor]],
        task_ids: List[str]
    ):
        """Meta-training across multiple tasks."""
        if self.meta_learning_type == 'maml':
            self.meta_learner.meta_update(tasks_data, task_ids)
        elif self.meta_learning_type == 'prototypical':
            # Prototypical networks don't have explicit meta-updates
            # Training is done through episodic learning
            pass
    
    def adapt_to_task(
        self,
        support_data: List[Data],
        support_labels: torch.Tensor,
        task_id: str,
        num_adaptation_steps: int = 5
    ):
        """Adapt model to a new task."""
        if self.meta_learning_type == 'maml':
            # Perform inner loop adaptation
            adapted_params = self.meta_learner.inner_loop_update(
                support_data, 
                support_labels, 
                task_id
            )
            return adapted_params
        elif self.meta_learning_type == 'prototypical':
            # For prototypical networks, adaptation is implicit through prototypes
            return None
    
    def evaluate_on_task(
        self,
        support_data: List[Data],
        support_labels: torch.Tensor,
        query_data: List[Data],
        query_labels: torch.Tensor,
        task_id: str = 'evaluation'
    ) -> Dict[str, float]:
        """Evaluate model on a few-shot task."""
        if self.meta_learning_type == 'maml':
            # Adapt to task
            adapted_params = self.adapt_to_task(support_data, support_labels, task_id)
            original_params = self.meta_learner.clone_parameters()
            
            # Set adapted parameters
            self.meta_learner.set_parameters(adapted_params)
            
            # Evaluate on query set
            correct = 0
            total = 0
            total_loss = 0.0
            
            with torch.no_grad():
                for data, label in zip(query_data, query_labels):
                    output = self.meta_learner(data, task_id)
                    
                    if self.meta_learner.num_classes == 1:
                        loss = F.mse_loss(output, label.float())
                        # For regression, count as correct if within 10% of target
                        correct += (torch.abs(output - label) < 0.1 * torch.abs(label)).sum().item()
                    else:
                        loss = F.cross_entropy(output, label.long())
                        pred = output.argmax(dim=1)
                        correct += (pred == label).sum().item()
                    
                    total += label.size(0)
                    total_loss += loss.item()
            
            # Restore original parameters
            self.meta_learner.set_parameters(original_params)
            
            return {
                'accuracy': correct / total,
                'loss': total_loss / len(query_data)
            }
        
        elif self.meta_learning_type == 'prototypical':
            # Use prototypical prediction
            predictions = self.meta_learner.predict(support_data, support_labels, query_data)
            predicted_labels = predictions.argmax(dim=1)
            
            correct = (predicted_labels == query_labels).sum().item()
            total = query_labels.size(0)
            
            return {
                'accuracy': correct / total,
                'loss': F.cross_entropy(predictions, query_labels.long()).item()
            }


def create_meta_learning_model(config: Dict) -> MetaLearningConnectomeGNN:
    """Factory function for creating meta-learning models."""
    return MetaLearningConnectomeGNN(
        node_features=config.get('node_features', 100),
        hidden_dim=config.get('hidden_dim', 256),
        num_layers=config.get('num_layers', 4),
        num_classes=config.get('num_classes', 2),
        meta_learning_type=config.get('meta_learning_type', 'maml'),
        meta_lr=config.get('meta_lr', 0.001),
        inner_lr=config.get('inner_lr', 0.01),
        inner_steps=config.get('inner_steps', 5),
        first_order=config.get('first_order', False),
        distance_metric=config.get('distance_metric', 'euclidean')
    )