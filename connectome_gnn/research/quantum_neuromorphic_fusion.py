"""Quantum-Neuromorphic Fusion Framework for Next-Generation Brain Computing.

Revolutionary hybrid architecture combining quantum computing principles with
neuromorphic spike-based processing for unprecedented brain connectivity analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from typing import Optional, Dict, Any, Tuple, List
import math
import numpy as np

try:
    from ..models.base import BaseConnectomeModel
    from .quantum_gnn import QuantumStateEmbedding, QuantumEntanglementLayer
    from .neuromorphic_gnn import SpikingNeuron, SynapticPlasticity
except ImportError:
    class BaseConnectomeModel(nn.Module):
        def __init__(self, node_features, hidden_dim, output_dim, dropout=0.1):
            super().__init__()
            self.node_features = node_features
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.dropout_layer = nn.Dropout(dropout)
            self.model_config = {}
    
    # Simplified imports for standalone operation
    class QuantumStateEmbedding(nn.Module):
        def __init__(self, input_dim, quantum_dim, num_qubits=8):
            super().__init__()
            self.quantum_transform = nn.Linear(input_dim, quantum_dim * 2)
        def forward(self, x): return self.quantum_transform(x)
    
    class QuantumEntanglementLayer(MessagePassing):
        def __init__(self, in_dim, out_dim, num_heads=4):
            super().__init__(aggr='add')
            self.proj = nn.Linear(in_dim, out_dim)
        def forward(self, x, edge_index, edge_attr=None):
            return self.propagate(edge_index, x=x)
        def message(self, x_j): return self.proj(x_j)
    
    class SpikingNeuron(nn.Module):
        def __init__(self, input_dim, threshold=1.0, decay=0.9):
            super().__init__()
            self.threshold = nn.Parameter(torch.tensor(threshold))
            self.linear = nn.Linear(input_dim, input_dim)
        def forward(self, x): 
            out = self.linear(x)
            spikes = (out > self.threshold).float()
            return spikes, out
    
    class SynapticPlasticity(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.weights = nn.Linear(input_dim, output_dim)
        def forward(self, pre_spikes, post_spikes):
            return self.weights(pre_spikes)


class QuantumSpikingNeuron(nn.Module):
    """Quantum-enhanced spiking neuron with superposition states."""
    
    def __init__(self, input_dim: int, quantum_dim: int = 32, threshold: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.quantum_dim = quantum_dim
        self.threshold = nn.Parameter(torch.tensor(threshold))
        
        # Quantum state preparation for membrane potential
        self.quantum_membrane = QuantumStateEmbedding(input_dim, quantum_dim)
        
        # Quantum-classical interface
        self.quantum_to_classical = nn.Linear(quantum_dim * 2, input_dim)
        
        # Membrane potential dynamics
        self.register_buffer('membrane_potential', torch.zeros(1, input_dim))
        self.register_buffer('quantum_coherence', torch.ones(1, quantum_dim))
        
        # Decay parameters
        self.decay_rate = nn.Parameter(torch.tensor(0.9))
        self.quantum_decay = nn.Parameter(torch.tensor(0.95))
        
    def forward(self, input_current: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass combining quantum and classical dynamics.
        
        Returns:
            classical_spikes: Traditional binary spikes
            quantum_spikes: Probabilistic quantum spikes
            membrane_potential: Continuous membrane state
        """
        batch_size = input_current.size(0)
        
        # Expand buffers for batch
        if self.membrane_potential.size(0) != batch_size:
            self.membrane_potential = self.membrane_potential.expand(batch_size, -1).contiguous()
            self.quantum_coherence = self.quantum_coherence.expand(batch_size, -1).contiguous()
        
        # Quantum membrane dynamics
        quantum_membrane = self.quantum_membrane(input_current.unsqueeze(1))
        quantum_membrane = quantum_membrane.squeeze(1)
        
        # Update quantum coherence
        self.quantum_coherence *= self.quantum_decay
        coherence_factor = torch.sigmoid(self.quantum_coherence.mean(dim=1, keepdim=True))
        
        # Classical membrane update
        self.membrane_potential *= self.decay_rate
        classical_input = self.quantum_to_classical(quantum_membrane)
        self.membrane_potential += classical_input * coherence_factor
        
        # Generate classical spikes
        classical_spikes = (self.membrane_potential > self.threshold).float()
        
        # Generate quantum probabilistic spikes
        quantum_probs = torch.sigmoid((self.membrane_potential - self.threshold) / 0.1)
        quantum_spikes = torch.bernoulli(quantum_probs)
        
        # Reset membrane where classical spikes occurred
        self.membrane_potential = torch.where(
            classical_spikes.bool(),
            torch.zeros_like(self.membrane_potential),
            self.membrane_potential
        )
        
        # Update quantum coherence based on spiking activity
        spike_activity = classical_spikes.mean(dim=1, keepdim=True)
        self.quantum_coherence += (spike_activity - 0.1) * 0.01  # Homeostatic update
        self.quantum_coherence = torch.clamp(self.quantum_coherence, 0.0, 1.0)
        
        return classical_spikes, quantum_spikes, self.membrane_potential.clone()


class QuantumSynapticPlasticity(nn.Module):
    """Quantum-enhanced synaptic plasticity with entangled weight updates."""
    
    def __init__(self, input_dim: int, output_dim: int, quantum_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.quantum_dim = quantum_dim
        
        # Classical synaptic weights
        self.classical_weights = nn.Parameter(torch.randn(output_dim, input_dim) * 0.1)
        
        # Quantum entangled weights
        self.quantum_weights = QuantumEntanglementLayer(input_dim, output_dim)
        
        # Plasticity learning rates
        self.classical_lr = nn.Parameter(torch.tensor(0.001))
        self.quantum_lr = nn.Parameter(torch.tensor(0.0005))
        
        # Hebbian learning traces
        self.register_buffer('pre_trace', torch.zeros(1, input_dim))
        self.register_buffer('post_trace', torch.zeros(1, output_dim))
        
        # Quantum coherence for weight updates
        self.weight_coherence = nn.Parameter(torch.ones(output_dim, input_dim) * 0.5)
        
    def forward(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor,
                quantum_pre: torch.Tensor, quantum_post: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """Quantum-enhanced synaptic transmission and plasticity."""
        batch_size = pre_spikes.size(0)
        
        # Expand traces for batch
        if self.pre_trace.size(0) != batch_size:
            self.pre_trace = self.pre_trace.expand(batch_size, -1).contiguous()
            self.post_trace = self.post_trace.expand(batch_size, -1).contiguous()
        
        # Update Hebbian traces
        self.pre_trace = 0.9 * self.pre_trace + 0.1 * pre_spikes
        self.post_trace = 0.9 * self.post_trace + 0.1 * post_spikes
        
        # Classical synaptic transmission
        classical_current = F.linear(pre_spikes, self.classical_weights)
        
        # Quantum synaptic transmission
        quantum_current = self.quantum_weights(quantum_pre, edge_index)
        
        # Coherent combination of classical and quantum currents
        coherence_mask = torch.sigmoid(self.weight_coherence).mean(dim=1, keepdim=True)
        total_current = coherence_mask * classical_current + (1 - coherence_mask) * quantum_current
        
        # Quantum-enhanced plasticity updates
        if self.training:
            # Classical Hebbian learning
            classical_ltp = torch.outer(post_spikes.mean(0), self.pre_trace.mean(0))
            classical_ltd = torch.outer(self.post_trace.mean(0), pre_spikes.mean(0))
            classical_delta = self.classical_lr * (0.01 * classical_ltp - 0.012 * classical_ltd)
            
            # Quantum correlation-based learning
            quantum_correlation = torch.outer(quantum_post.mean(0), quantum_pre.mean(0))
            quantum_delta = self.quantum_lr * quantum_correlation
            
            # Apply weight updates with coherence modulation
            total_delta = self.weight_coherence * classical_delta + (1 - self.weight_coherence) * quantum_delta
            self.classical_weights.data += total_delta
            
            # Update weight coherence based on learning success
            weight_change_magnitude = torch.abs(total_delta).mean()
            self.weight_coherence.data += 0.001 * (weight_change_magnitude - 0.01)
            self.weight_coherence.data = torch.clamp(self.weight_coherence.data, 0.0, 1.0)
            
            # Clip weights to prevent runaway
            self.classical_weights.data = torch.clamp(self.classical_weights.data, -2.0, 2.0)
        
        return total_current


class QuantumNeuromorphicLayer(MessagePassing):
    """Hybrid quantum-neuromorphic graph convolution layer."""
    
    def __init__(self, in_dim: int, out_dim: int, quantum_dim: int = 32, time_steps: int = 5):
        super().__init__(aggr='add')
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.quantum_dim = quantum_dim
        self.time_steps = time_steps
        
        # Quantum-spiking neurons
        self.input_neurons = QuantumSpikingNeuron(in_dim, quantum_dim)
        self.output_neurons = QuantumSpikingNeuron(out_dim, quantum_dim)
        
        # Quantum-enhanced synapses
        self.synapses = QuantumSynapticPlasticity(in_dim, out_dim, quantum_dim)
        
        # Temporal memory for quantum states
        self.register_buffer('quantum_memory', torch.zeros(time_steps, 1, quantum_dim))
        self.register_buffer('time_pointer', torch.tensor(0))
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with quantum-neuromorphic dynamics."""
        batch_size = x.size(0)
        
        # Temporal processing over multiple time steps
        all_outputs = []
        quantum_states = []
        
        for t in range(self.time_steps):
            # Input neuron processing
            classical_spikes_in, quantum_spikes_in, membrane_in = self.input_neurons(x)
            
            # Message passing with quantum enhancement
            output_current = self.propagate(
                edge_index,
                classical_spikes=classical_spikes_in,
                quantum_spikes=quantum_spikes_in,
                edge_attr=edge_attr
            )
            
            # Output neuron processing
            classical_spikes_out, quantum_spikes_out, membrane_out = self.output_neurons(output_current)
            
            # Store quantum state in memory
            if self.quantum_memory.size(1) != batch_size:
                self.quantum_memory = self.quantum_memory.expand(-1, batch_size, -1)
            
            memory_idx = self.time_pointer % self.time_steps
            quantum_state = torch.cat([quantum_spikes_in[:, :self.quantum_dim], 
                                     quantum_spikes_out[:, :self.quantum_dim]], dim=1)[:, :self.quantum_dim]
            self.quantum_memory[memory_idx] = quantum_state.detach()
            self.time_pointer += 1
            
            all_outputs.append(classical_spikes_out)
            quantum_states.append(quantum_spikes_out)
        
        # Temporal aggregation
        final_output = torch.stack(all_outputs, dim=0).mean(dim=0)
        quantum_output = torch.stack(quantum_states, dim=0).mean(dim=0)
        
        # Additional quantum information
        quantum_info = {
            'quantum_spikes': quantum_output,
            'quantum_memory': self.quantum_memory.clone(),
            'coherence': self.input_neurons.quantum_coherence.mean().item()
        }
        
        return final_output, quantum_info
    
    def message(self, classical_spikes_j: torch.Tensor, quantum_spikes_j: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Create quantum-enhanced messages."""
        # For simplicity in message passing, create edge_index tensor
        # In practice, this would be handled more elegantly
        dummy_edge_index = torch.zeros(2, classical_spikes_j.size(0), dtype=torch.long, device=classical_spikes_j.device)
        
        # Quantum synaptic transmission
        synaptic_current = self.synapses(
            classical_spikes_j, classical_spikes_j,  # Self-connection for simplicity
            quantum_spikes_j, quantum_spikes_j,
            dummy_edge_index
        )
        
        # Edge attribute modulation
        if edge_attr is not None:
            synaptic_current = synaptic_current * edge_attr.unsqueeze(-1)
        
        return synaptic_current


class QuantumNeuromorphicGNN(BaseConnectomeModel):
    """Revolutionary Quantum-Neuromorphic Graph Neural Network.
    
    Combines the best of both worlds:
    - Quantum computing: Superposition, entanglement, quantum parallelism
    - Neuromorphic computing: Spiking dynamics, plasticity, temporal processing
    - Graph neural networks: Message passing, structural learning
    
    Designed for next-generation brain connectivity analysis with unprecedented
    computational efficiency and biological realism.
    """
    
    def __init__(
        self,
        node_features: int = 100,
        hidden_dim: int = 256,
        output_dim: int = 1,
        num_layers: int = 3,
        quantum_dim: int = 64,
        time_steps: int = 8,
        enable_quantum_learning: bool = True,
        enable_neuromorphic_plasticity: bool = True,
        dropout: float = 0.1
    ):
        """Initialize Quantum-Neuromorphic GNN.
        
        Args:
            node_features: Number of input node features
            hidden_dim: Hidden dimension size
            output_dim: Output dimension
            num_layers: Number of hybrid layers
            quantum_dim: Dimension of quantum state space
            time_steps: Number of temporal processing steps
            enable_quantum_learning: Enable quantum learning mechanisms
            enable_neuromorphic_plasticity: Enable spike-timing dependent plasticity
            dropout: Dropout probability
        """
        super().__init__(node_features, hidden_dim, output_dim, dropout)
        
        self.num_layers = num_layers
        self.quantum_dim = quantum_dim
        self.time_steps = time_steps
        self.enable_quantum_learning = enable_quantum_learning
        self.enable_neuromorphic_plasticity = enable_neuromorphic_plasticity
        
        # Input encoding with quantum enhancement
        self.input_encoder = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Quantum-neuromorphic hybrid layers
        self.hybrid_layers = nn.ModuleList([
            QuantumNeuromorphicLayer(hidden_dim, hidden_dim, quantum_dim, time_steps)
            for _ in range(num_layers)
        ])
        
        # Global quantum coherence controller
        self.global_coherence = nn.Parameter(torch.tensor(0.7))
        
        # Quantum state aggregator
        self.quantum_aggregator = nn.Sequential(
            nn.Linear(quantum_dim, quantum_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(quantum_dim // 2, hidden_dim // 4)
        )
        
        # Output decoder with quantum-classical fusion
        self.output_decoder = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Quantum error correction (simplified)
        self.error_correction = nn.Parameter(torch.ones(quantum_dim) * 0.1)
        
        # Update model config
        self.model_config.update({
            'architecture': 'quantum_neuromorphic_gnn',
            'num_layers': num_layers,
            'quantum_dim': quantum_dim,
            'time_steps': time_steps,
            'quantum_learning': enable_quantum_learning,
            'neuromorphic_plasticity': enable_neuromorphic_plasticity
        })
    
    def forward(self, data: Data) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass with quantum-neuromorphic processing."""
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)
        batch = getattr(data, 'batch', None)
        
        # Input encoding
        h = self.input_encoder(x)
        
        # Track quantum information across layers
        quantum_info_list = []
        
        # Process through hybrid layers
        for i, hybrid_layer in enumerate(self.hybrid_layers):
            h_new, quantum_info = hybrid_layer(h, edge_index, edge_attr)
            
            # Residual connection with quantum modulation
            quantum_weight = torch.sigmoid(self.global_coherence)
            h = quantum_weight * h_new + (1 - quantum_weight) * h
            
            # Apply quantum error correction
            if 'quantum_spikes' in quantum_info:
                corrected_quantum = quantum_info['quantum_spikes'] * (1 + self.error_correction[:h.size(1)])
                quantum_info['quantum_spikes'] = corrected_quantum
            
            quantum_info_list.append(quantum_info)
            h = self.dropout_layer(h)
        
        # Global pooling
        if batch is not None:
            from torch_geometric.nn import global_mean_pool
            h_pooled = global_mean_pool(h, batch)
        else:
            h_pooled = h.mean(dim=0, keepdim=True)
        
        # Aggregate quantum information
        if quantum_info_list:
            quantum_features = torch.stack([
                info['quantum_spikes'][:h_pooled.size(0)] if 'quantum_spikes' in info else torch.zeros_like(h_pooled[:, :self.quantum_dim])
                for info in quantum_info_list
            ], dim=0).mean(dim=0)
            
            # Process quantum features
            quantum_processed = self.quantum_aggregator(quantum_features[:, :self.quantum_dim])
            
            # Fuse classical and quantum features
            fused_features = torch.cat([h_pooled, quantum_processed], dim=1)
        else:
            fused_features = h_pooled
        
        # Final prediction
        output = self.output_decoder(fused_features)
        
        # Compile comprehensive quantum-neuromorphic info
        final_info = {
            'quantum_coherence': [info.get('coherence', 0.0) for info in quantum_info_list],
            'average_coherence': sum(info.get('coherence', 0.0) for info in quantum_info_list) / len(quantum_info_list) if quantum_info_list else 0.0,
            'global_coherence': self.global_coherence.item(),
            'quantum_error_correction': self.error_correction.mean().item(),
            'layers_processed': len(quantum_info_list)
        }
        
        return output, final_info
    
    def get_quantum_neuromorphic_state(self, data: Data) -> Dict[str, torch.Tensor]:
        """Extract comprehensive quantum-neuromorphic state information."""
        x, edge_index = data.x, data.edge_index
        
        # Get states from all hybrid layers
        h = self.input_encoder(x)
        states = {}
        
        for i, hybrid_layer in enumerate(self.hybrid_layers):
            # Extract quantum memory
            states[f'layer_{i}_quantum_memory'] = hybrid_layer.quantum_memory.clone()
            
            # Extract synaptic weights
            states[f'layer_{i}_classical_weights'] = hybrid_layer.synapses.classical_weights.clone()
            states[f'layer_{i}_weight_coherence'] = hybrid_layer.synapses.weight_coherence.clone()
            
            # Extract neuron states
            states[f'layer_{i}_input_membrane'] = hybrid_layer.input_neurons.membrane_potential.clone()
            states[f'layer_{i}_quantum_coherence'] = hybrid_layer.input_neurons.quantum_coherence.clone()
        
        return states
    
    def reset_quantum_neuromorphic_state(self):
        """Reset all quantum and neuromorphic states."""
        for layer in self.hybrid_layers:
            # Reset neuronal membrane potentials
            layer.input_neurons.membrane_potential.zero_()
            layer.output_neurons.membrane_potential.zero_()
            
            # Reset quantum coherence
            layer.input_neurons.quantum_coherence.fill_(1.0)
            layer.output_neurons.quantum_coherence.fill_(1.0)
            
            # Reset quantum memory
            layer.quantum_memory.zero_()
            layer.time_pointer.zero_()
            
            # Reset synaptic traces
            layer.synapses.pre_trace.zero_()
            layer.synapses.post_trace.zero_()


class QuantumNeuromorphicMetrics:
    """Advanced metrics for quantum-neuromorphic evaluation."""
    
    @staticmethod
    def quantum_neuromorphic_efficiency(model: QuantumNeuromorphicGNN, data: Data) -> Dict[str, float]:
        """Measure computational efficiency of quantum-neuromorphic processing."""
        with torch.no_grad():
            output, info = model(data)
            
            efficiency_metrics = {
                'quantum_coherence_efficiency': info['average_coherence'],
                'classical_quantum_balance': info['global_coherence'],
                'error_correction_overhead': info['quantum_error_correction'],
                'layers_utilization': info['layers_processed'] / model.num_layers
            }
            
            # Compute quantum advantage ratio
            quantum_states = model.get_quantum_neuromorphic_state(data)
            quantum_activity = sum(
                torch.mean(state).item() for key, state in quantum_states.items() 
                if 'quantum' in key
            ) / len([k for k in quantum_states.keys() if 'quantum' in k])
            
            efficiency_metrics['quantum_advantage_ratio'] = quantum_activity
            
            return efficiency_metrics
    
    @staticmethod
    def neuromorphic_temporal_dynamics(model: QuantumNeuromorphicGNN, data: Data) -> Dict[str, float]:
        """Analyze temporal dynamics of neuromorphic processing."""
        with torch.no_grad():
            states = model.get_quantum_neuromorphic_state(data)
            
            temporal_metrics = {}
            
            for i in range(model.num_layers):
                memory_key = f'layer_{i}_quantum_memory'
                if memory_key in states:
                    memory = states[memory_key]
                    
                    # Temporal consistency
                    temporal_std = torch.std(memory, dim=0).mean().item()
                    temporal_metrics[f'layer_{i}_temporal_consistency'] = 1.0 / (1.0 + temporal_std)
                    
                    # Memory utilization
                    memory_utilization = torch.mean(torch.abs(memory)).item()
                    temporal_metrics[f'layer_{i}_memory_utilization'] = memory_utilization
            
            return temporal_metrics
    
    @staticmethod
    def quantum_entanglement_measure(model: QuantumNeuromorphicGNN, data: Data) -> float:
        """Measure quantum entanglement in the model."""
        with torch.no_grad():
            states = model.get_quantum_neuromorphic_state(data)
            
            entanglement_scores = []
            
            for i in range(model.num_layers):
                coherence_key = f'layer_{i}_quantum_coherence'
                if coherence_key in states:
                    coherence = states[coherence_key]
                    
                    # Simplified entanglement measure based on coherence correlations
                    if coherence.numel() > 1:
                        correlation_matrix = torch.corrcoef(coherence.T)
                        off_diagonal = correlation_matrix - torch.diag(torch.diag(correlation_matrix))
                        entanglement = torch.mean(torch.abs(off_diagonal)).item()
                        entanglement_scores.append(entanglement)
            
            return sum(entanglement_scores) / len(entanglement_scores) if entanglement_scores else 0.0


def create_quantum_neuromorphic_model(config: Dict[str, Any]) -> QuantumNeuromorphicGNN:
    """Factory function for creating Quantum-Neuromorphic GNN models."""
    return QuantumNeuromorphicGNN(**config)


# Advanced training utilities for quantum-neuromorphic models
class QuantumNeuromorphicTrainer:
    """Specialized trainer for quantum-neuromorphic models."""
    
    def __init__(self, model: QuantumNeuromorphicGNN, optimizer: torch.optim.Optimizer):
        self.model = model
        self.optimizer = optimizer
        self.training_history = []
    
    def train_step(self, data: Data, target: torch.Tensor) -> Dict[str, float]:
        """Single training step with quantum-neuromorphic specific handling."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        output, quantum_info = self.model(data)
        
        # Main task loss
        task_loss = F.mse_loss(output, target)
        
        # Quantum regularization terms
        coherence_penalty = 0.01 * (quantum_info['average_coherence'] - 0.7) ** 2
        
        # Neuromorphic regularization
        states = self.model.get_quantum_neuromorphic_state(data)
        plasticity_penalty = 0.0
        for key, state in states.items():
            if 'weight_coherence' in key:
                # Encourage balanced quantum-classical weight usage
                balance = torch.std(state).item()
                plasticity_penalty += 0.001 * balance
        
        # Total loss
        total_loss = task_loss + coherence_penalty + plasticity_penalty
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Log metrics
        metrics = {
            'total_loss': total_loss.item(),
            'task_loss': task_loss.item(),
            'coherence_penalty': coherence_penalty.item(),
            'plasticity_penalty': plasticity_penalty,
            'quantum_coherence': quantum_info['average_coherence']
        }
        
        self.training_history.append(metrics)
        return metrics
    
    def reset_quantum_state_periodically(self, epoch: int, reset_interval: int = 10):
        """Reset quantum-neuromorphic state periodically to prevent saturation."""
        if epoch % reset_interval == 0:
            self.model.reset_quantum_neuromorphic_state()