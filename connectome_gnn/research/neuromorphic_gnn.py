"""Neuromorphic Graph Neural Network for Brain-Like Computing.

Implements neuromorphic computing principles in graph neural networks,
including spiking neurons, synaptic plasticity, and event-driven processing
for biologically-inspired connectome analysis.
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
except ImportError:
    class BaseConnectomeModel(nn.Module):
        def __init__(self, node_features, hidden_dim, output_dim, dropout=0.1):
            super().__init__()
            self.node_features = node_features
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.dropout_layer = nn.Dropout(dropout)
            self.model_config = {}


class SpikingNeuron(nn.Module):
    """Leaky Integrate-and-Fire (LIF) spiking neuron model."""
    
    def __init__(self, input_dim: int, threshold: float = 1.0, decay: float = 0.9, 
                 reset_voltage: float = 0.0, refractory_period: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.threshold = nn.Parameter(torch.tensor(threshold))
        self.decay = nn.Parameter(torch.tensor(decay))
        self.reset_voltage = reset_voltage
        self.refractory_period = refractory_period
        
        # Membrane potential and refractory counter
        self.register_buffer('membrane_potential', torch.zeros(1, input_dim))
        self.register_buffer('refractory_counter', torch.zeros(1, input_dim, dtype=torch.int))
        
        # Learnable integration parameters
        self.integration_weights = nn.Parameter(torch.randn(input_dim))
        
    def forward(self, input_current: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of spiking neuron.
        
        Args:
            input_current: Input current [batch, features]
            
        Returns:
            spikes: Binary spike tensor [batch, features]
            membrane_potential: Continuous membrane potential [batch, features]
        """
        batch_size = input_current.size(0)
        
        # Expand membrane potential for batch
        if self.membrane_potential.size(0) != batch_size:
            self.membrane_potential = self.membrane_potential.expand(batch_size, -1).contiguous()
            self.refractory_counter = self.refractory_counter.expand(batch_size, -1).contiguous()
        
        # Decay membrane potential
        self.membrane_potential *= self.decay
        
        # Add input current (only for non-refractory neurons)
        active_mask = (self.refractory_counter == 0).float()
        weighted_input = input_current * self.integration_weights.abs()
        self.membrane_potential += weighted_input * active_mask
        
        # Generate spikes when threshold is exceeded
        spikes = (self.membrane_potential > self.threshold).float()
        
        # Reset membrane potential where spikes occurred
        self.membrane_potential = torch.where(
            spikes.bool(), 
            torch.full_like(self.membrane_potential, self.reset_voltage),
            self.membrane_potential
        )
        
        # Update refractory counters
        self.refractory_counter = torch.where(
            spikes.bool(),
            torch.full_like(self.refractory_counter, self.refractory_period),
            torch.clamp(self.refractory_counter - 1, min=0)
        )
        
        return spikes, self.membrane_potential.clone()
    
    def reset_state(self):
        """Reset neuron state."""
        self.membrane_potential.zero_()
        self.refractory_counter.zero_()


class SynapticPlasticity(nn.Module):
    """Spike-timing dependent plasticity (STDP) mechanism."""
    
    def __init__(self, input_dim: int, output_dim: int, 
                 tau_plus: float = 20.0, tau_minus: float = 20.0,
                 a_plus: float = 0.01, a_minus: float = 0.012):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.a_plus = a_plus
        self.a_minus = a_minus
        
        # Synaptic weights
        self.weights = nn.Parameter(torch.randn(output_dim, input_dim) * 0.1)
        
        # Spike timing traces
        self.register_buffer('pre_trace', torch.zeros(1, input_dim))
        self.register_buffer('post_trace', torch.zeros(1, output_dim))
        
        # Learning rate parameter
        self.learning_rate = nn.Parameter(torch.tensor(0.001))
        
    def forward(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor) -> torch.Tensor:
        """Apply synaptic transmission and plasticity.
        
        Args:
            pre_spikes: Presynaptic spikes [batch, input_dim]
            post_spikes: Postsynaptic spikes [batch, output_dim]
            
        Returns:
            synaptic_current: Output synaptic current [batch, output_dim]
        """
        batch_size = pre_spikes.size(0)
        
        # Expand traces for batch
        if self.pre_trace.size(0) != batch_size:
            self.pre_trace = self.pre_trace.expand(batch_size, -1).contiguous()
            self.post_trace = self.post_trace.expand(batch_size, -1).contiguous()
        
        # Update spike traces
        self.pre_trace = self.pre_trace * math.exp(-1/self.tau_plus) + pre_spikes
        self.post_trace = self.post_trace * math.exp(-1/self.tau_minus) + post_spikes
        
        # Compute synaptic current
        synaptic_current = F.linear(pre_spikes, self.weights)
        
        # STDP weight updates (when training)
        if self.training:
            # Long-term potentiation (LTP): post-spike strengthens recently active pre-synapses
            ltp = torch.outer(post_spikes.mean(0), self.pre_trace.mean(0))
            
            # Long-term depression (LTD): pre-spike weakens recently active post-synapses
            ltd = torch.outer(self.post_trace.mean(0), pre_spikes.mean(0))
            
            # Apply weight changes
            weight_delta = self.learning_rate * (self.a_plus * ltp - self.a_minus * ltd)
            self.weights.data += weight_delta
            
            # Clip weights to prevent runaway
            self.weights.data = torch.clamp(self.weights.data, -1.0, 1.0)
        
        return synaptic_current


class EventDrivenConv(MessagePassing):
    """Event-driven graph convolution with sparse spike processing."""
    
    def __init__(self, in_dim: int, out_dim: int, spike_threshold: float = 0.5):
        super().__init__(aggr='add')
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.spike_threshold = spike_threshold
        
        # Spiking neurons for input processing
        self.input_neurons = SpikingNeuron(in_dim)
        
        # Synaptic connections
        self.synapses = SynapticPlasticity(in_dim, out_dim)
        
        # Output integration
        self.output_neurons = SpikingNeuron(out_dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Event-driven forward pass.
        
        Returns:
            spike_output: Spike trains [num_nodes, out_dim]
            analog_output: Continuous membrane potentials [num_nodes, out_dim]
        """
        # Convert input to spikes
        input_spikes, input_membrane = self.input_neurons(x)
        
        # Only process nodes with significant activity (event-driven)
        active_nodes = torch.any(input_spikes > self.spike_threshold, dim=1)
        
        if not torch.any(active_nodes):
            # No active nodes, return zeros
            return torch.zeros_like(x[:, :self.out_dim]), torch.zeros_like(x[:, :self.out_dim])
        
        # Sparse message passing only for active nodes
        sparse_output = self.propagate(
            edge_index, 
            x=input_spikes, 
            edge_attr=edge_attr,
            active_mask=active_nodes
        )
        
        # Generate output spikes
        output_spikes, output_membrane = self.output_neurons(sparse_output)
        
        return output_spikes, output_membrane
    
    def message(self, x_j: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Create spike-based messages."""
        # Synaptic transmission
        synaptic_current = self.synapses(x_j, x_j)  # Self-connection for simplicity
        
        # Modulate by edge attributes if provided
        if edge_attr is not None:
            synaptic_current = synaptic_current * edge_attr.unsqueeze(-1)
        
        return synaptic_current
    
    def reset_neurons(self):
        """Reset all neuronal states."""
        self.input_neurons.reset_state()
        self.output_neurons.reset_state()


class AdaptiveThreshold(nn.Module):
    """Adaptive threshold mechanism based on recent activity."""
    
    def __init__(self, input_dim: int, adaptation_rate: float = 0.01):
        super().__init__()
        self.input_dim = input_dim
        self.adaptation_rate = adaptation_rate
        
        # Base and adaptive thresholds
        self.base_threshold = nn.Parameter(torch.ones(input_dim))
        self.register_buffer('adaptive_threshold', torch.ones(1, input_dim))
        self.register_buffer('activity_trace', torch.zeros(1, input_dim))
        
    def forward(self, activity: torch.Tensor) -> torch.Tensor:
        """Compute adaptive threshold based on recent activity."""
        batch_size = activity.size(0)
        
        # Expand buffers for batch
        if self.activity_trace.size(0) != batch_size:
            self.activity_trace = self.activity_trace.expand(batch_size, -1).contiguous()
            self.adaptive_threshold = self.adaptive_threshold.expand(batch_size, -1).contiguous()
        
        # Update activity trace
        self.activity_trace = 0.9 * self.activity_trace + 0.1 * activity
        
        # Adapt threshold based on recent activity
        target_activity = 0.1  # Target 10% activation
        threshold_adjustment = self.adaptation_rate * (self.activity_trace - target_activity)
        self.adaptive_threshold += threshold_adjustment
        
        # Combine base and adaptive components
        total_threshold = self.base_threshold + self.adaptive_threshold.mean(0)
        
        return total_threshold.clamp(min=0.1, max=2.0)


class NeuromorphicGNN(BaseConnectomeModel):
    """Neuromorphic Graph Neural Network with spiking dynamics.
    
    Implements brain-inspired computing principles:
    - Spiking neural dynamics with membrane potentials
    - Spike-timing dependent plasticity (STDP)
    - Event-driven sparse processing
    - Adaptive thresholds and homeostasis
    - Temporal dynamics and memory
    """
    
    def __init__(
        self,
        node_features: int = 100,
        hidden_dim: int = 256,
        output_dim: int = 1,
        num_layers: int = 3,
        spike_threshold: float = 1.0,
        time_steps: int = 10,
        enable_plasticity: bool = True,
        dropout: float = 0.1
    ):
        """Initialize Neuromorphic GNN.
        
        Args:
            node_features: Number of input node features
            hidden_dim: Hidden dimension size
            output_dim: Output dimension
            num_layers: Number of neuromorphic layers
            spike_threshold: Threshold for spike generation
            time_steps: Number of time steps for temporal dynamics
            enable_plasticity: Whether to enable synaptic plasticity
            dropout: Dropout probability
        """
        super().__init__(node_features, hidden_dim, output_dim, dropout)
        
        self.num_layers = num_layers
        self.spike_threshold = spike_threshold
        self.time_steps = time_steps
        self.enable_plasticity = enable_plasticity
        
        # Input encoding to spikes
        self.input_encoder = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Neuromorphic graph convolution layers
        self.neuromorphic_layers = nn.ModuleList([
            EventDrivenConv(hidden_dim, hidden_dim, spike_threshold)
            for _ in range(num_layers)
        ])
        
        # Adaptive threshold mechanisms
        self.adaptive_thresholds = nn.ModuleList([
            AdaptiveThreshold(hidden_dim) for _ in range(num_layers)
        ])
        
        # Temporal memory for spike history
        self.register_buffer('spike_history', torch.zeros(time_steps, 1, hidden_dim))
        self.register_buffer('time_step_counter', torch.tensor(0))
        
        # Output decoder from spikes to continuous values
        self.spike_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Homeostatic mechanisms
        self.homeostatic_controller = nn.Parameter(torch.ones(hidden_dim))
        
        # Update model config
        self.model_config.update({
            'architecture': 'neuromorphic_gnn',
            'num_layers': num_layers,
            'spike_threshold': spike_threshold,
            'time_steps': time_steps,
            'enable_plasticity': enable_plasticity
        })
    
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass with neuromorphic dynamics."""
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)
        batch = getattr(data, 'batch', None)
        
        batch_size = x.size(0)
        
        # Encode input to initial activation
        h = self.input_encoder(x)
        
        # Temporal processing over multiple time steps
        spike_outputs = []
        analog_outputs = []
        
        for t in range(self.time_steps):
            layer_input = h
            
            # Process through neuromorphic layers
            for i, (neuromophic_layer, adaptive_thresh) in enumerate(
                zip(self.neuromorphic_layers, self.adaptive_thresholds)
            ):
                # Get adaptive threshold for current activity
                current_threshold = adaptive_thresh(layer_input.abs())
                
                # Update layer threshold
                neuromophic_layer.input_neurons.threshold.data = current_threshold.mean()
                neuromophic_layer.output_neurons.threshold.data = current_threshold.mean()
                
                # Event-driven processing
                spikes, membrane = neuromophic_layer(layer_input, edge_index, edge_attr)
                
                # Apply homeostatic scaling
                spikes = spikes * self.homeostatic_controller
                membrane = membrane * self.homeostatic_controller
                
                # Use membrane potential for next layer
                layer_input = membrane
                
            spike_outputs.append(spikes)
            analog_outputs.append(membrane)
            
            # Store in temporal memory
            if self.spike_history.size(1) != batch_size:
                self.spike_history = self.spike_history.expand(-1, batch_size, -1)
            
            history_idx = self.time_step_counter % self.time_steps
            self.spike_history[history_idx] = spikes.detach()
            self.time_step_counter += 1
        
        # Aggregate temporal outputs
        final_spikes = torch.stack(spike_outputs, dim=0).mean(dim=0)  # Temporal average
        final_analog = torch.stack(analog_outputs, dim=0).mean(dim=0)
        
        # Global pooling
        if batch is not None:
            from torch_geometric.nn import global_mean_pool
            pooled_spikes = global_mean_pool(final_spikes, batch)
            pooled_analog = global_mean_pool(final_analog, batch)
        else:
            pooled_spikes = final_spikes.mean(dim=0, keepdim=True)
            pooled_analog = final_analog.mean(dim=0, keepdim=True)
        
        # Combine spike and analog information
        combined_output = 0.7 * pooled_spikes + 0.3 * pooled_analog
        
        # Decode to final prediction
        output = self.spike_decoder(combined_output)
        
        return output
    
    def get_spike_statistics(self) -> Dict[str, float]:
        """Get statistics about spiking activity."""
        stats = {}
        
        # Spike rate from history
        if torch.sum(self.spike_history) > 0:
            stats['spike_rate'] = torch.mean(self.spike_history).item()
            stats['spike_sparsity'] = torch.mean((self.spike_history > 0).float()).item()
        else:
            stats['spike_rate'] = 0.0
            stats['spike_sparsity'] = 0.0
        
        # Layer-wise statistics
        for i, layer in enumerate(self.neuromorphic_layers):
            if hasattr(layer.input_neurons, 'membrane_potential'):
                membrane_var = torch.var(layer.input_neurons.membrane_potential).item()
                stats[f'layer_{i}_membrane_variance'] = membrane_var
        
        return stats
    
    def reset_neuronal_states(self):
        """Reset all neuronal states in the network."""
        for layer in self.neuromorphic_layers:
            layer.reset_neurons()
        
        self.spike_history.zero_()
        self.time_step_counter.zero_()
    
    def enable_plasticity_learning(self, enable: bool = True):
        """Enable or disable synaptic plasticity."""
        for layer in self.neuromorphic_layers:
            layer.synapses.learning_rate.requires_grad = enable
            for param in layer.synapses.parameters():
                param.requires_grad = enable


class NeuromorphicMetrics:
    """Specialized metrics for neuromorphic GNN evaluation."""
    
    @staticmethod
    def spike_train_similarity(spikes1: torch.Tensor, spikes2: torch.Tensor) -> float:
        """Compute similarity between spike trains using Victor-Purpura distance."""
        # Simplified version - count coincident spikes
        coincident_spikes = torch.sum(spikes1 * spikes2)
        total_spikes = torch.sum(spikes1) + torch.sum(spikes2)
        
        if total_spikes == 0:
            return 1.0
        
        similarity = 2 * coincident_spikes / total_spikes
        return similarity.item()
    
    @staticmethod
    def temporal_precision(model: NeuromorphicGNN) -> float:
        """Measure temporal precision of spike patterns."""
        spike_history = model.spike_history
        
        # Compute inter-spike interval statistics
        spike_times = torch.nonzero(spike_history, as_tuple=False)
        if len(spike_times) < 2:
            return 0.0
        
        intervals = torch.diff(spike_times[:, 0].float())
        cv = torch.std(intervals) / torch.mean(intervals)  # Coefficient of variation
        precision = 1.0 / (1.0 + cv)  # Higher precision = lower CV
        
        return precision.item()
    
    @staticmethod
    def synaptic_efficacy(model: NeuromorphicGNN) -> Dict[str, float]:
        """Measure synaptic efficacy across layers."""
        efficacy_stats = {}
        
        for i, layer in enumerate(model.neuromorphic_layers):
            weights = layer.synapses.weights
            
            # Average synaptic strength
            avg_strength = torch.mean(torch.abs(weights)).item()
            efficacy_stats[f'layer_{i}_avg_strength'] = avg_strength
            
            # Weight distribution entropy
            weight_probs = F.softmax(weights.flatten(), dim=0)
            entropy = -torch.sum(weight_probs * torch.log(weight_probs + 1e-8)).item()
            efficacy_stats[f'layer_{i}_weight_entropy'] = entropy
        
        return efficacy_stats


def create_neuromorphic_model(config: Dict[str, Any]) -> NeuromorphicGNN:
    """Factory function for creating Neuromorphic GNN models."""
    return NeuromorphicGNN(**config)