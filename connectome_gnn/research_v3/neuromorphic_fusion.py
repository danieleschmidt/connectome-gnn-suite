"""Neuromorphic-Inspired Fusion GNN for Brain-Computer Interface Applications.

Combines neuromorphic computing principles with graph neural networks
for enhanced brain connectivity analysis and real-time processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data
import numpy as np
import math
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass


@dataclass 
class SpikePattern:
    """Container for spike train patterns."""
    timestamps: torch.Tensor
    neuron_ids: torch.Tensor
    amplitudes: torch.Tensor
    frequencies: torch.Tensor


class LeakyIntegrateFireNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron model for neuromorphic processing."""
    
    def __init__(
        self, 
        input_dim: int, 
        threshold: float = 1.0,
        leak_rate: float = 0.9,
        refractory_period: int = 2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.threshold = threshold
        self.leak_rate = leak_rate
        self.refractory_period = refractory_period
        
        # Learnable parameters
        self.weights = nn.Parameter(torch.randn(input_dim, 1) * 0.1)
        self.bias = nn.Parameter(torch.zeros(1))
        
        # State variables
        self.register_buffer('membrane_potential', torch.zeros(1))
        self.register_buffer('refractory_counter', torch.zeros(1, dtype=torch.long))
        
    def forward(self, x: torch.Tensor, reset_state: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through LIF neuron."""
        batch_size, seq_len = x.shape[0], x.shape[1] if x.dim() > 2 else 1
        
        if reset_state or self.membrane_potential.shape[0] != batch_size:
            self.membrane_potential = torch.zeros(batch_size, device=x.device)
            self.refractory_counter = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        
        spikes = []
        potentials = []
        
        # Handle both 2D [batch, features] and 3D [batch, time, features] inputs
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add time dimension
        
        for t in range(seq_len):
            current_input = x[:, t]  # [batch, features]
            
            # Apply weights and bias
            input_current = torch.matmul(current_input, self.weights).squeeze(-1) + self.bias
            
            # Update membrane potential (only if not in refractory period)
            active_mask = (self.refractory_counter <= 0)
            
            # Leak current potential
            self.membrane_potential = self.membrane_potential * self.leak_rate
            
            # Add input current
            self.membrane_potential = torch.where(
                active_mask,
                self.membrane_potential + input_current,
                self.membrane_potential
            )
            
            # Check for spikes
            spike_mask = (self.membrane_potential >= self.threshold) & active_mask
            current_spikes = spike_mask.float()
            
            # Reset membrane potential for spiking neurons
            self.membrane_potential = torch.where(
                spike_mask,
                torch.zeros_like(self.membrane_potential),
                self.membrane_potential
            )
            
            # Set refractory period
            self.refractory_counter = torch.where(
                spike_mask,
                torch.full_like(self.refractory_counter, self.refractory_period),
                torch.maximum(self.refractory_counter - 1, torch.zeros_like(self.refractory_counter))
            )
            
            spikes.append(current_spikes)
            potentials.append(self.membrane_potential.clone())
        
        spike_train = torch.stack(spikes, dim=1)  # [batch, time]
        potential_trace = torch.stack(potentials, dim=1)  # [batch, time]
        
        return spike_train, potential_trace


class SpikingGraphConvolution(MessagePassing):
    """Spiking neural network based graph convolution."""
    
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        spike_threshold: float = 1.0,
        temporal_window: int = 10
    ):
        super().__init__(aggr='add')
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.spike_threshold = spike_threshold
        self.temporal_window = temporal_window
        
        # Spiking neurons for each output dimension
        self.spiking_neurons = nn.ModuleList([
            LeakyIntegrateFireNeuron(in_dim * 2, threshold=spike_threshold)  # *2 for concat of node features
            for _ in range(out_dim)
        ])
        
        # Temporal integration
        self.temporal_conv = nn.Conv1d(out_dim, out_dim, kernel_size=temporal_window, padding=temporal_window//2)
        
        # Spike rate adaptation
        self.adaptation_weights = nn.Parameter(torch.ones(out_dim))
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through spiking graph convolution."""
        # Message passing with spiking dynamics
        aggregated = self.propagate(edge_index, x=x)
        
        # Convert to spike trains
        spike_outputs = []
        
        for i, neuron in enumerate(self.spiking_neurons):
            spike_train, _ = neuron(aggregated, reset_state=True)
            spike_outputs.append(spike_train)
        
        # Stack spike trains
        spike_matrix = torch.stack(spike_outputs, dim=-1)  # [batch, time, out_dim]
        
        # Temporal integration
        spike_matrix = spike_matrix.transpose(1, 2)  # [batch, out_dim, time]
        integrated_spikes = self.temporal_conv(spike_matrix)
        integrated_spikes = integrated_spikes.transpose(1, 2)  # [batch, time, out_dim]
        
        # Convert back to rate-coded representation
        spike_rates = integrated_spikes.mean(dim=1)  # [batch, out_dim]
        
        # Apply adaptive scaling
        output = spike_rates * self.adaptation_weights
        
        return output
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        """Compute messages between connected nodes."""
        # Concatenate source and target features
        return torch.cat([x_i, x_j], dim=1)


class AdaptivePlasticityMechanism(nn.Module):
    """Adaptive synaptic plasticity mechanism based on STDP."""
    
    def __init__(self, feature_dim: int, learning_rate: float = 0.01):
        super().__init__()
        self.feature_dim = feature_dim
        self.learning_rate = learning_rate
        
        # Plasticity parameters
        self.tau_plus = nn.Parameter(torch.tensor(20.0))  # LTP time constant
        self.tau_minus = nn.Parameter(torch.tensor(20.0))  # LTD time constant
        self.a_plus = nn.Parameter(torch.tensor(0.1))  # LTP strength
        self.a_minus = nn.Parameter(torch.tensor(0.12))  # LTD strength
        
        # Synaptic weights
        self.synaptic_weights = nn.Parameter(torch.randn(feature_dim, feature_dim) * 0.1)
        
        # Spike history buffers
        self.register_buffer('pre_spike_history', torch.zeros(100, feature_dim))
        self.register_buffer('post_spike_history', torch.zeros(100, feature_dim))
        self.register_buffer('time_step', torch.tensor(0))
        
    def forward(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor) -> torch.Tensor:
        """Apply STDP-based plasticity updates."""
        batch_size = pre_spikes.shape[0]
        
        # Update spike history
        current_time = self.time_step.item() % 100
        self.pre_spike_history[current_time] = pre_spikes.mean(dim=0)
        self.post_spike_history[current_time] = post_spikes.mean(dim=0)
        self.time_step += 1
        
        # Compute STDP updates
        weight_updates = torch.zeros_like(self.synaptic_weights)
        
        for dt in range(1, min(50, self.time_step.item() + 1)):
            pre_idx = (current_time - dt) % 100
            post_idx = current_time
            
            if dt > 0:  # Post-before-pre (LTD)
                pre_trace = self.pre_spike_history[pre_idx]
                post_trace = self.post_spike_history[post_idx]
                
                ltd_update = -self.a_minus * torch.exp(-dt / self.tau_minus)
                weight_updates += ltd_update * torch.outer(post_trace, pre_trace)
                
            if dt > 0:  # Pre-before-post (LTP)
                pre_idx = current_time
                post_idx = (current_time + dt) % 100
                
                pre_trace = self.pre_spike_history[pre_idx]
                post_trace = self.post_spike_history[post_idx]
                
                ltp_update = self.a_plus * torch.exp(-dt / self.tau_plus)
                weight_updates += ltp_update * torch.outer(post_trace, pre_trace)
        
        # Apply plasticity updates
        if self.training:
            self.synaptic_weights.data += self.learning_rate * weight_updates
            # Clip weights to prevent runaway dynamics
            self.synaptic_weights.data.clamp_(-1.0, 1.0)
        
        # Apply synaptic transformation
        output = torch.matmul(pre_spikes, self.synaptic_weights.T)
        
        return output


class NeuromorphicFusionGNN(nn.Module):
    """Neuromorphic-inspired GNN with adaptive plasticity and spiking dynamics."""
    
    def __init__(
        self,
        node_features: int = 100,
        hidden_dim: int = 256,
        num_layers: int = 4,
        spike_threshold: float = 1.0,
        temporal_window: int = 20,
        plasticity_enabled: bool = True,
        output_dim: int = 1
    ):
        super().__init__()
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.spike_threshold = spike_threshold
        self.temporal_window = temporal_window
        self.plasticity_enabled = plasticity_enabled
        
        # Input projection to spiking representation
        self.input_encoder = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Spiking graph convolution layers
        self.spiking_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = SpikingGraphConvolution(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                spike_threshold=spike_threshold,
                temporal_window=temporal_window
            )
            self.spiking_layers.append(layer)
        
        # Adaptive plasticity mechanisms
        if plasticity_enabled:
            self.plasticity_layers = nn.ModuleList()
            for i in range(num_layers):
                plasticity = AdaptivePlasticityMechanism(hidden_dim)
                self.plasticity_layers.append(plasticity)
        else:
            self.plasticity_layers = None
        
        # Neuromorphic attention mechanism
        self.spike_attention = SpikeBasedAttention(hidden_dim)
        
        # Temporal dynamics modeling
        self.temporal_integrator = TemporalIntegrator(hidden_dim, temporal_window)
        
        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Homeostatic mechanisms
        self.homeostatic_controller = HomeostaticController(hidden_dim)
        
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass through neuromorphic fusion GNN."""
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, 'batch', None)
        
        # Handle feature dimension mismatch
        if x.size(1) != self.node_features:
            if x.size(1) < self.node_features:
                padding = torch.zeros(x.size(0), self.node_features - x.size(1), device=x.device)
                x = torch.cat([x, padding], dim=1)
            else:
                x = x[:, :self.node_features]
        
        # Encode to hidden representation
        x = self.input_encoder(x)
        
        # Store layer outputs for analysis
        layer_spikes = []
        layer_potentials = []
        
        # Process through spiking layers
        for i, spiking_layer in enumerate(self.spiking_layers):
            # Apply spiking graph convolution
            x_prev = x
            x = spiking_layer(x, edge_index)
            
            # Apply plasticity if enabled
            if self.plasticity_layers is not None:
                # Convert to spike trains for plasticity
                pre_spikes = torch.sigmoid(x_prev)  # Rate-coded spikes
                post_spikes = torch.sigmoid(x)
                
                x = self.plasticity_layers[i](pre_spikes, post_spikes)
            
            # Apply spike-based attention
            x = self.spike_attention(x, edge_index)
            
            # Apply homeostatic control
            x = self.homeostatic_controller(x)
            
            layer_spikes.append(x.clone())
        
        # Temporal integration across layers
        x = self.temporal_integrator(torch.stack(layer_spikes, dim=1))
        
        # Global pooling
        if batch is None:
            graph_embedding = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        else:
            graph_embedding = global_mean_pool(x, batch)
        
        # Final prediction
        output = self.output_projection(graph_embedding)
        
        return output.squeeze(-1) if output.size(1) == 1 else output
    
    def get_spike_statistics(self) -> Dict[str, float]:
        """Get statistics about spiking activity."""
        stats = {}
        
        for i, layer in enumerate(self.spiking_layers):
            # Get average firing rates from adaptation weights
            avg_firing_rate = layer.adaptation_weights.mean().item()
            stats[f'layer_{i}_firing_rate'] = avg_firing_rate
        
        return stats


class SpikeBasedAttention(nn.Module):
    """Attention mechanism based on spike timing and rates."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Spike-timing dependent attention
        self.timing_weights = nn.Parameter(torch.randn(num_heads, hidden_dim))
        self.rate_weights = nn.Parameter(torch.randn(num_heads, hidden_dim))
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Apply spike-based attention."""
        batch_size, feature_dim = x.shape
        
        # Convert features to spike rates and timing
        spike_rates = torch.sigmoid(x)  # [0, 1] representing firing rates
        spike_timing = torch.tanh(x)  # [-1, 1] representing relative timing
        
        # Multi-head attention
        head_outputs = []
        
        for head in range(self.num_heads):
            # Compute attention weights based on spike timing and rates
            timing_attn = torch.matmul(spike_timing, self.timing_weights[head])
            rate_attn = torch.matmul(spike_rates, self.rate_weights[head])
            
            # Combine timing and rate attention
            combined_attn = F.softmax(timing_attn + rate_attn, dim=0)
            
            # Apply attention to features
            attended_features = combined_attn.unsqueeze(1) * x
            head_outputs.append(attended_features)
        
        # Concatenate heads and project
        multi_head_output = torch.cat(head_outputs, dim=1)
        output = self.output_proj(multi_head_output)
        
        # Residual connection
        return output + x


class TemporalIntegrator(nn.Module):
    """Integrates temporal dynamics across network layers."""
    
    def __init__(self, hidden_dim: int, temporal_window: int = 20):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temporal_window = temporal_window
        
        # Temporal convolution for integration
        self.temporal_conv = nn.Conv1d(
            hidden_dim, 
            hidden_dim, 
            kernel_size=min(temporal_window, 5),
            padding=min(temporal_window, 5)//2
        )
        
        # LSTM for long-term dependencies
        self.lstm = nn.LSTM(
            hidden_dim, 
            hidden_dim // 2, 
            batch_first=True, 
            bidirectional=True
        )
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(self, layer_features: torch.Tensor) -> torch.Tensor:
        """Integrate temporal dynamics."""
        # layer_features: [batch, num_layers, hidden_dim]
        batch_size, num_layers, hidden_dim = layer_features.shape
        
        # Apply temporal convolution
        conv_features = layer_features.transpose(1, 2)  # [batch, hidden_dim, num_layers]
        conv_output = self.temporal_conv(conv_features)
        conv_output = conv_output.transpose(1, 2)  # [batch, num_layers, hidden_dim]
        
        # Apply LSTM
        lstm_output, _ = self.lstm(conv_output)
        
        # Gating mechanism
        gate_weights = self.gate(lstm_output)
        gated_output = lstm_output * gate_weights
        
        # Return final temporal state
        return gated_output[:, -1, :]  # [batch, hidden_dim]


class HomeostaticController(nn.Module):
    """Homeostatic plasticity controller for network stability."""
    
    def __init__(self, hidden_dim: int, target_rate: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.target_rate = target_rate
        
        # Scaling factors for homeostatic control
        self.scaling_factors = nn.Parameter(torch.ones(hidden_dim))
        
        # Moving averages for activity tracking
        self.register_buffer('activity_avg', torch.ones(hidden_dim) * target_rate)
        self.register_buffer('update_count', torch.tensor(0.0))
        
        # Homeostatic time constant
        self.tau_homeostatic = 100.0
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply homeostatic scaling."""
        # Current activity (firing rates)
        current_activity = torch.sigmoid(x).mean(dim=0)
        
        # Update moving average during training
        if self.training:
            alpha = 1.0 / (self.tau_homeostatic + self.update_count)
            self.activity_avg.data = (1 - alpha) * self.activity_avg + alpha * current_activity
            self.update_count.data += 1
        
        # Compute homeostatic scaling
        activity_error = self.target_rate - self.activity_avg
        scaling_update = 0.01 * activity_error  # Small learning rate
        
        if self.training:
            self.scaling_factors.data += scaling_update
            # Prevent scaling factors from becoming too extreme
            self.scaling_factors.data.clamp_(0.1, 10.0)
        
        # Apply homeostatic scaling
        output = x * self.scaling_factors.unsqueeze(0)
        
        return output


def create_neuromorphic_model(config: Dict) -> NeuromorphicFusionGNN:
    """Factory function for creating neuromorphic fusion models."""
    return NeuromorphicFusionGNN(
        node_features=config.get('node_features', 100),
        hidden_dim=config.get('hidden_dim', 256),
        num_layers=config.get('num_layers', 4),
        spike_threshold=config.get('spike_threshold', 1.0),
        temporal_window=config.get('temporal_window', 20),
        plasticity_enabled=config.get('plasticity_enabled', True),
        output_dim=config.get('output_dim', 1)
    )