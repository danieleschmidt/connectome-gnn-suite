"""Edge Computing Optimization for Connectome-GNN Real-Time Processing.

Advanced optimization framework for deploying connectome analysis models on edge devices,
including mobile phones, embedded systems, and IoT devices for real-time brain monitoring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import Optional, Dict, Any, Tuple, List, Union
import math
import numpy as np
import time
from collections import OrderedDict

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


class QuantizedGraphConv(nn.Module):
    """Quantized graph convolution for edge deployment."""
    
    def __init__(self, in_features: int, out_features: int, bits: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        
        # Quantized weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Quantization parameters
        self.weight_scale = nn.Parameter(torch.ones(1))
        self.weight_zero_point = nn.Parameter(torch.zeros(1))
        
        # Activation quantization
        self.activation_scale = nn.Parameter(torch.ones(1))
        self.activation_zero_point = nn.Parameter(torch.zeros(1))
        
    def quantize_weights(self) -> torch.Tensor:
        """Quantize weights to specified bit precision."""
        qmin = -(2 ** (self.bits - 1))
        qmax = 2 ** (self.bits - 1) - 1
        
        # Scale and quantize
        scaled_weights = self.weight / self.weight_scale
        quantized = torch.clamp(torch.round(scaled_weights + self.weight_zero_point), qmin, qmax)
        
        # Dequantize for computation
        dequantized = (quantized - self.weight_zero_point) * self.weight_scale
        return dequantized
    
    def quantize_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize activations."""
        qmin = 0
        qmax = 2 ** self.bits - 1
        
        scaled_x = x / self.activation_scale
        quantized = torch.clamp(torch.round(scaled_x + self.activation_zero_point), qmin, qmax)
        dequantized = (quantized - self.activation_zero_point) * self.activation_scale
        return dequantized
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantization."""
        # Quantize inputs
        x_quant = self.quantize_activations(x)
        
        # Use quantized weights
        weight_quant = self.quantize_weights()
        
        # Linear transformation
        output = F.linear(x_quant, weight_quant, self.bias)
        
        return output


class MobileGraphBlock(nn.Module):
    """Mobile-optimized graph block with depthwise separable convolutions."""
    
    def __init__(self, in_dim: int, out_dim: int, expansion_factor: int = 6):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        expanded_dim = in_dim * expansion_factor
        
        # Depthwise convolution
        self.depthwise = nn.Conv1d(in_dim, expanded_dim, kernel_size=1, groups=in_dim, bias=False)
        self.depthwise_bn = nn.BatchNorm1d(expanded_dim)
        
        # Pointwise convolution
        self.pointwise = nn.Conv1d(expanded_dim, out_dim, kernel_size=1, bias=False)
        self.pointwise_bn = nn.BatchNorm1d(out_dim)
        
        # Squeeze and excitation
        self.se_reduce = nn.Conv1d(out_dim, out_dim // 4, kernel_size=1)
        self.se_expand = nn.Conv1d(out_dim // 4, out_dim, kernel_size=1)
        
        # Residual connection
        self.use_residual = (in_dim == out_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of mobile graph block."""
        residual = x
        
        # Reshape for conv1d: [batch*nodes, features] -> [batch*nodes, features, 1]
        x = x.unsqueeze(-1)
        
        # Depthwise separable convolution
        x = F.relu6(self.depthwise_bn(self.depthwise(x)))
        x = self.pointwise_bn(self.pointwise(x))
        
        # Squeeze and excitation
        se = F.adaptive_avg_pool1d(x, 1)
        se = F.relu(self.se_reduce(se))
        se = torch.sigmoid(self.se_expand(se))
        x = x * se
        
        # Remove extra dimension
        x = x.squeeze(-1)
        
        # Residual connection
        if self.use_residual:
            x = x + residual
        
        return F.relu6(x)


class SparseGraphConv(nn.Module):
    """Sparse graph convolution optimized for edge devices."""
    
    def __init__(self, in_features: int, out_features: int, sparsity_ratio: float = 0.8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity_ratio = sparsity_ratio
        
        # Dense weights for training
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Sparse mask
        self.register_buffer('sparse_mask', torch.ones_like(self.weight))
        
        # Initialize sparse pattern
        self._initialize_sparsity()
    
    def _initialize_sparsity(self):
        """Initialize sparse pattern using magnitude-based pruning."""
        with torch.no_grad():
            weight_magnitude = torch.abs(self.weight)
            threshold = torch.quantile(weight_magnitude, self.sparsity_ratio)
            self.sparse_mask = (weight_magnitude > threshold).float()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sparse weights."""
        sparse_weight = self.weight * self.sparse_mask
        return F.linear(x, sparse_weight, self.bias)
    
    def update_sparsity(self):
        """Update sparse pattern during training."""
        if self.training:
            self._initialize_sparsity()


class EdgeOptimizedGNN(BaseConnectomeModel):
    """Edge-optimized Graph Neural Network for real-time deployment."""
    
    def __init__(
        self,
        node_features: int = 100,
        hidden_dim: int = 128,  # Reduced for edge deployment
        output_dim: int = 1,
        num_layers: int = 2,  # Fewer layers for edge
        quantization_bits: int = 8,
        sparsity_ratio: float = 0.7,
        use_mobile_blocks: bool = True,
        dropout: float = 0.1
    ):
        super().__init__(node_features, hidden_dim, output_dim, dropout)
        
        self.num_layers = num_layers
        self.quantization_bits = quantization_bits
        self.sparsity_ratio = sparsity_ratio
        self.use_mobile_blocks = use_mobile_blocks
        
        # Input projection with quantization
        self.input_proj = QuantizedGraphConv(node_features, hidden_dim, quantization_bits)
        
        # Graph layers
        if use_mobile_blocks:
            self.graph_layers = nn.ModuleList([
                MobileGraphBlock(hidden_dim, hidden_dim)
                for _ in range(num_layers)
            ])
        else:
            self.graph_layers = nn.ModuleList([
                SparseGraphConv(hidden_dim, hidden_dim, sparsity_ratio)
                for _ in range(num_layers)
            ])
        
        # Lightweight attention for global pooling
        self.attention_weights = nn.Parameter(torch.ones(hidden_dim) / hidden_dim)
        
        # Output layers with quantization
        self.output_layers = nn.Sequential(
            QuantizedGraphConv(hidden_dim, hidden_dim // 2, quantization_bits),
            nn.ReLU6(),  # ReLU6 is more hardware-friendly
            nn.Dropout(dropout),
            QuantizedGraphConv(hidden_dim // 2, output_dim, quantization_bits)
        )
        
        # Batch norm for stability
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        # Optimization flags
        self.is_quantized = False
        self.is_pruned = False
        
        self.model_config.update({
            'architecture': 'edge_optimized_gnn',
            'num_layers': num_layers,
            'quantization_bits': quantization_bits,
            'sparsity_ratio': sparsity_ratio,
            'use_mobile_blocks': use_mobile_blocks
        })
    
    def forward(self, data: Data) -> torch.Tensor:
        """Optimized forward pass for edge deployment."""
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, 'batch', None)
        
        # Input projection
        h = self.input_proj(x)
        h = F.relu6(h)  # Hardware-friendly activation
        
        # Graph layers with efficient message passing
        for i, (layer, bn) in enumerate(zip(self.graph_layers, self.batch_norms)):
            if self.use_mobile_blocks:
                h = layer(h)
            else:
                # Simple neighbor aggregation for sparse layers
                h_new = layer(h)
                
                # Efficient neighbor aggregation
                if edge_index.numel() > 0:
                    row, col = edge_index
                    aggregated = torch.zeros_like(h_new)
                    aggregated.index_add_(0, row, h_new[col])
                    h_new = h_new + aggregated
                
                h = h_new
            
            # Batch normalization
            h = bn(h)
            h = F.relu6(h)
        
        # Lightweight global pooling with attention
        if batch is not None:
            # Batch-wise pooling
            batch_size = batch.max().item() + 1
            pooled = torch.zeros(batch_size, h.size(1), device=h.device)
            
            for i in range(batch_size):
                mask = (batch == i)
                if mask.sum() > 0:
                    node_features = h[mask]
                    # Attention-based pooling
                    attention_scores = torch.softmax(
                        torch.sum(node_features * self.attention_weights, dim=1), dim=0
                    )
                    pooled[i] = torch.sum(node_features * attention_scores.unsqueeze(1), dim=0)
            h = pooled
        else:
            # Simple attention pooling
            attention_scores = torch.softmax(
                torch.sum(h * self.attention_weights, dim=1), dim=0
            )
            h = torch.sum(h * attention_scores.unsqueeze(1), dim=0, keepdim=True)
        
        # Output prediction
        output = self.output_layers(h)
        return output
    
    def quantize_model(self):
        """Apply post-training quantization."""
        self.is_quantized = True
        
        # Update quantization parameters for all quantized layers
        for module in self.modules():
            if isinstance(module, QuantizedGraphConv):
                # Simple quantization parameter estimation
                with torch.no_grad():
                    weight_min, weight_max = module.weight.min(), module.weight.max()
                    module.weight_scale.data = (weight_max - weight_min) / (2 ** module.bits - 1)
                    module.weight_zero_point.data = torch.round(-weight_min / module.weight_scale)
    
    def prune_model(self):
        """Apply structured pruning to the model."""
        self.is_pruned = True
        
        for module in self.modules():
            if isinstance(module, SparseGraphConv):
                module.update_sparsity()


class EdgeInferenceEngine:
    """Optimized inference engine for edge deployment."""
    
    def __init__(self, model: EdgeOptimizedGNN, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Performance tracking
        self.inference_times = []
        self.memory_usage = []
        
        # Optimization flags
        self.use_half_precision = (device != 'cpu')
        
        if self.use_half_precision:
            self.model = self.model.half()
    
    def preprocess_graph(self, data: Data) -> Data:
        """Preprocess graph for edge inference."""
        # Convert to appropriate precision
        if self.use_half_precision:
            data.x = data.x.half()
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                data.edge_attr = data.edge_attr.half()
        
        # Move to device
        data = data.to(self.device)
        
        return data
    
    def predict(self, data: Data) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Optimized prediction with performance monitoring."""
        start_time = time.time()
        
        # Preprocess
        data = self.preprocess_graph(data)
        
        # Inference
        with torch.no_grad():
            if self.use_half_precision:
                with torch.cuda.amp.autocast():
                    output = self.model(data)
            else:
                output = self.model(data)
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        # Memory usage (approximation)
        if torch.cuda.is_available() and self.device != 'cpu':
            memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
        else:
            memory_mb = 0  # CPU memory tracking is more complex
        
        # Track performance
        self.inference_times.append(inference_time)
        self.memory_usage.append(memory_mb)
        
        performance_metrics = {
            'inference_time_ms': inference_time * 1000,
            'memory_usage_mb': memory_mb,
            'throughput_fps': 1.0 / inference_time if inference_time > 0 else float('inf')
        }
        
        return output, performance_metrics
    
    def batch_predict(self, data_list: List[Data]) -> Tuple[List[torch.Tensor], Dict[str, float]]:
        """Batch prediction for improved throughput."""
        start_time = time.time()
        
        outputs = []
        for data in data_list:
            data = self.preprocess_graph(data)
            with torch.no_grad():
                if self.use_half_precision:
                    with torch.cuda.amp.autocast():
                        output = self.model(data)
                else:
                    output = self.model(data)
            outputs.append(output)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        performance_metrics = {
            'total_time_ms': total_time * 1000,
            'avg_time_per_sample_ms': (total_time / len(data_list)) * 1000,
            'batch_throughput_fps': len(data_list) / total_time if total_time > 0 else float('inf')
        }
        
        return outputs, performance_metrics
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get comprehensive performance summary."""
        if not self.inference_times:
            return {}
        
        return {
            'avg_inference_time_ms': np.mean(self.inference_times) * 1000,
            'p50_inference_time_ms': np.percentile(self.inference_times, 50) * 1000,
            'p95_inference_time_ms': np.percentile(self.inference_times, 95) * 1000,
            'p99_inference_time_ms': np.percentile(self.inference_times, 99) * 1000,
            'avg_memory_usage_mb': np.mean(self.memory_usage) if self.memory_usage else 0,
            'max_memory_usage_mb': np.max(self.memory_usage) if self.memory_usage else 0,
            'avg_throughput_fps': len(self.inference_times) / sum(self.inference_times) if sum(self.inference_times) > 0 else 0
        }


class ModelCompressor:
    """Comprehensive model compression for edge deployment."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.original_size = self._calculate_model_size()
    
    def _calculate_model_size(self) -> float:
        """Calculate model size in MB."""
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        return (param_size + buffer_size) / 1024 / 1024
    
    def magnitude_pruning(self, sparsity_ratio: float = 0.5) -> nn.Module:
        """Apply magnitude-based pruning."""
        for module in self.model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                with torch.no_grad():
                    weight = module.weight
                    threshold = torch.quantile(torch.abs(weight), sparsity_ratio)
                    mask = torch.abs(weight) > threshold
                    module.weight.data *= mask.float()
        
        return self.model
    
    def structured_pruning(self, prune_ratio: float = 0.3) -> nn.Module:
        """Apply structured pruning (remove entire channels/neurons)."""
        import torch.nn.utils.prune as prune
        
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=prune_ratio)
                prune.remove(module, 'weight')
        
        return self.model
    
    def knowledge_distillation(self, teacher_model: nn.Module, student_model: nn.Module,
                             train_loader, temperature: float = 3.0, alpha: float = 0.7) -> nn.Module:
        """Knowledge distillation for model compression."""
        teacher_model.eval()
        student_model.train()
        
        optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        for data in train_loader:
            optimizer.zero_grad()
            
            # Teacher predictions
            with torch.no_grad():
                teacher_output = teacher_model(data)
            
            # Student predictions
            student_output = student_model(data)
            
            # Distillation loss
            distillation_loss = criterion(
                F.softmax(student_output / temperature, dim=1),
                F.softmax(teacher_output / temperature, dim=1)
            )
            
            distillation_loss.backward()
            optimizer.step()
        
        return student_model
    
    def export_to_onnx(self, model: nn.Module, dummy_input: torch.Tensor, 
                      filepath: str = 'model.onnx') -> str:
        """Export model to ONNX format for deployment."""
        model.eval()
        
        torch.onnx.export(
            model,
            dummy_input,
            filepath,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        return filepath
    
    def get_compression_summary(self) -> Dict[str, Any]:
        """Get compression summary statistics."""
        compressed_size = self._calculate_model_size()
        compression_ratio = self.original_size / compressed_size if compressed_size > 0 else float('inf')
        
        return {
            'original_size_mb': self.original_size,
            'compressed_size_mb': compressed_size,
            'compression_ratio': compression_ratio,
            'size_reduction_percent': (1 - compressed_size / self.original_size) * 100
        }


class EdgeDeploymentPipeline:
    """Complete pipeline for edge deployment."""
    
    def __init__(self, model: nn.Module, target_device: str = 'cpu'):
        self.original_model = model
        self.target_device = target_device
        self.optimized_model = None
        self.inference_engine = None
        
    def optimize_for_edge(self, 
                         quantization_bits: int = 8,
                         sparsity_ratio: float = 0.7,
                         enable_mobile_blocks: bool = True) -> EdgeOptimizedGNN:
        """Create edge-optimized version of the model."""
        
        # Create edge-optimized architecture
        config = {
            'node_features': self.original_model.node_features,
            'hidden_dim': min(128, self.original_model.hidden_dim // 2),  # Reduce for edge
            'output_dim': self.original_model.output_dim,
            'num_layers': min(3, getattr(self.original_model, 'num_layers', 2)),
            'quantization_bits': quantization_bits,
            'sparsity_ratio': sparsity_ratio,
            'use_mobile_blocks': enable_mobile_blocks
        }
        
        self.optimized_model = EdgeOptimizedGNN(**config)
        
        # Transfer knowledge from original model (simplified)
        if hasattr(self.original_model, 'state_dict'):
            self._transfer_weights()
        
        # Apply optimizations
        self.optimized_model.quantize_model()
        self.optimized_model.prune_model()
        
        return self.optimized_model
    
    def _transfer_weights(self):
        """Transfer weights from original to optimized model."""
        original_state = self.original_model.state_dict()
        optimized_state = self.optimized_model.state_dict()
        
        # Simple weight transfer for compatible layers
        for name, param in optimized_state.items():
            if name in original_state:
                original_param = original_state[name]
                if param.shape == original_param.shape:
                    param.data.copy_(original_param.data)
    
    def create_inference_engine(self) -> EdgeInferenceEngine:
        """Create optimized inference engine."""
        if self.optimized_model is None:
            raise ValueError("Must optimize model first using optimize_for_edge()")
        
        self.inference_engine = EdgeInferenceEngine(self.optimized_model, self.target_device)
        return self.inference_engine
    
    def benchmark_performance(self, test_data: List[Data], 
                            num_runs: int = 100) -> Dict[str, Any]:
        """Benchmark edge deployment performance."""
        if self.inference_engine is None:
            raise ValueError("Must create inference engine first")
        
        # Warmup
        for _ in range(10):
            if test_data:
                self.inference_engine.predict(test_data[0])
        
        # Benchmark
        all_metrics = []
        for i in range(num_runs):
            data_idx = i % len(test_data)
            _, metrics = self.inference_engine.predict(test_data[data_idx])
            all_metrics.append(metrics)
        
        # Aggregate results
        performance_summary = self.inference_engine.get_performance_summary()
        
        # Model size analysis
        compressor = ModelCompressor(self.optimized_model)
        compression_summary = compressor.get_compression_summary()
        
        return {
            'performance': performance_summary,
            'compression': compression_summary,
            'device': self.target_device,
            'quantization_bits': getattr(self.optimized_model, 'quantization_bits', 32),
            'model_optimizations': {
                'quantized': getattr(self.optimized_model, 'is_quantized', False),
                'pruned': getattr(self.optimized_model, 'is_pruned', False),
                'mobile_blocks': getattr(self.optimized_model, 'use_mobile_blocks', False)
            }
        }


def create_edge_optimized_model(config: Dict[str, Any]) -> EdgeOptimizedGNN:
    """Factory function for creating edge-optimized models."""
    return EdgeOptimizedGNN(**config)


# Example usage and utilities
class EdgeMetrics:
    """Specialized metrics for edge deployment evaluation."""
    
    @staticmethod
    def latency_analysis(inference_engine: EdgeInferenceEngine, 
                        test_data: List[Data]) -> Dict[str, float]:
        """Comprehensive latency analysis."""
        latencies = []
        
        for data in test_data:
            _, metrics = inference_engine.predict(data)
            latencies.append(metrics['inference_time_ms'])
        
        return {
            'mean_latency_ms': np.mean(latencies),
            'median_latency_ms': np.median(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'latency_std_ms': np.std(latencies)
        }
    
    @staticmethod
    def energy_efficiency_estimate(performance_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Estimate energy efficiency (simplified model)."""
        # Simplified energy model based on inference time and memory usage
        inference_time_s = performance_metrics['performance']['avg_inference_time_ms'] / 1000
        memory_mb = performance_metrics['performance']['avg_memory_usage_mb']
        
        # Rough energy estimates (device-dependent)
        cpu_power_w = 2.0  # Typical mobile CPU power
        memory_power_w = 0.1 * memory_mb / 1000  # Memory power scaling
        
        energy_per_inference_mj = (cpu_power_w + memory_power_w) * inference_time_s
        
        return {
            'energy_per_inference_mj': energy_per_inference_mj,
            'estimated_battery_life_hours': 10000 / (energy_per_inference_mj * 3600) if energy_per_inference_mj > 0 else float('inf'),
            'efficiency_score': 1.0 / energy_per_inference_mj if energy_per_inference_mj > 0 else float('inf')
        }
    
    @staticmethod
    def accuracy_vs_efficiency_tradeoff(original_model: nn.Module, 
                                       edge_model: EdgeOptimizedGNN,
                                       test_data: List[Data],
                                       test_labels: List[torch.Tensor]) -> Dict[str, float]:
        """Analyze accuracy vs efficiency tradeoff."""
        # Accuracy comparison
        original_model.eval()
        edge_model.eval()
        
        original_preds = []
        edge_preds = []
        
        with torch.no_grad():
            for data in test_data:
                orig_pred = original_model(data)
                edge_pred = edge_model(data)
                original_preds.append(orig_pred)
                edge_preds.append(edge_pred)
        
        # Calculate accuracy metrics
        original_acc = sum(torch.abs(pred - label).mean().item() 
                          for pred, label in zip(original_preds, test_labels)) / len(test_labels)
        edge_acc = sum(torch.abs(pred - label).mean().item() 
                      for pred, label in zip(edge_preds, test_labels)) / len(test_labels)
        
        accuracy_degradation = (original_acc - edge_acc) / original_acc * 100 if original_acc > 0 else 0
        
        # Size comparison
        original_size = sum(p.numel() for p in original_model.parameters())
        edge_size = sum(p.numel() for p in edge_model.parameters())
        size_reduction = (1 - edge_size / original_size) * 100 if original_size > 0 else 0
        
        return {
            'original_accuracy': original_acc,
            'edge_accuracy': edge_acc,
            'accuracy_degradation_percent': accuracy_degradation,
            'model_size_reduction_percent': size_reduction,
            'efficiency_ratio': size_reduction / max(accuracy_degradation, 0.1)  # Avoid division by zero
        }