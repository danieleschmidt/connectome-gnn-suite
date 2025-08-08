"""Optimized inference utilities for production deployment."""

import torch
import torch.nn as nn
import torch.jit as jit
import torch.onnx as onnx
import numpy as np
import time
import queue
import threading
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, Future
import json
import pickle

from ..models.base import BaseConnectomeModel


@dataclass
class InferenceConfig:
    """Configuration for optimized inference."""
    
    # Model optimization
    use_jit: bool = True
    use_half_precision: bool = False
    optimize_for_mobile: bool = False
    
    # Batching
    batch_size: int = 32
    max_batch_wait_time: float = 0.1  # seconds
    
    # Caching
    enable_caching: bool = True
    cache_size: int = 1000
    
    # Threading
    num_workers: int = 4
    queue_size: int = 100
    
    # Monitoring
    enable_profiling: bool = False
    log_predictions: bool = False


class InferenceOptimizer:
    """Optimizer for connectome model inference."""
    
    def __init__(
        self,
        model: BaseConnectomeModel,
        config: InferenceConfig = None,
        device: Optional[torch.device] = None
    ):
        """Initialize inference optimizer.
        
        Args:
            model: Connectome model to optimize
            config: Inference configuration
            device: Device for inference
        """
        self.original_model = model
        self.config = config or InferenceConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Optimize model
        self.optimized_model = self._optimize_model()
        
        # Setup caching
        if self.config.enable_caching:
            self.cache = {}
            self.cache_hits = 0
            self.cache_misses = 0
        
        # Performance tracking
        self.inference_times = []
        self.batch_sizes = []
        
        print(f"Inference optimizer initialized on {self.device}")
        if self.config.use_jit:
            print("JIT compilation enabled")
        if self.config.use_half_precision:
            print("Half precision inference enabled")
    
    def _optimize_model(self) -> nn.Module:
        """Optimize model for inference."""
        
        # Move to device and set eval mode
        model = self.original_model.to(self.device)
        model.eval()
        
        # Apply half precision if requested
        if self.config.use_half_precision and self.device.type == 'cuda':
            model = model.half()
        
        # Apply JIT compilation
        if self.config.use_jit:
            try:
                # Trace the model with a dummy input
                dummy_input = self._create_dummy_input()
                model = jit.trace(model, dummy_input)
                print("JIT tracing successful")
            except Exception as e:
                print(f"JIT tracing failed: {e}")
                print("Falling back to regular model")
        
        # Optimize for mobile if requested
        if self.config.optimize_for_mobile:
            try:
                model = torch.utils.mobile_optimizer.optimize_for_mobile(model)
                print("Mobile optimization applied")
            except Exception as e:
                print(f"Mobile optimization failed: {e}")
        
        return model
    
    def _create_dummy_input(self):
        """Create dummy input for model tracing."""
        
        # Create a simple dummy graph
        num_nodes = 90  # Standard AAL atlas size
        node_features = 100
        
        # Create dummy data
        x = torch.randn(num_nodes, node_features, device=self.device)
        
        # Create dummy edges (fully connected)
        edge_index = torch.randperm(num_nodes * num_nodes)[:num_nodes * 10]  # Sample edges
        edge_index = torch.stack([
            edge_index // num_nodes,
            edge_index % num_nodes
        ]).to(self.device)
        
        edge_attr = torch.randn(edge_index.size(1), 1, device=self.device)
        
        # Create data object (simplified)
        from torch_geometric.data import Data
        dummy_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        # Apply precision
        if self.config.use_half_precision and self.device.type == 'cuda':
            dummy_data.x = dummy_data.x.half()
            dummy_data.edge_attr = dummy_data.edge_attr.half()
        
        return dummy_data
    
    def predict(self, data, use_cache: bool = True) -> torch.Tensor:
        """Single prediction with optimizations.
        
        Args:
            data: Input connectome data
            use_cache: Whether to use caching
            
        Returns:
            Model prediction
        """
        
        # Check cache if enabled
        if self.config.enable_caching and use_cache:
            cache_key = self._get_cache_key(data)
            if cache_key in self.cache:
                self.cache_hits += 1
                return self.cache[cache_key]
            else:
                self.cache_misses += 1
        
        # Prepare input
        data = data.to(self.device)
        if self.config.use_half_precision and self.device.type == 'cuda':
            data.x = data.x.half()
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                data.edge_attr = data.edge_attr.half()
        
        # Inference
        start_time = time.time()
        
        with torch.no_grad():
            prediction = self.optimized_model(data)
        
        inference_time = time.time() - start_time
        
        # Track performance
        self.inference_times.append(inference_time)
        self.batch_sizes.append(1)
        
        # Cache result
        if self.config.enable_caching and use_cache:
            if len(self.cache) < self.config.cache_size:
                self.cache[cache_key] = prediction.clone()
        
        return prediction
    
    def predict_batch(self, data_list: List, return_individual: bool = False):
        """Batch prediction with optimizations.
        
        Args:
            data_list: List of connectome data
            return_individual: Whether to return individual predictions
            
        Returns:
            Batch predictions or list of individual predictions
        """
        
        if not data_list:
            return []
        
        # Check if we can use simple batching (same structure)
        if self._can_batch_data(data_list):
            return self._predict_simple_batch(data_list, return_individual)
        else:
            return self._predict_complex_batch(data_list)
    
    def _can_batch_data(self, data_list: List) -> bool:
        """Check if data can be easily batched."""
        
        if len(data_list) <= 1:
            return True
        
        # Check if all data have same structure
        first_data = data_list[0]
        
        for data in data_list[1:]:
            if (data.x.size() != first_data.x.size() or
                data.edge_index.size() != first_data.edge_index.size()):
                return False
        
        return True
    
    def _predict_simple_batch(self, data_list: List, return_individual: bool):
        """Predict on homogeneous batch."""
        
        # Stack data
        batch_x = torch.stack([data.x for data in data_list])
        
        # For edges, we need to offset indices for each graph in batch
        batch_edge_indices = []
        batch_edge_attrs = []
        batch_indicators = []
        
        node_offset = 0
        for i, data in enumerate(data_list):
            # Offset edge indices
            offset_edge_index = data.edge_index + node_offset
            batch_edge_indices.append(offset_edge_index)
            
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                batch_edge_attrs.append(data.edge_attr)
            
            # Create batch indicators
            num_nodes = data.x.size(0)
            batch_indicators.extend([i] * num_nodes)
            node_offset += num_nodes
        
        # Combine edges
        batch_edge_index = torch.cat(batch_edge_indices, dim=1)
        batch_edge_attr = torch.cat(batch_edge_attrs, dim=0) if batch_edge_attrs else None
        batch_indicator = torch.tensor(batch_indicators, dtype=torch.long)
        
        # Create batched data
        from torch_geometric.data import Batch
        
        # Manually create batch
        batch_data = type(data_list[0])()
        batch_data.x = torch.cat([data.x for data in data_list], dim=0)
        batch_data.edge_index = batch_edge_index
        batch_data.edge_attr = batch_edge_attr
        batch_data.batch = batch_indicator
        
        # Move to device and apply precision
        batch_data = batch_data.to(self.device)
        if self.config.use_half_precision and self.device.type == 'cuda':
            batch_data.x = batch_data.x.half()
            if batch_data.edge_attr is not None:
                batch_data.edge_attr = batch_data.edge_attr.half()
        
        # Inference
        start_time = time.time()
        
        with torch.no_grad():
            batch_predictions = self.optimized_model(batch_data)
        
        inference_time = time.time() - start_time
        
        # Track performance
        self.inference_times.append(inference_time)
        self.batch_sizes.append(len(data_list))
        
        if return_individual:
            # Split predictions back to individual
            return [batch_predictions[i:i+1] for i in range(len(data_list))]
        else:
            return batch_predictions
    
    def _predict_complex_batch(self, data_list: List):
        """Predict on heterogeneous batch (fallback to individual predictions)."""
        
        predictions = []
        
        for data in data_list:
            pred = self.predict(data, use_cache=True)
            predictions.append(pred)
        
        return predictions
    
    def _get_cache_key(self, data) -> str:
        """Generate cache key for data."""
        
        # Simple hash of node features and edge structure
        x_hash = hash(data.x.data.tobytes())
        edge_hash = hash(data.edge_index.data.tobytes())
        
        return f"{x_hash}_{edge_hash}"
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get inference performance statistics."""
        
        if not self.inference_times:
            return {"message": "No inference data available"}
        
        stats = {
            'total_inferences': len(self.inference_times),
            'mean_inference_time': np.mean(self.inference_times),
            'std_inference_time': np.std(self.inference_times),
            'min_inference_time': np.min(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'mean_batch_size': np.mean(self.batch_sizes),
            'total_samples_processed': sum(self.batch_sizes),
            'throughput_samples_per_sec': sum(self.batch_sizes) / sum(self.inference_times)
        }
        
        if self.config.enable_caching:
            cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
            stats.update({
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'cache_hit_rate': cache_hit_rate,
                'cache_size': len(self.cache)
            })
        
        return stats
    
    def export_onnx(self, output_path: str, opset_version: int = 11):
        """Export model to ONNX format.
        
        Args:
            output_path: Path to save ONNX model
            opset_version: ONNX opset version
        """
        
        dummy_input = self._create_dummy_input()
        
        # Export to ONNX
        torch.onnx.export(
            self.optimized_model,
            dummy_input,
            output_path,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['connectome_data'],
            output_names=['prediction'],
            dynamic_axes={
                'connectome_data': {0: 'num_nodes'},
                'prediction': {0: 'batch_size'}
            }
        )
        
        print(f"Model exported to ONNX: {output_path}")
    
    def save_optimized_model(self, output_path: str):
        """Save optimized model for later use.
        
        Args:
            output_path: Path to save model
        """
        
        save_data = {
            'model': self.optimized_model,
            'config': self.config,
            'performance_stats': self.get_performance_stats()
        }
        
        torch.save(save_data, output_path)
        print(f"Optimized model saved: {output_path}")


class BatchedInference:
    """Batched inference with automatic batching and queueing."""
    
    def __init__(
        self,
        optimizer: InferenceOptimizer,
        max_batch_size: int = 32,
        max_wait_time: float = 0.1,
        num_workers: int = 2
    ):
        """Initialize batched inference.
        
        Args:
            optimizer: Inference optimizer
            max_batch_size: Maximum batch size
            max_wait_time: Maximum wait time for batching
            num_workers: Number of worker threads
        """
        self.optimizer = optimizer
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.num_workers = num_workers
        
        # Queue for incoming requests
        self.request_queue = queue.Queue()
        
        # Threading setup
        self.workers = []
        self.running = False
        
        # Performance tracking
        self.requests_processed = 0
        self.batches_processed = 0
    
    def start(self):
        """Start the batched inference service."""
        
        self.running = True
        
        # Start worker threads
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._worker_loop, args=(i,))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
        
        print(f"Batched inference started with {self.num_workers} workers")
    
    def stop(self):
        """Stop the batched inference service."""
        
        self.running = False
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=1.0)
        
        print("Batched inference stopped")
    
    def predict_async(self, data) -> Future:
        """Submit prediction request asynchronously.
        
        Args:
            data: Input connectome data
            
        Returns:
            Future object for the prediction result
        """
        
        future = Future()
        request = {
            'data': data,
            'future': future,
            'timestamp': time.time()
        }
        
        self.request_queue.put(request)
        
        return future
    
    def predict_sync(self, data, timeout: float = 10.0):
        """Submit prediction request synchronously.
        
        Args:
            data: Input connectome data
            timeout: Timeout for prediction
            
        Returns:
            Prediction result
        """
        
        future = self.predict_async(data)
        return future.result(timeout=timeout)
    
    def _worker_loop(self, worker_id: int):
        """Worker loop for processing batched requests."""
        
        while self.running:
            try:
                # Collect batch
                batch_requests = self._collect_batch()
                
                if not batch_requests:
                    time.sleep(0.001)  # Short sleep to avoid busy waiting
                    continue
                
                # Process batch
                self._process_batch(batch_requests)
                
                self.batches_processed += 1
                
            except Exception as e:
                print(f"Error in worker {worker_id}: {e}")
    
    def _collect_batch(self) -> List[Dict]:
        """Collect a batch of requests from the queue."""
        
        batch = []
        start_time = time.time()
        
        while (len(batch) < self.max_batch_size and 
               (time.time() - start_time) < self.max_wait_time):
            
            try:
                request = self.request_queue.get_nowait()
                batch.append(request)
            except queue.Empty:
                if batch:  # If we have some requests, wait a bit more
                    time.sleep(0.001)
                else:
                    break
        
        return batch
    
    def _process_batch(self, batch_requests: List[Dict]):
        """Process a batch of requests."""
        
        if not batch_requests:
            return
        
        try:
            # Extract data
            data_list = [req['data'] for req in batch_requests]
            
            # Run batch inference
            predictions = self.optimizer.predict_batch(data_list, return_individual=True)
            
            # Set results
            for request, prediction in zip(batch_requests, predictions):
                request['future'].set_result(prediction)
            
            self.requests_processed += len(batch_requests)
            
        except Exception as e:
            # Set exception for all requests in batch
            for request in batch_requests:
                request['future'].set_exception(e)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batched inference statistics."""
        
        return {
            'requests_processed': self.requests_processed,
            'batches_processed': self.batches_processed,
            'queue_size': self.request_queue.qsize(),
            'avg_batch_size': self.requests_processed / max(1, self.batches_processed),
            'workers_active': sum(1 for w in self.workers if w.is_alive())
        }


class ModelServer:
    """HTTP server for connectome model inference."""
    
    def __init__(
        self,
        optimizer: InferenceOptimizer,
        host: str = "0.0.0.0",
        port: int = 8080,
        enable_batching: bool = True
    ):
        """Initialize model server.
        
        Args:
            optimizer: Inference optimizer
            host: Server host
            port: Server port
            enable_batching: Enable request batching
        """
        self.optimizer = optimizer
        self.host = host
        self.port = port
        
        # Setup batched inference if enabled
        if enable_batching:
            self.batched_inference = BatchedInference(optimizer)
        else:
            self.batched_inference = None
        
        # Server setup
        self.app = self._create_app()
    
    def _create_app(self):
        """Create Flask app for the server."""
        
        try:
            from flask import Flask, request, jsonify
            import flask
        except ImportError:
            raise ImportError("Flask is required for ModelServer. Install with: pip install flask")
        
        app = Flask(__name__)
        
        @app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'model_type': self.optimizer.original_model.__class__.__name__,
                'device': str(self.optimizer.device)
            })
        
        @app.route('/predict', methods=['POST'])
        def predict():
            """Prediction endpoint."""
            try:
                # Parse request
                data = request.get_json()
                
                if 'connectome_data' not in data:
                    return jsonify({'error': 'Missing connectome_data'}), 400
                
                # Convert to tensor
                connectome_tensor = torch.tensor(data['connectome_data'], dtype=torch.float32)
                
                # Create minimal data object
                from torch_geometric.data import Data
                connectome_data = Data(x=connectome_tensor)
                
                # Predict
                if self.batched_inference:
                    prediction = self.batched_inference.predict_sync(connectome_data)
                else:
                    prediction = self.optimizer.predict(connectome_data)
                
                # Convert to response
                result = {
                    'prediction': prediction.cpu().numpy().tolist(),
                    'status': 'success'
                }
                
                return jsonify(result)
                
            except Exception as e:
                return jsonify({
                    'error': str(e),
                    'status': 'error'
                }), 500
        
        @app.route('/stats', methods=['GET'])
        def get_stats():
            """Get server statistics."""
            stats = {
                'optimizer_stats': self.optimizer.get_performance_stats()
            }
            
            if self.batched_inference:
                stats['batching_stats'] = self.batched_inference.get_stats()
            
            return jsonify(stats)
        
        return app
    
    def start(self, debug: bool = False):
        """Start the model server.
        
        Args:
            debug: Enable debug mode
        """
        
        if self.batched_inference:
            self.batched_inference.start()
        
        print(f"Starting model server on {self.host}:{self.port}")
        
        self.app.run(
            host=self.host,
            port=self.port,
            debug=debug,
            threaded=True
        )
    
    def stop(self):
        """Stop the model server."""
        
        if self.batched_inference:
            self.batched_inference.stop()
        
        print("Model server stopped")


class InferenceBenchmark:
    """Benchmark tool for inference optimization."""
    
    def __init__(self, model: BaseConnectomeModel):
        """Initialize inference benchmark.
        
        Args:
            model: Model to benchmark
        """
        self.model = model
        self.results = {}
    
    def benchmark_configurations(
        self,
        test_data: List,
        configurations: List[InferenceConfig],
        num_runs: int = 100
    ) -> Dict[str, Any]:
        """Benchmark different inference configurations.
        
        Args:
            test_data: Test data for benchmarking
            configurations: List of configurations to test
            num_runs: Number of runs per configuration
            
        Returns:
            Benchmark results
        """
        
        results = {}
        
        for i, config in enumerate(configurations):
            config_name = f"config_{i}"
            print(f"Benchmarking {config_name}...")
            
            # Create optimizer
            optimizer = InferenceOptimizer(self.model, config)
            
            # Warmup
            for data in test_data[:5]:
                optimizer.predict(data)
            
            # Benchmark
            start_time = time.time()
            
            for _ in range(num_runs):
                for data in test_data:
                    optimizer.predict(data)
            
            total_time = time.time() - start_time
            
            # Collect results
            stats = optimizer.get_performance_stats()
            stats.update({
                'config': config,
                'total_benchmark_time': total_time,
                'samples_per_second': (num_runs * len(test_data)) / total_time
            })
            
            results[config_name] = stats
        
        self.results = results
        return results
    
    def compare_results(self) -> Dict[str, Any]:
        """Compare benchmark results."""
        
        if not self.results:
            return {"error": "No benchmark results available"}
        
        comparison = {
            'best_throughput': None,
            'best_latency': None,
            'summary': {}
        }
        
        best_throughput = 0
        best_latency = float('inf')
        
        for config_name, stats in self.results.items():
            throughput = stats.get('samples_per_second', 0)
            latency = stats.get('mean_inference_time', float('inf'))
            
            if throughput > best_throughput:
                best_throughput = throughput
                comparison['best_throughput'] = config_name
            
            if latency < best_latency:
                best_latency = latency
                comparison['best_latency'] = config_name
            
            comparison['summary'][config_name] = {
                'throughput': throughput,
                'latency': latency,
                'cache_hit_rate': stats.get('cache_hit_rate', 0)
            }
        
        return comparison