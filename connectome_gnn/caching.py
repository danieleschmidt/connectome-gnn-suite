"""Intelligent caching system for connectome data and model outputs."""

import os
import pickle
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union, Callable
import threading
import time
from functools import wraps
import numpy as np
import torch


class ConnectomeCache:
    """Intelligent cache for connectome data with adaptive policies."""
    
    def __init__(
        self,
        cache_dir: Union[str, Path] = None,
        max_size_gb: float = 1.0,
        ttl_seconds: int = 3600,
        cleanup_interval: int = 300
    ):
        """Initialize connectome cache.
        
        Args:
            cache_dir: Directory for cache storage
            max_size_gb: Maximum cache size in GB  
            ttl_seconds: Time-to-live for cache entries
            cleanup_interval: Cleanup interval in seconds
        """
        self.cache_dir = Path(cache_dir or Path.home() / ".connectome_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_size_bytes = int(max_size_gb * 1024 ** 3)
        self.ttl_seconds = ttl_seconds
        self.cleanup_interval = cleanup_interval
        
        # Thread-safe access
        self._lock = threading.RLock()
        self._access_times = {}
        self._file_sizes = {}
        
        # Metadata file
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self._load_metadata()
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._periodic_cleanup, daemon=True
        )
        self._cleanup_thread.start()
    
    def _load_metadata(self) -> None:
        """Load cache metadata."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                    self._access_times = metadata.get('access_times', {})
                    self._file_sizes = metadata.get('file_sizes', {})
        except Exception:
            self._access_times = {}
            self._file_sizes = {}
    
    def _save_metadata(self) -> None:
        """Save cache metadata."""
        try:
            metadata = {
                'access_times': self._access_times,
                'file_sizes': self._file_sizes
            }
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f)
        except Exception:
            pass
    
    def _compute_key_hash(self, key: str) -> str:
        """Compute hash for cache key."""
        return hashlib.sha256(key.encode()).hexdigest()[:16]
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key."""
        key_hash = self._compute_key_hash(key)
        return self.cache_dir / f"cache_{key_hash}.pkl"
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached item or None if not found
        """
        with self._lock:
            cache_path = self._get_cache_path(key)
            
            if not cache_path.exists():
                return None
            
            # Check TTL
            current_time = time.time()
            if key in self._access_times:
                if current_time - self._access_times[key] > self.ttl_seconds:
                    self._remove_item(key)
                    return None
            
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Update access time
                self._access_times[key] = current_time
                
                return data
                
            except Exception:
                self._remove_item(key)
                return None
    
    def put(self, key: str, value: Any) -> bool:
        """Put item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            
        Returns:
            Whether item was successfully cached
        """
        with self._lock:
            try:
                cache_path = self._get_cache_path(key)
                
                # Serialize data
                with open(cache_path, 'wb') as f:
                    pickle.dump(value, f)
                
                # Update metadata
                file_size = cache_path.stat().st_size
                current_time = time.time()
                
                self._access_times[key] = current_time
                self._file_sizes[key] = file_size
                
                # Check if cache exceeds size limit
                self._enforce_size_limit()
                
                return True
                
            except Exception:
                return False
    
    def _remove_item(self, key: str) -> None:
        """Remove item from cache."""
        cache_path = self._get_cache_path(key)
        
        try:
            if cache_path.exists():
                cache_path.unlink()
        except Exception:
            pass
        
        # Update metadata
        self._access_times.pop(key, None)
        self._file_sizes.pop(key, None)
    
    def _enforce_size_limit(self) -> None:
        """Enforce cache size limit using LRU eviction."""
        total_size = sum(self._file_sizes.values())
        
        if total_size <= self.max_size_bytes:
            return
        
        # Sort by access time (least recently used first)
        sorted_items = sorted(
            self._access_times.items(),
            key=lambda x: x[1]
        )
        
        # Remove items until under size limit
        for key, _ in sorted_items:
            if total_size <= self.max_size_bytes:
                break
            
            total_size -= self._file_sizes.get(key, 0)
            self._remove_item(key)
    
    def _periodic_cleanup(self) -> None:
        """Periodic cleanup of expired items."""
        while True:
            time.sleep(self.cleanup_interval)
            
            with self._lock:
                current_time = time.time()
                expired_keys = []
                
                for key, access_time in self._access_times.items():
                    if current_time - access_time > self.ttl_seconds:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    self._remove_item(key)
                
                self._save_metadata()
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            for key in list(self._access_times.keys()):
                self._remove_item(key)
            
            self._save_metadata()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_size = sum(self._file_sizes.values())
            
            return {
                'total_items': len(self._access_times),
                'total_size_mb': total_size / (1024 ** 2),
                'cache_utilization': total_size / self.max_size_bytes,
                'cache_dir': str(self.cache_dir),
                'max_size_gb': self.max_size_bytes / (1024 ** 3)
            }


# Global cache instance
_global_cache = None


def get_cache() -> ConnectomeCache:
    """Get global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = ConnectomeCache()
    return _global_cache


def cached(
    ttl: Optional[int] = None,
    cache_key_func: Optional[Callable] = None
) -> Callable:
    """Decorator for caching function results.
    
    Args:
        ttl: Time-to-live for cached result
        cache_key_func: Function to generate cache key
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache()
            
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Compute result and cache it
            result = func(*args, **kwargs)
            cache.put(cache_key, result)
            
            return result
        
        return wrapper
    return decorator


class ModelOutputCache:
    """Specialized cache for model outputs and embeddings."""
    
    def __init__(
        self,
        cache_dir: Union[str, Path] = None,
        max_size_gb: float = 2.0
    ):
        """Initialize model output cache.
        
        Args:
            cache_dir: Directory for cache storage
            max_size_gb: Maximum cache size in GB
        """
        self.cache_dir = Path(cache_dir or Path.home() / ".connectome_model_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_size_bytes = int(max_size_gb * 1024 ** 3)
        self._lock = threading.RLock()
    
    def cache_model_output(
        self,
        model_name: str,
        input_hash: str,
        output: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> bool:
        """Cache model output.
        
        Args:
            model_name: Name/identifier of model
            input_hash: Hash of input data
            output: Model output to cache
            
        Returns:
            Whether caching was successful
        """
        with self._lock:
            try:
                cache_key = f"{model_name}_{input_hash}"
                cache_path = self.cache_dir / f"{cache_key}.pt"
                
                # Save tensor(s)
                if isinstance(output, torch.Tensor):
                    torch.save({'output': output}, cache_path)
                elif isinstance(output, dict):
                    torch.save(output, cache_path)
                else:
                    return False
                
                return True
                
            except Exception:
                return False
    
    def get_model_output(
        self,
        model_name: str,
        input_hash: str
    ) -> Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Get cached model output.
        
        Args:
            model_name: Name/identifier of model
            input_hash: Hash of input data
            
        Returns:
            Cached output or None
        """
        with self._lock:
            try:
                cache_key = f"{model_name}_{input_hash}"
                cache_path = self.cache_dir / f"{cache_key}.pt"
                
                if not cache_path.exists():
                    return None
                
                data = torch.load(cache_path, map_location='cpu', weights_only=True)
                
                if 'output' in data:
                    return data['output']
                else:
                    return data
                    
            except Exception:
                return None
    
    def compute_input_hash(self, data: Any) -> str:
        """Compute hash of input data.
        
        Args:
            data: Input data
            
        Returns:
            Hash string
        """
        if isinstance(data, torch.Tensor):
            # Use tensor shape, dtype, and a sample of values for hash
            sample_values = data.flatten()[:100].tolist() if data.numel() > 0 else []
            hash_input = f"{data.shape}_{data.dtype}_{sample_values}"
        elif hasattr(data, 'x') and hasattr(data, 'edge_index'):
            # PyTorch Geometric Data object
            x_sample = data.x.flatten()[:100].tolist() if data.x.numel() > 0 else []
            edge_sample = data.edge_index.flatten()[:100].tolist() if data.edge_index.numel() > 0 else []
            hash_input = f"{data.x.shape}_{data.edge_index.shape}_{x_sample}_{edge_sample}"
        else:
            hash_input = str(data)
        
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]


class AdaptiveCache:
    """Adaptive cache that adjusts policies based on access patterns."""
    
    def __init__(self, base_cache: ConnectomeCache):
        """Initialize adaptive cache.
        
        Args:
            base_cache: Base cache to wrap
        """
        self.base_cache = base_cache
        self.hit_rates = {}
        self.access_patterns = {}
        self._adjustment_interval = 3600  # 1 hour
        self._last_adjustment = time.time()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item with adaptive caching."""
        # Track access pattern
        current_time = time.time()
        
        if key not in self.access_patterns:
            self.access_patterns[key] = []
        
        self.access_patterns[key].append(current_time)
        
        # Get from base cache
        result = self.base_cache.get(key)
        
        # Track hit/miss
        if key not in self.hit_rates:
            self.hit_rates[key] = {'hits': 0, 'misses': 0}
        
        if result is not None:
            self.hit_rates[key]['hits'] += 1
        else:
            self.hit_rates[key]['misses'] += 1
        
        # Periodically adjust cache policies
        if current_time - self._last_adjustment > self._adjustment_interval:
            self._adjust_cache_policies()
            self._last_adjustment = current_time
        
        return result
    
    def put(self, key: str, value: Any) -> bool:
        """Put item with adaptive caching."""
        # Predict if this item is worth caching
        if self._should_cache(key):
            return self.base_cache.put(key, value)
        return False
    
    def _should_cache(self, key: str) -> bool:
        """Determine if item should be cached based on patterns."""
        if key not in self.hit_rates:
            return True  # Cache new items by default
        
        hit_rate = self._compute_hit_rate(key)
        access_frequency = self._compute_access_frequency(key)
        
        # Cache if hit rate is high or access frequency is high
        return hit_rate > 0.5 or access_frequency > 0.1  # per hour
    
    def _compute_hit_rate(self, key: str) -> float:
        """Compute hit rate for key."""
        if key not in self.hit_rates:
            return 0.0
        
        hits = self.hit_rates[key]['hits']
        misses = self.hit_rates[key]['misses']
        total = hits + misses
        
        return hits / total if total > 0 else 0.0
    
    def _compute_access_frequency(self, key: str) -> float:
        """Compute access frequency for key (accesses per hour)."""
        if key not in self.access_patterns:
            return 0.0
        
        current_time = time.time()
        recent_accesses = [
            t for t in self.access_patterns[key]
            if current_time - t <= 3600  # Last hour
        ]
        
        return len(recent_accesses)
    
    def _adjust_cache_policies(self) -> None:
        """Adjust cache policies based on access patterns."""
        # Remove old access pattern data
        current_time = time.time()
        cutoff_time = current_time - 3600 * 24  # Keep 24 hours of data
        
        for key in self.access_patterns:
            self.access_patterns[key] = [
                t for t in self.access_patterns[key]
                if t > cutoff_time
            ]
        
        # Could implement more sophisticated policy adjustments here
        # such as adjusting TTL based on access patterns