"""Intelligent caching system for improved performance."""

import hashlib
import pickle
import json
import time
import threading
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass
import logging
import weakref

from ..core.utils import safe_import, Timer
from ..robust.logging_config import get_logger


@dataclass
class CacheEntry:
    """Represents a cached entry with metadata."""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    last_access: float = 0.0
    size_bytes: int = 0
    ttl: Optional[float] = None  # Time to live in seconds


class LRUCache:
    """Thread-safe LRU cache implementation."""
    
    def __init__(self, max_size: int = 1000, logger: Optional[logging.Logger] = None):
        self.max_size = max_size
        self.logger = logger or get_logger("lru_cache")
        self._cache = OrderedDict()
        self._lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            self._stats['total_requests'] += 1
            
            if key in self._cache:
                # Move to end (most recently used)
                entry = self._cache.pop(key)
                entry.access_count += 1
                entry.last_access = time.time()
                self._cache[key] = entry
                
                # Check TTL
                if entry.ttl and (time.time() - entry.timestamp) > entry.ttl:
                    del self._cache[key]
                    self._stats['misses'] += 1
                    return None
                
                self._stats['hits'] += 1
                return entry.value
            else:
                self._stats['misses'] += 1
                return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        """Put value in cache."""
        with self._lock:
            # Calculate size estimate
            try:
                size_bytes = len(pickle.dumps(value))
            except:
                size_bytes = 0
            
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                last_access=time.time(),
                size_bytes=size_bytes,
                ttl=ttl
            )
            
            if key in self._cache:
                # Update existing entry
                del self._cache[key]
            
            self._cache[key] = entry
            
            # Evict if necessary
            while len(self._cache) > self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._stats['evictions'] += 1
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_size_bytes = sum(entry.size_bytes for entry in self._cache.values())
            hit_rate = self._stats['hits'] / max(1, self._stats['total_requests'])
            
            return {
                **self._stats,
                'current_size': len(self._cache),
                'max_size': self.max_size,
                'hit_rate': hit_rate,
                'total_size_bytes': total_size_bytes,
                'avg_size_bytes': total_size_bytes / max(1, len(self._cache))
            }


class SmartCache:
    """Intelligent cache with multiple eviction strategies and persistence."""
    
    def __init__(self, 
                 max_memory_mb: float = 100,
                 persist_to_disk: bool = False,
                 cache_dir: Optional[str] = None,
                 eviction_strategy: str = 'lru',
                 logger: Optional[logging.Logger] = None):
        
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.persist_to_disk = persist_to_disk
        self.cache_dir = Path(cache_dir or '/tmp/connectome_gnn_cache')
        self.eviction_strategy = eviction_strategy
        self.logger = logger or get_logger("smart_cache")
        
        # Cache storage
        self._cache = {}
        self._lock = threading.RLock()
        self._current_memory = 0
        
        # Statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'disk_reads': 0,
            'disk_writes': 0,
            'total_requests': 0
        }
        
        # Setup disk cache if enabled
        if self.persist_to_disk:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_disk_cache_index()
    
    def _compute_key_hash(self, key: str) -> str:
        """Compute hash for cache key."""
        return hashlib.sha256(key.encode()).hexdigest()[:16]
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value."""
        try:
            return len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        except:
            # Fallback estimation
            if hasattr(value, '__sizeof__'):
                return value.__sizeof__()
            return 1000  # Default estimate
    
    def _evict_entries(self, target_bytes: int):
        """Evict entries based on strategy."""
        if self.eviction_strategy == 'lru':
            # Evict least recently used
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: x[1].last_access
            )
        elif self.eviction_strategy == 'lfu':
            # Evict least frequently used
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: x[1].access_count
            )
        elif self.eviction_strategy == 'size':
            # Evict largest entries first
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: x[1].size_bytes,
                reverse=True
            )
        else:
            # Default to LRU
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: x[1].last_access
            )
        
        bytes_freed = 0
        for key, entry in sorted_entries:
            if bytes_freed >= target_bytes:
                break
            
            # Move to disk if persistence enabled
            if self.persist_to_disk:
                self._write_to_disk(key, entry)
            
            bytes_freed += entry.size_bytes
            self._current_memory -= entry.size_bytes
            del self._cache[key]
            self._stats['evictions'] += 1
        
        self.logger.debug(f"Evicted {bytes_freed} bytes from cache")
    
    def _write_to_disk(self, key: str, entry: CacheEntry):
        """Write cache entry to disk."""
        try:
            key_hash = self._compute_key_hash(key)
            filepath = self.cache_dir / f"{key_hash}.cache"
            
            with open(filepath, 'wb') as f:
                pickle.dump(entry, f)
            
            self._stats['disk_writes'] += 1
            self.logger.debug(f"Wrote cache entry to disk: {key}")
            
        except Exception as e:
            self.logger.warning(f"Failed to write cache entry to disk: {e}")
    
    def _read_from_disk(self, key: str) -> Optional[CacheEntry]:
        """Read cache entry from disk."""
        try:
            key_hash = self._compute_key_hash(key)
            filepath = self.cache_dir / f"{key_hash}.cache"
            
            if not filepath.exists():
                return None
            
            with open(filepath, 'rb') as f:
                entry = pickle.load(f)
            
            self._stats['disk_reads'] += 1
            self.logger.debug(f"Read cache entry from disk: {key}")
            
            return entry
            
        except Exception as e:
            self.logger.warning(f"Failed to read cache entry from disk: {e}")
            return None
    
    def _load_disk_cache_index(self):
        """Load existing disk cache entries."""
        if not self.cache_dir.exists():
            return
        
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                with open(cache_file, 'rb') as f:
                    entry = pickle.load(f)
                
                # Check if entry is still valid
                if entry.ttl and (time.time() - entry.timestamp) > entry.ttl:
                    cache_file.unlink()  # Remove expired file
                    continue
                
                # Add to index (but don't load value into memory yet)
                # This is just for tracking what's available on disk
                
            except Exception as e:
                self.logger.debug(f"Failed to load cache file {cache_file}: {e}")
                # Remove corrupted file
                try:
                    cache_file.unlink()
                except:
                    pass
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            self._stats['total_requests'] += 1
            
            # Check memory cache first
            if key in self._cache:
                entry = self._cache[key]
                
                # Check TTL
                if entry.ttl and (time.time() - entry.timestamp) > entry.ttl:
                    del self._cache[key]
                    self._current_memory -= entry.size_bytes
                    self._stats['misses'] += 1
                    return None
                
                # Update access statistics
                entry.access_count += 1
                entry.last_access = time.time()
                
                self._stats['hits'] += 1
                return entry.value
            
            # Check disk cache if enabled
            if self.persist_to_disk:
                disk_entry = self._read_from_disk(key)
                if disk_entry:
                    # Check TTL
                    if disk_entry.ttl and (time.time() - disk_entry.timestamp) > disk_entry.ttl:
                        # Remove expired disk entry
                        key_hash = self._compute_key_hash(key)
                        filepath = self.cache_dir / f"{key_hash}.cache"
                        try:
                            filepath.unlink()
                        except:
                            pass
                        self._stats['misses'] += 1
                        return None
                    
                    # Load back into memory cache
                    disk_entry.access_count += 1
                    disk_entry.last_access = time.time()
                    
                    # Check if we need to evict first
                    if self._current_memory + disk_entry.size_bytes > self.max_memory_bytes:
                        self._evict_entries(disk_entry.size_bytes)
                    
                    self._cache[key] = disk_entry
                    self._current_memory += disk_entry.size_bytes
                    
                    self._stats['hits'] += 1
                    return disk_entry.value
            
            self._stats['misses'] += 1
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        """Put value in cache."""
        with self._lock:
            size_bytes = self._estimate_size(value)
            
            # Check if value is too large for cache
            if size_bytes > self.max_memory_bytes:
                self.logger.warning(f"Value too large for cache: {size_bytes} bytes")
                return
            
            # Remove existing entry if present
            if key in self._cache:
                old_entry = self._cache[key]
                self._current_memory -= old_entry.size_bytes
                del self._cache[key]
            
            # Evict entries if necessary
            if self._current_memory + size_bytes > self.max_memory_bytes:
                target_eviction = self._current_memory + size_bytes - self.max_memory_bytes
                self._evict_entries(target_eviction)
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                last_access=time.time(),
                size_bytes=size_bytes,
                ttl=ttl
            )
            
            self._cache[key] = entry
            self._current_memory += size_bytes
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._current_memory = 0
            
            # Clear disk cache if enabled
            if self.persist_to_disk:
                for cache_file in self.cache_dir.glob("*.cache"):
                    try:
                        cache_file.unlink()
                    except:
                        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            hit_rate = self._stats['hits'] / max(1, self._stats['total_requests'])
            
            return {
                **self._stats,
                'memory_usage_mb': self._current_memory / 1024 / 1024,
                'memory_limit_mb': self.max_memory_bytes / 1024 / 1024,
                'memory_utilization': self._current_memory / self.max_memory_bytes,
                'entries_in_memory': len(self._cache),
                'hit_rate': hit_rate,
                'eviction_strategy': self.eviction_strategy,
                'disk_persistence': self.persist_to_disk
            }


class ResultCache:
    """Cache for function results with automatic key generation."""
    
    def __init__(self, cache: Union[LRUCache, SmartCache], 
                 logger: Optional[logging.Logger] = None):
        self.cache = cache
        self.logger = logger or get_logger("result_cache")
    
    def _generate_key(self, func: Callable, args: Tuple, kwargs: Dict) -> str:
        """Generate cache key for function call."""
        key_parts = [
            func.__name__,
            func.__module__,
            str(hash(args)),
            str(hash(tuple(sorted(kwargs.items()))))
        ]
        return "|".join(key_parts)
    
    def cached_call(self, func: Callable, *args, ttl: Optional[float] = None, **kwargs) -> Any:
        """Call function with caching."""
        cache_key = self._generate_key(func, args, kwargs)
        
        # Try to get from cache
        result = self.cache.get(cache_key)
        if result is not None:
            self.logger.debug(f"Cache hit for {func.__name__}")
            return result
        
        # Call function and cache result
        self.logger.debug(f"Cache miss for {func.__name__}, executing function")
        result = func(*args, **kwargs)
        self.cache.put(cache_key, result, ttl=ttl)
        
        return result
    
    def cache_decorator(self, ttl: Optional[float] = None):
        """Decorator for caching function results."""
        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                return self.cached_call(func, *args, ttl=ttl, **kwargs)
            return wrapper
        return decorator


class ModelCache:
    """Specialized cache for machine learning models."""
    
    def __init__(self, cache_dir: str, 
                 max_models: int = 5,
                 logger: Optional[logging.Logger] = None):
        self.cache_dir = Path(cache_dir)
        self.max_models = max_models
        self.logger = logger or get_logger("model_cache")
        self.torch = safe_import('torch')
        
        # Model metadata
        self.model_index = {}
        self._lock = threading.RLock()
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing model index
        self._load_model_index()
    
    def _load_model_index(self):
        """Load model index from disk."""
        index_file = self.cache_dir / "model_index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    self.model_index = json.load(f)
                self.logger.info(f"Loaded model index with {len(self.model_index)} entries")
            except Exception as e:
                self.logger.warning(f"Failed to load model index: {e}")
                self.model_index = {}
    
    def _save_model_index(self):
        """Save model index to disk."""
        index_file = self.cache_dir / "model_index.json"
        try:
            with open(index_file, 'w') as f:
                json.dump(self.model_index, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save model index: {e}")
    
    def _generate_model_key(self, model_config: Dict[str, Any]) -> str:
        """Generate unique key for model configuration."""
        config_str = json.dumps(model_config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def cache_model(self, model: Any, model_config: Dict[str, Any], 
                   model_name: str = None) -> str:
        """Cache a trained model."""
        if not self.torch:
            self.logger.error("PyTorch required for model caching")
            return ""
        
        with self._lock:
            model_key = self._generate_model_key(model_config)
            model_path = self.cache_dir / f"model_{model_key}.pt"
            
            try:
                # Save model
                self.torch.save(model.state_dict(), model_path)
                
                # Update index
                self.model_index[model_key] = {
                    'config': model_config,
                    'model_name': model_name or f"model_{model_key}",
                    'cached_time': time.time(),
                    'file_path': str(model_path),
                    'file_size_bytes': model_path.stat().st_size,
                    'access_count': 0,
                    'last_access': time.time()
                }
                
                # Evict old models if necessary
                self._evict_old_models()
                
                # Save index
                self._save_model_index()
                
                self.logger.info(f"Cached model with key {model_key}")
                return model_key
                
            except Exception as e:
                self.logger.error(f"Failed to cache model: {e}")
                return ""
    
    def load_model(self, model_key: str, model_class: type) -> Optional[Any]:
        """Load cached model."""
        if not self.torch:
            self.logger.error("PyTorch required for model loading")
            return None
        
        with self._lock:
            if model_key not in self.model_index:
                self.logger.warning(f"Model key {model_key} not found in cache")
                return None
            
            model_info = self.model_index[model_key]
            model_path = Path(model_info['file_path'])
            
            if not model_path.exists():
                self.logger.error(f"Model file not found: {model_path}")
                # Remove from index
                del self.model_index[model_key]
                self._save_model_index()
                return None
            
            try:
                # Create model instance
                model_config = model_info['config']
                model = model_class(**model_config)
                
                # Load state dict
                state_dict = self.torch.load(model_path, map_location='cpu')
                model.load_state_dict(state_dict)
                
                # Update access statistics
                model_info['access_count'] += 1
                model_info['last_access'] = time.time()
                self._save_model_index()
                
                self.logger.info(f"Loaded cached model with key {model_key}")
                return model
                
            except Exception as e:
                self.logger.error(f"Failed to load model: {e}")
                return None
    
    def _evict_old_models(self):
        """Evict old models if cache is full."""
        if len(self.model_index) <= self.max_models:
            return
        
        # Sort by last access time
        sorted_models = sorted(
            self.model_index.items(),
            key=lambda x: x[1]['last_access']
        )
        
        # Remove oldest models
        models_to_remove = len(self.model_index) - self.max_models
        for i in range(models_to_remove):
            model_key, model_info = sorted_models[i]
            
            # Delete model file
            model_path = Path(model_info['file_path'])
            try:
                model_path.unlink()
            except:
                pass
            
            # Remove from index
            del self.model_index[model_key]
            
            self.logger.info(f"Evicted cached model {model_key}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get model cache statistics."""
        with self._lock:
            total_size_bytes = sum(
                info['file_size_bytes'] for info in self.model_index.values()
            )
            
            return {
                'total_models': len(self.model_index),
                'max_models': self.max_models,
                'total_size_mb': total_size_bytes / 1024 / 1024,
                'cache_dir': str(self.cache_dir),
                'models': list(self.model_index.keys())
            }


def cache_function_results(cache: Union[LRUCache, SmartCache], 
                          ttl: Optional[float] = None):
    """Decorator for caching function results."""
    result_cache = ResultCache(cache)
    return result_cache.cache_decorator(ttl=ttl)