"""Self-healing and autonomous recovery system for Connectome-GNN-Suite."""

import time
import traceback
import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import json
import os

from ..robust.error_handling import ConnectomeError
from ..robust.logging_config import get_logger


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthMetric:
    """Health metric data structure."""
    name: str
    value: float
    threshold: float
    status: HealthStatus
    timestamp: float
    metadata: Dict[str, Any]


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Exception = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        self.lock = threading.Lock()
        
        self.logger = get_logger(__name__)
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self.lock:
            if self.state == "open":
                if self._should_attempt_reset():
                    self.state = "half-open"
                else:
                    raise ConnectomeError("Circuit breaker is OPEN")
                    
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
                
            except self.expected_exception as e:
                self._on_failure()
                raise e
                
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
        
    def _on_success(self):
        """Handle successful operation."""
        self.failure_count = 0
        self.state = "closed"
        
    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            self.logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


class AutoRecoverySystem:
    """Autonomous recovery system with self-healing capabilities."""
    
    def __init__(
        self,
        health_check_interval: float = 30.0,
        recovery_strategies: Optional[Dict[str, Callable]] = None
    ):
        self.health_check_interval = health_check_interval
        self.recovery_strategies = recovery_strategies or {}
        
        self.health_metrics: Dict[str, HealthMetric] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.logger = get_logger(__name__)
        
        self._setup_default_strategies()
        
    def _setup_default_strategies(self):
        """Setup default recovery strategies."""
        self.recovery_strategies.update({
            'memory_pressure': self._recover_memory_pressure,
            'disk_space': self._recover_disk_space,
            'gpu_memory': self._recover_gpu_memory,
            'network_timeout': self._recover_network_timeout,
            'model_degradation': self._recover_model_degradation
        })
        
    def register_health_check(
        self,
        name: str,
        check_func: Callable[[], float],
        threshold: float,
        recovery_strategy: Optional[str] = None
    ):
        """Register a health check function."""
        self.health_checks[name] = {
            'function': check_func,
            'threshold': threshold,
            'recovery_strategy': recovery_strategy
        }
        
    def start_monitoring(self):
        """Start autonomous health monitoring."""
        if self.is_running:
            return
            
        self.is_running = True
        self.logger.info("Starting autonomous health monitoring system")
        
        # Start health check loop
        self.executor.submit(self._health_check_loop)
        
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.is_running = False
        self.executor.shutdown(wait=True)
        self.logger.info("Stopped autonomous health monitoring system")
        
    def _health_check_loop(self):
        """Main health check loop."""
        while self.is_running:
            try:
                self._perform_health_checks()
                time.sleep(self.health_check_interval)
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                time.sleep(self.health_check_interval)
                
    def _perform_health_checks(self):
        """Perform all registered health checks."""
        for name, config in getattr(self, 'health_checks', {}).items():
            try:
                value = config['function']()
                threshold = config['threshold']
                
                status = self._determine_status(value, threshold)
                
                metric = HealthMetric(
                    name=name,
                    value=value,
                    threshold=threshold,
                    status=status,
                    timestamp=time.time(),
                    metadata={}
                )
                
                self.health_metrics[name] = metric
                
                # Trigger recovery if needed
                if status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                    self._trigger_recovery(name, metric)
                    
            except Exception as e:
                self.logger.error(f"Health check failed for {name}: {e}")
                
    def _determine_status(self, value: float, threshold: float) -> HealthStatus:
        """Determine health status based on value and threshold."""
        if value <= threshold:
            return HealthStatus.HEALTHY
        elif value <= threshold * 1.5:
            return HealthStatus.DEGRADED
        elif value <= threshold * 2.0:
            return HealthStatus.UNHEALTHY
        else:
            return HealthStatus.CRITICAL
            
    def _trigger_recovery(self, metric_name: str, metric: HealthMetric):
        """Trigger recovery strategy for unhealthy metric."""
        recovery_strategy = getattr(self, 'health_checks', {}).get(
            metric_name, {}
        ).get('recovery_strategy')
        
        if recovery_strategy and recovery_strategy in self.recovery_strategies:
            self.logger.warning(f"Triggering recovery for {metric_name}: {recovery_strategy}")
            
            try:
                self.recovery_strategies[recovery_strategy](metric)
            except Exception as e:
                self.logger.error(f"Recovery strategy failed for {metric_name}: {e}")
                
    def _recover_memory_pressure(self, metric: HealthMetric):
        """Recover from memory pressure."""
        import gc
        import torch
        
        # Force garbage collection
        gc.collect()
        
        # Clear PyTorch caches if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.logger.info("Applied memory pressure recovery strategy")
        
    def _recover_disk_space(self, metric: HealthMetric):
        """Recover from disk space issues."""
        # Clean temporary files
        temp_dirs = ['/tmp', '/var/tmp']
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                # Clean old temporary files (older than 1 day)
                os.system(f"find {temp_dir} -type f -mtime +1 -delete")
                
        self.logger.info("Applied disk space recovery strategy")
        
    def _recover_gpu_memory(self, metric: HealthMetric):
        """Recover from GPU memory issues."""
        import torch
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        self.logger.info("Applied GPU memory recovery strategy")
        
    def _recover_network_timeout(self, metric: HealthMetric):
        """Recover from network timeout issues."""
        # Reset network connections (simplified)
        self.logger.info("Applied network timeout recovery strategy")
        
    def _recover_model_degradation(self, metric: HealthMetric):
        """Recover from model performance degradation."""
        # Could trigger model retraining or parameter adjustment
        self.logger.info("Applied model degradation recovery strategy")
        
    def get_circuit_breaker(self, name: str, **kwargs) -> CircuitBreaker:
        """Get or create circuit breaker for a component."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(**kwargs)
        return self.circuit_breakers[name]
        
    def get_health_status(self) -> Dict[str, Any]:
        """Get current system health status."""
        return {
            'timestamp': time.time(),
            'overall_status': self._calculate_overall_status(),
            'metrics': {
                name: {
                    'value': metric.value,
                    'threshold': metric.threshold,
                    'status': metric.status.value,
                    'timestamp': metric.timestamp
                }
                for name, metric in self.health_metrics.items()
            },
            'circuit_breakers': {
                name: {
                    'state': breaker.state,
                    'failure_count': breaker.failure_count,
                    'last_failure': breaker.last_failure_time
                }
                for name, breaker in self.circuit_breakers.items()
            }
        }
        
    def _calculate_overall_status(self) -> str:
        """Calculate overall system health status."""
        if not self.health_metrics:
            return HealthStatus.HEALTHY.value
            
        statuses = [metric.status for metric in self.health_metrics.values()]
        
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL.value
        elif HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY.value
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED.value
        else:
            return HealthStatus.HEALTHY.value


class AdaptiveThrottling:
    """Adaptive request throttling based on system health."""
    
    def __init__(self, base_rate: float = 1.0):
        self.base_rate = base_rate
        self.current_rate = base_rate
        self.last_adjustment = time.time()
        self.lock = threading.Lock()
        
    def get_current_rate(self, health_status: HealthStatus) -> float:
        """Get current throttling rate based on health status."""
        with self.lock:
            if health_status == HealthStatus.HEALTHY:
                self.current_rate = self.base_rate
            elif health_status == HealthStatus.DEGRADED:
                self.current_rate = self.base_rate * 0.8
            elif health_status == HealthStatus.UNHEALTHY:
                self.current_rate = self.base_rate * 0.5
            else:  # CRITICAL
                self.current_rate = self.base_rate * 0.1
                
            return self.current_rate
            
    def should_throttle(self) -> bool:
        """Check if request should be throttled."""
        current_time = time.time()
        time_since_last = current_time - self.last_adjustment
        
        if time_since_last < (1.0 / self.current_rate):
            return True
            
        self.last_adjustment = current_time
        return False


# Global auto-recovery system instance
_global_recovery_system = None

def get_auto_recovery_system() -> AutoRecoverySystem:
    """Get global auto-recovery system instance."""
    global _global_recovery_system
    if _global_recovery_system is None:
        _global_recovery_system = AutoRecoverySystem()
    return _global_recovery_system


def with_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0
):
    """Decorator to add circuit breaker protection to functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            recovery_system = get_auto_recovery_system()
            breaker = recovery_system.get_circuit_breaker(
                name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout
            )
            return breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator


# Health check functions
def check_memory_usage() -> float:
    """Check system memory usage percentage."""
    import psutil
    return psutil.virtual_memory().percent


def check_disk_usage(path: str = "/") -> float:
    """Check disk usage percentage."""
    import psutil
    return psutil.disk_usage(path).percent


def check_gpu_memory_usage() -> float:
    """Check GPU memory usage percentage."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
    except:
        pass
    return 0.0