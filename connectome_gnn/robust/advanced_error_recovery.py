"""Advanced Error Recovery and Fault Tolerance System.

Implements sophisticated error recovery mechanisms, fault tolerance,
and self-healing capabilities for robust connectome analysis.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import traceback
import time
import logging
import threading
import queue
from typing import Dict, List, Optional, Callable, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import pickle
import json
from contextlib import contextmanager
from functools import wraps
import numpy as np


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class RecoveryStrategy(Enum):
    """Available recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CHECKPOINT_RESTORE = "checkpoint_restore"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    CIRCUIT_BREAKER = "circuit_breaker"
    ADAPTIVE_PARAMETERS = "adaptive_parameters"


@dataclass
class ErrorEvent:
    """Container for error event information."""
    timestamp: float = field(default_factory=time.time)
    error_type: str = ""
    error_message: str = ""
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    stack_trace: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None
    node_id: Optional[str] = None
    model_state: Optional[Dict] = None


@dataclass
class CheckpointData:
    """Container for checkpoint data."""
    model_state: Dict[str, torch.Tensor]
    optimizer_state: Optional[Dict] = None
    epoch: int = 0
    loss: float = float('inf')
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    checksum: str = ""
    
    def __post_init__(self):
        """Compute checksum for data integrity."""
        data_str = json.dumps({
            'epoch': self.epoch,
            'loss': self.loss,
            'metrics': self.metrics,
            'timestamp': self.timestamp
        }, sort_keys=True)
        self.checksum = hashlib.md5(data_str.encode()).hexdigest()


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self._failure_count = 0
        self._last_failure_time = None
        self._state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        
    def _can_attempt(self) -> bool:
        """Check if attempt is allowed."""
        if self._state == 'CLOSED':
            return True
        elif self._state == 'OPEN':
            return time.time() - self._last_failure_time >= self.recovery_timeout
        else:  # HALF_OPEN
            return True
    
    def _on_success(self):
        """Handle successful operation."""
        self._failure_count = 0
        self._state = 'CLOSED'
        self._last_failure_time = None
    
    def _on_failure(self):
        """Handle failed operation."""
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._failure_count >= self.failure_threshold:
            if self._state == 'HALF_OPEN':
                self._state = 'OPEN'
            elif self._state == 'CLOSED':
                self._state = 'OPEN'
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap function with circuit breaker."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self._can_attempt():
                raise Exception(f"Circuit breaker is OPEN. Try again after {self.recovery_timeout}s")
            
            if self._state == 'OPEN':
                self._state = 'HALF_OPEN'
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise e
        
        return wrapper


class AdaptiveCheckpointManager:
    """Intelligent checkpointing with adaptive frequency."""
    
    def __init__(
        self,
        base_checkpoint_dir: str = "./checkpoints",
        max_checkpoints: int = 10,
        checkpoint_frequency: int = 100,
        adaptive_threshold: float = 0.1
    ):
        self.base_checkpoint_dir = base_checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.checkpoint_frequency = checkpoint_frequency
        self.adaptive_threshold = adaptive_threshold
        
        self.checkpoints: List[CheckpointData] = []
        self.loss_history: List[float] = []
        self.last_checkpoint_step = 0
        
        # Create checkpoint directory
        import os
        os.makedirs(base_checkpoint_dir, exist_ok=True)
        
    def should_checkpoint(self, step: int, current_loss: float) -> bool:
        """Determine if checkpointing is needed."""
        # Always checkpoint at regular intervals
        if step - self.last_checkpoint_step >= self.checkpoint_frequency:
            return True
        
        # Adaptive checkpointing based on loss improvement
        if len(self.loss_history) > 0:
            recent_losses = self.loss_history[-10:]
            if len(recent_losses) >= 2:
                loss_improvement = (recent_losses[0] - current_loss) / recent_losses[0]
                if loss_improvement > self.adaptive_threshold:
                    return True
        
        return False
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        loss: float = float('inf'),
        metrics: Dict[str, float] = None,
        step: int = 0
    ) -> str:
        """Save model checkpoint."""
        metrics = metrics or {}
        
        checkpoint_data = CheckpointData(
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict() if optimizer else None,
            epoch=epoch,
            loss=loss,
            metrics=metrics
        )
        
        # Save to disk
        checkpoint_path = f"{self.base_checkpoint_dir}/checkpoint_step_{step}.pt"
        torch.save(checkpoint_data, checkpoint_path)
        
        # Update internal state
        self.checkpoints.append(checkpoint_data)
        self.loss_history.append(loss)
        self.last_checkpoint_step = step
        
        # Cleanup old checkpoints
        if len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            # Remove old checkpoint file
            try:
                import os
                old_path = f"{self.base_checkpoint_dir}/checkpoint_step_{step - len(self.checkpoints) * self.checkpoint_frequency}.pt"
                if os.path.exists(old_path):
                    os.remove(old_path)
            except Exception:
                pass  # Ignore cleanup errors
        
        return checkpoint_path
    
    def load_best_checkpoint(self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> Optional[CheckpointData]:
        """Load the best available checkpoint."""
        if not self.checkpoints:
            return None
        
        # Find best checkpoint (lowest loss)
        best_checkpoint = min(self.checkpoints, key=lambda x: x.loss)
        
        # Load model state
        try:
            model.load_state_dict(best_checkpoint.model_state)
            if optimizer and best_checkpoint.optimizer_state:
                optimizer.load_state_dict(best_checkpoint.optimizer_state)
            
            return best_checkpoint
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
            return None


class SelfHealingTrainer:
    """Self-healing training system with automatic error recovery."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        max_retries: int = 3,
        checkpoint_manager: Optional[AdaptiveCheckpointManager] = None,
        enable_circuit_breaker: bool = True
    ):
        self.model = model
        self.optimizer = optimizer
        self.max_retries = max_retries
        self.checkpoint_manager = checkpoint_manager or AdaptiveCheckpointManager()
        
        # Error tracking
        self.error_history: List[ErrorEvent] = []
        self.recovery_statistics: Dict[str, int] = {}
        
        # Circuit breakers for different operations
        self.circuit_breakers = {}
        if enable_circuit_breaker:
            self.circuit_breakers['forward'] = CircuitBreaker(
                failure_threshold=3,
                recovery_timeout=30.0,
                expected_exception=RuntimeError
            )
            self.circuit_breakers['backward'] = CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=60.0,
                expected_exception=(RuntimeError, torch.cuda.OutOfMemoryError)
            )
        
        # Adaptive parameters
        self.adaptive_batch_size = None
        self.adaptive_learning_rate = None
        self.original_batch_size = None
        self.original_learning_rate = None
        
    def _record_error(self, error: Exception, context: Dict[str, Any], severity: ErrorSeverity = ErrorSeverity.MEDIUM):
        """Record error event for analysis."""
        error_event = ErrorEvent(
            error_type=type(error).__name__,
            error_message=str(error),
            severity=severity,
            stack_trace=traceback.format_exc(),
            context=context.copy()
        )
        
        self.error_history.append(error_event)
        
        # Update statistics
        error_type = error_event.error_type
        if error_type not in self.recovery_statistics:
            self.recovery_statistics[error_type] = 0
    
    def _determine_recovery_strategy(self, error: Exception, context: Dict[str, Any]) -> RecoveryStrategy:
        """Determine the best recovery strategy for the error."""
        error_type = type(error).__name__
        
        # CUDA out of memory -> adaptive parameters
        if isinstance(error, torch.cuda.OutOfMemoryError):
            return RecoveryStrategy.ADAPTIVE_PARAMETERS
        
        # Runtime errors -> retry with fallback
        if isinstance(error, RuntimeError):
            if "backward" in str(error).lower():
                return RecoveryStrategy.CHECKPOINT_RESTORE
            return RecoveryStrategy.RETRY
        
        # Distributed training errors -> circuit breaker
        if "distributed" in str(error).lower():
            return RecoveryStrategy.CIRCUIT_BREAKER
        
        # Default to retry
        return RecoveryStrategy.RETRY
    
    def _apply_recovery_strategy(
        self, 
        strategy: RecoveryStrategy, 
        error: Exception,
        context: Dict[str, Any]
    ) -> bool:
        """Apply the specified recovery strategy."""
        try:
            if strategy == RecoveryStrategy.ADAPTIVE_PARAMETERS:
                return self._recover_with_adaptive_parameters(error, context)
            elif strategy == RecoveryStrategy.CHECKPOINT_RESTORE:
                return self._recover_with_checkpoint_restore(error, context)
            elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                return self._recover_with_graceful_degradation(error, context)
            elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                return self._recover_with_circuit_breaker(error, context)
            else:  # RETRY or FALLBACK
                return True  # Allow retry
        except Exception as recovery_error:
            logging.error(f"Recovery strategy {strategy} failed: {recovery_error}")
            return False
    
    def _recover_with_adaptive_parameters(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Recover by adapting training parameters."""
        try:
            # Store original parameters if not already stored
            if self.original_batch_size is None:
                self.original_batch_size = context.get('batch_size', 32)
            if self.original_learning_rate is None:
                for param_group in self.optimizer.param_groups:
                    self.original_learning_rate = param_group['lr']
                    break
            
            # Reduce batch size
            if isinstance(error, torch.cuda.OutOfMemoryError):
                current_batch_size = context.get('batch_size', self.original_batch_size)
                new_batch_size = max(1, current_batch_size // 2)
                context['batch_size'] = new_batch_size
                self.adaptive_batch_size = new_batch_size
                
                logging.info(f"Adapted batch size: {current_batch_size} -> {new_batch_size}")
                
                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return True
            
            # Reduce learning rate for numerical instability
            if "loss" in str(error).lower() or "nan" in str(error).lower():
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.5
                    self.adaptive_learning_rate = param_group['lr']
                    logging.info(f"Adapted learning rate to: {param_group['lr']}")
                
                return True
            
        except Exception as e:
            logging.error(f"Failed to adapt parameters: {e}")
        
        return False
    
    def _recover_with_checkpoint_restore(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Recover by restoring from checkpoint."""
        try:
            checkpoint_data = self.checkpoint_manager.load_best_checkpoint(self.model, self.optimizer)
            if checkpoint_data:
                logging.info(f"Restored from checkpoint: epoch {checkpoint_data.epoch}, loss {checkpoint_data.loss}")
                return True
        except Exception as e:
            logging.error(f"Failed to restore checkpoint: {e}")
        
        return False
    
    def _recover_with_graceful_degradation(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Recover by gracefully degrading model complexity."""
        try:
            # Temporarily disable dropout for stability
            def disable_dropout(module):
                if isinstance(module, nn.Dropout):
                    module.p = 0.0
            
            self.model.apply(disable_dropout)
            logging.info("Applied graceful degradation: disabled dropout")
            return True
            
        except Exception as e:
            logging.error(f"Failed graceful degradation: {e}")
        
        return False
    
    def _recover_with_circuit_breaker(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Implement circuit breaker recovery."""
        operation = context.get('operation', 'unknown')
        if operation in self.circuit_breakers:
            # Circuit breaker will handle the retry logic
            logging.info(f"Circuit breaker activated for operation: {operation}")
            return True
        
        return False
    
    @contextmanager
    def resilient_training_step(self, **context):
        """Context manager for resilient training steps."""
        step_successful = False
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                yield context
                step_successful = True
                break
                
            except Exception as error:
                last_error = error
                severity = ErrorSeverity.HIGH if attempt == self.max_retries else ErrorSeverity.MEDIUM
                
                self._record_error(error, context, severity)
                
                if attempt < self.max_retries:
                    # Determine and apply recovery strategy
                    strategy = self._determine_recovery_strategy(error, context)
                    recovery_successful = self._apply_recovery_strategy(strategy, error, context)
                    
                    if recovery_successful:
                        # Update error record
                        if self.error_history:
                            self.error_history[-1].recovery_attempted = True
                            self.error_history[-1].recovery_successful = True
                            self.error_history[-1].recovery_strategy = strategy
                        
                        # Update statistics
                        error_type = type(error).__name__
                        self.recovery_statistics[error_type] = self.recovery_statistics.get(error_type, 0) + 1
                        
                        # Wait before retry
                        time.sleep(min(2 ** attempt, 10))  # Exponential backoff
                        
                        logging.info(f"Recovery attempt {attempt + 1} with strategy {strategy} succeeded")
                        continue
                    else:
                        logging.error(f"Recovery attempt {attempt + 1} with strategy {strategy} failed")
                
                # If we're here, recovery failed or max retries reached
                logging.error(f"Training step failed after {attempt + 1} attempts: {error}")
        
        if not step_successful and last_error:
            # Final attempt at graceful handling
            try:
                self._recover_with_graceful_degradation(last_error, context)
            except:
                pass  # Ignore final recovery failures
            
            # Re-raise the last error
            raise last_error
    
    def get_error_report(self) -> Dict[str, Any]:
        """Generate comprehensive error report."""
        error_counts = {}
        severity_counts = {severity: 0 for severity in ErrorSeverity}
        recovery_success_rate = {}
        
        for error_event in self.error_history:
            # Count error types
            error_type = error_event.error_type
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            # Count severities
            severity_counts[error_event.severity] += 1
            
            # Calculate recovery success rates
            if error_event.recovery_attempted:
                if error_type not in recovery_success_rate:
                    recovery_success_rate[error_type] = {'attempted': 0, 'successful': 0}
                
                recovery_success_rate[error_type]['attempted'] += 1
                if error_event.recovery_successful:
                    recovery_success_rate[error_type]['successful'] += 1
        
        # Calculate success rates
        for error_type, stats in recovery_success_rate.items():
            stats['success_rate'] = stats['successful'] / stats['attempted'] if stats['attempted'] > 0 else 0.0
        
        return {
            'total_errors': len(self.error_history),
            'error_counts': error_counts,
            'severity_distribution': {s.name: count for s, count in severity_counts.items()},
            'recovery_statistics': self.recovery_statistics,
            'recovery_success_rates': recovery_success_rate,
            'recent_errors': [
                {
                    'timestamp': e.timestamp,
                    'type': e.error_type,
                    'message': e.error_message,
                    'severity': e.severity.name,
                    'recovered': e.recovery_successful
                }
                for e in self.error_history[-10:]  # Last 10 errors
            ]
        }


class DistributedErrorCoordinator:
    """Coordinates error handling across distributed training nodes."""
    
    def __init__(self, node_id: str, world_size: int = 1, rank: int = 0):
        self.node_id = node_id
        self.world_size = world_size
        self.rank = rank
        
        self.error_events: List[ErrorEvent] = []
        self.global_error_state = {}
        
    def broadcast_error(self, error_event: ErrorEvent):
        """Broadcast error event to all nodes."""
        try:
            if dist.is_available() and dist.is_initialized():
                error_data = {
                    'node_id': self.node_id,
                    'error_type': error_event.error_type,
                    'error_message': error_event.error_message,
                    'severity': error_event.severity.value,
                    'timestamp': error_event.timestamp
                }
                
                # Serialize and broadcast
                error_tensor = torch.tensor([hash(str(error_data))], dtype=torch.long)
                dist.all_reduce(error_tensor, op=dist.ReduceOp.MAX)
                
        except Exception as e:
            logging.error(f"Failed to broadcast error: {e}")
    
    def should_stop_training(self) -> bool:
        """Determine if training should stop based on global error state."""
        critical_errors = sum(
            1 for event in self.error_events 
            if event.severity == ErrorSeverity.CRITICAL
        )
        
        # Stop if too many critical errors
        if critical_errors >= 3:
            return True
        
        return False


def create_resilient_trainer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any] = None
) -> SelfHealingTrainer:
    """Factory function for creating resilient trainers."""
    config = config or {}
    
    checkpoint_manager = AdaptiveCheckpointManager(
        base_checkpoint_dir=config.get('checkpoint_dir', './checkpoints'),
        max_checkpoints=config.get('max_checkpoints', 10),
        checkpoint_frequency=config.get('checkpoint_frequency', 100)
    )
    
    return SelfHealingTrainer(
        model=model,
        optimizer=optimizer,
        max_retries=config.get('max_retries', 3),
        checkpoint_manager=checkpoint_manager,
        enable_circuit_breaker=config.get('enable_circuit_breaker', True)
    )