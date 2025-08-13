"""Comprehensive error handling and exception management."""

import traceback
import logging
import sys
from typing import Dict, Any, Optional, Union, Callable
from datetime import datetime
from pathlib import Path
import json


class ConnectomeError(Exception):
    """Base exception for connectome-related errors."""
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "CONNECTOME_ERROR"
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()


class ModelError(ConnectomeError):
    """Exception for model-related errors."""
    
    def __init__(self, message: str, model_type: str = None, **kwargs):
        super().__init__(message, error_code="MODEL_ERROR", **kwargs)
        self.model_type = model_type


class DataError(ConnectomeError):
    """Exception for data-related errors."""
    
    def __init__(self, message: str, data_type: str = None, **kwargs):
        super().__init__(message, error_code="DATA_ERROR", **kwargs)
        self.data_type = data_type


class TrainingError(ConnectomeError):
    """Exception for training-related errors."""
    
    def __init__(self, message: str, epoch: int = None, **kwargs):
        super().__init__(message, error_code="TRAINING_ERROR", **kwargs)
        self.epoch = epoch


class ValidationError(ConnectomeError):
    """Exception for validation-related errors."""
    
    def __init__(self, message: str, validation_type: str = None, **kwargs):
        super().__init__(message, error_code="VALIDATION_ERROR", **kwargs)
        self.validation_type = validation_type


class SecurityError(ConnectomeError):
    """Exception for security-related errors."""
    
    def __init__(self, message: str, security_issue: str = None, **kwargs):
        super().__init__(message, error_code="SECURITY_ERROR", **kwargs)
        self.security_issue = security_issue


class ErrorHandler:
    """Centralized error handling and logging."""
    
    def __init__(self, logger: Optional[logging.Logger] = None, 
                 error_log_file: Optional[str] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.error_log_file = error_log_file
        self.error_stats = {
            'total_errors': 0,
            'errors_by_type': {},
            'errors_by_code': {}
        }
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle an error with comprehensive logging and tracking."""
        error_info = self._extract_error_info(error, context or {})
        
        # Log the error
        self._log_error(error_info)
        
        # Update statistics
        self._update_error_stats(error_info)
        
        # Save detailed error report if configured
        if self.error_log_file:
            self._save_error_report(error_info)
        
        return error_info
    
    def _extract_error_info(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive information about an error."""
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'traceback': traceback.format_exc(),
            'system_info': {
                'python_version': sys.version,
                'platform': sys.platform
            }
        }
        
        # Add specific information for ConnectomeError and subclasses
        if isinstance(error, ConnectomeError):
            error_info.update({
                'error_code': error.error_code,
                'details': error.details,
                'connectome_timestamp': error.timestamp
            })
            
            # Add type-specific information
            if isinstance(error, ModelError):
                error_info['model_type'] = error.model_type
            elif isinstance(error, DataError):
                error_info['data_type'] = error.data_type
            elif isinstance(error, TrainingError):
                error_info['epoch'] = error.epoch
            elif isinstance(error, ValidationError):
                error_info['validation_type'] = error.validation_type
            elif isinstance(error, SecurityError):
                error_info['security_issue'] = error.security_issue
        
        return error_info
    
    def _log_error(self, error_info: Dict[str, Any]):
        """Log error information at appropriate level."""
        error_type = error_info['error_type']
        message = error_info['error_message']
        
        # Determine log level based on error type
        if error_type in ['SecurityError', 'ValidationError']:
            log_level = logging.ERROR
        elif error_type in ['ModelError', 'TrainingError']:
            log_level = logging.WARNING
        else:
            log_level = logging.ERROR
        
        # Log the error
        self.logger.log(log_level, f"{error_type}: {message}")
        
        # Log additional context if available
        if error_info.get('context'):
            self.logger.debug(f"Error context: {error_info['context']}")
    
    def _update_error_stats(self, error_info: Dict[str, Any]):
        """Update error statistics."""
        self.error_stats['total_errors'] += 1
        
        error_type = error_info['error_type']
        self.error_stats['errors_by_type'][error_type] = \
            self.error_stats['errors_by_type'].get(error_type, 0) + 1
        
        error_code = error_info.get('error_code', 'UNKNOWN')
        self.error_stats['errors_by_code'][error_code] = \
            self.error_stats['errors_by_code'].get(error_code, 0) + 1
    
    def _save_error_report(self, error_info: Dict[str, Any]):
        """Save detailed error report to file."""
        try:
            error_file = Path(self.error_log_file)
            error_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Load existing errors if file exists
            if error_file.exists():
                with open(error_file, 'r') as f:
                    existing_errors = json.load(f)
            else:
                existing_errors = []
            
            # Add new error
            existing_errors.append(error_info)
            
            # Save updated errors
            with open(error_file, 'w') as f:
                json.dump(existing_errors, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Failed to save error report: {e}")
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get current error statistics."""
        return self.error_stats.copy()
    
    def reset_error_stats(self):
        """Reset error statistics."""
        self.error_stats = {
            'total_errors': 0,
            'errors_by_type': {},
            'errors_by_code': {}
        }


def error_handler_decorator(handler: ErrorHandler = None, 
                          reraise: bool = True,
                          default_return: Any = None):
    """Decorator for automatic error handling."""
    
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Use provided handler or create default
                error_handler = handler or ErrorHandler()
                
                # Handle the error
                context = {
                    'function': func.__name__,
                    'args': str(args)[:100],  # Truncate for privacy
                    'kwargs': {k: str(v)[:100] for k, v in kwargs.items()}
                }
                error_handler.handle_error(e, context)
                
                # Re-raise or return default
                if reraise:
                    raise
                else:
                    return default_return
        
        return wrapper
    return decorator


class CircuitBreaker:
    """Circuit breaker pattern for handling repeated failures."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
    
    def call(self, func: Callable, *args, **kwargs):
        """Call function with circuit breaker protection."""
        if self.state == 'open':
            if self._should_attempt_reset():
                self.state = 'half-open'
            else:
                raise ConnectomeError("Circuit breaker is open", error_code="CIRCUIT_BREAKER_OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = datetime.now().timestamp() - self.last_failure_time
        return time_since_failure >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = 'closed'
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now().timestamp()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'open'


class RetryManager:
    """Manages retry logic with exponential backoff."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
    
    def retry(self, func: Callable, *args, **kwargs):
        """Execute function with retry logic."""
        import time
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    break
                
                # Calculate delay with exponential backoff
                delay = min(
                    self.base_delay * (self.backoff_factor ** attempt),
                    self.max_delay
                )
                
                time.sleep(delay)
        
        # If we get here, all retries failed
        raise ConnectomeError(
            f"Function failed after {self.max_retries + 1} attempts",
            error_code="MAX_RETRIES_EXCEEDED",
            details={'last_exception': str(last_exception)}
        )


def safe_execute(func: Callable, error_handler: ErrorHandler = None, 
                circuit_breaker: CircuitBreaker = None,
                retry_manager: RetryManager = None) -> Any:
    """Safely execute a function with comprehensive error handling."""
    
    # Set up default handlers if not provided
    if error_handler is None:
        error_handler = ErrorHandler()
    
    # Define execution function
    def execute():
        return func()
    
    # Apply circuit breaker if provided
    if circuit_breaker:
        execute = lambda: circuit_breaker.call(func)
    
    # Apply retry logic if provided
    if retry_manager:
        execute = lambda: retry_manager.retry(execute)
    
    try:
        return execute()
    except Exception as e:
        error_handler.handle_error(e, {'function': func.__name__})
        raise


class HealthChecker:
    """Monitors system health and detects issues."""
    
    def __init__(self):
        self.checks = {}
        self.last_check_time = None
        self.health_status = 'unknown'
    
    def add_check(self, name: str, check_func: Callable[[], bool], 
                 description: str = ""):
        """Add a health check."""
        self.checks[name] = {
            'func': check_func,
            'description': description,
            'last_result': None,
            'last_checked': None
        }
    
    def run_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = {}
        all_passed = True
        
        for name, check in self.checks.items():
            try:
                result = check['func']()
                check['last_result'] = result
                check['last_checked'] = datetime.now().isoformat()
                
                results[name] = {
                    'passed': result,
                    'description': check['description'],
                    'last_checked': check['last_checked']
                }
                
                if not result:
                    all_passed = False
                    
            except Exception as e:
                check['last_result'] = False
                check['last_checked'] = datetime.now().isoformat()
                
                results[name] = {
                    'passed': False,
                    'error': str(e),
                    'description': check['description'],
                    'last_checked': check['last_checked']
                }
                all_passed = False
        
        self.health_status = 'healthy' if all_passed else 'unhealthy'
        self.last_check_time = datetime.now().isoformat()
        
        return {
            'overall_status': self.health_status,
            'last_check_time': self.last_check_time,
            'checks': results
        }