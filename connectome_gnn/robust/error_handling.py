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


class ConfigurationError(ConnectomeError):
    """Exception for configuration-related errors."""
    
    def __init__(self, message: str, config_key: str = None, **kwargs):
        super().__init__(message, error_code="CONFIG_ERROR", **kwargs)
        self.config_key = config_key


class SecurityError(ConnectomeError):
    """Exception for security-related errors."""
    
    def __init__(self, message: str, security_type: str = None, **kwargs):
        super().__init__(message, error_code="SECURITY_ERROR", **kwargs)
        self.security_type = security_type


class ErrorHandler:
    """Centralized error handling and logging."""
    
    def __init__(self, log_dir: Optional[str] = None, enable_file_logging: bool = True):
        self.log_dir = Path(log_dir) if log_dir else Path.cwd() / ".connectome_logs"
        self.enable_file_logging = enable_file_logging
        
        # Create log directory
        if self.enable_file_logging:
            self.log_dir.mkdir(exist_ok=True)
        
        # Setup logger
        self.logger = self._setup_logger()
        
        # Error statistics
        self.error_counts = {}
        self.recent_errors = []
    
    def _setup_logger(self) -> logging.Logger:
        """Setup comprehensive logging."""
        logger = logging.getLogger("connectome_error_handler")
        logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
        
        # File handler
        if self.enable_file_logging:
            error_file = self.log_dir / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
            file_handler = logging.FileHandler(error_file)
            file_handler.setLevel(logging.DEBUG)
            file_format = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)
        
        return logger
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle and log errors with context."""
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat(),
            'traceback': traceback.format_exc(),
            'context': context or {}
        }
        
        # Add specific error details
        if isinstance(error, ConnectomeError):
            error_info.update({
                'error_code': error.error_code,
                'details': error.details
            })
        
        # Log the error
        self.logger.error(f"Error occurred: {error_info['error_type']}", extra=error_info)
        
        # Update statistics
        error_type = error_info['error_type']
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Keep recent errors (max 100)
        self.recent_errors.append(error_info)
        if len(self.recent_errors) > 100:
            self.recent_errors.pop(0)
        
        # Save error report if file logging enabled
        if self.enable_file_logging:
            self._save_error_report(error_info)
        
        return error_info
    
    def _save_error_report(self, error_info: Dict[str, Any]):
        """Save detailed error report."""
        error_file = self.log_dir / f"error_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(error_file, 'w') as f:
                json.dump(error_info, f, indent=2, default=str)
        except Exception as e:
            self.logger.warning(f"Failed to save error report: {e}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        total_errors = sum(self.error_counts.values())
        return {
            'total_errors': total_errors,
            'error_counts_by_type': self.error_counts.copy(),
            'recent_errors_count': len(self.recent_errors),
            'most_common_error': max(self.error_counts.items(), key=lambda x: x[1])[0] if self.error_counts else None
        }
    
    def create_error_context(self, function_name: str, **kwargs) -> Dict[str, Any]:
        """Create error context for better debugging."""
        return {
            'function': function_name,
            'timestamp': datetime.now().isoformat(),
            'python_version': sys.version,
            'arguments': {k: str(v) if not callable(v) else f"<callable: {v.__name__}>" for k, v in kwargs.items()}
        }


def error_handler_decorator(error_handler: ErrorHandler):
    """Decorator for automatic error handling."""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = error_handler.create_error_context(func.__name__, **kwargs)
                error_info = error_handler.handle_error(e, context)
                
                # Re-raise with additional context
                if isinstance(e, ConnectomeError):
                    e.details.update({'handler_context': context})
                    raise e
                else:
                    # Wrap in ConnectomeError
                    raise ConnectomeError(
                        f"Error in {func.__name__}: {str(e)}",
                        error_code="WRAPPED_ERROR",
                        details={'original_error': str(e), 'context': context}
                    ) from e
        
        return wrapper
    return decorator


def safe_execute(func: Callable, *args, error_handler: ErrorHandler = None, 
                default_return=None, **kwargs):
    """Safely execute a function with error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if error_handler:
            context = error_handler.create_error_context(func.__name__, **kwargs)
            error_handler.handle_error(e, context)
        else:
            logging.error(f"Error in {func.__name__}: {e}")
        
        return default_return


class RetryHandler:
    """Handle retries with exponential backoff."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
    
    def retry(self, func: Callable, *args, **kwargs):
        """Retry function execution with exponential backoff."""
        import time
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = min(self.base_delay * (self.backoff_factor ** attempt), self.max_delay)
                    logging.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                else:
                    logging.error(f"All {self.max_retries + 1} attempts failed")
        
        # If we get here, all retries failed
        raise last_exception


# Global error handler instance
_global_error_handler = None

def get_global_error_handler() -> ErrorHandler:
    """Get global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler


def set_global_error_handler(error_handler: ErrorHandler):
    """Set global error handler instance."""
    global _global_error_handler
    _global_error_handler = error_handler


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