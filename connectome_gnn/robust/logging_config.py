"""Comprehensive logging configuration for the connectome GNN framework."""

import logging
import logging.handlers
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
from datetime import datetime
import traceback


class ConnectomeFormatter(logging.Formatter):
    """Custom formatter for connectome-specific logging."""
    
    def __init__(self, include_context: bool = True):
        self.include_context = include_context
        super().__init__()
    
    def format(self, record):
        # Base format
        log_format = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        
        # Add context information if available
        if self.include_context and hasattr(record, 'context'):
            log_format += " | Context: %(context)s"
        
        # Add function info for debug level
        if record.levelno <= logging.DEBUG:
            log_format += " | %(funcName)s:%(lineno)d"
        
        formatter = logging.Formatter(log_format)
        return formatter.format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = traceback.format_exception(*record.exc_info)
        
        # Add extra context if present
        if hasattr(record, 'context'):
            log_entry['context'] = record.context
        
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        
        if hasattr(record, 'session_id'):
            log_entry['session_id'] = record.session_id
        
        return json.dumps(log_entry)


class PerformanceFilter(logging.Filter):
    """Filter for performance-related logs."""
    
    def filter(self, record):
        # Only allow performance-related logs
        performance_keywords = ['training', 'inference', 'memory', 'gpu', 'timing', 'benchmark']
        message = record.getMessage().lower()
        return any(keyword in message for keyword in performance_keywords)


class SecurityFilter(logging.Filter):
    """Filter for security-related logs."""
    
    def filter(self, record):
        # Only allow security-related logs
        security_keywords = ['security', 'authentication', 'authorization', 'access', 'breach', 'attack']
        message = record.getMessage().lower()
        return any(keyword in message for keyword in security_keywords)


class ErrorAggregator(logging.Handler):
    """Aggregates errors for monitoring and alerting."""
    
    def __init__(self, max_errors: int = 1000):
        super().__init__()
        self.max_errors = max_errors
        self.errors = []
        self.error_counts = {}
    
    def emit(self, record):
        if record.levelno >= logging.ERROR:
            error_info = {
                'timestamp': datetime.utcnow().isoformat(),
                'level': record.levelname,
                'message': record.getMessage(),
                'logger': record.name,
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }
            
            # Add to errors list
            self.errors.append(error_info)
            
            # Maintain max size
            if len(self.errors) > self.max_errors:
                self.errors.pop(0)
            
            # Update error counts
            error_key = f"{record.name}:{record.levelname}"
            self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors."""
        return {
            'total_errors': len(self.errors),
            'error_counts': self.error_counts.copy(),
            'recent_errors': self.errors[-10:] if self.errors else []
        }
    
    def clear_errors(self):
        """Clear error history."""
        self.errors = []
        self.error_counts = {}


def setup_comprehensive_logging(
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    enable_console: bool = True,
    enable_file: bool = True,
    enable_json: bool = False,
    enable_performance: bool = False,
    enable_security: bool = False,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    logger_name: str = "connectome_gnn"
) -> Dict[str, Any]:
    """Set up comprehensive logging for the framework."""
    
    # Create main logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create log directory if specified
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
    
    handlers = {}
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ConnectomeFormatter())
        console_handler.setLevel(getattr(logging, log_level.upper()))
        logger.addHandler(console_handler)
        handlers['console'] = console_handler
    
    # File handler (rotating)
    if enable_file and log_dir:
        file_handler = logging.handlers.RotatingFileHandler(
            filename=Path(log_dir) / f"{logger_name}.log",
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setFormatter(ConnectomeFormatter())
        file_handler.setLevel(logging.DEBUG)  # File gets all logs
        logger.addHandler(file_handler)
        handlers['file'] = file_handler
    
    # JSON handler for structured logging
    if enable_json and log_dir:
        json_handler = logging.handlers.RotatingFileHandler(
            filename=Path(log_dir) / f"{logger_name}_structured.json",
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        json_handler.setFormatter(JSONFormatter())
        json_handler.setLevel(logging.DEBUG)
        logger.addHandler(json_handler)
        handlers['json'] = json_handler
    
    # Performance logging
    if enable_performance and log_dir:
        perf_handler = logging.handlers.RotatingFileHandler(
            filename=Path(log_dir) / f"{logger_name}_performance.log",
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        perf_handler.setFormatter(ConnectomeFormatter())
        perf_handler.addFilter(PerformanceFilter())
        perf_handler.setLevel(logging.INFO)
        logger.addHandler(perf_handler)
        handlers['performance'] = perf_handler
    
    # Security logging
    if enable_security and log_dir:
        security_handler = logging.handlers.RotatingFileHandler(
            filename=Path(log_dir) / f"{logger_name}_security.log",
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        security_handler.setFormatter(ConnectomeFormatter())
        security_handler.addFilter(SecurityFilter())
        security_handler.setLevel(logging.WARNING)
        logger.addHandler(security_handler)
        handlers['security'] = security_handler
    
    # Error aggregator
    error_aggregator = ErrorAggregator()
    error_aggregator.setLevel(logging.ERROR)
    logger.addHandler(error_aggregator)
    handlers['error_aggregator'] = error_aggregator
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return {
        'logger': logger,
        'handlers': handlers,
        'log_dir': log_dir,
        'log_level': log_level
    }


def get_logger(name: str = None) -> logging.Logger:
    """Get a logger instance."""
    if name:
        return logging.getLogger(f"connectome_gnn.{name}")
    else:
        return logging.getLogger("connectome_gnn")


class LogContext:
    """Context manager for adding context to log messages."""
    
    def __init__(self, logger: logging.Logger, **context):
        self.logger = logger
        self.context = context
        self.old_factory = None
    
    def __enter__(self):
        # Store old record factory
        self.old_factory = logging.getLogRecordFactory()
        
        # Create new factory that adds context
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            record.context = self.context
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore old factory
        logging.setLogRecordFactory(self.old_factory)


def log_function_call(logger: logging.Logger = None):
    """Decorator to log function calls."""
    if logger is None:
        logger = get_logger()
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            logger.debug(f"Calling function: {func_name}")
            
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Function {func_name} completed successfully")
                return result
            except Exception as e:
                logger.error(f"Function {func_name} failed with error: {e}")
                raise
        
        return wrapper
    return decorator


def log_performance(logger: logging.Logger = None):
    """Decorator to log function performance."""
    if logger is None:
        logger = get_logger()
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            
            func_name = func.__name__
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                duration = end_time - start_time
                
                logger.info(f"Performance: {func_name} completed in {duration:.4f}s")
                return result
            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                logger.error(f"Performance: {func_name} failed after {duration:.4f}s with error: {e}")
                raise
        
        return wrapper
    return decorator


class MetricsLogger:
    """Logger for training and model metrics."""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or get_logger("metrics")
    
    def log_training_metrics(self, epoch: int, metrics: Dict[str, float], 
                           prefix: str = "train"):
        """Log training metrics."""
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch} - {prefix} metrics: {metrics_str}")
    
    def log_model_info(self, model_name: str, num_parameters: int, 
                      model_size_mb: float = None):
        """Log model information."""
        info = f"Model: {model_name}, Parameters: {num_parameters:,}"
        if model_size_mb:
            info += f", Size: {model_size_mb:.2f}MB"
        self.logger.info(info)
    
    def log_dataset_info(self, dataset_name: str, num_samples: int, 
                        num_features: int = None):
        """Log dataset information."""
        info = f"Dataset: {dataset_name}, Samples: {num_samples:,}"
        if num_features:
            info += f", Features: {num_features}"
        self.logger.info(info)
    
    def log_hardware_info(self, device: str, memory_gb: float = None):
        """Log hardware information."""
        info = f"Device: {device}"
        if memory_gb:
            info += f", Memory: {memory_gb:.2f}GB"
        self.logger.info(info)


def setup_model_logging(model_name: str, experiment_id: str = None,
                       log_dir: str = "logs") -> Dict[str, Any]:
    """Set up logging specifically for model training/inference."""
    
    # Create experiment-specific log directory
    if experiment_id:
        log_path = Path(log_dir) / model_name / experiment_id
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(log_dir) / model_name / timestamp
    
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Set up comprehensive logging
    logging_config = setup_comprehensive_logging(
        log_dir=str(log_path),
        enable_console=True,
        enable_file=True,
        enable_json=True,
        enable_performance=True,
        logger_name=f"connectome_gnn.{model_name}"
    )
    
    # Create metrics logger
    metrics_logger = MetricsLogger(logging_config['logger'])
    
    # Save logging configuration
    config_info = {
        'model_name': model_name,
        'experiment_id': experiment_id,
        'log_directory': str(log_path),
        'timestamp': datetime.now().isoformat(),
        'log_level': logging_config['log_level']
    }
    
    with open(log_path / "logging_config.json", 'w') as f:
        json.dump(config_info, f, indent=2)
    
    return {
        **logging_config,
        'metrics_logger': metrics_logger,
        'experiment_path': log_path,
        'config_info': config_info
    }