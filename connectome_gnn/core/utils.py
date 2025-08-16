"""Core utility functions that don't depend on external ML libraries."""

import os
import sys
import logging
import time
import json
import hashlib
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from pathlib import Path
import warnings


def safe_import(module_name: str, package: str = None) -> Optional[Any]:
    """Safely import a module, returning None if not available."""
    try:
        if package:
            module = __import__(module_name, fromlist=[package])
            return getattr(module, package)
        else:
            return __import__(module_name)
    except ImportError:
        return None


def validate_input(data: Any, validators: List[Callable[[Any], bool]], 
                  error_msg: str = "Input validation failed") -> bool:
    """Validate input data using a list of validator functions."""
    for validator in validators:
        if not validator(data):
            raise ValueError(f"{error_msg}: Failed validator {validator.__name__}")
    return True


def get_device_info() -> Dict[str, Any]:
    """Get information about available compute devices."""
    info = {
        'python_version': sys.version,
        'platform': sys.platform,
        'cpu_count': os.cpu_count(),
    }
    
    # Try to detect GPU availability
    torch = safe_import('torch')
    if torch:
        info['torch_available'] = True
        info['torch_version'] = torch.__version__
        info['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['gpu_count'] = torch.cuda.device_count()
            info['gpu_name'] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
    else:
        info['torch_available'] = False
        
    return info


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger('connectome_gnn')
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Set level
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def create_directory(path: Union[str, Path], exist_ok: bool = True) -> Path:
    """Create directory with proper error handling."""
    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=exist_ok)
        return path
    except Exception as e:
        raise OSError(f"Failed to create directory {path}: {e}")


def save_json(data: Dict[str, Any], filepath: Union[str, Path], indent: int = 2):
    """Save data to JSON file with error handling."""
    filepath = Path(filepath)
    try:
        # Create parent directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent, default=str)  # default=str handles non-serializable objects
    except Exception as e:
        raise IOError(f"Failed to save JSON to {filepath}: {e}")


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load data from JSON file with error handling."""
    filepath = Path(filepath)
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file {filepath}: {e}")
    except Exception as e:
        raise IOError(f"Failed to load JSON from {filepath}: {e}")


def compute_file_hash(filepath: Union[str, Path], algorithm: str = 'sha256') -> str:
    """Compute hash of a file."""
    filepath = Path(filepath)
    hash_func = hashlib.new(algorithm)
    
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        return hash_func.hexdigest()
    except Exception as e:
        raise IOError(f"Failed to compute hash for {filepath}: {e}")


def validate_file_path(filepath: Union[str, Path], must_exist: bool = True, 
                      extensions: Optional[List[str]] = None) -> Path:
    """Validate file path with various checks."""
    filepath = Path(filepath)
    
    if must_exist and not filepath.exists():
        raise FileNotFoundError(f"File does not exist: {filepath}")
    
    if extensions and filepath.suffix.lower() not in [ext.lower() for ext in extensions]:
        raise ValueError(f"File must have one of these extensions: {extensions}")
    
    return filepath


def validate_directory_path(dirpath: Union[str, Path], must_exist: bool = True, 
                           create_if_missing: bool = False) -> Path:
    """Validate directory path."""
    dirpath = Path(dirpath)
    
    if not dirpath.exists():
        if must_exist and not create_if_missing:
            raise FileNotFoundError(f"Directory does not exist: {dirpath}")
        elif create_if_missing:
            dirpath.mkdir(parents=True, exist_ok=True)
    
    if dirpath.exists() and not dirpath.is_dir():
        raise ValueError(f"Path exists but is not a directory: {dirpath}")
    
    return dirpath


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, description: str = "Operation", logger: Optional[logging.Logger] = None):
        self.description = description
        self.logger = logger
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        if self.logger:
            self.logger.info(f"Starting {self.description}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        if exc_type is None:
            message = f"Completed {self.description} in {duration:.2f}s"
        else:
            message = f"Failed {self.description} after {duration:.2f}s"
        
        if self.logger:
            if exc_type is None:
                self.logger.info(message)
            else:
                self.logger.error(message)
        else:
            print(message)
    
    @property
    def duration(self) -> Optional[float]:
        """Get duration if timing is complete."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


def batch_process(items: List[Any], batch_size: int, 
                 process_func: Callable[[List[Any]], Any]) -> List[Any]:
    """Process items in batches."""
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_result = process_func(batch)
        
        if isinstance(batch_result, list):
            results.extend(batch_result)
        else:
            results.append(batch_result)
    
    return results


def memory_usage() -> Dict[str, float]:
    """Get current memory usage information."""
    import psutil
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
        'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
        'percent': process.memory_percent()
    }


def ensure_reproducibility(seed: int = 42):
    """Ensure reproducible results by setting random seeds."""
    import random
    import os
    
    # Python random
    random.seed(seed)
    
    # NumPy random
    numpy = safe_import('numpy')
    if numpy:
        numpy.random.seed(seed)
    
    # PyTorch random
    torch = safe_import('torch')
    if torch:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    # Environment variable for hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)


def set_random_seed(seed: int = 42):
    """Alias for ensure_reproducibility for compatibility."""
    ensure_reproducibility(seed)


def check_system_requirements() -> Dict[str, bool]:
    """Check if system meets minimum requirements."""
    requirements = {
        'python_version_ok': sys.version_info >= (3, 8),
        'disk_space_ok': True,  # Simplified check
        'memory_ok': True,      # Simplified check
    }
    
    # Check available packages
    torch = safe_import('torch')
    requirements['torch_available'] = torch is not None
    
    torch_geometric = safe_import('torch_geometric')
    requirements['torch_geometric_available'] = torch_geometric is not None
    
    sklearn = safe_import('sklearn')
    requirements['sklearn_available'] = sklearn is not None
    
    return requirements


def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Perform safe division avoiding division by zero."""
    try:
        if abs(denominator) < 1e-10:  # Near zero
            return default
        return numerator / denominator
    except (ZeroDivisionError, TypeError):
        return default


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file system operations."""
    import re
    
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove control characters
    filename = ''.join(char for char in filename if ord(char) >= 32)
    
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255-len(ext)] + ext
    
    return filename


def deprecated(func):
    """Decorator to mark functions as deprecated."""
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"Function {func.__name__} is deprecated and will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2
        )
        return func(*args, **kwargs)
    return wrapper


class ProgressTracker:
    """Simple progress tracker for long-running operations."""
    
    def __init__(self, total: int, description: str = "Progress"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
    
    def update(self, increment: int = 1):
        """Update progress."""
        self.current += increment
        self._print_progress()
    
    def _print_progress(self):
        """Print progress to console."""
        if self.total > 0:
            percent = (self.current / self.total) * 100
            elapsed = time.time() - self.start_time
            
            if self.current > 0:
                eta = (elapsed / self.current) * (self.total - self.current)
                eta_str = f", ETA: {eta:.1f}s"
            else:
                eta_str = ""
            
            print(f"\r{self.description}: {self.current}/{self.total} ({percent:.1f}%){eta_str}", 
                  end="", flush=True)
            
            if self.current >= self.total:
                print()  # New line when complete