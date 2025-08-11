"""Security utilities for connectome GNN analysis."""

import os
import hashlib
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import torch
import logging
import warnings

# Configure secure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)  # Only log warnings and errors by default


class SecureDataLoader:
    """Secure data loading utilities with input validation."""
    
    def __init__(self, allowed_extensions: Optional[List[str]] = None):
        """Initialize secure data loader.
        
        Args:
            allowed_extensions: List of allowed file extensions
        """
        self.allowed_extensions = allowed_extensions or ['.pt', '.pth', '.npy', '.npz', '.csv']
        self.max_file_size = 1024 * 1024 * 1024  # 1GB max file size
    
    def validate_file_path(self, file_path: Union[str, Path]) -> Path:
        """Validate and sanitize file path.
        
        Args:
            file_path: Input file path
            
        Returns:
            Validated Path object
            
        Raises:
            SecurityError: If path is unsafe
        """
        try:
            path = Path(file_path).resolve()
        except (OSError, ValueError) as e:
            raise SecurityError(f"Invalid file path: {e}")
        
        # Check for directory traversal attempts
        if '..' in str(path) or '~' in str(path):
            raise SecurityError(f"Potentially unsafe path: {path}")
        
        # Check file extension
        if path.suffix.lower() not in self.allowed_extensions:
            raise SecurityError(f"File extension {path.suffix} not allowed")
        
        # Check if file exists and is within size limits
        if path.exists():
            file_size = path.stat().st_size
            if file_size > self.max_file_size:
                raise SecurityError(f"File too large: {file_size} bytes > {self.max_file_size}")
        
        return path
    
    def safe_load_tensor(self, file_path: Union[str, Path]) -> torch.Tensor:
        """Safely load tensor data.
        
        Args:
            file_path: Path to tensor file
            
        Returns:
            Loaded tensor
            
        Raises:
            SecurityError: If loading is unsafe
        """
        validated_path = self.validate_file_path(file_path)
        
        try:
            if validated_path.suffix == '.pt' or validated_path.suffix == '.pth':
                # Use weights_only=True for security
                data = torch.load(validated_path, map_location='cpu', weights_only=True)
            elif validated_path.suffix == '.npy':
                import numpy as np
                data = torch.from_numpy(np.load(validated_path))
            else:
                raise SecurityError(f"Unsupported tensor format: {validated_path.suffix}")
            
            # Validate tensor properties
            self._validate_tensor_security(data)
            
            return data
            
        except Exception as e:
            raise SecurityError(f"Failed to safely load tensor: {e}")
    
    def _validate_tensor_security(self, tensor: torch.Tensor) -> None:
        """Validate tensor for security issues.
        
        Args:
            tensor: Tensor to validate
            
        Raises:
            SecurityError: If tensor has security issues
        """
        # Check tensor size (prevent memory bombs)
        max_elements = 100_000_000  # 100M elements max
        if tensor.numel() > max_elements:
            raise SecurityError(f"Tensor too large: {tensor.numel()} elements > {max_elements}")
        
        # Check for NaN/Inf values that could cause issues
        if torch.isnan(tensor).any():
            warnings.warn("Tensor contains NaN values", UserWarning)
        
        if torch.isinf(tensor).any():
            warnings.warn("Tensor contains infinite values", UserWarning)
        
        # Check for extremely large values that could cause overflow
        if tensor.dtype == torch.float32:
            max_safe_val = 1e30
            if torch.abs(tensor).max() > max_safe_val:
                warnings.warn(f"Tensor contains very large values: {torch.abs(tensor).max()}", UserWarning)


class ModelSecurityChecker:
    """Security checker for neural network models."""
    
    def __init__(self):
        self.suspicious_patterns = [
            'exec', 'eval', 'import', '__import__',
            'subprocess', 'os.system', 'pickle'
        ]
    
    def check_model_security(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Check model for potential security issues.
        
        Args:
            model: PyTorch model to check
            
        Returns:
            Security check results
        """
        results = {
            'is_secure': True,
            'warnings': [],
            'risks': [],
            'checks': {}
        }
        
        # Check for suspicious code patterns
        self._check_suspicious_code(model, results)
        
        # Check model size and complexity
        self._check_model_complexity(model, results)
        
        # Check for custom layers that might be risky
        self._check_custom_layers(model, results)
        
        results['is_secure'] = len(results['risks']) == 0
        
        return results
    
    def _check_suspicious_code(self, model: torch.nn.Module, results: Dict) -> None:
        """Check for suspicious code patterns in model."""
        try:
            model_source = str(model)
            
            for pattern in self.suspicious_patterns:
                if pattern in model_source.lower():
                    results['risks'].append(f"Suspicious pattern found: {pattern}")
                    
        except Exception:
            results['warnings'].append("Could not analyze model source code")
        
        results['checks']['suspicious_code'] = 'completed'
    
    def _check_model_complexity(self, model: torch.nn.Module, results: Dict) -> None:
        """Check model complexity for potential issues."""
        total_params = sum(p.numel() for p in model.parameters())
        
        # Very large models might indicate malicious intent
        if total_params > 1_000_000_000:  # 1B parameters
            results['warnings'].append(f"Very large model: {total_params:,} parameters")
        
        # Check memory usage
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        memory_gb = param_memory / (1024**3)
        
        if memory_gb > 10:  # 10GB
            results['warnings'].append(f"High memory usage: {memory_gb:.1f} GB")
        
        results['checks']['complexity'] = {
            'parameters': total_params,
            'memory_gb': memory_gb
        }
    
    def _check_custom_layers(self, model: torch.nn.Module, results: Dict) -> None:
        """Check for potentially risky custom layers."""
        standard_modules = {
            'Linear', 'Conv1d', 'Conv2d', 'BatchNorm1d', 'BatchNorm2d',
            'ReLU', 'ELU', 'GELU', 'Dropout', 'LayerNorm', 'ModuleList',
            'ModuleDict', 'Sequential', 'GCNConv', 'GATConv', 'SAGEConv'
        }
        
        custom_modules = []
        for name, module in model.named_modules():
            module_type = type(module).__name__
            if module_type not in standard_modules and not module_type.startswith('_'):
                custom_modules.append((name, module_type))
        
        if custom_modules:
            results['warnings'].append(f"Found {len(custom_modules)} custom modules")
            for name, module_type in custom_modules[:5]:  # Limit output
                results['warnings'].append(f"Custom module: {name} ({module_type})")
        
        results['checks']['custom_layers'] = len(custom_modules)


class SecureTrainingEnvironment:
    """Secure environment setup for model training."""
    
    def __init__(self, work_dir: Optional[Path] = None):
        """Initialize secure training environment.
        
        Args:
            work_dir: Working directory for training (creates temp if None)
        """
        self.work_dir = work_dir or Path(tempfile.mkdtemp(prefix='connectome_training_'))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Security settings
        self.max_disk_usage = 10 * 1024**3  # 10GB max disk usage
        self.allowed_operations = {
            'read_data', 'write_checkpoints', 'create_logs', 'save_results'
        }
    
    def __enter__(self):
        """Enter secure training context."""
        logger.info(f"Starting secure training in {self.work_dir}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up secure training environment."""
        self._cleanup_environment()
    
    def validate_operation(self, operation: str, file_path: Path) -> bool:
        """Validate if operation is allowed.
        
        Args:
            operation: Type of operation to perform
            file_path: Path for the operation
            
        Returns:
            Whether operation is allowed
        """
        if operation not in self.allowed_operations:
            logger.warning(f"Operation {operation} not allowed")
            return False
        
        # Check if path is within work directory
        try:
            file_path.resolve().relative_to(self.work_dir.resolve())
        except ValueError:
            logger.warning(f"Path {file_path} outside work directory")
            return False
        
        return True
    
    def check_disk_usage(self) -> Dict[str, Any]:
        """Check current disk usage in work directory.
        
        Returns:
            Disk usage information
        """
        total_size = 0
        file_count = 0
        
        for file_path in self.work_dir.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1
        
        usage_info = {
            'total_size_gb': total_size / (1024**3),
            'file_count': file_count,
            'within_limit': total_size <= self.max_disk_usage
        }
        
        if not usage_info['within_limit']:
            logger.warning(f"Disk usage {usage_info['total_size_gb']:.1f} GB exceeds limit")
        
        return usage_info
    
    def _cleanup_environment(self) -> None:
        """Clean up training environment."""
        try:
            if self.work_dir.exists() and self.work_dir.name.startswith('connectome_training_'):
                shutil.rmtree(self.work_dir)
                logger.info(f"Cleaned up training directory: {self.work_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup training directory: {e}")


class InputSanitizer:
    """Sanitize inputs for connectome analysis pipeline."""
    
    @staticmethod
    def sanitize_string(input_str: str, max_length: int = 1000) -> str:
        """Sanitize string input.
        
        Args:
            input_str: Input string to sanitize
            max_length: Maximum allowed string length
            
        Returns:
            Sanitized string
            
        Raises:
            ValueError: If input is invalid
        """
        if not isinstance(input_str, str):
            raise ValueError("Input must be a string")
        
        if len(input_str) > max_length:
            raise ValueError(f"String too long: {len(input_str)} > {max_length}")
        
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '&', '"', "'", '\\', '/', '\x00']
        sanitized = input_str
        
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        return sanitized.strip()
    
    @staticmethod
    def sanitize_numeric_input(
        value: Union[int, float],
        min_val: Optional[float] = None,
        max_val: Optional[float] = None
    ) -> Union[int, float]:
        """Sanitize numeric input.
        
        Args:
            value: Numeric value to sanitize
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            Sanitized numeric value
            
        Raises:
            ValueError: If value is invalid
        """
        if not isinstance(value, (int, float)):
            raise ValueError("Input must be numeric")
        
        if not isinstance(value, bool) and (np.isnan(value) or np.isinf(value)):
            raise ValueError("Input cannot be NaN or infinite")
        
        if min_val is not None and value < min_val:
            raise ValueError(f"Value {value} below minimum {min_val}")
        
        if max_val is not None and value > max_val:
            raise ValueError(f"Value {value} above maximum {max_val}")
        
        return value


class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of file for integrity checking.
    
    Args:
        file_path: Path to file
        
    Returns:
        SHA256 hash as hex string
    """
    hash_sha256 = hashlib.sha256()
    
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        raise SecurityError(f"Failed to compute file hash: {e}")


def setup_secure_environment() -> Dict[str, Any]:
    """Setup secure environment for connectome analysis.
    
    Returns:
        Environment configuration
    """
    config = {
        'torch_settings': {},
        'warnings_filtered': [],
        'security_measures': []
    }
    
    # Configure PyTorch for security
    try:
        # Disable JIT compilation which could be exploited
        torch.jit.set_enabled(False)
        config['torch_settings']['jit_disabled'] = True
        config['security_measures'].append('JIT compilation disabled')
    except Exception:
        pass
    
    # Filter specific warnings that could leak information
    warnings.filterwarnings('ignore', category=UserWarning, module='torch')
    config['warnings_filtered'].append('torch UserWarnings')
    
    # Set secure random seed if needed
    torch.manual_seed(42)  # Deterministic for reproducibility
    config['torch_settings']['manual_seed'] = 42
    
    return config