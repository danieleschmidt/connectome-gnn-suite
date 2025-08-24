"""Comprehensive input and model validation."""

import numpy as np
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from pathlib import Path
import re
import json

from .error_handling import ValidationError, DataError


def validate_connectome_data(connectivity_matrix: np.ndarray) -> bool:
    """Validate connectome connectivity matrix.
    
    Args:
        connectivity_matrix: Square connectivity matrix
        
    Returns:
        True if valid, raises ValidationError otherwise
        
    Raises:
        ValidationError: If data is invalid
    """
    if not isinstance(connectivity_matrix, np.ndarray):
        raise ValidationError("Connectivity matrix must be numpy array")
        
    if len(connectivity_matrix.shape) != 2:
        raise ValidationError("Connectivity matrix must be 2D")
        
    if connectivity_matrix.shape[0] != connectivity_matrix.shape[1]:
        raise ValidationError("Connectivity matrix must be square")
        
    if connectivity_matrix.shape[0] == 0:
        raise ValidationError("Connectivity matrix cannot be empty")
        
    if not np.isfinite(connectivity_matrix).all():
        raise ValidationError("Connectivity matrix contains non-finite values")
        
    return True


class BaseValidator:
    """Base class for all validators."""
    
    def __init__(self, strict: bool = True, logger = None):
        self.strict = strict
        self.logger = logger
        self.validation_errors = []
    
    def validate(self, data: Any) -> bool:
        """Main validation method - to be implemented by subclasses."""
        raise NotImplementedError
    
    def add_error(self, message: str, error_code: str = None):
        """Add validation error."""
        error = {
            'message': message,
            'error_code': error_code or 'VALIDATION_ERROR',
            'validator': self.__class__.__name__
        }
        self.validation_errors.append(error)
        
        if self.logger:
            self.logger.warning(f"Validation error: {message}")
    
    def clear_errors(self):
        """Clear validation errors."""
        self.validation_errors = []
    
    def get_errors(self) -> List[Dict[str, str]]:
        """Get all validation errors."""
        return self.validation_errors.copy()
    
    def has_errors(self) -> bool:
        """Check if there are validation errors."""
        return len(self.validation_errors) > 0
    
    def raise_if_errors(self):
        """Raise ValidationError if there are any errors."""
        if self.has_errors():
            error_messages = [err['message'] for err in self.validation_errors]
            raise ValidationError(
                f"Validation failed with {len(error_messages)} errors: {'; '.join(error_messages)}",
                validation_type=self.__class__.__name__,
                details={'errors': self.validation_errors}
            )


class InputValidator(BaseValidator):
    """Validates various types of inputs."""
    
    def validate_array(self, array: Any, 
                      expected_shape: Optional[Tuple] = None,
                      expected_dtype: Optional[type] = None,
                      min_value: Optional[float] = None,
                      max_value: Optional[float] = None,
                      allow_nan: bool = False,
                      allow_inf: bool = False) -> bool:
        """Validate array inputs."""
        self.clear_errors()
        
        # Check if it's array-like
        if not hasattr(array, 'shape'):
            try:
                array = np.array(array)
            except:
                self.add_error("Input cannot be converted to array", "INVALID_ARRAY")
                return not self.strict
        
        # Check shape
        if expected_shape is not None:
            if array.shape != expected_shape:
                # Allow flexible dimensions (use -1 for any size)
                shape_match = True
                if len(array.shape) != len(expected_shape):
                    shape_match = False
                else:
                    for actual, expected in zip(array.shape, expected_shape):
                        if expected != -1 and actual != expected:
                            shape_match = False
                            break
                
                if not shape_match:
                    self.add_error(
                        f"Array shape {array.shape} doesn't match expected {expected_shape}",
                        "SHAPE_MISMATCH"
                    )
        
        # Check dtype
        if expected_dtype is not None and array.dtype != expected_dtype:
            self.add_error(
                f"Array dtype {array.dtype} doesn't match expected {expected_dtype}",
                "DTYPE_MISMATCH"
            )
        
        # Check for NaN values
        if not allow_nan and np.isnan(array).any():
            self.add_error("Array contains NaN values", "NAN_VALUES")
        
        # Check for infinite values
        if not allow_inf and np.isinf(array).any():
            self.add_error("Array contains infinite values", "INF_VALUES")
        
        # Check value range
        if min_value is not None and np.any(array < min_value):
            self.add_error(f"Array contains values below minimum {min_value}", "VALUE_TOO_LOW")
        
        if max_value is not None and np.any(array > max_value):
            self.add_error(f"Array contains values above maximum {max_value}", "VALUE_TOO_HIGH")
        
        if self.strict:
            self.raise_if_errors()
        
        return not self.has_errors()
    
    def validate_connectivity_matrix(self, matrix: Any) -> bool:
        """Validate brain connectivity matrix."""
        self.clear_errors()
        
        # Basic array validation
        if not self.validate_array(matrix, min_value=-1, max_value=1):
            return False
        
        matrix = np.array(matrix)
        
        # Check if square
        if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
            self.add_error("Connectivity matrix must be square", "NOT_SQUARE")
        
        # Check if symmetric (with tolerance)
        if not np.allclose(matrix, matrix.T, atol=1e-6):
            self.add_error("Connectivity matrix should be symmetric", "NOT_SYMMETRIC")
        
        # Check diagonal (should be zeros for most cases)
        diagonal_values = np.diag(matrix)
        if not np.allclose(diagonal_values, 0, atol=1e-6):
            self.add_error("Connectivity matrix diagonal should be zero", "NON_ZERO_DIAGONAL")
        
        if self.strict:
            self.raise_if_errors()
        
        return not self.has_errors()
    
    def validate_node_features(self, features: Any, num_nodes: int) -> bool:
        """Validate node feature matrix."""
        self.clear_errors()
        
        if not self.validate_array(features):
            return False
        
        features = np.array(features)
        
        # Check number of nodes
        if features.shape[0] != num_nodes:
            self.add_error(
                f"Number of node features {features.shape[0]} doesn't match expected {num_nodes}",
                "NODE_COUNT_MISMATCH"
            )
        
        # Check feature dimension
        if len(features.shape) != 2:
            self.add_error("Node features must be 2D (nodes x features)", "INVALID_FEATURE_SHAPE")
        
        if self.strict:
            self.raise_if_errors()
        
        return not self.has_errors()
    
    def validate_edge_list(self, edge_index: Any, num_nodes: int) -> bool:
        """Validate edge index format."""
        self.clear_errors()
        
        if not self.validate_array(edge_index):
            return False
        
        edge_index = np.array(edge_index)
        
        # Check shape (should be 2 x num_edges)
        if len(edge_index.shape) != 2 or edge_index.shape[0] != 2:
            self.add_error("Edge index must be 2 x num_edges", "INVALID_EDGE_SHAPE")
        
        # Check node indices are valid
        max_node_idx = edge_index.max()
        min_node_idx = edge_index.min()
        
        if min_node_idx < 0:
            self.add_error("Edge indices cannot be negative", "NEGATIVE_NODE_INDEX")
        
        if max_node_idx >= num_nodes:
            self.add_error(
                f"Edge index {max_node_idx} exceeds number of nodes {num_nodes}",
                "NODE_INDEX_OUT_OF_RANGE"
            )
        
        if self.strict:
            self.raise_if_errors()
        
        return not self.has_errors()
    
    def validate_batch_data(self, data: Any) -> bool:
        """Validate batch data structure."""
        self.clear_errors()
        
        # Check if data has required attributes
        required_attrs = ['x', 'edge_index']
        for attr in required_attrs:
            if not hasattr(data, attr):
                self.add_error(f"Data missing required attribute: {attr}", "MISSING_ATTRIBUTE")
        
        # Validate x and edge_index if present
        if hasattr(data, 'x') and hasattr(data, 'edge_index'):
            num_nodes = data.x.shape[0] if hasattr(data.x, 'shape') else len(data.x)
            self.validate_edge_list(data.edge_index, num_nodes)
        
        if self.strict:
            self.raise_if_errors()
        
        return not self.has_errors()


class ModelValidator(BaseValidator):
    """Validates model configurations and states."""
    
    def validate_model_config(self, config: Dict[str, Any]) -> bool:
        """Validate model configuration."""
        self.clear_errors()
        
        # Required fields
        required_fields = ['model_type', 'node_features', 'hidden_dim', 'output_dim']
        for field in required_fields:
            if field not in config:
                self.add_error(f"Missing required config field: {field}", "MISSING_CONFIG_FIELD")
        
        # Validate numeric parameters
        numeric_fields = {
            'node_features': (1, None),
            'hidden_dim': (1, None),
            'output_dim': (1, None),
            'num_layers': (1, 20),
            'dropout': (0.0, 1.0)
        }
        
        for field, (min_val, max_val) in numeric_fields.items():
            if field in config:
                value = config[field]
                if not isinstance(value, (int, float)):
                    self.add_error(f"Config field {field} must be numeric", "INVALID_TYPE")
                elif value < min_val:
                    self.add_error(f"Config field {field} must be >= {min_val}", "VALUE_TOO_LOW")
                elif max_val is not None and value > max_val:
                    self.add_error(f"Config field {field} must be <= {max_val}", "VALUE_TOO_HIGH")
        
        # Validate activation function
        if 'activation' in config:
            valid_activations = ['relu', 'gelu', 'swish', 'elu', 'leaky_relu', 'tanh']
            if config['activation'] not in valid_activations:
                self.add_error(
                    f"Invalid activation function: {config['activation']}",
                    "INVALID_ACTIVATION"
                )
        
        if self.strict:
            self.raise_if_errors()
        
        return not self.has_errors()
    
    def validate_model_state(self, model: Any) -> bool:
        """Validate model state and parameters."""
        self.clear_errors()
        
        # Check if model has parameters
        if hasattr(model, 'parameters'):
            param_count = 0
            for param in model.parameters():
                param_count += 1
                
                # Check for NaN parameters
                if hasattr(param, 'data'):
                    if np.isnan(param.data.cpu().numpy()).any():
                        self.add_error("Model contains NaN parameters", "NAN_PARAMETERS")
                
                # Check for extreme parameter values
                if hasattr(param, 'data'):
                    param_values = param.data.cpu().numpy()
                    if np.abs(param_values).max() > 1e6:
                        self.add_error("Model contains extreme parameter values", "EXTREME_PARAMETERS")
            
            if param_count == 0:
                self.add_error("Model has no parameters", "NO_PARAMETERS")
        
        if self.strict:
            self.raise_if_errors()
        
        return not self.has_errors()


class DataValidator(BaseValidator):
    """Validates datasets and data loaders."""
    
    def validate_dataset(self, dataset: Any) -> bool:
        """Validate dataset structure and contents."""
        self.clear_errors()
        
        # Check if dataset has required methods
        required_methods = ['__len__', '__getitem__']
        for method in required_methods:
            if not hasattr(dataset, method):
                self.add_error(f"Dataset missing required method: {method}", "MISSING_METHOD")
        
        # Check dataset size
        if hasattr(dataset, '__len__'):
            try:
                length = len(dataset)
                if length == 0:
                    self.add_error("Dataset is empty", "EMPTY_DATASET")
                elif length < 0:
                    self.add_error("Dataset has negative length", "NEGATIVE_LENGTH")
            except Exception as e:
                self.add_error(f"Cannot determine dataset length: {e}", "LENGTH_ERROR")
        
        # Sample check - try to get first item
        if hasattr(dataset, '__getitem__') and hasattr(dataset, '__len__'):
            try:
                if len(dataset) > 0:
                    sample = dataset[0]
                    # Validate sample structure
                    self.validate_batch_data(sample)
            except Exception as e:
                self.add_error(f"Cannot access dataset items: {e}", "ACCESS_ERROR")
        
        if self.strict:
            self.raise_if_errors()
        
        return not self.has_errors()
    
    def validate_data_splits(self, train_size: int, val_size: int, test_size: int) -> bool:
        """Validate data split sizes."""
        self.clear_errors()
        
        # Check all sizes are positive
        for split_name, size in [('train', train_size), ('val', val_size), ('test', test_size)]:
            if size < 0:
                self.add_error(f"{split_name} split size cannot be negative", "NEGATIVE_SPLIT_SIZE")
        
        # Check total size is reasonable
        total_size = train_size + val_size + test_size
        if total_size == 0:
            self.add_error("Total dataset size is zero", "ZERO_TOTAL_SIZE")
        
        # Check train set is not empty
        if train_size == 0:
            self.add_error("Training set cannot be empty", "EMPTY_TRAIN_SET")
        
        if self.strict:
            self.raise_if_errors()
        
        return not self.has_errors()


class FileValidator(BaseValidator):
    """Validates file paths and contents."""
    
    def validate_file_path(self, filepath: str, must_exist: bool = True,
                          allowed_extensions: Optional[List[str]] = None) -> bool:
        """Validate file path."""
        self.clear_errors()
        
        path = Path(filepath)
        
        # Check if file exists
        if must_exist and not path.exists():
            self.add_error(f"File does not exist: {filepath}", "FILE_NOT_FOUND")
        
        # Check file extension
        if allowed_extensions:
            if path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
                self.add_error(
                    f"File extension {path.suffix} not in allowed extensions {allowed_extensions}",
                    "INVALID_EXTENSION"
                )
        
        # Check if path is safe (no directory traversal)
        try:
            path.resolve()
        except Exception:
            self.add_error(f"Unsafe file path: {filepath}", "UNSAFE_PATH")
        
        if self.strict:
            self.raise_if_errors()
        
        return not self.has_errors()
    
    def validate_json_file(self, filepath: str) -> bool:
        """Validate JSON file content."""
        self.clear_errors()
        
        # First validate the file path
        if not self.validate_file_path(filepath, allowed_extensions=['.json']):
            return False
        
        # Try to parse JSON
        try:
            with open(filepath, 'r') as f:
                json.load(f)
        except json.JSONDecodeError as e:
            self.add_error(f"Invalid JSON in file {filepath}: {e}", "INVALID_JSON")
        except Exception as e:
            self.add_error(f"Cannot read JSON file {filepath}: {e}", "JSON_READ_ERROR")
        
        if self.strict:
            self.raise_if_errors()
        
        return not self.has_errors()


class SecurityValidator(BaseValidator):
    """Validates inputs for security issues."""
    
    def validate_user_input(self, user_input: str, max_length: int = 1000) -> bool:
        """Validate user input for security issues."""
        self.clear_errors()
        
        # Check length
        if len(user_input) > max_length:
            self.add_error(f"Input too long: {len(user_input)} > {max_length}", "INPUT_TOO_LONG")
        
        # Check for potentially dangerous patterns
        dangerous_patterns = [
            r'<script[^>]*>',  # Script tags
            r'javascript:',     # JavaScript URLs
            r'on\w+\s*=',      # Event handlers
            r'\.\./',          # Directory traversal
            r'\\.\\.\\',       # Windows directory traversal
            r'[;&|`]',         # Command injection
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                self.add_error("Input contains potentially dangerous pattern", "DANGEROUS_PATTERN")
                break
        
        if self.strict:
            self.raise_if_errors()
        
        return not self.has_errors()
    
    def validate_file_upload(self, filename: str, content: bytes, 
                           max_size: int = 10 * 1024 * 1024) -> bool:  # 10MB default
        """Validate uploaded file for security."""
        self.clear_errors()
        
        # Check file size
        if len(content) > max_size:
            self.add_error(f"File too large: {len(content)} > {max_size}", "FILE_TOO_LARGE")
        
        # Check filename for dangerous characters
        if re.search(r'[<>:"/\\|?*]', filename):
            self.add_error("Filename contains dangerous characters", "DANGEROUS_FILENAME")
        
        # Check for executable file extensions
        dangerous_extensions = ['.exe', '.bat', '.sh', '.cmd', '.scr', '.pif']
        file_ext = Path(filename).suffix.lower()
        if file_ext in dangerous_extensions:
            self.add_error(f"Dangerous file extension: {file_ext}", "DANGEROUS_EXTENSION")
        
        # Check file content for embedded scripts (simple check)
        content_str = content[:1000].decode('utf-8', errors='ignore').lower()
        if '<script' in content_str or 'javascript:' in content_str:
            self.add_error("File content contains potentially dangerous scripts", "DANGEROUS_CONTENT")
        
        if self.strict:
            self.raise_if_errors()
        
        return not self.has_errors()


def validate_all(data: Any, validators: List[BaseValidator]) -> Dict[str, Any]:
    """Run multiple validators on data."""
    results = {}
    all_passed = True
    
    for validator in validators:
        validator_name = validator.__class__.__name__
        
        try:
            result = validator.validate(data)
            results[validator_name] = {
                'passed': result,
                'errors': validator.get_errors()
            }
            
            if not result:
                all_passed = False
                
        except Exception as e:
            results[validator_name] = {
                'passed': False,
                'errors': [{'message': str(e), 'error_code': 'VALIDATOR_ERROR'}]
            }
            all_passed = False
    
    results['overall_passed'] = all_passed
    return results