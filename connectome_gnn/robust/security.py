"""Security utilities and input sanitization."""

import hashlib
import hmac
import os
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import tempfile
import shutil
import logging
import numpy as np


class SecurityError(Exception):
    """Security-related exception."""
    def __init__(self, message: str, security_type: str = None):
        super().__init__(message)
        self.security_type = security_type


class SecurityManager:
    """Manages security-related operations."""
    
    def __init__(self, secret_key: Optional[str] = None, logger: Optional[logging.Logger] = None):
        self.secret_key = secret_key or os.urandom(32).hex()
        self.logger = logger or logging.getLogger(__name__)
        
        # Security policies
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        self.allowed_file_extensions = {'.pt', '.pth', '.json', '.csv', '.npy', '.npz', '.mat'}
        self.blocked_patterns = [
            r'\.\./',          # Directory traversal
            r'\\.\\.\\',       # Windows directory traversal
            r'<script.*?>',    # Script tags
            r'javascript:',    # JavaScript URLs
            r'on\w+\s*=',     # Event handlers
            r'[;&|`$]',       # Command injection characters
        ]
        
    def validate_file_upload(self, filename: str, content: bytes) -> Tuple[bool, List[str]]:
        """Validate uploaded file for security issues."""
        issues = []
        
        # Check file size
        if len(content) > self.max_file_size:
            issues.append(f"File too large: {len(content)} bytes > {self.max_file_size}")
        
        # Check filename for dangerous patterns
        filename = Path(filename).name  # Remove any path components
        
        # Check file extension
        file_ext = Path(filename).suffix.lower()
        if file_ext not in self.allowed_file_extensions:
            issues.append(f"File extension {file_ext} not allowed")
        
        # Check for dangerous filename patterns
        for pattern in self.blocked_patterns:
            if re.search(pattern, filename, re.IGNORECASE):
                issues.append(f"Dangerous pattern in filename: {pattern}")
        
        # Check content for malicious patterns (for text files)
        if file_ext in {'.json', '.csv', '.txt'}:
            try:
                content_str = content.decode('utf-8', errors='ignore')
                for pattern in self.blocked_patterns:
                    if re.search(pattern, content_str, re.IGNORECASE):
                        issues.append(f"Dangerous content pattern detected: {pattern}")
            except Exception:
                pass  # Binary content, skip text analysis
        
        return len(issues) == 0, issues
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage."""
        # Get only the filename part
        filename = Path(filename).name
        
        # Remove or replace dangerous characters
        safe_filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        safe_filename = re.sub(r'\.{2,}', '.', safe_filename)  # Multiple dots
        safe_filename = safe_filename.strip('. ')  # Leading/trailing dots and spaces
        
        # Ensure it's not empty
        if not safe_filename:
            safe_filename = f"file_{os.urandom(8).hex()}"
        
        return safe_filename
    
    def validate_json_input(self, data: Union[str, Dict]) -> Tuple[bool, List[str], Optional[Dict]]:
        """Validate JSON input for security issues."""
        issues = []
        parsed_data = None
        
        try:
            if isinstance(data, str):
                parsed_data = json.loads(data)
            else:
                parsed_data = data
        except json.JSONDecodeError as e:
            issues.append(f"Invalid JSON: {e}")
            return False, issues, None
        
        # Check for dangerous keys or values
        dangerous_keys = ['__import__', 'eval', 'exec', 'open', 'file', '__builtins__']
        
        def check_dict_recursively(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    # Check for dangerous keys
                    if key in dangerous_keys:
                        issues.append(f"Dangerous key detected: {current_path}")
                    
                    # Check for dangerous values
                    if isinstance(value, str):
                        for pattern in self.blocked_patterns:
                            if re.search(pattern, value, re.IGNORECASE):
                                issues.append(f"Dangerous value at {current_path}: {pattern}")
                    
                    # Recurse into nested structures
                    check_dict_recursively(value, current_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    current_path = f"{path}[{i}]" if path else f"[{i}]"
                    check_dict_recursively(item, current_path)
        
        check_dict_recursively(parsed_data)
        
        return len(issues) == 0, issues, parsed_data
    
    def create_secure_temp_file(self, suffix: str = '', prefix: str = 'connectome_') -> str:
        """Create a secure temporary file."""
        # Create temporary file with secure permissions
        fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
        os.close(fd)
        
        # Set restrictive permissions (owner read/write only)
        os.chmod(temp_path, 0o600)
        
        return temp_path
    
    def secure_file_copy(self, src_path: str, dst_path: str) -> bool:
        """Securely copy a file with validation."""
        try:
            src_path = Path(src_path).resolve()
            dst_path = Path(dst_path).resolve()
            
            # Validate source file
            if not src_path.exists():
                self.logger.error(f"Source file does not exist: {src_path}")
                return False
            
            if not src_path.is_file():
                self.logger.error(f"Source is not a file: {src_path}")
                return False
            
            # Check file size
            file_size = src_path.stat().st_size
            if file_size > self.max_file_size:
                self.logger.error(f"File too large: {file_size} bytes")
                return False
            
            # Validate destination directory
            dst_dir = dst_path.parent
            dst_dir.mkdir(parents=True, exist_ok=True)
            
            # Perform secure copy
            shutil.copy2(src_path, dst_path)
            
            # Set secure permissions on destination
            os.chmod(dst_path, 0o644)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Secure file copy failed: {e}")
            return False
    
    def generate_file_hash(self, filepath: str, algorithm: str = 'sha256') -> str:
        """Generate secure hash of a file."""
        hash_func = hashlib.new(algorithm)
        
        try:
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_func.update(chunk)
            return hash_func.hexdigest()
        except Exception as e:
            self.logger.error(f"Hash generation failed: {e}")
            raise SecurityError(f"Failed to generate hash: {e}", security_type="hash_generation")
    
    def verify_file_integrity(self, filepath: str, expected_hash: str, 
                            algorithm: str = 'sha256') -> bool:
        """Verify file integrity using hash."""
        try:
            actual_hash = self.generate_file_hash(filepath, algorithm)
            return hmac.compare_digest(expected_hash.lower(), actual_hash.lower())
        except Exception as e:
            self.logger.error(f"Integrity verification failed: {e}")
            return False
    
    def sanitize_path(self, path: str, base_dir: Optional[str] = None) -> Optional[str]:
        """Sanitize file path to prevent directory traversal."""
        try:
            # Resolve the path
            clean_path = Path(path).resolve()
            
            # If base directory specified, ensure path is within it
            if base_dir:
                base_path = Path(base_dir).resolve()
                try:
                    clean_path.relative_to(base_path)
                except ValueError:
                    self.logger.warning(f"Path outside base directory: {path}")
                    return None
            
            return str(clean_path)
            
        except Exception as e:
            self.logger.error(f"Path sanitization failed: {e}")
            return None
    
    def create_api_key(self, identifier: str = None) -> str:
        """Create a secure API key."""
        # Generate random component
        random_part = os.urandom(24)
        
        # Add identifier component if provided
        if identifier:
            identifier_bytes = identifier.encode('utf-8')
            combined = identifier_bytes + random_part
        else:
            combined = random_part
        
        # Create HMAC signature
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            combined,
            hashlib.sha256
        ).digest()
        
        # Combine and encode
        api_key_bytes = combined + signature
        api_key = api_key_bytes.hex()
        
        return api_key
    
    def validate_api_key(self, api_key: str, identifier: str = None) -> bool:
        """Validate an API key."""
        try:
            # Decode the key
            api_key_bytes = bytes.fromhex(api_key)
            
            # Split components
            if identifier:
                identifier_bytes = identifier.encode('utf-8')
                expected_len = len(identifier_bytes) + 24 + 32  # identifier + random + signature
                if len(api_key_bytes) != expected_len:
                    return False
                
                received_identifier = api_key_bytes[:len(identifier_bytes)]
                received_random = api_key_bytes[len(identifier_bytes):-32]
                received_signature = api_key_bytes[-32:]
                
                if received_identifier != identifier_bytes:
                    return False
                
                combined = identifier_bytes + received_random
            else:
                expected_len = 24 + 32  # random + signature
                if len(api_key_bytes) != expected_len:
                    return False
                
                received_random = api_key_bytes[:-32]
                received_signature = api_key_bytes[-32:]
                combined = received_random
            
            # Verify signature
            expected_signature = hmac.new(
                self.secret_key.encode('utf-8'),
                combined,
                hashlib.sha256
            ).digest()
            
            return hmac.compare_digest(received_signature, expected_signature)
            
        except Exception as e:
            self.logger.error(f"API key validation failed: {e}")
            return False


def validate_model_input(data: Any, max_nodes: int = 10000, max_edges: int = 100000) -> Tuple[bool, List[str]]:
    """Validate model input data for security and size limits."""
    issues = []
    
    try:
        # Check if it's a PyTorch tensor or similar
        if hasattr(data, 'shape'):
            # Check tensor dimensions
            if hasattr(data, 'size') and len(data.shape) > 4:
                issues.append(f"Tensor has too many dimensions: {len(data.shape)}")
            
            # Check tensor size
            if hasattr(data, 'numel'):
                total_elements = data.numel()
                if total_elements > max_nodes * max_edges:
                    issues.append(f"Tensor too large: {total_elements} elements")
        
        # Check for graph data
        if hasattr(data, 'edge_index'):
            num_edges = data.edge_index.shape[1] if hasattr(data.edge_index, 'shape') else 0
            if num_edges > max_edges:
                issues.append(f"Too many edges: {num_edges} > {max_edges}")
        
        if hasattr(data, 'x') and hasattr(data.x, 'shape'):
            num_nodes = data.x.shape[0]
            if num_nodes > max_nodes:
                issues.append(f"Too many nodes: {num_nodes} > {max_nodes}")
        
    except Exception as e:
        issues.append(f"Input validation error: {e}")
    
    return len(issues) == 0, issues


class InputSanitizer:
    """Sanitize various types of inputs."""
    
    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000, allowed_chars: str = None) -> str:
        """Sanitize string input."""
        if not isinstance(value, str):
            raise ValueError("Input must be a string")
        
        # Truncate if too long
        if len(value) > max_length:
            value = value[:max_length]
        
        # Remove dangerous characters if allowed_chars specified
        if allowed_chars:
            value = ''.join(c for c in value if c in allowed_chars)
        
        # Remove control characters
        value = ''.join(c for c in value if ord(c) >= 32 or c in '\t\n\r')
        
        return value
    
    @staticmethod
    def sanitize_number(value: Union[int, float], min_val: float = None, 
                       max_val: float = None) -> Union[int, float]:
        """Sanitize numeric input."""
        if not isinstance(value, (int, float)):
            raise ValueError("Input must be a number")
        
        # Check for NaN or infinity
        if isinstance(value, float):
            if not np.isfinite(value):
                raise ValueError("Value must be finite")
        
        # Apply bounds
        if min_val is not None and value < min_val:
            value = min_val
        if max_val is not None and value > max_val:
            value = max_val
        
        return value
    
    @staticmethod
    def sanitize_list(value: List, max_length: int = 1000, 
                     item_sanitizer: Callable = None) -> List:
        """Sanitize list input."""
        if not isinstance(value, list):
            raise ValueError("Input must be a list")
        
        # Truncate if too long
        if len(value) > max_length:
            value = value[:max_length]
        
        # Sanitize individual items
        if item_sanitizer:
            value = [item_sanitizer(item) for item in value]
        
        return value


# Create global security manager
_global_security_manager = None

def get_security_manager() -> SecurityManager:
    """Get global security manager instance."""
    global _global_security_manager
    if _global_security_manager is None:
        _global_security_manager = SecurityManager()
    return _global_security_manager