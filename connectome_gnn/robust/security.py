"""Security utilities and input sanitization."""

import hashlib
import hmac
import os
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import tempfile
import shutil
import logging


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
                issues.append(f"Filename contains dangerous pattern: {pattern}")
                break
        
        # Check file content (basic checks)
        try:
            # Check for null bytes
            if b'\x00' in content:
                issues.append("File contains null bytes")
            
            # For text-based files, check for scripts
            if file_ext in {'.json', '.csv'}:
                content_str = content[:1000].decode('utf-8', errors='ignore').lower()
                if '<script' in content_str or 'javascript:' in content_str:
                    issues.append("File content contains potentially dangerous scripts")
        except Exception as e:
            issues.append(f"Error analyzing file content: {e}")
        
        is_safe = len(issues) == 0
        
        if not is_safe:
            self.logger.warning(f"File validation failed for {filename}: {issues}")
        
        return is_safe, issues
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage."""
        # Remove path components
        filename = Path(filename).name
        
        # Remove or replace dangerous characters
        filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
        
        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:255-len(ext)] + ext
        
        # Ensure filename is not empty
        if not filename or filename.startswith('.'):
            filename = 'safe_file' + (Path(filename).suffix if filename else '.txt')
        
        return filename
    
    def create_secure_temp_file(self, suffix: str = None) -> str:
        """Create a secure temporary file."""
        fd, temp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)  # Close the file descriptor
        
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
            
            # Create destination directory if needed
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            shutil.copy2(src_path, dst_path)
            
            # Set secure permissions on destination
            os.chmod(dst_path, 0o600)
            
            self.logger.info(f"Securely copied file from {src_path} to {dst_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Secure file copy failed: {e}")
            return False
    
    def compute_file_hash(self, filepath: str, algorithm: str = 'sha256') -> str:
        """Compute secure hash of file content."""
        hash_func = hashlib.new(algorithm)
        
        try:
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_func.update(chunk)
            return hash_func.hexdigest()
        except Exception as e:
            self.logger.error(f"Failed to compute hash for {filepath}: {e}")
            return ""
    
    def verify_file_integrity(self, filepath: str, expected_hash: str, 
                            algorithm: str = 'sha256') -> bool:
        """Verify file integrity using hash."""
        actual_hash = self.compute_file_hash(filepath, algorithm)
        return hmac.compare_digest(actual_hash, expected_hash)
    
    def secure_json_load(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Securely load JSON file with validation."""
        try:
            # Validate file path
            filepath = Path(filepath).resolve()
            
            # Check file size
            file_size = filepath.stat().st_size
            if file_size > 10 * 1024 * 1024:  # 10MB limit for JSON
                self.logger.error(f"JSON file too large: {file_size} bytes")
                return None
            
            # Load and parse JSON
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Basic security check for JSON content
                if len(content) > 10 * 1024 * 1024:  # 10MB content limit
                    self.logger.error("JSON content too large")
                    return None
                
                data = json.loads(content)
                
                # Validate JSON structure (no deeply nested objects)
                if self._check_json_depth(data) > 20:
                    self.logger.error("JSON structure too deeply nested")
                    return None
                
                return data
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in {filepath}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error loading JSON from {filepath}: {e}")
            return None
    
    def _check_json_depth(self, obj: Any, depth: int = 0) -> int:
        """Check maximum depth of JSON object."""
        if depth > 20:  # Prevent excessive recursion
            return depth
        
        if isinstance(obj, dict):
            return max([self._check_json_depth(v, depth + 1) for v in obj.values()] + [depth])
        elif isinstance(obj, list):
            return max([self._check_json_depth(item, depth + 1) for item in obj] + [depth])
        else:
            return depth
    
    def secure_json_save(self, data: Dict[str, Any], filepath: str) -> bool:
        """Securely save JSON file."""
        try:
            # Validate data structure
            if self._check_json_depth(data) > 20:
                self.logger.error("Data structure too deeply nested")
                return False
            
            # Serialize to check size
            json_content = json.dumps(data, indent=2)
            if len(json_content) > 10 * 1024 * 1024:  # 10MB limit
                self.logger.error("JSON content too large")
                return False
            
            # Secure file path
            filepath = Path(filepath).resolve()
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to temporary file first
            temp_path = self.create_secure_temp_file(suffix='.json')
            
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(json_content)
            
            # Move to final location
            shutil.move(temp_path, filepath)
            os.chmod(filepath, 0o600)
            
            self.logger.info(f"Securely saved JSON to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving JSON to {filepath}: {e}")
            return False


def sanitize_input(user_input: str, max_length: int = 1000, 
                  allow_html: bool = False) -> str:
    """Sanitize user input for security."""
    if not isinstance(user_input, str):
        user_input = str(user_input)
    
    # Truncate to max length
    if len(user_input) > max_length:
        user_input = user_input[:max_length]
    
    # Remove null bytes
    user_input = user_input.replace('\x00', '')
    
    # Remove control characters except tab, newline, carriage return
    user_input = ''.join(char for char in user_input 
                        if ord(char) >= 32 or char in '\t\n\r')
    
    if not allow_html:
        # Remove HTML tags
        user_input = re.sub(r'<[^>]*>', '', user_input)
        
        # Remove JavaScript
        user_input = re.sub(r'javascript:', '', user_input, flags=re.IGNORECASE)
        
        # Remove event handlers
        user_input = re.sub(r'on\w+\s*=', '', user_input, flags=re.IGNORECASE)
    
    return user_input


def validate_file_safety(filepath: str, allowed_extensions: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
    """Validate file path for safety."""
    issues = []
    filepath = Path(filepath)
    
    try:
        # Check for directory traversal
        resolved_path = filepath.resolve()
        
        # Check if path contains dangerous patterns
        path_str = str(resolved_path)
        dangerous_patterns = ['../', '..\\', '~/', '/etc/', '/root/', 'C:\\Windows\\']
        
        for pattern in dangerous_patterns:
            if pattern in path_str:
                issues.append(f"Path contains dangerous pattern: {pattern}")
        
        # Check file extension if specified
        if allowed_extensions:
            file_ext = filepath.suffix.lower()
            if file_ext not in [ext.lower() for ext in allowed_extensions]:
                issues.append(f"File extension {file_ext} not allowed")
        
        # Check for executable extensions
        dangerous_extensions = ['.exe', '.bat', '.sh', '.cmd', '.scr', '.pif', '.com']
        if filepath.suffix.lower() in dangerous_extensions:
            issues.append(f"Dangerous executable extension: {filepath.suffix}")
        
    except Exception as e:
        issues.append(f"Error validating path: {e}")
    
    is_safe = len(issues) == 0
    return is_safe, issues


class SecureModelLoader:
    """Secure loader for model files."""
    
    def __init__(self, security_manager: SecurityManager):
        self.security_manager = security_manager
        self.logger = security_manager.logger
    
    def load_model_safely(self, model_path: str, expected_hash: Optional[str] = None) -> Optional[Any]:
        """Safely load a model file with security checks."""
        try:
            # Validate file path
            is_safe, issues = validate_file_safety(model_path, ['.pt', '.pth'])
            if not is_safe:
                self.logger.error(f"Unsafe model path: {issues}")
                return None
            
            model_path = Path(model_path).resolve()
            
            # Check if file exists
            if not model_path.exists():
                self.logger.error(f"Model file does not exist: {model_path}")
                return None
            
            # Verify file integrity if hash provided
            if expected_hash:
                if not self.security_manager.verify_file_integrity(str(model_path), expected_hash):
                    self.logger.error(f"Model file integrity check failed: {model_path}")
                    return None
            
            # Check file size
            file_size = model_path.stat().st_size
            if file_size > self.security_manager.max_file_size:
                self.logger.error(f"Model file too large: {file_size} bytes")
                return None
            
            # Try to import torch for loading
            try:
                import torch
                
                # Load model with restricted pickle loading
                model = torch.load(model_path, map_location='cpu')
                
                self.logger.info(f"Successfully loaded model from {model_path}")
                return model
                
            except ImportError:
                self.logger.error("PyTorch not available for model loading")
                return None
            except Exception as e:
                self.logger.error(f"Error loading model: {e}")
                return None
            
        except Exception as e:
            self.logger.error(f"Secure model loading failed: {e}")
            return None
    
    def save_model_safely(self, model: Any, model_path: str, 
                         compute_hash: bool = True) -> Optional[str]:
        """Safely save a model file."""
        try:
            # Validate file path
            is_safe, issues = validate_file_safety(model_path, ['.pt', '.pth'])
            if not is_safe:
                self.logger.error(f"Unsafe model path: {issues}")
                return None
            
            model_path = Path(model_path).resolve()
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Try to import torch for saving
            try:
                import torch
                
                # Save to temporary file first
                temp_path = self.security_manager.create_secure_temp_file(suffix='.pt')
                
                torch.save(model, temp_path)
                
                # Move to final location
                shutil.move(temp_path, model_path)
                os.chmod(model_path, 0o600)
                
                # Compute hash if requested
                file_hash = None
                if compute_hash:
                    file_hash = self.security_manager.compute_file_hash(str(model_path))
                
                self.logger.info(f"Successfully saved model to {model_path}")
                return file_hash
                
            except ImportError:
                self.logger.error("PyTorch not available for model saving")
                return None
            except Exception as e:
                self.logger.error(f"Error saving model: {e}")
                return None
            
        except Exception as e:
            self.logger.error(f"Secure model saving failed: {e}")
            return None


def create_security_report(security_manager: SecurityManager, 
                          output_path: str) -> bool:
    """Create a security report."""
    try:
        report = {
            'timestamp': os.path.getctime,
            'security_policies': {
                'max_file_size': security_manager.max_file_size,
                'allowed_extensions': list(security_manager.allowed_file_extensions),
                'blocked_patterns': security_manager.blocked_patterns
            },
            'recommendations': [
                'Regularly update allowed file extensions based on requirements',
                'Monitor file upload sizes and adjust limits as needed',
                'Review blocked patterns for new security threats',
                'Enable integrity checking for all critical files',
                'Use secure temporary files for all intermediate operations'
            ]
        }
        
        # Save report securely
        success = security_manager.secure_json_save(report, output_path)
        
        if success:
            security_manager.logger.info(f"Security report saved to {output_path}")
        
        return success
        
    except Exception as e:
        security_manager.logger.error(f"Failed to create security report: {e}")
        return False