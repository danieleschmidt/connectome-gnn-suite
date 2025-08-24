"""Advanced security framework for Connectome-GNN-Suite."""

import hashlib
import hmac
import secrets
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import json
import os
from pathlib import Path

from ..robust.logging_config import get_logger


class SecurityLevel(Enum):
    """Security level enumeration."""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(Enum):
    """Threat type enumeration."""
    DATA_POISONING = "data_poisoning"
    MODEL_INVERSION = "model_inversion"
    ADVERSARIAL_ATTACK = "adversarial_attack"
    PRIVACY_BREACH = "privacy_breach"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


@dataclass
class SecurityEvent:
    """Security event data structure."""
    event_id: str
    threat_type: ThreatType
    severity: SecurityLevel
    timestamp: float
    source_ip: Optional[str]
    user_id: Optional[str]
    details: Dict[str, Any]
    resolved: bool = False


class SecurityMonitor:
    """Real-time security monitoring system."""
    
    def __init__(self):
        self.events: List[SecurityEvent] = []
        self.threat_patterns: Dict[ThreatType, Dict[str, Any]] = {}
        self.security_policies: Dict[str, Dict[str, Any]] = {}
        
        self.logger = get_logger(__name__)
        self.lock = threading.Lock()
        
        self._setup_default_patterns()
        self._setup_default_policies()
        
    def _setup_default_patterns(self):
        """Setup default threat detection patterns."""
        self.threat_patterns = {
            ThreatType.DATA_POISONING: {
                'anomaly_threshold': 3.0,
                'feature_drift_threshold': 0.5,
                'label_consistency_threshold': 0.1
            },
            ThreatType.MODEL_INVERSION: {
                'gradient_similarity_threshold': 0.9,
                'reconstruction_accuracy_threshold': 0.8
            },
            ThreatType.ADVERSARIAL_ATTACK: {
                'perturbation_magnitude_threshold': 0.1,
                'prediction_confidence_drop': 0.5
            },
            ThreatType.PRIVACY_BREACH: {
                'membership_inference_threshold': 0.7,
                'attribute_inference_threshold': 0.6
            },
            ThreatType.UNAUTHORIZED_ACCESS: {
                'failed_login_threshold': 5,
                'suspicious_access_patterns': True
            },
            ThreatType.RESOURCE_EXHAUSTION: {
                'memory_threshold': 0.9,
                'cpu_threshold': 0.95,
                'request_rate_threshold': 1000
            }
        }
        
    def _setup_default_policies(self):
        """Setup default security policies."""
        self.security_policies = {
            'data_validation': {
                'required': True,
                'strict_mode': True,
                'quarantine_suspicious': True
            },
            'model_protection': {
                'differential_privacy': True,
                'noise_multiplier': 1.0,
                'max_grad_norm': 1.0
            },
            'access_control': {
                'require_authentication': True,
                'session_timeout': 3600,  # 1 hour
                'rate_limiting': True
            },
            'audit_logging': {
                'enabled': True,
                'include_data_access': True,
                'retention_days': 90
            }
        }
        
    def detect_threat(
        self,
        data: Dict[str, Any],
        threat_type: ThreatType
    ) -> Optional[SecurityEvent]:
        """Detect potential security threats."""
        patterns = self.threat_patterns.get(threat_type)
        if not patterns:
            return None
            
        # Threat-specific detection logic
        if threat_type == ThreatType.DATA_POISONING:
            return self._detect_data_poisoning(data, patterns)
        elif threat_type == ThreatType.ADVERSARIAL_ATTACK:
            return self._detect_adversarial_attack(data, patterns)
        elif threat_type == ThreatType.PRIVACY_BREACH:
            return self._detect_privacy_breach(data, patterns)
        elif threat_type == ThreatType.UNAUTHORIZED_ACCESS:
            return self._detect_unauthorized_access(data, patterns)
        elif threat_type == ThreatType.RESOURCE_EXHAUSTION:
            return self._detect_resource_exhaustion(data, patterns)
            
        return None
        
    def _detect_data_poisoning(
        self,
        data: Dict[str, Any],
        patterns: Dict[str, Any]
    ) -> Optional[SecurityEvent]:
        """Detect data poisoning attacks."""
        anomaly_score = data.get('anomaly_score', 0.0)
        feature_drift = data.get('feature_drift', 0.0)
        label_consistency = data.get('label_consistency', 1.0)
        
        if (
            anomaly_score > patterns['anomaly_threshold'] or
            feature_drift > patterns['feature_drift_threshold'] or
            label_consistency < patterns['label_consistency_threshold']
        ):
            return SecurityEvent(
                event_id=self._generate_event_id(),
                threat_type=ThreatType.DATA_POISONING,
                severity=SecurityLevel.HIGH,
                timestamp=time.time(),
                source_ip=data.get('source_ip'),
                user_id=data.get('user_id'),
                details={
                    'anomaly_score': anomaly_score,
                    'feature_drift': feature_drift,
                    'label_consistency': label_consistency
                }
            )
        return None
        
    def _detect_adversarial_attack(
        self,
        data: Dict[str, Any], 
        patterns: Dict[str, Any]
    ) -> Optional[SecurityEvent]:
        """Detect adversarial attacks."""
        perturbation_mag = data.get('perturbation_magnitude', 0.0)
        confidence_drop = data.get('confidence_drop', 0.0)
        
        if (
            perturbation_mag > patterns['perturbation_magnitude_threshold'] or
            confidence_drop > patterns['prediction_confidence_drop']
        ):
            return SecurityEvent(
                event_id=self._generate_event_id(),
                threat_type=ThreatType.ADVERSARIAL_ATTACK,
                severity=SecurityLevel.HIGH,
                timestamp=time.time(),
                source_ip=data.get('source_ip'),
                user_id=data.get('user_id'),
                details={
                    'perturbation_magnitude': perturbation_mag,
                    'confidence_drop': confidence_drop
                }
            )
        return None
        
    def _detect_privacy_breach(
        self,
        data: Dict[str, Any],
        patterns: Dict[str, Any]
    ) -> Optional[SecurityEvent]:
        """Detect privacy breach attempts."""
        membership_score = data.get('membership_inference_score', 0.0)
        attribute_score = data.get('attribute_inference_score', 0.0)
        
        if (
            membership_score > patterns['membership_inference_threshold'] or
            attribute_score > patterns['attribute_inference_threshold']
        ):
            return SecurityEvent(
                event_id=self._generate_event_id(),
                threat_type=ThreatType.PRIVACY_BREACH,
                severity=SecurityLevel.CRITICAL,
                timestamp=time.time(),
                source_ip=data.get('source_ip'),
                user_id=data.get('user_id'),
                details={
                    'membership_inference_score': membership_score,
                    'attribute_inference_score': attribute_score
                }
            )
        return None
        
    def _detect_unauthorized_access(
        self,
        data: Dict[str, Any],
        patterns: Dict[str, Any]
    ) -> Optional[SecurityEvent]:
        """Detect unauthorized access attempts."""
        failed_logins = data.get('failed_login_count', 0)
        suspicious_pattern = data.get('suspicious_access', False)
        
        if (
            failed_logins > patterns['failed_login_threshold'] or
            suspicious_pattern
        ):
            return SecurityEvent(
                event_id=self._generate_event_id(),
                threat_type=ThreatType.UNAUTHORIZED_ACCESS,
                severity=SecurityLevel.MEDIUM,
                timestamp=time.time(),
                source_ip=data.get('source_ip'),
                user_id=data.get('user_id'),
                details={
                    'failed_login_count': failed_logins,
                    'suspicious_access': suspicious_pattern
                }
            )
        return None
        
    def _detect_resource_exhaustion(
        self,
        data: Dict[str, Any],
        patterns: Dict[str, Any]
    ) -> Optional[SecurityEvent]:
        """Detect resource exhaustion attacks."""
        memory_usage = data.get('memory_usage', 0.0)
        cpu_usage = data.get('cpu_usage', 0.0)
        request_rate = data.get('request_rate', 0)
        
        if (
            memory_usage > patterns['memory_threshold'] or
            cpu_usage > patterns['cpu_threshold'] or
            request_rate > patterns['request_rate_threshold']
        ):
            return SecurityEvent(
                event_id=self._generate_event_id(),
                threat_type=ThreatType.RESOURCE_EXHAUSTION,
                severity=SecurityLevel.HIGH,
                timestamp=time.time(),
                source_ip=data.get('source_ip'),
                user_id=data.get('user_id'),
                details={
                    'memory_usage': memory_usage,
                    'cpu_usage': cpu_usage,
                    'request_rate': request_rate
                }
            )
        return None
        
    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        return f"SEC_{int(time.time())}_{secrets.token_hex(4)}"
        
    def log_event(self, event: SecurityEvent):
        """Log security event."""
        with self.lock:
            self.events.append(event)
            
        self.logger.warning(
            f"Security Event: {event.threat_type.value} "
            f"[{event.severity.value}] - {event.event_id}"
        )
        
    def get_recent_events(
        self,
        hours: int = 24,
        threat_type: Optional[ThreatType] = None
    ) -> List[SecurityEvent]:
        """Get recent security events."""
        cutoff_time = time.time() - (hours * 3600)
        
        with self.lock:
            events = [
                event for event in self.events
                if event.timestamp > cutoff_time
            ]
            
            if threat_type:
                events = [
                    event for event in events
                    if event.threat_type == threat_type
                ]
                
            return sorted(events, key=lambda x: x.timestamp, reverse=True)


class SecureDataHandler:
    """Secure data handling with encryption and validation."""
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        self.encryption_key = encryption_key or self._generate_key()
        self.logger = get_logger(__name__)
        
    def _generate_key(self) -> bytes:
        """Generate encryption key."""
        return secrets.token_bytes(32)  # 256-bit key
        
    def encrypt_data(self, data: bytes) -> Dict[str, Any]:
        """Encrypt sensitive data."""
        from cryptography.fernet import Fernet
        
        key = Fernet.generate_key()
        cipher = Fernet(key)
        
        encrypted_data = cipher.encrypt(data)
        
        return {
            'encrypted_data': encrypted_data,
            'key': key,
            'timestamp': time.time()
        }
        
    def decrypt_data(self, encrypted_package: Dict[str, Any]) -> bytes:
        """Decrypt sensitive data."""
        from cryptography.fernet import Fernet
        
        cipher = Fernet(encrypted_package['key'])
        return cipher.decrypt(encrypted_package['encrypted_data'])
        
    def validate_data_integrity(self, data: bytes, signature: str) -> bool:
        """Validate data integrity using HMAC."""
        expected_signature = hmac.new(
            self.encryption_key,
            data,
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(expected_signature, signature)
        
    def sign_data(self, data: bytes) -> str:
        """Create HMAC signature for data."""
        return hmac.new(
            self.encryption_key,
            data,
            hashlib.sha256
        ).hexdigest()


class DifferentialPrivacyManager:
    """Differential privacy implementation for model training."""
    
    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        noise_multiplier: float = 1.0
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.noise_multiplier = noise_multiplier
        
        self.privacy_spent = 0.0
        self.logger = get_logger(__name__)
        
    def add_noise_to_gradients(self, gradients: Any) -> Any:
        """Add calibrated noise to gradients for differential privacy."""
        import torch
        
        if isinstance(gradients, torch.Tensor):
            noise = torch.normal(
                0.0,
                self.noise_multiplier,
                size=gradients.shape,
                device=gradients.device
            )
            return gradients + noise
        
        return gradients
        
    def clip_gradients(self, gradients: Any, max_norm: float = 1.0) -> Any:
        """Clip gradients to bounded norm."""
        import torch
        
        if isinstance(gradients, torch.Tensor):
            return torch.clamp(gradients, -max_norm, max_norm)
            
        return gradients
        
    def compute_privacy_spent(self, steps: int) -> float:
        """Compute total privacy budget spent."""
        # Simplified RDP accounting
        self.privacy_spent += steps * (self.epsilon / steps)
        return self.privacy_spent
        
    def check_privacy_budget(self) -> bool:
        """Check if privacy budget is exhausted."""
        return self.privacy_spent < self.epsilon


class AccessControlManager:
    """Role-based access control system."""
    
    def __init__(self):
        self.users: Dict[str, Dict[str, Any]] = {}
        self.roles: Dict[str, List[str]] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        
        self.logger = get_logger(__name__)
        self.lock = threading.Lock()
        
        self._setup_default_roles()
        
    def _setup_default_roles(self):
        """Setup default user roles."""
        self.roles = {
            'admin': [
                'read_data', 'write_data', 'train_model', 'deploy_model',
                'manage_users', 'view_logs', 'configure_system'
            ],
            'researcher': [
                'read_data', 'train_model', 'view_results', 'export_results'
            ],
            'analyst': [
                'read_data', 'view_results', 'export_results'
            ],
            'viewer': [
                'view_results'
            ]
        }
        
    def create_user(
        self,
        user_id: str,
        password: str,
        role: str = 'viewer',
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create new user account."""
        if role not in self.roles:
            return False
            
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        with self.lock:
            self.users[user_id] = {
                'password_hash': password_hash,
                'role': role,
                'created_at': time.time(),
                'last_login': None,
                'failed_attempts': 0,
                'locked': False,
                'metadata': metadata or {}
            }
            
        self.logger.info(f"Created user: {user_id} with role: {role}")
        return True
        
    def authenticate_user(self, user_id: str, password: str) -> Optional[str]:
        """Authenticate user and create session."""
        with self.lock:
            user = self.users.get(user_id)
            
            if not user or user['locked']:
                return None
                
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            
            if user['password_hash'] != password_hash:
                user['failed_attempts'] += 1
                if user['failed_attempts'] >= 5:
                    user['locked'] = True
                    self.logger.warning(f"User {user_id} locked due to failed attempts")
                return None
                
            # Reset failed attempts on successful login
            user['failed_attempts'] = 0
            user['last_login'] = time.time()
            
            # Create session
            session_id = secrets.token_hex(32)
            self.sessions[session_id] = {
                'user_id': user_id,
                'role': user['role'],
                'created_at': time.time(),
                'expires_at': time.time() + 3600,  # 1 hour
                'permissions': self.roles[user['role']]
            }
            
            self.logger.info(f"User {user_id} authenticated successfully")
            return session_id
            
    def check_permission(self, session_id: str, permission: str) -> bool:
        """Check if session has required permission."""
        with self.lock:
            session = self.sessions.get(session_id)
            
            if not session:
                return False
                
            # Check if session expired
            if time.time() > session['expires_at']:
                del self.sessions[session_id]
                return False
                
            return permission in session['permissions']
            
    def logout_user(self, session_id: str):
        """Logout user and invalidate session."""
        with self.lock:
            if session_id in self.sessions:
                user_id = self.sessions[session_id]['user_id']
                del self.sessions[session_id]
                self.logger.info(f"User {user_id} logged out")


# Global instances
_security_monitor = None
_access_control = None

def get_security_monitor() -> SecurityMonitor:
    """Get global security monitor instance."""
    global _security_monitor
    if _security_monitor is None:
        _security_monitor = SecurityMonitor()
    return _security_monitor


def get_access_control() -> AccessControlManager:
    """Get global access control manager instance."""
    global _access_control
    if _access_control is None:
        _access_control = AccessControlManager()
    return _access_control


def require_permission(permission: str):
    """Decorator to require specific permission for function access."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # In a real implementation, you'd extract session_id from request context
            session_id = kwargs.pop('session_id', None)
            
            if not session_id:
                raise PermissionError("Authentication required")
                
            access_control = get_access_control()
            if not access_control.check_permission(session_id, permission):
                raise PermissionError(f"Permission denied: {permission}")
                
            return func(*args, **kwargs)
        return wrapper
    return decorator