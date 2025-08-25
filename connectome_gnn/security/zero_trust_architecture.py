"""Zero Trust Security Architecture with AI-powered threat detection."""

import hashlib
import secrets
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pickle
from collections import defaultdict, deque

from .advanced_security import SecurityMonitor, ThreatType, SecurityLevel
from ..robust.logging_config import get_logger


class TrustLevel(Enum):
    """Trust level enumeration."""
    ZERO = 0  # Never trust
    MINIMAL = 1  # Verify everything
    LOW = 2  # Basic verification
    MEDIUM = 3  # Standard verification
    HIGH = 4  # Trusted but verify
    IMPLICIT = 5  # Highly trusted (dangerous!)


class BehaviorPattern(Enum):
    """User behavior pattern types."""
    NORMAL = "normal"
    SUSPICIOUS = "suspicious"
    ANOMALOUS = "anomalous"
    MALICIOUS = "malicious"
    UNKNOWN = "unknown"


@dataclass
class TrustContext:
    """Trust evaluation context."""
    user_id: str
    device_fingerprint: str
    location_hash: str
    time_of_day: int  # Hour 0-23
    day_of_week: int  # 0-6
    access_patterns: List[str]
    resource_requested: str
    historical_behavior: List[float] = field(default_factory=list)
    risk_score: float = 0.0
    

@dataclass
class BehaviorProfile:
    """User behavior profile for ML-based trust scoring."""
    user_id: str
    feature_vector: np.ndarray
    access_times: deque = field(default_factory=lambda: deque(maxlen=100))
    resource_frequencies: Dict[str, int] = field(default_factory=dict)
    location_patterns: Set[str] = field(default_factory=set)
    device_patterns: Set[str] = field(default_factory=set)
    anomaly_scores: deque = field(default_factory=lambda: deque(maxlen=50))
    last_updated: float = field(default_factory=time.time)
    trust_decay_rate: float = 0.01
    
    
class AIThreatDetector:
    """AI-powered threat detection using behavior analysis."""
    
    def __init__(self):
        self.behavior_profiles: Dict[str, BehaviorProfile] = {}
        self.global_patterns: Dict[str, np.ndarray] = {}
        self.anomaly_threshold = 2.5  # Standard deviations
        self.learning_rate = 0.01
        
        # Simple neural network weights for threat detection
        self.threat_weights = np.random.normal(0, 0.1, (20, 1))
        self.threat_bias = 0.0
        
        self.logger = get_logger(__name__)
        
    def extract_features(self, context: TrustContext) -> np.ndarray:
        """Extract behavioral features from trust context."""
        features = np.zeros(20)
        
        # Temporal features
        features[0] = context.time_of_day / 24.0
        features[1] = context.day_of_week / 7.0
        features[2] = np.sin(2 * np.pi * context.time_of_day / 24.0)  # Circular encoding
        features[3] = np.cos(2 * np.pi * context.time_of_day / 24.0)
        
        # Access pattern features
        features[4] = len(context.access_patterns)
        features[5] = len(set(context.access_patterns))  # Unique patterns
        
        # Historical behavior statistics
        if context.historical_behavior:
            features[6] = np.mean(context.historical_behavior)
            features[7] = np.std(context.historical_behavior)
            features[8] = np.median(context.historical_behavior)
            
        # Hash-based features (simplified)
        device_hash = int(hashlib.md5(context.device_fingerprint.encode()).hexdigest()[:8], 16)
        location_hash = int(hashlib.md5(context.location_hash.encode()).hexdigest()[:8], 16)
        resource_hash = int(hashlib.md5(context.resource_requested.encode()).hexdigest()[:8], 16)
        
        features[9] = (device_hash % 10000) / 10000.0
        features[10] = (location_hash % 10000) / 10000.0
        features[11] = (resource_hash % 10000) / 10000.0
        
        # Risk-based features
        features[12] = context.risk_score
        
        # Frequency features
        current_hour_freq = sum(1 for t in context.historical_behavior 
                              if abs(t - (context.time_of_day * 3600)) < 3600)
        features[13] = current_hour_freq / max(len(context.historical_behavior), 1)
        
        # Novelty features
        features[14] = 1.0 if context.device_fingerprint not in self.global_patterns else 0.0
        features[15] = 1.0 if context.location_hash not in self.global_patterns else 0.0
        
        # Additional statistical features
        features[16] = len(context.access_patterns) / max(context.time_of_day, 1)  # Access density
        features[17] = context.risk_score ** 2  # Squared risk for non-linear relationship
        features[18] = 1.0 / (1 + len(context.historical_behavior))  # Novelty of user
        features[19] = np.random.normal(0, 0.01)  # Small random component for regularization
        
        return features
        
    def update_behavior_profile(self, context: TrustContext, threat_detected: bool):
        """Update user behavior profile with new data."""
        user_id = context.user_id
        
        if user_id not in self.behavior_profiles:
            self.behavior_profiles[user_id] = BehaviorProfile(
                user_id=user_id,
                feature_vector=np.zeros(20)
            )
            
        profile = self.behavior_profiles[user_id]
        
        # Update temporal patterns
        profile.access_times.append(time.time())
        
        # Update resource access frequencies
        resource = context.resource_requested
        profile.resource_frequencies[resource] = profile.resource_frequencies.get(resource, 0) + 1
        
        # Update location and device patterns
        profile.location_patterns.add(context.location_hash)
        profile.device_patterns.add(context.device_fingerprint)
        
        # Update feature vector with exponential moving average
        new_features = self.extract_features(context)
        profile.feature_vector = (
            (1 - self.learning_rate) * profile.feature_vector + 
            self.learning_rate * new_features
        )
        
        # Calculate and store anomaly score
        anomaly_score = self.calculate_anomaly_score(new_features, profile.feature_vector)
        profile.anomaly_scores.append(anomaly_score)
        
        profile.last_updated = time.time()
        
        # Update global patterns
        self.global_patterns[context.device_fingerprint] = new_features
        self.global_patterns[context.location_hash] = new_features
        
        # Train threat detection model
        self.train_threat_model(new_features, threat_detected)
        
    def calculate_anomaly_score(self, new_features: np.ndarray, baseline_features: np.ndarray) -> float:
        """Calculate anomaly score using statistical methods."""
        if np.allclose(baseline_features, 0):
            return 0.0
            
        # Euclidean distance normalized by feature magnitudes
        distance = np.linalg.norm(new_features - baseline_features)
        baseline_magnitude = np.linalg.norm(baseline_features)
        
        if baseline_magnitude > 0:
            normalized_distance = distance / baseline_magnitude
        else:
            normalized_distance = distance
            
        return normalized_distance
        
    def train_threat_model(self, features: np.ndarray, is_threat: bool):
        """Online training of threat detection model."""
        # Simple gradient descent update
        prediction = self.predict_threat_probability(features)
        target = 1.0 if is_threat else 0.0
        
        # Binary cross-entropy gradient
        error = prediction - target
        
        # Update weights
        self.threat_weights -= self.learning_rate * error * features.reshape(-1, 1)
        self.threat_bias -= self.learning_rate * error
        
    def predict_threat_probability(self, features: np.ndarray) -> float:
        """Predict threat probability using trained model."""
        logit = np.dot(features, self.threat_weights.flatten()) + self.threat_bias
        probability = 1.0 / (1.0 + np.exp(-logit))  # Sigmoid
        return probability
        
    def classify_behavior(self, context: TrustContext) -> BehaviorPattern:
        """Classify user behavior pattern."""
        features = self.extract_features(context)
        threat_prob = self.predict_threat_probability(features)
        
        user_id = context.user_id
        if user_id in self.behavior_profiles:
            profile = self.behavior_profiles[user_id]
            if profile.anomaly_scores:
                avg_anomaly = np.mean(list(profile.anomaly_scores))
            else:
                avg_anomaly = 0.0
        else:
            avg_anomaly = 0.0
            
        # Classification thresholds
        if threat_prob > 0.8 or avg_anomaly > 3.0:
            return BehaviorPattern.MALICIOUS
        elif threat_prob > 0.6 or avg_anomaly > 2.0:
            return BehaviorPattern.ANOMALOUS
        elif threat_prob > 0.4 or avg_anomaly > 1.0:
            return BehaviorPattern.SUSPICIOUS
        elif user_id not in self.behavior_profiles:
            return BehaviorPattern.UNKNOWN
        else:
            return BehaviorPattern.NORMAL


class ZeroTrustEngine:
    """Zero Trust Architecture enforcement engine."""
    
    def __init__(self, security_monitor: Optional[SecurityMonitor] = None):
        self.security_monitor = security_monitor
        self.ai_detector = AIThreatDetector()
        
        self.trust_policies: Dict[str, Dict[str, Any]] = {}
        self.resource_policies: Dict[str, Dict[str, Any]] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Adaptive trust thresholds
        self.trust_thresholds = {
            TrustLevel.ZERO: 0.0,
            TrustLevel.MINIMAL: 0.2,
            TrustLevel.LOW: 0.4,
            TrustLevel.MEDIUM: 0.6,
            TrustLevel.HIGH: 0.8,
            TrustLevel.IMPLICIT: 0.95
        }
        
        self.verification_methods = {
            'multi_factor': self._verify_mfa,
            'biometric': self._verify_biometric,
            'device_cert': self._verify_device_certificate,
            'behavior_auth': self._verify_behavior,
            'location_geo': self._verify_geolocation,
            'time_based': self._verify_time_constraints
        }
        
        self.logger = get_logger(__name__)
        self.lock = threading.Lock()
        
        self._setup_default_policies()
        
    def _setup_default_policies(self):
        """Setup default zero trust policies."""
        # Default resource policies
        self.resource_policies = {
            'high_security': {
                'required_trust_level': TrustLevel.HIGH,
                'verification_methods': ['multi_factor', 'biometric', 'device_cert'],
                'max_session_duration': 1800,  # 30 minutes
                'continuous_verification': True,
                'allowed_locations': 'restricted',
                'risk_tolerance': 0.1
            },
            'medium_security': {
                'required_trust_level': TrustLevel.MEDIUM,
                'verification_methods': ['multi_factor', 'behavior_auth'],
                'max_session_duration': 3600,  # 1 hour
                'continuous_verification': False,
                'allowed_locations': 'known',
                'risk_tolerance': 0.3
            },
            'low_security': {
                'required_trust_level': TrustLevel.LOW,
                'verification_methods': ['behavior_auth'],
                'max_session_duration': 7200,  # 2 hours
                'continuous_verification': False,
                'allowed_locations': 'any',
                'risk_tolerance': 0.5
            }
        }
        
        # Default trust policies
        self.trust_policies = {
            'default': {
                'initial_trust': TrustLevel.ZERO,
                'trust_decay_rate': 0.1,  # Per hour
                'anomaly_penalty': 0.5,
                'success_boost': 0.1,
                'max_trust_level': TrustLevel.HIGH
            }
        }
        
    def evaluate_trust(self, context: TrustContext) -> Tuple[float, TrustLevel, Dict[str, Any]]:
        """Evaluate trust score for given context."""
        # AI-based behavioral analysis
        behavior_pattern = self.ai_detector.classify_behavior(context)
        threat_probability = self.ai_detector.predict_threat_probability(
            self.ai_detector.extract_features(context)
        )
        
        # Base trust calculation
        base_trust = self._calculate_base_trust(context, behavior_pattern)
        
        # Risk adjustments
        risk_adjusted_trust = self._apply_risk_adjustments(base_trust, context, threat_probability)
        
        # Determine trust level
        trust_level = self._determine_trust_level(risk_adjusted_trust)
        
        # Generate trust report
        trust_report = {
            'base_trust': base_trust,
            'risk_adjusted_trust': risk_adjusted_trust,
            'behavior_pattern': behavior_pattern.value,
            'threat_probability': threat_probability,
            'factors': self._get_trust_factors(context),
            'timestamp': time.time()
        }
        
        self.logger.info(
            f"Trust evaluation for {context.user_id}: {risk_adjusted_trust:.3f} "
            f"({trust_level.name}) - {behavior_pattern.value}"
        )
        
        return risk_adjusted_trust, trust_level, trust_report
        
    def _calculate_base_trust(self, context: TrustContext, behavior_pattern: BehaviorPattern) -> float:
        """Calculate base trust score."""
        base_trust = 0.5  # Neutral starting point
        
        # Behavior pattern adjustment
        behavior_adjustments = {
            BehaviorPattern.NORMAL: 0.3,
            BehaviorPattern.SUSPICIOUS: -0.2,
            BehaviorPattern.ANOMALOUS: -0.4,
            BehaviorPattern.MALICIOUS: -0.8,
            BehaviorPattern.UNKNOWN: -0.1
        }
        
        base_trust += behavior_adjustments.get(behavior_pattern, 0.0)
        
        # Historical behavior boost
        if context.historical_behavior:
            avg_historical = np.mean(context.historical_behavior)
            if avg_historical > 0.7:
                base_trust += 0.2
            elif avg_historical < 0.3:
                base_trust -= 0.2
                
        # Consistency bonus
        user_id = context.user_id
        if user_id in self.ai_detector.behavior_profiles:
            profile = self.ai_detector.behavior_profiles[user_id]
            
            # Device consistency
            if context.device_fingerprint in profile.device_patterns:
                base_trust += 0.1
                
            # Location consistency
            if context.location_hash in profile.location_patterns:
                base_trust += 0.1
                
            # Time pattern consistency
            current_time = context.time_of_day
            if profile.access_times:
                typical_times = [t % 86400 // 3600 for t in profile.access_times]
                if current_time in set(typical_times):
                    base_trust += 0.1
                    
        return max(0.0, min(1.0, base_trust))
        
    def _apply_risk_adjustments(self, base_trust: float, context: TrustContext, threat_prob: float) -> float:
        """Apply risk-based adjustments to trust score."""
        adjusted_trust = base_trust
        
        # Threat probability penalty
        adjusted_trust -= threat_prob * 0.5
        
        # Context risk penalty
        adjusted_trust -= context.risk_score * 0.3
        
        # Time-based adjustments
        if context.time_of_day < 6 or context.time_of_day > 22:  # Off hours
            adjusted_trust -= 0.1
            
        # Weekend adjustment
        if context.day_of_week in [5, 6]:  # Saturday, Sunday
            adjusted_trust -= 0.05
            
        return max(0.0, min(1.0, adjusted_trust))
        
    def _determine_trust_level(self, trust_score: float) -> TrustLevel:
        """Determine trust level from numerical score."""
        for level in reversed(list(TrustLevel)):
            if trust_score >= self.trust_thresholds[level]:
                return level
        return TrustLevel.ZERO
        
    def _get_trust_factors(self, context: TrustContext) -> Dict[str, Any]:
        """Get detailed trust factors for transparency."""
        return {
            'user_id': context.user_id,
            'device_known': context.device_fingerprint in self.ai_detector.global_patterns,
            'location_known': context.location_hash in self.ai_detector.global_patterns,
            'time_typical': 6 <= context.time_of_day <= 22,
            'weekend_access': context.day_of_week in [5, 6],
            'resource_frequency': len(context.access_patterns),
            'historical_success': np.mean(context.historical_behavior) if context.historical_behavior else 0.5
        }
        
    def authorize_access(self, context: TrustContext, resource: str) -> Tuple[bool, Dict[str, Any]]:
        """Authorize access request using zero trust principles."""
        # Evaluate trust
        trust_score, trust_level, trust_report = self.evaluate_trust(context)
        
        # Get resource policy
        resource_policy = self._get_resource_policy(resource)
        required_trust_level = resource_policy['required_trust_level']
        
        # Check if trust level meets requirements
        access_granted = trust_level.value >= required_trust_level.value
        
        # Additional risk checks
        if access_granted and trust_score < resource_policy.get('risk_tolerance', 0.5):
            access_granted = False
            
        # Perform additional verification if needed
        verification_results = {}
        if access_granted and resource_policy.get('verification_methods'):
            verification_results = self._perform_verification(
                context, resource_policy['verification_methods']
            )
            access_granted = all(verification_results.values())
            
        # Create session if access granted
        session_id = None
        if access_granted:
            session_id = self._create_session(context, resource, resource_policy)
            
        # Update behavior profile
        self.ai_detector.update_behavior_profile(context, not access_granted)
        
        # Security monitoring
        if self.security_monitor and not access_granted:
            threat_data = {
                'user_id': context.user_id,
                'resource': resource,
                'trust_score': trust_score,
                'trust_level': trust_level.name,
                'source_ip': getattr(context, 'source_ip', None)
            }
            
            threat_event = self.security_monitor.detect_threat(
                threat_data, ThreatType.UNAUTHORIZED_ACCESS
            )
            
            if threat_event:
                self.security_monitor.log_event(threat_event)
                
        authorization_result = {
            'access_granted': access_granted,
            'session_id': session_id,
            'trust_score': trust_score,
            'trust_level': trust_level.name,
            'verification_results': verification_results,
            'trust_report': trust_report,
            'required_trust_level': required_trust_level.name,
            'timestamp': time.time()
        }
        
        self.logger.info(
            f"Access {'GRANTED' if access_granted else 'DENIED'} for {context.user_id} "
            f"to {resource} (trust: {trust_score:.3f})"
        )
        
        return access_granted, authorization_result
        
    def _get_resource_policy(self, resource: str) -> Dict[str, Any]:
        """Get security policy for resource."""
        # Simple resource classification
        if 'admin' in resource.lower() or 'config' in resource.lower():
            return self.resource_policies['high_security']
        elif 'data' in resource.lower() or 'model' in resource.lower():
            return self.resource_policies['medium_security']
        else:
            return self.resource_policies['low_security']
            
    def _perform_verification(self, context: TrustContext, methods: List[str]) -> Dict[str, bool]:
        """Perform additional verification methods."""
        results = {}
        
        for method in methods:
            if method in self.verification_methods:
                try:
                    results[method] = self.verification_methods[method](context)
                except Exception as e:
                    self.logger.error(f"Verification method {method} failed: {e}")
                    results[method] = False
            else:
                results[method] = False
                
        return results
        
    def _create_session(self, context: TrustContext, resource: str, policy: Dict[str, Any]) -> str:
        """Create authenticated session."""
        session_id = secrets.token_hex(32)
        
        with self.lock:
            self.active_sessions[session_id] = {
                'user_id': context.user_id,
                'resource': resource,
                'created_at': time.time(),
                'expires_at': time.time() + policy.get('max_session_duration', 3600),
                'device_fingerprint': context.device_fingerprint,
                'location_hash': context.location_hash,
                'continuous_verification': policy.get('continuous_verification', False)
            }
            
        return session_id
        
    # Simplified verification method implementations
    def _verify_mfa(self, context: TrustContext) -> bool:
        """Multi-factor authentication verification."""
        # Simplified - would integrate with actual MFA system
        return np.random.random() > 0.1  # 90% success rate
        
    def _verify_biometric(self, context: TrustContext) -> bool:
        """Biometric verification."""
        # Simplified - would integrate with biometric system
        return np.random.random() > 0.05  # 95% success rate
        
    def _verify_device_certificate(self, context: TrustContext) -> bool:
        """Device certificate verification."""
        # Simplified - would check actual device certificates
        return context.device_fingerprint in self.ai_detector.global_patterns
        
    def _verify_behavior(self, context: TrustContext) -> bool:
        """Behavioral authentication."""
        pattern = self.ai_detector.classify_behavior(context)
        return pattern in [BehaviorPattern.NORMAL, BehaviorPattern.SUSPICIOUS]
        
    def _verify_geolocation(self, context: TrustContext) -> bool:
        """Geolocation verification."""
        # Simplified - would use actual geolocation services
        return context.location_hash in self.ai_detector.global_patterns
        
    def _verify_time_constraints(self, context: TrustContext) -> bool:
        """Time-based access constraints."""
        # Allow access during business hours
        return 6 <= context.time_of_day <= 22
        
    def continuous_verification_loop(self):
        """Continuous verification of active sessions."""
        current_time = time.time()
        
        with self.lock:
            expired_sessions = []
            
            for session_id, session_data in self.active_sessions.items():
                # Check expiration
                if current_time > session_data['expires_at']:
                    expired_sessions.append(session_id)
                    continue
                    
                # Continuous verification if enabled
                if session_data.get('continuous_verification', False):
                    # Create context for verification
                    verify_context = TrustContext(
                        user_id=session_data['user_id'],
                        device_fingerprint=session_data['device_fingerprint'],
                        location_hash=session_data['location_hash'],
                        time_of_day=int(current_time % 86400 // 3600),
                        day_of_week=int(current_time // 86400) % 7,
                        access_patterns=[session_data['resource']],
                        resource_requested=session_data['resource']
                    )
                    
                    # Re-evaluate trust
                    trust_score, trust_level, _ = self.evaluate_trust(verify_context)
                    
                    # Revoke session if trust drops too low
                    if trust_score < 0.3:
                        expired_sessions.append(session_id)
                        self.logger.warning(
                            f"Session {session_id} revoked due to low trust: {trust_score:.3f}"
                        )
                        
            # Remove expired sessions
            for session_id in expired_sessions:
                del self.active_sessions[session_id]
                
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get zero trust security metrics."""
        with self.lock:
            active_session_count = len(self.active_sessions)
            
        behavior_profiles_count = len(self.ai_detector.behavior_profiles)
        
        # Calculate average trust scores
        trust_scores = []
        for profile in self.ai_detector.behavior_profiles.values():
            if profile.anomaly_scores:
                avg_anomaly = np.mean(list(profile.anomaly_scores))
                trust_score = max(0.0, 1.0 - avg_anomaly / 3.0)  # Normalize
                trust_scores.append(trust_score)
                
        return {
            'active_sessions': active_session_count,
            'behavior_profiles': behavior_profiles_count,
            'average_trust_score': np.mean(trust_scores) if trust_scores else 0.0,
            'threat_model_weights_norm': np.linalg.norm(self.ai_detector.threat_weights),
            'global_patterns_count': len(self.ai_detector.global_patterns),
            'timestamp': time.time()
        }


# Global zero trust engine instance
_global_zero_trust = None

def get_zero_trust_engine() -> ZeroTrustEngine:
    """Get global zero trust engine instance."""
    global _global_zero_trust
    if _global_zero_trust is None:
        _global_zero_trust = ZeroTrustEngine()
    return _global_zero_trust
