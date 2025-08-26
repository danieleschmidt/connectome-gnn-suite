"""Adaptive Validation Framework with Self-Correcting Mechanisms.

Implements intelligent validation that adapts to data patterns,
detects anomalies, and provides self-correcting capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
import warnings
from collections import deque, defaultdict
import statistics
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report
import json


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


class ValidationResult(Enum):
    """Validation result types."""
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    CORRECTED = "CORRECTED"


@dataclass
class ValidationEvent:
    """Container for validation event information."""
    timestamp: float = field(default_factory=time.time)
    validator_name: str = ""
    severity: ValidationSeverity = ValidationSeverity.INFO
    result: ValidationResult = ValidationResult.PASS
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    corrective_action: Optional[str] = None
    data_signature: Optional[str] = None


@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics."""
    total_validations: int = 0
    passed_validations: int = 0
    failed_validations: int = 0
    warnings: int = 0
    corrections_applied: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    average_validation_time: float = 0.0
    adaptation_rate: float = 0.0
    
    @property
    def pass_rate(self) -> float:
        return self.passed_validations / max(self.total_validations, 1)
    
    @property
    def correction_rate(self) -> float:
        return self.corrections_applied / max(self.failed_validations, 1)


class BaseValidator:
    """Base class for all validators."""
    
    def __init__(self, name: str, severity: ValidationSeverity = ValidationSeverity.ERROR):
        self.name = name
        self.severity = severity
        self.enabled = True
        self.adaptation_enabled = True
        
        # Adaptive parameters
        self.threshold_history = deque(maxlen=100)
        self.performance_history = deque(maxlen=50)
        
    def validate(self, data: Any, context: Dict[str, Any] = None) -> ValidationEvent:
        """Perform validation on data."""
        raise NotImplementedError
    
    def adapt_thresholds(self, feedback: Dict[str, Any]):
        """Adapt validation thresholds based on feedback."""
        if not self.adaptation_enabled:
            return
        
        # Override in subclasses
        pass
    
    def get_adaptation_status(self) -> Dict[str, Any]:
        """Get current adaptation status."""
        return {
            'name': self.name,
            'enabled': self.enabled,
            'adaptation_enabled': self.adaptation_enabled,
            'threshold_history_size': len(self.threshold_history),
            'performance_history_size': len(self.performance_history)
        }


class TensorShapeValidator(BaseValidator):
    """Validates tensor shapes and dimensions."""
    
    def __init__(self, expected_shapes: Dict[str, Tuple], tolerance: float = 0.1):
        super().__init__("TensorShapeValidator")
        self.expected_shapes = expected_shapes
        self.tolerance = tolerance
        self.dynamic_shapes: Dict[str, List[Tuple]] = defaultdict(list)
        
    def validate(self, data: Any, context: Dict[str, Any] = None) -> ValidationEvent:
        """Validate tensor shapes."""
        context = context or {}
        start_time = time.time()
        
        try:
            if not isinstance(data, (torch.Tensor, dict)):
                return ValidationEvent(
                    validator_name=self.name,
                    severity=ValidationSeverity.ERROR,
                    result=ValidationResult.FAIL,
                    message=f"Expected tensor or dict, got {type(data)}"
                )
            
            if isinstance(data, torch.Tensor):
                data = {'default': data}
            
            issues = []
            corrections = []
            
            for key, tensor in data.items():
                if not isinstance(tensor, torch.Tensor):
                    continue
                
                current_shape = tuple(tensor.shape)
                
                # Check against expected shapes
                if key in self.expected_shapes:
                    expected_shape = self.expected_shapes[key]
                    
                    if not self._shapes_compatible(current_shape, expected_shape):
                        # Attempt correction
                        corrected_tensor = self._attempt_shape_correction(tensor, expected_shape)
                        
                        if corrected_tensor is not None:
                            corrections.append(f"Corrected {key}: {current_shape} -> {tuple(corrected_tensor.shape)}")
                            data[key] = corrected_tensor
                        else:
                            issues.append(f"Shape mismatch for {key}: expected {expected_shape}, got {current_shape}")
                
                # Update dynamic shape learning
                if self.adaptation_enabled:
                    self.dynamic_shapes[key].append(current_shape)
                    if len(self.dynamic_shapes[key]) > 20:
                        self.dynamic_shapes[key].pop(0)
            
            # Determine result
            if issues and not corrections:
                result = ValidationResult.FAIL
                severity = ValidationSeverity.ERROR
                message = "; ".join(issues)
            elif corrections:
                result = ValidationResult.CORRECTED
                severity = ValidationSeverity.WARNING
                message = "; ".join(corrections)
            else:
                result = ValidationResult.PASS
                severity = ValidationSeverity.INFO
                message = "All tensor shapes valid"
            
            return ValidationEvent(
                validator_name=self.name,
                severity=severity,
                result=result,
                message=message,
                details={
                    'validation_time': time.time() - start_time,
                    'shapes_checked': len(data),
                    'issues_found': len(issues),
                    'corrections_applied': len(corrections)
                },
                corrective_action="; ".join(corrections) if corrections else None
            )
            
        except Exception as e:
            return ValidationEvent(
                validator_name=self.name,
                severity=ValidationSeverity.CRITICAL,
                result=ValidationResult.FAIL,
                message=f"Validation error: {e}",
                details={'error_type': type(e).__name__}
            )
    
    def _shapes_compatible(self, actual: Tuple, expected: Tuple) -> bool:
        """Check if shapes are compatible within tolerance."""
        if len(actual) != len(expected):
            return False
        
        for a, e in zip(actual, expected):
            if e == -1:  # -1 means any size
                continue
            if abs(a - e) / max(e, 1) > self.tolerance:
                return False
        
        return True
    
    def _attempt_shape_correction(self, tensor: torch.Tensor, target_shape: Tuple) -> Optional[torch.Tensor]:
        """Attempt to correct tensor shape."""
        try:
            current_shape = tensor.shape
            
            # Try reshaping if total elements match
            target_numel = np.prod([s for s in target_shape if s != -1])
            if tensor.numel() == target_numel:
                return tensor.reshape(target_shape)
            
            # Try padding/cropping for 2D tensors
            if len(current_shape) == len(target_shape) == 2:
                h_curr, w_curr = current_shape
                h_target, w_target = target_shape
                
                # Pad if needed
                if h_curr < h_target or w_curr < w_target:
                    pad_h = max(0, h_target - h_curr)
                    pad_w = max(0, w_target - w_curr)
                    tensor = F.pad(tensor, (0, pad_w, 0, pad_h))
                
                # Crop if needed
                tensor = tensor[:h_target, :w_target]
                return tensor
            
            return None
            
        except Exception:
            return None
    
    def adapt_thresholds(self, feedback: Dict[str, Any]):
        """Adapt shape validation based on observed patterns."""
        if not self.adaptation_enabled:
            return
        
        # Learn common shapes from dynamic history
        for key, shapes in self.dynamic_shapes.items():
            if len(shapes) >= 10:  # Enough samples
                # Find most common shape
                shape_counts = defaultdict(int)
                for shape in shapes:
                    shape_counts[shape] += 1
                
                most_common_shape = max(shape_counts, key=shape_counts.get)
                
                # Update expected shapes if pattern is strong
                if shape_counts[most_common_shape] / len(shapes) > 0.8:
                    if key not in self.expected_shapes or self.expected_shapes[key] != most_common_shape:
                        self.expected_shapes[key] = most_common_shape
                        logging.info(f"Adapted expected shape for {key}: {most_common_shape}")


class NumericStabilityValidator(BaseValidator):
    """Validates numeric stability (NaN, Inf, extreme values)."""
    
    def __init__(self, max_value: float = 1e6, min_value: float = -1e6):
        super().__init__("NumericStabilityValidator")
        self.max_value = max_value
        self.min_value = min_value
        self.value_history = deque(maxlen=1000)
        
    def validate(self, data: Any, context: Dict[str, Any] = None) -> ValidationEvent:
        """Validate numeric stability."""
        start_time = time.time()
        
        try:
            if isinstance(data, torch.Tensor):
                tensors = {'default': data}
            elif isinstance(data, dict):
                tensors = {k: v for k, v in data.items() if isinstance(v, torch.Tensor)}
            else:
                return ValidationEvent(
                    validator_name=self.name,
                    severity=ValidationSeverity.ERROR,
                    result=ValidationResult.FAIL,
                    message=f"Unsupported data type: {type(data)}"
                )
            
            issues = []
            corrections = []
            stats = {}
            
            for key, tensor in tensors.items():
                # Check for NaN and Inf
                has_nan = torch.isnan(tensor).any().item()
                has_inf = torch.isinf(tensor).any().item()
                
                # Statistical analysis
                tensor_stats = {
                    'mean': tensor.mean().item(),
                    'std': tensor.std().item(),
                    'min': tensor.min().item(),
                    'max': tensor.max().item(),
                    'has_nan': has_nan,
                    'has_inf': has_inf
                }
                stats[key] = tensor_stats
                
                # Record values for adaptation
                if self.adaptation_enabled:
                    self.value_history.extend([tensor_stats['min'], tensor_stats['max']])
                
                # Detect issues
                if has_nan:
                    issues.append(f"NaN values detected in {key}")
                    # Attempt correction
                    corrected = torch.nan_to_num(tensor, nan=0.0)
                    tensors[key] = corrected
                    corrections.append(f"Replaced NaN values with 0 in {key}")
                
                if has_inf:
                    issues.append(f"Infinite values detected in {key}")
                    # Attempt correction
                    corrected = torch.nan_to_num(tensor, posinf=self.max_value, neginf=self.min_value)
                    tensors[key] = corrected
                    corrections.append(f"Clamped infinite values in {key}")
                
                # Check extreme values
                if tensor_stats['max'] > self.max_value or tensor_stats['min'] < self.min_value:
                    issues.append(f"Extreme values in {key}: min={tensor_stats['min']:.2e}, max={tensor_stats['max']:.2e}")
                    # Attempt correction
                    corrected = torch.clamp(tensor, self.min_value, self.max_value)
                    tensors[key] = corrected
                    corrections.append(f"Clamped extreme values in {key}")
            
            # Determine result
            if issues and not corrections:
                result = ValidationResult.FAIL
                severity = ValidationSeverity.ERROR
            elif corrections:
                result = ValidationResult.CORRECTED
                severity = ValidationSeverity.WARNING
            else:
                result = ValidationResult.PASS
                severity = ValidationSeverity.INFO
            
            message = "Numeric stability check completed"
            if issues:
                message += f": {len(issues)} issues found"
            if corrections:
                message += f", {len(corrections)} corrections applied"
            
            return ValidationEvent(
                validator_name=self.name,
                severity=severity,
                result=result,
                message=message,
                details={
                    'validation_time': time.time() - start_time,
                    'statistics': stats,
                    'issues': issues,
                    'corrections': corrections
                },
                corrective_action="; ".join(corrections) if corrections else None
            )
            
        except Exception as e:
            return ValidationEvent(
                validator_name=self.name,
                severity=ValidationSeverity.CRITICAL,
                result=ValidationResult.FAIL,
                message=f"Validation error: {e}",
                details={'error_type': type(e).__name__}
            )
    
    def adapt_thresholds(self, feedback: Dict[str, Any]):
        """Adapt numeric thresholds based on observed values."""
        if not self.adaptation_enabled or len(self.value_history) < 50:
            return
        
        # Calculate adaptive thresholds based on percentiles
        values = list(self.value_history)
        
        # Use 1st and 99th percentiles as new bounds
        new_min = np.percentile(values, 1)
        new_max = np.percentile(values, 99)
        
        # Only update if the change is significant
        min_change = abs((new_min - self.min_value) / self.min_value)
        max_change = abs((new_max - self.max_value) / self.max_value)
        
        if min_change > 0.1:  # 10% change threshold
            self.min_value = new_min
            logging.info(f"Adapted min_value threshold to: {self.min_value:.2e}")
        
        if max_change > 0.1:
            self.max_value = new_max
            logging.info(f"Adapted max_value threshold to: {self.max_value:.2e}")


class AnomalyDetectionValidator(BaseValidator):
    """Validates data using anomaly detection algorithms."""
    
    def __init__(self, contamination: float = 0.1, algorithm: str = 'isolation_forest'):
        super().__init__("AnomalyDetectionValidator")
        self.contamination = contamination
        self.algorithm = algorithm
        
        # Initialize anomaly detector
        if algorithm == 'isolation_forest':
            self.detector = IsolationForest(contamination=contamination, random_state=42)
        else:
            raise ValueError(f"Unsupported anomaly detection algorithm: {algorithm}")
        
        self.training_data = []
        self.is_fitted = False
        
    def validate(self, data: Any, context: Dict[str, Any] = None) -> ValidationEvent:
        """Validate data for anomalies."""
        start_time = time.time()
        
        try:
            # Convert data to feature vector
            features = self._extract_features(data)
            
            if features is None:
                return ValidationEvent(
                    validator_name=self.name,
                    severity=ValidationSeverity.ERROR,
                    result=ValidationResult.FAIL,
                    message="Could not extract features for anomaly detection"
                )
            
            # Train detector if not fitted and we have enough data
            if not self.is_fitted:
                self.training_data.append(features)
                
                if len(self.training_data) >= 50:  # Minimum training samples
                    self.detector.fit(self.training_data)
                    self.is_fitted = True
                    logging.info("Anomaly detector fitted with training data")
                
                # Cannot validate without fitted detector
                return ValidationEvent(
                    validator_name=self.name,
                    severity=ValidationSeverity.INFO,
                    result=ValidationResult.PASS,
                    message="Collecting training data for anomaly detection",
                    details={'training_samples': len(self.training_data)}
                )
            
            # Predict anomaly
            anomaly_score = self.detector.decision_function([features])[0]
            is_anomaly = self.detector.predict([features])[0] == -1
            
            # Determine result
            if is_anomaly:
                severity = ValidationSeverity.WARNING if anomaly_score > -0.2 else ValidationSeverity.ERROR
                result = ValidationResult.FAIL
                message = f"Anomaly detected (score: {anomaly_score:.3f})"
            else:
                severity = ValidationSeverity.INFO
                result = ValidationResult.PASS
                message = f"No anomaly detected (score: {anomaly_score:.3f})"
            
            return ValidationEvent(
                validator_name=self.name,
                severity=severity,
                result=result,
                message=message,
                details={
                    'validation_time': time.time() - start_time,
                    'anomaly_score': anomaly_score,
                    'is_anomaly': is_anomaly,
                    'features_count': len(features)
                }
            )
            
        except Exception as e:
            return ValidationEvent(
                validator_name=self.name,
                severity=ValidationSeverity.CRITICAL,
                result=ValidationResult.FAIL,
                message=f"Anomaly detection error: {e}",
                details={'error_type': type(e).__name__}
            )
    
    def _extract_features(self, data: Any) -> Optional[List[float]]:
        """Extract features from data for anomaly detection."""
        try:
            if isinstance(data, torch.Tensor):
                tensor = data.detach().cpu()
                
                # Extract statistical features
                features = [
                    tensor.mean().item(),
                    tensor.std().item(),
                    tensor.min().item(),
                    tensor.max().item(),
                    tensor.median().item(),
                    torch.quantile(tensor.float(), 0.25).item(),
                    torch.quantile(tensor.float(), 0.75).item(),
                    tensor.numel(),
                    len(tensor.shape)
                ]
                
                # Add shape features
                features.extend(list(tensor.shape))
                
                return features
                
            elif isinstance(data, dict):
                # Aggregate features from multiple tensors
                all_features = []
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        tensor_features = self._extract_features(value)
                        if tensor_features:
                            all_features.extend(tensor_features)
                
                return all_features if all_features else None
            
            return None
            
        except Exception:
            return None
    
    def adapt_thresholds(self, feedback: Dict[str, Any]):
        """Adapt anomaly detection based on feedback."""
        if not self.adaptation_enabled:
            return
        
        # Retrain detector if we have new feedback
        false_positives = feedback.get('false_positives', 0)
        false_negatives = feedback.get('false_negatives', 0)
        
        if false_positives > 3:  # Too many false positives
            self.contamination = min(self.contamination * 1.2, 0.5)
            self.detector = IsolationForest(contamination=self.contamination, random_state=42)
            self.is_fitted = False
            logging.info(f"Increased contamination parameter to: {self.contamination}")
        
        elif false_negatives > 2:  # Missing real anomalies
            self.contamination = max(self.contamination * 0.8, 0.01)
            self.detector = IsolationForest(contamination=self.contamination, random_state=42)
            self.is_fitted = False
            logging.info(f"Decreased contamination parameter to: {self.contamination}")


class AdaptiveValidationFramework:
    """Comprehensive adaptive validation framework."""
    
    def __init__(self, validators: List[BaseValidator] = None):
        self.validators = validators or [
            TensorShapeValidator({}),
            NumericStabilityValidator(),
            AnomalyDetectionValidator()
        ]
        
        self.metrics = ValidationMetrics()
        self.event_history: List[ValidationEvent] = []
        self.performance_monitor = PerformanceMonitor()
        
        # Adaptive learning
        self.learning_enabled = True
        self.feedback_history: List[Dict[str, Any]] = []
        
    def validate(self, data: Any, context: Dict[str, Any] = None) -> List[ValidationEvent]:
        """Run all validators on data."""
        context = context or {}
        start_time = time.time()
        
        events = []
        corrections_applied = False
        
        for validator in self.validators:
            if not validator.enabled:
                continue
            
            event = validator.validate(data, context)
            events.append(event)
            
            # Update metrics
            self.metrics.total_validations += 1
            
            if event.result == ValidationResult.PASS:
                self.metrics.passed_validations += 1
            elif event.result == ValidationResult.FAIL:
                self.metrics.failed_validations += 1
            elif event.result == ValidationResult.WARNING:
                self.metrics.warnings += 1
            elif event.result == ValidationResult.CORRECTED:
                self.metrics.corrections_applied += 1
                corrections_applied = True
            
            # Store event
            self.event_history.append(event)
        
        # Update average validation time
        validation_time = time.time() - start_time
        self.metrics.average_validation_time = (
            self.metrics.average_validation_time * 0.9 + validation_time * 0.1
        )
        
        # Keep history manageable
        if len(self.event_history) > 1000:
            self.event_history = self.event_history[-1000:]
        
        return events
    
    def provide_feedback(self, feedback: Dict[str, Any]):
        """Provide feedback for adaptive learning."""
        self.feedback_history.append({
            'timestamp': time.time(),
            'feedback': feedback.copy()
        })
        
        if self.learning_enabled:
            # Adapt validators based on feedback
            for validator in self.validators:
                validator.adapt_thresholds(feedback)
        
        # Update false positive/negative counts
        if 'false_positives' in feedback:
            self.metrics.false_positives += feedback['false_positives']
        if 'false_negatives' in feedback:
            self.metrics.false_negatives += feedback['false_negatives']
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        # Recent event analysis
        recent_events = self.event_history[-100:]  # Last 100 events
        
        event_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        validator_performance = defaultdict(lambda: {'pass': 0, 'fail': 0, 'warning': 0, 'corrected': 0})
        
        for event in recent_events:
            event_counts[event.result.value] += 1
            severity_counts[event.severity.name] += 1
            validator_performance[event.validator_name][event.result.value.lower()] += 1
        
        return {
            'metrics': {
                'total_validations': self.metrics.total_validations,
                'pass_rate': self.metrics.pass_rate,
                'correction_rate': self.metrics.correction_rate,
                'false_positives': self.metrics.false_positives,
                'false_negatives': self.metrics.false_negatives,
                'average_validation_time_ms': self.metrics.average_validation_time * 1000
            },
            'recent_events': {
                'total': len(recent_events),
                'by_result': dict(event_counts),
                'by_severity': dict(severity_counts)
            },
            'validator_performance': dict(validator_performance),
            'validators': [validator.get_adaptation_status() for validator in self.validators],
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for improving validation."""
        recommendations = []
        
        # Check pass rate
        if self.metrics.pass_rate < 0.8:
            recommendations.append("Low pass rate detected. Consider adjusting validator thresholds.")
        
        # Check false positive rate
        if self.metrics.false_positives > 10:
            recommendations.append("High false positive rate. Consider providing more feedback for adaptation.")
        
        # Check validation time
        if self.metrics.average_validation_time > 0.1:  # 100ms threshold
            recommendations.append("Validation time is high. Consider optimizing validators or reducing frequency.")
        
        # Check recent performance
        recent_failures = sum(
            1 for event in self.event_history[-50:] 
            if event.result == ValidationResult.FAIL
        )
        
        if recent_failures > 10:
            recommendations.append("High recent failure rate. Investigate data quality issues.")
        
        return recommendations


class PerformanceMonitor:
    """Monitors validation performance over time."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.performance_history = deque(maxlen=window_size)
        
    def record_performance(self, event: ValidationEvent):
        """Record performance metrics from validation event."""
        performance_data = {
            'timestamp': event.timestamp,
            'validator': event.validator_name,
            'result': event.result.value,
            'severity': event.severity.value,
            'validation_time': event.details.get('validation_time', 0.0)
        }
        
        self.performance_history.append(performance_data)
    
    def get_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends."""
        if len(self.performance_history) < 10:
            return {'status': 'insufficient_data'}
        
        # Calculate trends
        recent_times = [p['validation_time'] for p in list(self.performance_history)[-20:]]
        older_times = [p['validation_time'] for p in list(self.performance_history)[-50:-20]] if len(self.performance_history) >= 50 else []
        
        time_trend = 'stable'
        if older_times:
            recent_avg = statistics.mean(recent_times)
            older_avg = statistics.mean(older_times)
            
            if recent_avg > older_avg * 1.2:
                time_trend = 'increasing'
            elif recent_avg < older_avg * 0.8:
                time_trend = 'decreasing'
        
        # Success rate trend
        recent_results = [p['result'] for p in list(self.performance_history)[-20:]]
        success_rate = recent_results.count('PASS') / len(recent_results)
        
        return {
            'validation_time_trend': time_trend,
            'recent_success_rate': success_rate,
            'total_samples': len(self.performance_history),
            'average_validation_time': statistics.mean(recent_times)
        }


def create_adaptive_validation_framework(config: Dict[str, Any] = None) -> AdaptiveValidationFramework:
    """Factory function for creating adaptive validation frameworks."""
    config = config or {}
    
    validators = []
    
    # Shape validator
    if config.get('enable_shape_validation', True):
        expected_shapes = config.get('expected_shapes', {})
        validators.append(TensorShapeValidator(expected_shapes))
    
    # Numeric stability validator
    if config.get('enable_numeric_validation', True):
        max_value = config.get('max_value', 1e6)
        min_value = config.get('min_value', -1e6)
        validators.append(NumericStabilityValidator(max_value, min_value))
    
    # Anomaly detection validator
    if config.get('enable_anomaly_detection', True):
        contamination = config.get('contamination', 0.1)
        validators.append(AnomalyDetectionValidator(contamination))
    
    return AdaptiveValidationFramework(validators)