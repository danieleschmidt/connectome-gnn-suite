"""Comprehensive testing and quality gates for Connectome-GNN-Suite."""

import unittest
import pytest
import torch
import numpy as np
import time
import psutil
import threading
import concurrent.futures
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
import json
import os
from pathlib import Path

from ..robust.logging_config import get_logger
from ..robust.error_handling import ConnectomeError, ValidationError
from ..security.advanced_security import SecurityMonitor, ThreatType
from ..validation.advanced_validation import ValidationPipeline, ConnectomeDataValidator
from ..autonomous.self_healing import get_auto_recovery_system, HealthStatus


@dataclass
class TestResult:
    """Test result container."""
    test_name: str
    passed: bool
    score: float
    execution_time: float
    memory_usage: float
    details: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    timestamp: float


@dataclass
class QualityGateConfig:
    """Configuration for quality gates."""
    min_test_coverage: float = 0.85
    max_memory_usage_mb: float = 1000.0
    max_execution_time_s: float = 300.0
    min_performance_score: float = 0.8
    max_security_incidents: int = 0
    min_validation_score: float = 0.9
    enable_stress_testing: bool = True
    enable_security_testing: bool = True
    enable_performance_testing: bool = True


class BaseQualityGate(ABC):
    """Base class for quality gates."""
    
    def __init__(self, name: str, config: QualityGateConfig):
        self.name = name
        self.config = config
        self.logger = get_logger(__name__)
        
    @abstractmethod
    def run_tests(self, test_data: Any) -> List[TestResult]:
        """Run quality gate tests."""
        pass
        
    def _create_result(
        self,
        test_name: str,
        passed: bool,
        score: float,
        execution_time: float = 0.0,
        memory_usage: float = 0.0,
        details: Dict[str, Any] = None,
        errors: List[str] = None,
        warnings: List[str] = None
    ) -> TestResult:
        """Create test result."""
        return TestResult(
            test_name=test_name,
            passed=passed,
            score=score,
            execution_time=execution_time,
            memory_usage=memory_usage,
            details=details or {},
            errors=errors or [],
            warnings=warnings or [],
            timestamp=time.time()
        )


class UnitTestGate(BaseQualityGate):
    """Unit testing quality gate."""
    
    def __init__(self, config: QualityGateConfig):
        super().__init__("UnitTestGate", config)
        
    def run_tests(self, test_data: Any) -> List[TestResult]:
        """Run unit tests."""
        results = []
        
        # Test data loading
        results.append(self._test_data_loading())
        
        # Test model creation
        results.append(self._test_model_creation())
        
        # Test training pipeline
        results.append(self._test_training_pipeline())
        
        # Test validation framework
        results.append(self._test_validation_framework())
        
        # Test security components
        results.append(self._test_security_components())
        
        return results
        
    def _test_data_loading(self) -> TestResult:
        """Test data loading functionality."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        errors = []
        warnings = []
        
        try:
            # Test basic data structures
            from ..core.base_types import ConnectomeData
            
            # Create synthetic data
            data = ConnectomeData(
                x=torch.randn(100, 50),
                edge_index=torch.randint(0, 100, (2, 500)),
                y=torch.randn(100, 1)
            )
            
            # Validate data structure
            if not hasattr(data, 'x') or not hasattr(data, 'edge_index'):
                errors.append("Missing required data attributes")
                
            if data.x.shape[0] != data.y.shape[0]:
                errors.append("Inconsistent data dimensions")
                
            # Test data processing
            from ..robust.validation import validate_connectome_data
            validation_result = validate_connectome_data(data)
            
            if not validation_result:
                warnings.append("Data validation returned warnings")
                
        except Exception as e:
            errors.append(f"Data loading test failed: {str(e)}")
            
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        return self._create_result(
            test_name="data_loading",
            passed=len(errors) == 0,
            score=1.0 if len(errors) == 0 else 0.5 if len(warnings) == 0 else 0.0,
            execution_time=end_time - start_time,
            memory_usage=end_memory - start_memory,
            details={"data_points": 100, "features": 50, "edges": 500},
            errors=errors,
            warnings=warnings
        )
        
    def _test_model_creation(self) -> TestResult:
        """Test model creation and basic operations."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        errors = []
        warnings = []
        
        try:
            # Test model imports
            from ..models import BaseConnectomeModel
            
            # Create simple model for testing
            class TestModel(BaseConnectomeModel):
                def __init__(self):
                    super().__init__()
                    self.linear = torch.nn.Linear(50, 1)
                    
                def forward(self, data):
                    return self.linear(data.x)
                    
            model = TestModel()
            
            # Test forward pass
            test_data = type('Data', (), {
                'x': torch.randn(10, 50),
                'edge_index': torch.randint(0, 10, (2, 20))
            })()
            
            output = model(test_data)
            
            if output.shape[0] != 10 or output.shape[1] != 1:
                errors.append(f"Unexpected output shape: {output.shape}")
                
            # Test parameter counting
            param_count = sum(p.numel() for p in model.parameters())
            if param_count == 0:
                errors.append("Model has no trainable parameters")
                
        except Exception as e:
            errors.append(f"Model creation test failed: {str(e)}")
            
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        return self._create_result(
            test_name="model_creation",
            passed=len(errors) == 0,
            score=1.0 if len(errors) == 0 else 0.0,
            execution_time=end_time - start_time,
            memory_usage=end_memory - start_memory,
            details={"parameters": param_count if 'param_count' in locals() else 0},
            errors=errors,
            warnings=warnings
        )
        
    def _test_training_pipeline(self) -> TestResult:
        """Test training pipeline components."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        errors = []
        warnings = []
        
        try:
            # Test optimizer creation
            from ..optimization.adaptive_optimization import OptimizerFactory, OptimizationConfig
            
            config = OptimizationConfig(learning_rate=0.001)
            
            # Create dummy model
            model = torch.nn.Linear(10, 1)
            
            # Test optimizer factory
            optimizer = OptimizerFactory.create_optimizer(model, "adam", config)
            
            if optimizer is None:
                errors.append("Failed to create optimizer")
                
            # Test scheduler creation
            scheduler = OptimizerFactory.create_scheduler(optimizer, "adaptive", config)
            
            if scheduler is None:
                errors.append("Failed to create scheduler")
                
            # Test basic training step
            dummy_input = torch.randn(5, 10)
            dummy_target = torch.randn(5, 1)
            
            output = model(dummy_input)
            loss = torch.nn.MSELoss()(output, dummy_target)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if loss.item() != loss.item():  # Check for NaN
                errors.append("Loss became NaN during training")
                
        except Exception as e:
            errors.append(f"Training pipeline test failed: {str(e)}")
            
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        return self._create_result(
            test_name="training_pipeline",
            passed=len(errors) == 0,
            score=1.0 if len(errors) == 0 else 0.0,
            execution_time=end_time - start_time,
            memory_usage=end_memory - start_memory,
            details={"loss": loss.item() if 'loss' in locals() else float('nan')},
            errors=errors,
            warnings=warnings
        )
        
    def _test_validation_framework(self) -> TestResult:
        """Test validation framework."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        errors = []
        warnings = []
        
        try:
            from ..validation.advanced_validation import ConnectomeDataValidator
            
            validator = ConnectomeDataValidator()
            
            # Create test data
            test_data = type('Data', (), {
                'x': torch.randn(100, 50),
                'edge_index': torch.randint(0, 100, (2, 500)),
                'edge_attr': torch.randn(500, 1)
            })()
            
            result = validator.validate(test_data)
            
            if result is None:
                errors.append("Validator returned None")
            elif not hasattr(result, 'passed'):
                errors.append("Invalid validation result structure")
                
        except Exception as e:
            errors.append(f"Validation framework test failed: {str(e)}")
            
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        return self._create_result(
            test_name="validation_framework",
            passed=len(errors) == 0,
            score=1.0 if len(errors) == 0 else 0.0,
            execution_time=end_time - start_time,
            memory_usage=end_memory - start_memory,
            details={},
            errors=errors,
            warnings=warnings
        )
        
    def _test_security_components(self) -> TestResult:
        """Test security components."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        errors = []
        warnings = []
        
        try:
            from ..security.advanced_security import SecurityMonitor, ThreatType
            
            monitor = SecurityMonitor()
            
            # Test threat detection
            test_data = {
                'anomaly_score': 2.0,
                'feature_drift': 0.3,
                'label_consistency': 0.8,
                'source_ip': '127.0.0.1',
                'user_id': 'test_user'
            }
            
            event = monitor.detect_threat(test_data, ThreatType.DATA_POISONING)
            
            if event and hasattr(event, 'threat_type'):
                # Security event detected as expected
                pass
            elif event is None:
                # No threat detected, which is also fine for normal data
                pass
            else:
                errors.append("Invalid security event structure")
                
        except Exception as e:
            errors.append(f"Security components test failed: {str(e)}")
            
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        return self._create_result(
            test_name="security_components",
            passed=len(errors) == 0,
            score=1.0 if len(errors) == 0 else 0.0,
            execution_time=end_time - start_time,
            memory_usage=end_memory - start_memory,
            details={},
            errors=errors,
            warnings=warnings
        )


class PerformanceTestGate(BaseQualityGate):
    """Performance testing quality gate."""
    
    def __init__(self, config: QualityGateConfig):
        super().__init__("PerformanceTestGate", config)
        
    def run_tests(self, test_data: Any) -> List[TestResult]:
        """Run performance tests."""
        results = []
        
        if self.config.enable_performance_testing:
            results.append(self._test_memory_efficiency())
            results.append(self._test_training_speed())
            results.append(self._test_inference_speed())
            results.append(self._test_scalability())
            
        return results
        
    def _test_memory_efficiency(self) -> TestResult:
        """Test memory efficiency."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        errors = []
        warnings = []
        peak_memory = start_memory
        
        try:
            # Create large dataset to test memory usage
            large_data = torch.randn(1000, 100)
            large_edges = torch.randint(0, 1000, (2, 5000))
            
            # Monitor memory usage
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            peak_memory = max(peak_memory, current_memory)
            
            # Process data in batches to test memory efficiency
            batch_size = 100
            for i in range(0, len(large_data), batch_size):
                batch = large_data[i:i+batch_size]
                
                # Simulate processing
                result = torch.matmul(batch, batch.t())
                
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                peak_memory = max(peak_memory, current_memory)
                
                # Clean up
                del result
                
            memory_increase = peak_memory - start_memory
            
            if memory_increase > self.config.max_memory_usage_mb:
                errors.append(f"Excessive memory usage: {memory_increase:.2f}MB")
            elif memory_increase > self.config.max_memory_usage_mb * 0.8:
                warnings.append(f"High memory usage: {memory_increase:.2f}MB")
                
        except Exception as e:
            errors.append(f"Memory efficiency test failed: {str(e)}")
            
        end_time = time.time()
        
        return self._create_result(
            test_name="memory_efficiency",
            passed=len(errors) == 0,
            score=max(0.0, 1.0 - (peak_memory - start_memory) / self.config.max_memory_usage_mb),
            execution_time=end_time - start_time,
            memory_usage=peak_memory - start_memory,
            details={
                "peak_memory_mb": peak_memory,
                "memory_increase_mb": peak_memory - start_memory
            },
            errors=errors,
            warnings=warnings
        )
        
    def _test_training_speed(self) -> TestResult:
        """Test training speed."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        errors = []
        warnings = []
        
        try:
            # Create test model and data
            model = torch.nn.Sequential(
                torch.nn.Linear(100, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 1)
            )
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.MSELoss()
            
            # Training benchmark
            train_start = time.time()
            num_iterations = 100
            
            for i in range(num_iterations):
                # Generate batch data
                batch_x = torch.randn(32, 100)
                batch_y = torch.randn(32, 1)
                
                # Forward pass
                output = model(batch_x)
                loss = criterion(output, batch_y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            train_end = time.time()
            training_time = train_end - train_start
            iterations_per_second = num_iterations / training_time
            
            # Check if training speed is acceptable
            min_speed = 10.0  # iterations per second
            if iterations_per_second < min_speed:
                errors.append(f"Training too slow: {iterations_per_second:.2f} it/s")
            elif iterations_per_second < min_speed * 2:
                warnings.append(f"Training speed below optimal: {iterations_per_second:.2f} it/s")
                
        except Exception as e:
            errors.append(f"Training speed test failed: {str(e)}")
            iterations_per_second = 0.0
            
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        return self._create_result(
            test_name="training_speed",
            passed=len(errors) == 0,
            score=min(1.0, iterations_per_second / 50.0),  # Scale to score
            execution_time=end_time - start_time,
            memory_usage=end_memory - start_memory,
            details={
                "iterations_per_second": iterations_per_second,
                "total_iterations": num_iterations if 'num_iterations' in locals() else 0
            },
            errors=errors,
            warnings=warnings
        )
        
    def _test_inference_speed(self) -> TestResult:
        """Test inference speed."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        errors = []
        warnings = []
        
        try:
            # Create test model
            model = torch.nn.Sequential(
                torch.nn.Linear(100, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 1)
            )
            model.eval()
            
            # Inference benchmark
            inference_start = time.time()
            num_samples = 1000
            
            with torch.no_grad():
                for i in range(num_samples):
                    sample_x = torch.randn(1, 100)
                    output = model(sample_x)
                    
            inference_end = time.time()
            inference_time = inference_end - inference_start
            samples_per_second = num_samples / inference_time
            
            # Check if inference speed is acceptable
            min_speed = 100.0  # samples per second
            if samples_per_second < min_speed:
                errors.append(f"Inference too slow: {samples_per_second:.2f} samples/s")
            elif samples_per_second < min_speed * 2:
                warnings.append(f"Inference speed below optimal: {samples_per_second:.2f} samples/s")
                
        except Exception as e:
            errors.append(f"Inference speed test failed: {str(e)}")
            samples_per_second = 0.0
            
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        return self._create_result(
            test_name="inference_speed",
            passed=len(errors) == 0,
            score=min(1.0, samples_per_second / 500.0),  # Scale to score
            execution_time=end_time - start_time,
            memory_usage=end_memory - start_memory,
            details={
                "samples_per_second": samples_per_second,
                "total_samples": num_samples if 'num_samples' in locals() else 0
            },
            errors=errors,
            warnings=warnings
        )
        
    def _test_scalability(self) -> TestResult:
        """Test scalability with increasing data sizes."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        errors = []
        warnings = []
        scalability_scores = []
        
        try:
            # Test with different data sizes
            data_sizes = [100, 500, 1000, 2000]
            model = torch.nn.Linear(50, 1)
            
            for size in data_sizes:
                size_start = time.time()
                
                # Create data of given size
                data = torch.randn(size, 50)
                
                # Process data
                with torch.no_grad():
                    output = model(data)
                    
                size_end = time.time()
                processing_time = size_end - size_start
                
                # Calculate processing rate
                rate = size / processing_time
                scalability_scores.append(rate)
                
            # Check if scalability is reasonable (should be roughly linear)
            if len(scalability_scores) > 1:
                rate_variance = np.var(scalability_scores) / np.mean(scalability_scores)
                if rate_variance > 0.5:
                    warnings.append(f"High variance in processing rates: {rate_variance:.3f}")
                    
        except Exception as e:
            errors.append(f"Scalability test failed: {str(e)}")
            
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        return self._create_result(
            test_name="scalability",
            passed=len(errors) == 0,
            score=1.0 if len(errors) == 0 else 0.5 if len(warnings) == 0 else 0.0,
            execution_time=end_time - start_time,
            memory_usage=end_memory - start_memory,
            details={
                "data_sizes": data_sizes if 'data_sizes' in locals() else [],
                "processing_rates": scalability_scores
            },
            errors=errors,
            warnings=warnings
        )


class SecurityTestGate(BaseQualityGate):
    """Security testing quality gate."""
    
    def __init__(self, config: QualityGateConfig):
        super().__init__("SecurityTestGate", config)
        
    def run_tests(self, test_data: Any) -> List[TestResult]:
        """Run security tests."""
        results = []
        
        if self.config.enable_security_testing:
            results.append(self._test_threat_detection())
            results.append(self._test_access_control())
            results.append(self._test_data_encryption())
            results.append(self._test_privacy_protection())
            
        return results
        
    def _test_threat_detection(self) -> TestResult:
        """Test threat detection capabilities."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        errors = []
        warnings = []
        detected_threats = 0
        
        try:
            from ..security.advanced_security import SecurityMonitor, ThreatType
            
            monitor = SecurityMonitor()
            
            # Test different threat scenarios
            threat_scenarios = [
                {
                    'type': ThreatType.DATA_POISONING,
                    'data': {
                        'anomaly_score': 5.0,  # High anomaly
                        'feature_drift': 0.8,  # High drift
                        'label_consistency': 0.05,  # Low consistency
                        'source_ip': '192.168.1.100'
                    }
                },
                {
                    'type': ThreatType.ADVERSARIAL_ATTACK,
                    'data': {
                        'perturbation_magnitude': 0.2,  # High perturbation
                        'confidence_drop': 0.7,  # High confidence drop
                        'source_ip': '10.0.0.1'
                    }
                },
                {
                    'type': ThreatType.UNAUTHORIZED_ACCESS,
                    'data': {
                        'failed_login_count': 10,  # Many failed attempts
                        'suspicious_access': True,
                        'source_ip': '203.0.113.1'
                    }
                }
            ]
            
            for scenario in threat_scenarios:
                event = monitor.detect_threat(scenario['data'], scenario['type'])
                
                if event is not None:
                    detected_threats += 1
                    monitor.log_event(event)
                    
            # Should detect most threats
            detection_rate = detected_threats / len(threat_scenarios)
            if detection_rate < 0.8:
                errors.append(f"Low threat detection rate: {detection_rate:.2f}")
            elif detection_rate < 0.9:
                warnings.append(f"Moderate threat detection rate: {detection_rate:.2f}")
                
        except Exception as e:
            errors.append(f"Threat detection test failed: {str(e)}")
            
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        return self._create_result(
            test_name="threat_detection",
            passed=len(errors) == 0,
            score=detected_threats / 3.0 if 'threat_scenarios' in locals() else 0.0,
            execution_time=end_time - start_time,
            memory_usage=end_memory - start_memory,
            details={
                "detected_threats": detected_threats,
                "total_scenarios": 3,
                "detection_rate": detected_threats / 3.0
            },
            errors=errors,
            warnings=warnings
        )
        
    def _test_access_control(self) -> TestResult:
        """Test access control mechanisms."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        errors = []
        warnings = []
        
        try:
            from ..security.advanced_security import AccessControlManager
            
            acl = AccessControlManager()
            
            # Create test users
            user_created = acl.create_user("test_user", "test_password", "researcher")
            if not user_created:
                errors.append("Failed to create test user")
                
            # Test authentication
            session_id = acl.authenticate_user("test_user", "test_password")
            if session_id is None:
                errors.append("Failed to authenticate valid user")
                
            # Test wrong password
            wrong_session = acl.authenticate_user("test_user", "wrong_password")
            if wrong_session is not None:
                errors.append("Authenticated with wrong password")
                
            # Test permissions
            if session_id:
                can_read = acl.check_permission(session_id, "read_data")
                if not can_read:
                    errors.append("User should have read_data permission")
                    
                can_admin = acl.check_permission(session_id, "manage_users")
                if can_admin:
                    errors.append("User should not have admin permissions")
                    
        except Exception as e:
            errors.append(f"Access control test failed: {str(e)}")
            
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        return self._create_result(
            test_name="access_control",
            passed=len(errors) == 0,
            score=1.0 if len(errors) == 0 else 0.0,
            execution_time=end_time - start_time,
            memory_usage=end_memory - start_memory,
            details={},
            errors=errors,
            warnings=warnings
        )
        
    def _test_data_encryption(self) -> TestResult:
        """Test data encryption functionality."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        errors = []
        warnings = []
        
        try:
            from ..security.advanced_security import SecureDataHandler
            
            handler = SecureDataHandler()
            
            # Test data encryption/decryption
            test_data = b"This is sensitive connectome data"
            
            # Encrypt data
            encrypted_package = handler.encrypt_data(test_data)
            
            if 'encrypted_data' not in encrypted_package:
                errors.append("Encryption package missing encrypted_data")
                
            # Decrypt data
            decrypted_data = handler.decrypt_data(encrypted_package)
            
            if decrypted_data != test_data:
                errors.append("Decrypted data doesn't match original")
                
            # Test data signing
            signature = handler.sign_data(test_data)
            if not signature:
                errors.append("Failed to sign data")
                
            # Test signature verification
            is_valid = handler.validate_data_integrity(test_data, signature)
            if not is_valid:
                errors.append("Failed to verify data signature")
                
        except ImportError:
            warnings.append("Cryptography module not available, skipping encryption tests")
        except Exception as e:
            errors.append(f"Data encryption test failed: {str(e)}")
            
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        return self._create_result(
            test_name="data_encryption",
            passed=len(errors) == 0,
            score=1.0 if len(errors) == 0 else 0.5 if len(warnings) > 0 else 0.0,
            execution_time=end_time - start_time,
            memory_usage=end_memory - start_memory,
            details={},
            errors=errors,
            warnings=warnings
        )
        
    def _test_privacy_protection(self) -> TestResult:
        """Test privacy protection mechanisms."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        errors = []
        warnings = []
        
        try:
            from ..security.advanced_security import DifferentialPrivacyManager
            
            dp_manager = DifferentialPrivacyManager(epsilon=1.0, delta=1e-5)
            
            # Test gradient noise addition
            test_gradients = torch.randn(100, 50)
            noisy_gradients = dp_manager.add_noise_to_gradients(test_gradients)
            
            if torch.equal(test_gradients, noisy_gradients):
                errors.append("No noise was added to gradients")
                
            # Test gradient clipping
            large_gradients = torch.randn(100, 50) * 10  # Large gradients
            clipped_gradients = dp_manager.clip_gradients(large_gradients, max_norm=1.0)
            
            max_norm = torch.norm(clipped_gradients, dim=1).max()
            if max_norm > 1.1:  # Allow small tolerance
                errors.append(f"Gradients not properly clipped: max_norm={max_norm}")
                
            # Test privacy budget tracking
            initial_spent = dp_manager.compute_privacy_spent(0)
            after_steps = dp_manager.compute_privacy_spent(100)
            
            if after_steps <= initial_spent:
                errors.append("Privacy budget not properly tracked")
                
        except Exception as e:
            errors.append(f"Privacy protection test failed: {str(e)}")
            
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        return self._create_result(
            test_name="privacy_protection",
            passed=len(errors) == 0,
            score=1.0 if len(errors) == 0 else 0.0,
            execution_time=end_time - start_time,
            memory_usage=end_memory - start_memory,
            details={},
            errors=errors,
            warnings=warnings
        )


class QualityGateOrchestrator:
    """Orchestrates all quality gates and provides comprehensive reporting."""
    
    def __init__(self, config: QualityGateConfig = None):
        self.config = config or QualityGateConfig()
        self.gates = [
            UnitTestGate(self.config),
            PerformanceTestGate(self.config),
            SecurityTestGate(self.config)
        ]
        
        self.logger = get_logger(__name__)
        
    def run_all_quality_gates(self, test_data: Any = None) -> Dict[str, Any]:
        """Run all quality gates and return comprehensive report."""
        start_time = time.time()
        
        all_results = {}
        gate_summaries = {}
        
        self.logger.info("Starting comprehensive quality gate execution")
        
        for gate in self.gates:
            gate_start = time.time()
            
            try:
                results = gate.run_tests(test_data)
                all_results[gate.name] = results
                
                # Compute gate summary
                passed_tests = sum(1 for r in results if r.passed)
                total_tests = len(results)
                avg_score = sum(r.score for r in results) / total_tests if total_tests > 0 else 0.0
                total_time = sum(r.execution_time for r in results)
                total_memory = sum(r.memory_usage for r in results)
                
                gate_summaries[gate.name] = {
                    'passed_tests': passed_tests,
                    'total_tests': total_tests,
                    'pass_rate': passed_tests / total_tests if total_tests > 0 else 0.0,
                    'average_score': avg_score,
                    'total_execution_time': total_time,
                    'total_memory_usage': total_memory,
                    'gate_execution_time': time.time() - gate_start
                }
                
                self.logger.info(
                    f"Completed {gate.name}: {passed_tests}/{total_tests} passed "
                    f"(score: {avg_score:.3f})"
                )
                
            except Exception as e:
                self.logger.error(f"Quality gate {gate.name} failed: {e}")
                gate_summaries[gate.name] = {
                    'error': str(e),
                    'passed_tests': 0,
                    'total_tests': 0,
                    'pass_rate': 0.0,
                    'average_score': 0.0
                }
                
        # Compute overall summary
        total_passed = sum(s.get('passed_tests', 0) for s in gate_summaries.values())
        total_tests = sum(s.get('total_tests', 0) for s in gate_summaries.values())
        overall_pass_rate = total_passed / total_tests if total_tests > 0 else 0.0
        
        overall_score = sum(
            s.get('average_score', 0.0) * s.get('total_tests', 0) 
            for s in gate_summaries.values()
        ) / total_tests if total_tests > 0 else 0.0
        
        # Determine if quality gates passed
        quality_gates_passed = (
            overall_pass_rate >= 0.9 and
            overall_score >= self.config.min_validation_score and
            all(s.get('pass_rate', 0.0) >= 0.8 for s in gate_summaries.values())
        )
        
        report = {
            'timestamp': time.time(),
            'total_execution_time': time.time() - start_time,
            'quality_gates_passed': quality_gates_passed,
            'overall_pass_rate': overall_pass_rate,
            'overall_score': overall_score,
            'total_passed_tests': total_passed,
            'total_tests': total_tests,
            'gate_summaries': gate_summaries,
            'detailed_results': all_results,
            'config': {
                'min_test_coverage': self.config.min_test_coverage,
                'max_memory_usage_mb': self.config.max_memory_usage_mb,
                'max_execution_time_s': self.config.max_execution_time_s,
                'min_performance_score': self.config.min_performance_score,
                'min_validation_score': self.config.min_validation_score
            }
        }
        
        if quality_gates_passed:
            self.logger.info(
                f"✅ All quality gates PASSED - Overall score: {overall_score:.3f}"
            )
        else:
            self.logger.warning(
                f"❌ Quality gates FAILED - Overall score: {overall_score:.3f}"
            )
            
        return report
        
    def save_report(self, report: Dict[str, Any], filepath: str):
        """Save quality gate report to file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"Quality gate report saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save report: {e}")


# Global quality gate orchestrator
_global_orchestrator = None

def get_quality_gate_orchestrator(config: QualityGateConfig = None) -> QualityGateOrchestrator:
    """Get global quality gate orchestrator."""
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = QualityGateOrchestrator(config)
    return _global_orchestrator


def run_comprehensive_quality_gates(test_data: Any = None) -> Dict[str, Any]:
    """Run comprehensive quality gates and return report."""
    orchestrator = get_quality_gate_orchestrator()
    return orchestrator.run_all_quality_gates(test_data)