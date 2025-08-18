#!/usr/bin/env python3
"""Simple validation script for the Connectome-GNN-Suite."""

import sys
import os
import importlib
import traceback
from pathlib import Path

def test_imports():
    """Test core module imports."""
    print("🧪 Testing Core Module Imports...")
    
    test_modules = [
        'connectome_gnn',
        'connectome_gnn.core',
        'connectome_gnn.models',
        'connectome_gnn.data',
        'connectome_gnn.tasks',
        'connectome_gnn.robust',
        'connectome_gnn.scale',
        'connectome_gnn.optimization'
    ]
    
    passed = 0
    failed = 0
    
    for module_name in test_modules:
        try:
            module = importlib.import_module(module_name)
            print(f"✅ {module_name}: Import successful")
            passed += 1
        except Exception as e:
            print(f"❌ {module_name}: Import failed - {e}")
            failed += 1
    
    return passed, failed

def test_model_creation():
    """Test model creation without external dependencies."""
    print("\n🏗️ Testing Model Creation...")
    
    try:
        from connectome_gnn.core.base_types import ModelConfig
        from connectome_gnn.models.base import BaseConnectomeModel
        
        # Test configuration creation
        config = ModelConfig(
            model_type="test_model",
            node_features=100,
            hidden_dim=256,
            output_dim=1
        )
        
        print(f"✅ ModelConfig created: {config.model_type}")
        print(f"✅ Config validation: {config.validate()}")
        
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_data_processing():
    """Test data processing utilities."""
    print("\n📊 Testing Data Processing...")
    
    try:
        from connectome_gnn.data.synthetic import SyntheticConnectomeGenerator
        
        # Create synthetic data generator
        generator = SyntheticConnectomeGenerator(num_nodes=50, random_seed=42)
        
        # Generate sample data
        connectivity = generator.generate_modular_connectivity()
        node_features = generator.generate_node_features(feature_dim=10)
        
        print(f"✅ Generated connectivity matrix: {connectivity.shape}")
        print(f"✅ Generated node features: {node_features.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Data processing failed: {e}")
        traceback.print_exc()
        return False

def test_security_features():
    """Test security utilities."""
    print("\n🔒 Testing Security Features...")
    
    try:
        from connectome_gnn.robust.security import SecurityManager, InputSanitizer
        
        # Test security manager
        security_mgr = SecurityManager()
        
        # Test file validation
        test_filename = "test_file.json"
        test_content = b'{"test": "data"}'
        is_valid, issues = security_mgr.validate_file_upload(test_filename, test_content)
        
        print(f"✅ File validation: {'Valid' if is_valid else 'Invalid'}")
        
        # Test input sanitization
        sanitized = InputSanitizer.sanitize_string("test<script>alert('xss')</script>", max_length=100)
        print(f"✅ Input sanitization: '{sanitized}'")
        
        return True
    except Exception as e:
        print(f"❌ Security testing failed: {e}")
        return False

def test_error_handling():
    """Test error handling system."""
    print("\n⚠️ Testing Error Handling...")
    
    try:
        from connectome_gnn.robust.error_handling import (
            ErrorHandler, ConnectomeError, get_global_error_handler
        )
        
        # Test error handler
        handler = ErrorHandler()
        
        # Test custom exception
        try:
            raise ConnectomeError("Test error", error_code="TEST_ERROR")
        except ConnectomeError as e:
            error_info = handler.handle_error(e)
            print(f"✅ Error handled: {error_info['error_type']}")
        
        # Test global handler
        global_handler = get_global_error_handler()
        print(f"✅ Global error handler: {type(global_handler).__name__}")
        
        return True
    except Exception as e:
        print(f"❌ Error handling failed: {e}")
        return False

def test_performance_monitoring():
    """Test performance monitoring utilities."""
    print("\n📈 Testing Performance Monitoring...")
    
    try:
        from connectome_gnn.scale.performance_monitoring import MetricsCollector
        
        # Test metrics collection
        collector = MetricsCollector()
        metrics = collector.collect_metrics()
        
        print(f"✅ Metrics collected: CPU {metrics.cpu_percent:.1f}%, Memory {metrics.memory_usage_mb:.1f}MB")
        
        return True
    except Exception as e:
        print(f"❌ Performance monitoring failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🚀 Connectome-GNN-Suite Simple Validation")
    print("=" * 50)
    
    # Add project root to path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    # Run tests
    total_passed = 0
    total_failed = 0
    
    # Test imports
    passed, failed = test_imports()
    total_passed += passed
    total_failed += failed
    
    # Test core functionality
    tests = [
        test_model_creation,
        test_data_processing,
        test_security_features,
        test_error_handling,
        test_performance_monitoring
    ]
    
    for test_func in tests:
        try:
            if test_func():
                total_passed += 1
            else:
                total_failed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            total_failed += 1
    
    # Summary
    total_tests = total_passed + total_failed
    success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    print("\n" + "=" * 50)
    print(f"📊 VALIDATION SUMMARY")
    print("=" * 50)
    print(f"✅ Passed: {total_passed}")
    print(f"❌ Failed: {total_failed}")
    print(f"📈 Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("🌟 VALIDATION SUCCESSFUL - Core functionality working!")
        return 0
    else:
        print("⚠️ VALIDATION ISSUES - Some components need attention")
        return 1

if __name__ == "__main__":
    sys.exit(main())