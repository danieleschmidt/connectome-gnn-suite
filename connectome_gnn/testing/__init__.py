"""Comprehensive testing framework for the connectome GNN suite."""

# Import with graceful degradation
__all__ = []

try:
    from .test_framework import TestFramework, TestCase, TestSuite
    __all__.extend(["TestFramework", "TestCase", "TestSuite"])
except ImportError:
    pass

try:
    from .quality_gates import QualityGateManager, QualityGate, TestResult
    __all__.extend(["QualityGateManager", "QualityGate", "TestResult"])
except ImportError:
    pass

# Optional components
try:
    from .unit_tests import UnitTestRunner, ModelTestSuite, DataTestSuite
    __all__.extend(["UnitTestRunner", "ModelTestSuite", "DataTestSuite"])
except ImportError:
    pass

try:
    from .integration_tests import IntegrationTestRunner, PipelineTestSuite
    __all__.extend(["IntegrationTestRunner", "PipelineTestSuite"])
except ImportError:
    pass

try:
    from .performance_tests import PerformanceTestRunner, BenchmarkTestSuite
    __all__.extend(["PerformanceTestRunner", "BenchmarkTestSuite"])
except ImportError:
    pass

try:
    from .security_tests import SecurityTestRunner, VulnerabilityScanner
    __all__.extend(["SecurityTestRunner", "VulnerabilityScanner"])
except ImportError:
    pass