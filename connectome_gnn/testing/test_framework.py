"""Core testing framework for the connectome GNN suite."""

import time
import traceback
import inspect
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import json

from ..robust.logging_config import get_logger
from ..robust.error_handling import ErrorHandler


class TestStatus(Enum):
    """Test execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestResult:
    """Results from test execution."""
    name: str
    status: TestStatus
    execution_time: float = 0.0
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    assertions: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'status': self.status.value,
            'execution_time': self.execution_time,
            'error_message': self.error_message,
            'error_traceback': self.error_traceback,
            'details': self.details,
            'assertions': self.assertions,
            'passed': self.status == TestStatus.PASSED
        }


class TestCase:
    """Base class for test cases."""
    
    def __init__(self, name: str, description: str = "", 
                 logger: Optional[logging.Logger] = None):
        self.name = name
        self.description = description
        self.logger = logger or get_logger(f"test.{name}")
        self.setup_called = False
        self.teardown_called = False
        
        # Test state
        self.assertions = []
        self.test_data = {}
        
    def setup(self):
        """Setup method called before test execution."""
        self.setup_called = True
        self.logger.debug(f"Setup for test {self.name}")
    
    def teardown(self):
        """Teardown method called after test execution."""
        self.teardown_called = True
        self.logger.debug(f"Teardown for test {self.name}")
    
    def assert_true(self, condition: bool, message: str = ""):
        """Assert that condition is true."""
        assertion = {
            'type': 'assert_true',
            'condition': condition,
            'message': message,
            'passed': condition
        }
        self.assertions.append(assertion)
        
        if not condition:
            raise AssertionError(f"Assertion failed: {message}")
    
    def assert_false(self, condition: bool, message: str = ""):
        """Assert that condition is false."""
        self.assert_true(not condition, message)
    
    def assert_equal(self, actual: Any, expected: Any, message: str = ""):
        """Assert that actual equals expected."""
        condition = actual == expected
        full_message = message or f"Expected {expected}, got {actual}"
        
        assertion = {
            'type': 'assert_equal',
            'actual': str(actual),
            'expected': str(expected),
            'message': full_message,
            'passed': condition
        }
        self.assertions.append(assertion)
        
        if not condition:
            raise AssertionError(full_message)
    
    def assert_not_equal(self, actual: Any, expected: Any, message: str = ""):
        """Assert that actual does not equal expected."""
        condition = actual != expected
        full_message = message or f"Expected not {expected}, got {actual}"
        
        assertion = {
            'type': 'assert_not_equal',
            'actual': str(actual),
            'expected': str(expected),
            'message': full_message,
            'passed': condition
        }
        self.assertions.append(assertion)
        
        if not condition:
            raise AssertionError(full_message)
    
    def assert_raises(self, exception_type: type, func: Callable, *args, **kwargs):
        """Assert that function raises specified exception."""
        try:
            func(*args, **kwargs)
            assertion = {
                'type': 'assert_raises',
                'exception_type': exception_type.__name__,
                'message': f"Expected {exception_type.__name__} to be raised",
                'passed': False
            }
            self.assertions.append(assertion)
            raise AssertionError(f"Expected {exception_type.__name__} to be raised")
        except exception_type:
            assertion = {
                'type': 'assert_raises',
                'exception_type': exception_type.__name__,
                'message': f"Successfully caught {exception_type.__name__}",
                'passed': True
            }
            self.assertions.append(assertion)
        except Exception as e:
            assertion = {
                'type': 'assert_raises',
                'exception_type': exception_type.__name__,
                'actual_exception': type(e).__name__,
                'message': f"Expected {exception_type.__name__}, got {type(e).__name__}",
                'passed': False
            }
            self.assertions.append(assertion)
            raise AssertionError(f"Expected {exception_type.__name__}, got {type(e).__name__}")
    
    def run_test(self) -> TestResult:
        """Run the test and return results."""
        result = TestResult(name=self.name, status=TestStatus.PENDING)
        start_time = time.time()
        
        try:
            result.status = TestStatus.RUNNING
            self.logger.info(f"Running test: {self.name}")
            
            # Setup
            self.setup()
            
            # Find and run test method
            test_method = None
            for method_name in dir(self):
                if method_name.startswith('test_') and callable(getattr(self, method_name)):
                    test_method = getattr(self, method_name)
                    break
            
            if not test_method:
                raise RuntimeError(f"No test method found in {self.__class__.__name__}")
            
            # Execute test
            test_method()
            
            result.status = TestStatus.PASSED
            result.assertions = self.assertions.copy()
            
        except AssertionError as e:
            result.status = TestStatus.FAILED
            result.error_message = str(e)
            result.assertions = self.assertions.copy()
            self.logger.error(f"Test {self.name} failed: {e}")
            
        except Exception as e:
            result.status = TestStatus.ERROR
            result.error_message = str(e)
            result.error_traceback = traceback.format_exc()
            result.assertions = self.assertions.copy()
            self.logger.error(f"Test {self.name} error: {e}")
            
        finally:
            # Teardown
            try:
                self.teardown()
            except Exception as e:
                self.logger.warning(f"Teardown failed for {self.name}: {e}")
            
            result.execution_time = time.time() - start_time
        
        return result


class TestSuite:
    """Collection of test cases."""
    
    def __init__(self, name: str, description: str = "",
                 logger: Optional[logging.Logger] = None):
        self.name = name
        self.description = description
        self.logger = logger or get_logger(f"testsuite.{name}")
        
        self.test_cases = []
        self.setup_suite_func = None
        self.teardown_suite_func = None
    
    def add_test(self, test_case: TestCase):
        """Add a test case to the suite."""
        self.test_cases.append(test_case)
        self.logger.debug(f"Added test case: {test_case.name}")
    
    def setup_suite(self, func: Callable):
        """Set suite-level setup function."""
        self.setup_suite_func = func
        return func
    
    def teardown_suite(self, func: Callable):
        """Set suite-level teardown function."""
        self.teardown_suite_func = func
        return func
    
    def run_tests(self, stop_on_failure: bool = False) -> Dict[str, TestResult]:
        """Run all tests in the suite."""
        self.logger.info(f"Running test suite: {self.name} ({len(self.test_cases)} tests)")
        
        results = {}
        
        try:
            # Suite setup
            if self.setup_suite_func:
                self.logger.debug("Running suite setup")
                self.setup_suite_func()
            
            # Run tests
            for test_case in self.test_cases:
                result = test_case.run_test()
                results[test_case.name] = result
                
                if stop_on_failure and result.status in [TestStatus.FAILED, TestStatus.ERROR]:
                    self.logger.warning(f"Stopping test suite due to failure in {test_case.name}")
                    break
            
        finally:
            # Suite teardown
            if self.teardown_suite_func:
                try:
                    self.logger.debug("Running suite teardown")
                    self.teardown_suite_func()
                except Exception as e:
                    self.logger.error(f"Suite teardown failed: {e}")
        
        # Log summary
        passed = sum(1 for r in results.values() if r.status == TestStatus.PASSED)
        failed = sum(1 for r in results.values() if r.status == TestStatus.FAILED)
        errors = sum(1 for r in results.values() if r.status == TestStatus.ERROR)
        
        self.logger.info(f"Test suite {self.name} completed: "
                        f"{passed} passed, {failed} failed, {errors} errors")
        
        return results


class TestFramework:
    """Main testing framework coordinator."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger("test_framework")
        self.error_handler = ErrorHandler(logger)
        
        self.test_suites = {}
        self.test_results = {}
        self.global_setup_func = None
        self.global_teardown_func = None
    
    def register_suite(self, test_suite: TestSuite):
        """Register a test suite."""
        self.test_suites[test_suite.name] = test_suite
        self.logger.info(f"Registered test suite: {test_suite.name}")
    
    def global_setup(self, func: Callable):
        """Set global setup function."""
        self.global_setup_func = func
        return func
    
    def global_teardown(self, func: Callable):
        """Set global teardown function."""
        self.global_teardown_func = func
        return func
    
    def run_suite(self, suite_name: str, **kwargs) -> Dict[str, TestResult]:
        """Run a specific test suite."""
        if suite_name not in self.test_suites:
            raise ValueError(f"Test suite '{suite_name}' not found")
        
        suite = self.test_suites[suite_name]
        return suite.run_tests(**kwargs)
    
    def run_all_suites(self, stop_on_failure: bool = False) -> Dict[str, Dict[str, TestResult]]:
        """Run all registered test suites."""
        self.logger.info(f"Running all test suites ({len(self.test_suites)} suites)")
        
        all_results = {}
        
        try:
            # Global setup
            if self.global_setup_func:
                self.logger.debug("Running global setup")
                self.global_setup_func()
            
            # Run suites
            for suite_name, suite in self.test_suites.items():
                self.logger.info(f"Running suite: {suite_name}")
                
                try:
                    results = suite.run_tests(stop_on_failure=stop_on_failure)
                    all_results[suite_name] = results
                    
                    # Check for failures if stop_on_failure is enabled
                    if stop_on_failure:
                        has_failures = any(
                            r.status in [TestStatus.FAILED, TestStatus.ERROR] 
                            for r in results.values()
                        )
                        if has_failures:
                            self.logger.warning(f"Stopping execution due to failures in {suite_name}")
                            break
                            
                except Exception as e:
                    self.error_handler.handle_error(e, {'suite': suite_name})
                    all_results[suite_name] = {}
        
        finally:
            # Global teardown
            if self.global_teardown_func:
                try:
                    self.logger.debug("Running global teardown")
                    self.global_teardown_func()
                except Exception as e:
                    self.logger.error(f"Global teardown failed: {e}")
        
        self.test_results = all_results
        return all_results
    
    def generate_test_report(self, output_path: str = None) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        if not self.test_results:
            self.logger.warning("No test results available for report generation")
            return {}
        
        # Calculate overall statistics
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_errors = 0
        total_execution_time = 0.0
        
        suite_summaries = {}
        
        for suite_name, results in self.test_results.items():
            suite_stats = {
                'total': len(results),
                'passed': 0,
                'failed': 0,
                'errors': 0,
                'execution_time': 0.0,
                'tests': {}
            }
            
            for test_name, result in results.items():
                suite_stats['tests'][test_name] = result.to_dict()
                suite_stats['execution_time'] += result.execution_time
                
                if result.status == TestStatus.PASSED:
                    suite_stats['passed'] += 1
                elif result.status == TestStatus.FAILED:
                    suite_stats['failed'] += 1
                elif result.status == TestStatus.ERROR:
                    suite_stats['errors'] += 1
            
            suite_summaries[suite_name] = suite_stats
            
            total_tests += suite_stats['total']
            total_passed += suite_stats['passed']
            total_failed += suite_stats['failed']
            total_errors += suite_stats['errors']
            total_execution_time += suite_stats['execution_time']
        
        # Create report
        report = {
            'timestamp': time.time(),
            'summary': {
                'total_tests': total_tests,
                'total_passed': total_passed,
                'total_failed': total_failed,
                'total_errors': total_errors,
                'success_rate': total_passed / max(1, total_tests),
                'total_execution_time': total_execution_time
            },
            'suites': suite_summaries
        }
        
        # Save report if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Test report saved to {output_path}")
        
        return report
    
    def get_failed_tests(self) -> List[Dict[str, Any]]:
        """Get list of failed tests across all suites."""
        failed_tests = []
        
        for suite_name, results in self.test_results.items():
            for test_name, result in results.items():
                if result.status in [TestStatus.FAILED, TestStatus.ERROR]:
                    failed_tests.append({
                        'suite': suite_name,
                        'test': test_name,
                        'status': result.status.value,
                        'error': result.error_message,
                        'execution_time': result.execution_time
                    })
        
        return failed_tests
    
    def create_test_class(self, class_name: str, test_methods: Dict[str, Callable]) -> type:
        """Dynamically create a test class."""
        def __init__(self, name: str):
            TestCase.__init__(self, name)
        
        # Create class attributes
        attrs = {'__init__': __init__}
        attrs.update(test_methods)
        
        # Create the class
        test_class = type(class_name, (TestCase,), attrs)
        
        return test_class


def create_simple_test(name: str, test_func: Callable, 
                      setup_func: Callable = None,
                      teardown_func: Callable = None) -> TestCase:
    """Create a simple test case from a function."""
    
    class SimpleTestCase(TestCase):
        def __init__(self):
            super().__init__(name)
        
        def setup(self):
            super().setup()
            if setup_func:
                setup_func()
        
        def teardown(self):
            if teardown_func:
                teardown_func()
            super().teardown()
        
        def test_function(self):
            test_func(self)
    
    return SimpleTestCase()


def test_decorator(name: str = None):
    """Decorator to mark functions as tests."""
    def decorator(func: Callable) -> Callable:
        test_name = name or func.__name__
        func._is_test = True
        func._test_name = test_name
        return func
    return decorator