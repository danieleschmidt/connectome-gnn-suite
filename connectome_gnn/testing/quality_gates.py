"""Quality gates for ensuring code and model quality."""

import time
import json
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging

from .test_framework import TestFramework, TestResult, TestStatus
from ..robust.logging_config import get_logger
from ..scale.performance_monitoring import PerformanceMonitor
from ..robust.validation import validate_all, BaseValidator


class QualityGateStatus(Enum):
    """Quality gate execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class QualityGateResult:
    """Results from quality gate execution."""
    name: str
    status: QualityGateStatus
    score: float = 0.0  # 0-100 score
    threshold: float = 0.0
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'status': self.status.value,
            'score': self.score,
            'threshold': self.threshold,
            'message': self.message,
            'details': self.details,
            'execution_time': self.execution_time,
            'passed': self.status == QualityGateStatus.PASSED
        }


class QualityGate:
    """Base class for quality gates."""
    
    def __init__(self, name: str, threshold: float = 80.0,
                 is_required: bool = True,
                 logger: Optional[logging.Logger] = None):
        self.name = name
        self.threshold = threshold
        self.is_required = is_required
        self.logger = logger or get_logger(f"quality_gate.{name}")
    
    def execute(self, context: Dict[str, Any]) -> QualityGateResult:
        """Execute the quality gate."""
        start_time = time.time()
        
        try:
            self.logger.info(f"Executing quality gate: {self.name}")
            
            # Run the quality check
            score, details = self._check_quality(context)
            
            # Determine status
            if score >= self.threshold:
                status = QualityGateStatus.PASSED
                message = f"Quality gate passed with score {score:.1f} (threshold: {self.threshold})"
            elif self.is_required:
                status = QualityGateStatus.FAILED
                message = f"Quality gate failed with score {score:.1f} (threshold: {self.threshold})"
            else:
                status = QualityGateStatus.WARNING
                message = f"Quality gate warning with score {score:.1f} (threshold: {self.threshold})"
            
            result = QualityGateResult(
                name=self.name,
                status=status,
                score=score,
                threshold=self.threshold,
                message=message,
                details=details,
                execution_time=time.time() - start_time
            )
            
            self.logger.info(f"Quality gate {self.name}: {status.value} (score: {score:.1f})")
            return result
            
        except Exception as e:
            self.logger.error(f"Quality gate {self.name} failed with error: {e}")
            return QualityGateResult(
                name=self.name,
                status=QualityGateStatus.FAILED,
                score=0.0,
                threshold=self.threshold,
                message=f"Quality gate error: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    def _check_quality(self, context: Dict[str, Any]) -> tuple[float, Dict[str, Any]]:
        """Check quality and return score and details. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _check_quality")


class TestCoverageGate(QualityGate):
    """Quality gate for test coverage."""
    
    def __init__(self, minimum_coverage: float = 80.0, **kwargs):
        super().__init__("test_coverage", threshold=minimum_coverage, **kwargs)
        self.minimum_coverage = minimum_coverage
    
    def _check_quality(self, context: Dict[str, Any]) -> tuple[float, Dict[str, Any]]:
        """Check test coverage quality."""
        test_results = context.get('test_results', {})
        
        if not test_results:
            return 0.0, {'error': 'No test results provided'}
        
        # Calculate coverage metrics
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        
        for suite_results in test_results.values():
            for result in suite_results.values():
                total_tests += 1
                if result.status == TestStatus.PASSED:
                    passed_tests += 1
                elif result.status == TestStatus.FAILED:
                    failed_tests += 1
        
        if total_tests == 0:
            coverage_score = 0.0
        else:
            coverage_score = (passed_tests / total_tests) * 100
        
        details = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'coverage_percentage': coverage_score,
            'missing_coverage': max(0, self.minimum_coverage - coverage_score)
        }
        
        return coverage_score, details


class PerformanceGate(QualityGate):
    """Quality gate for performance metrics."""
    
    def __init__(self, max_inference_time: float = 1.0,
                 max_memory_mb: float = 1000.0, **kwargs):
        super().__init__("performance", threshold=70.0, **kwargs)
        self.max_inference_time = max_inference_time
        self.max_memory_mb = max_memory_mb
    
    def _check_quality(self, context: Dict[str, Any]) -> tuple[float, Dict[str, Any]]:
        """Check performance quality."""
        performance_results = context.get('performance_results', {})
        
        if not performance_results:
            return 0.0, {'error': 'No performance results provided'}
        
        # Calculate performance score
        scores = []
        details = {}
        
        # Check inference time
        inference_time = performance_results.get('avg_inference_time', float('inf'))
        if inference_time <= self.max_inference_time:
            time_score = 100.0
        else:
            time_score = max(0, 100 * (self.max_inference_time / inference_time))
        
        scores.append(time_score)
        details['inference_time'] = {
            'actual': inference_time,
            'threshold': self.max_inference_time,
            'score': time_score
        }
        
        # Check memory usage
        memory_usage = performance_results.get('memory_usage', float('inf'))
        if memory_usage <= self.max_memory_mb:
            memory_score = 100.0
        else:
            memory_score = max(0, 100 * (self.max_memory_mb / memory_usage))
        
        scores.append(memory_score)
        details['memory_usage'] = {
            'actual': memory_usage,
            'threshold': self.max_memory_mb,
            'score': memory_score
        }
        
        # Overall performance score
        overall_score = sum(scores) / len(scores) if scores else 0.0
        details['overall_score'] = overall_score
        
        return overall_score, details


class SecurityGate(QualityGate):
    """Quality gate for security checks."""
    
    def __init__(self, max_vulnerabilities: int = 0, **kwargs):
        super().__init__("security", threshold=90.0, **kwargs)
        self.max_vulnerabilities = max_vulnerabilities
    
    def _check_quality(self, context: Dict[str, Any]) -> tuple[float, Dict[str, Any]]:
        """Check security quality."""
        security_results = context.get('security_results', {})
        
        # Count vulnerabilities by severity
        critical = security_results.get('critical_vulnerabilities', 0)
        high = security_results.get('high_vulnerabilities', 0)
        medium = security_results.get('medium_vulnerabilities', 0)
        low = security_results.get('low_vulnerabilities', 0)
        
        # Calculate weighted vulnerability score
        vulnerability_score = (critical * 10) + (high * 5) + (medium * 2) + (low * 1)
        
        # Convert to quality score (lower vulnerabilities = higher score)
        if vulnerability_score == 0:
            security_score = 100.0
        else:
            # Penalize based on severity
            security_score = max(0, 100 - vulnerability_score)
        
        details = {
            'critical_vulnerabilities': critical,
            'high_vulnerabilities': high,
            'medium_vulnerabilities': medium,
            'low_vulnerabilities': low,
            'total_vulnerabilities': critical + high + medium + low,
            'vulnerability_score': vulnerability_score,
            'security_score': security_score
        }
        
        return security_score, details


class CodeQualityGate(QualityGate):
    """Quality gate for code quality metrics."""
    
    def __init__(self, max_complexity: int = 10,
                 min_documentation: float = 70.0, **kwargs):
        super().__init__("code_quality", threshold=80.0, **kwargs)
        self.max_complexity = max_complexity
        self.min_documentation = min_documentation
    
    def _check_quality(self, context: Dict[str, Any]) -> tuple[float, Dict[str, Any]]:
        """Check code quality."""
        code_metrics = context.get('code_metrics', {})
        
        scores = []
        details = {}
        
        # Check complexity
        avg_complexity = code_metrics.get('average_complexity', 0)
        if avg_complexity <= self.max_complexity:
            complexity_score = 100.0
        else:
            complexity_score = max(0, 100 * (self.max_complexity / avg_complexity))
        
        scores.append(complexity_score)
        details['complexity'] = {
            'actual': avg_complexity,
            'threshold': self.max_complexity,
            'score': complexity_score
        }
        
        # Check documentation
        doc_coverage = code_metrics.get('documentation_coverage', 0)
        if doc_coverage >= self.min_documentation:
            doc_score = 100.0
        else:
            doc_score = max(0, (doc_coverage / self.min_documentation) * 100)
        
        scores.append(doc_score)
        details['documentation'] = {
            'actual': doc_coverage,
            'threshold': self.min_documentation,
            'score': doc_score
        }
        
        # Overall code quality score
        overall_score = sum(scores) / len(scores) if scores else 0.0
        details['overall_score'] = overall_score
        
        return overall_score, details


class ModelQualityGate(QualityGate):
    """Quality gate for ML model quality."""
    
    def __init__(self, min_accuracy: float = 0.8,
                 max_bias: float = 0.1, **kwargs):
        super().__init__("model_quality", threshold=85.0, **kwargs)
        self.min_accuracy = min_accuracy
        self.max_bias = max_bias
    
    def _check_quality(self, context: Dict[str, Any]) -> tuple[float, Dict[str, Any]]:
        """Check model quality."""
        model_metrics = context.get('model_metrics', {})
        
        scores = []
        details = {}
        
        # Check accuracy
        accuracy = model_metrics.get('accuracy', 0.0)
        if accuracy >= self.min_accuracy:
            accuracy_score = 100.0
        else:
            accuracy_score = (accuracy / self.min_accuracy) * 100
        
        scores.append(accuracy_score)
        details['accuracy'] = {
            'actual': accuracy,
            'threshold': self.min_accuracy,
            'score': accuracy_score
        }
        
        # Check bias (if available)
        bias = model_metrics.get('bias', 0.0)
        if bias <= self.max_bias:
            bias_score = 100.0
        else:
            bias_score = max(0, 100 * (self.max_bias / bias))
        
        scores.append(bias_score)
        details['bias'] = {
            'actual': bias,
            'threshold': self.max_bias,
            'score': bias_score
        }
        
        # Check other metrics
        precision = model_metrics.get('precision', 0.0)
        recall = model_metrics.get('recall', 0.0)
        f1_score = model_metrics.get('f1_score', 0.0)
        
        if precision > 0:
            scores.append(precision * 100)
            details['precision'] = precision
        
        if recall > 0:
            scores.append(recall * 100)
            details['recall'] = recall
        
        if f1_score > 0:
            scores.append(f1_score * 100)
            details['f1_score'] = f1_score
        
        # Overall model quality score
        overall_score = sum(scores) / len(scores) if scores else 0.0
        details['overall_score'] = overall_score
        
        return overall_score, details


class QualityGateManager:
    """Manages and executes quality gates."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger("quality_gate_manager")
        self.gates = {}
        self.execution_results = {}
        
        # Default gates
        self._register_default_gates()
    
    def _register_default_gates(self):
        """Register default quality gates."""
        self.register_gate(TestCoverageGate())
        self.register_gate(PerformanceGate())
        self.register_gate(SecurityGate())
        self.register_gate(CodeQualityGate())
        self.register_gate(ModelQualityGate())
    
    def register_gate(self, gate: QualityGate):
        """Register a quality gate."""
        self.gates[gate.name] = gate
        self.logger.info(f"Registered quality gate: {gate.name}")
    
    def execute_gate(self, gate_name: str, context: Dict[str, Any]) -> QualityGateResult:
        """Execute a specific quality gate."""
        if gate_name not in self.gates:
            raise ValueError(f"Quality gate '{gate_name}' not found")
        
        gate = self.gates[gate_name]
        result = gate.execute(context)
        
        self.execution_results[gate_name] = result
        return result
    
    def execute_all_gates(self, context: Dict[str, Any],
                         stop_on_failure: bool = False) -> Dict[str, QualityGateResult]:
        """Execute all registered quality gates."""
        self.logger.info(f"Executing {len(self.gates)} quality gates")
        
        results = {}
        
        for gate_name, gate in self.gates.items():
            result = gate.execute(context)
            results[gate_name] = result
            
            # Stop on failure if required gate fails
            if (stop_on_failure and 
                gate.is_required and 
                result.status == QualityGateStatus.FAILED):
                self.logger.error(f"Stopping execution due to failed required gate: {gate_name}")
                break
        
        self.execution_results = results
        
        # Log summary
        passed = sum(1 for r in results.values() if r.status == QualityGateStatus.PASSED)
        failed = sum(1 for r in results.values() if r.status == QualityGateStatus.FAILED)
        warnings = sum(1 for r in results.values() if r.status == QualityGateStatus.WARNING)
        
        self.logger.info(f"Quality gates completed: {passed} passed, {failed} failed, {warnings} warnings")
        
        return results
    
    def generate_quality_report(self, output_path: str = None) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        if not self.execution_results:
            self.logger.warning("No quality gate results available")
            return {}
        
        # Calculate overall quality score
        total_score = 0.0
        required_gates = 0
        optional_gates = 0
        
        gate_results = {}
        
        for gate_name, result in self.execution_results.items():
            gate = self.gates[gate_name]
            gate_results[gate_name] = result.to_dict()
            
            if gate.is_required:
                total_score += result.score
                required_gates += 1
            else:
                optional_gates += 1
        
        overall_score = total_score / max(1, required_gates)
        
        # Determine overall status
        failed_required = any(
            r.status == QualityGateStatus.FAILED and self.gates[name].is_required
            for name, r in self.execution_results.items()
        )
        
        if failed_required:
            overall_status = "FAILED"
        elif overall_score >= 90:
            overall_status = "EXCELLENT"
        elif overall_score >= 80:
            overall_status = "GOOD"
        elif overall_score >= 70:
            overall_status = "FAIR"
        else:
            overall_status = "POOR"
        
        report = {
            'timestamp': time.time(),
            'overall_status': overall_status,
            'overall_score': overall_score,
            'summary': {
                'total_gates': len(self.execution_results),
                'required_gates': required_gates,
                'optional_gates': optional_gates,
                'passed': sum(1 for r in self.execution_results.values() 
                             if r.status == QualityGateStatus.PASSED),
                'failed': sum(1 for r in self.execution_results.values() 
                             if r.status == QualityGateStatus.FAILED),
                'warnings': sum(1 for r in self.execution_results.values() 
                               if r.status == QualityGateStatus.WARNING)
            },
            'gates': gate_results
        }
        
        # Save report if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Quality report saved to {output_path}")
        
        return report
    
    def get_failed_gates(self) -> List[str]:
        """Get list of failed quality gates."""
        return [
            name for name, result in self.execution_results.items()
            if result.status == QualityGateStatus.FAILED
        ]
    
    def is_quality_acceptable(self) -> bool:
        """Check if overall quality is acceptable."""
        # All required gates must pass
        for gate_name, result in self.execution_results.items():
            gate = self.gates[gate_name]
            if gate.is_required and result.status == QualityGateStatus.FAILED:
                return False
        
        return True
    
    def create_custom_gate(self, name: str, check_function: Callable,
                          threshold: float = 80.0, is_required: bool = True) -> QualityGate:
        """Create a custom quality gate from a function."""
        
        class CustomQualityGate(QualityGate):
            def _check_quality(self, context: Dict[str, Any]) -> tuple[float, Dict[str, Any]]:
                return check_function(context)
        
        gate = CustomQualityGate(name, threshold, is_required, self.logger)
        self.register_gate(gate)
        
        return gate


def create_quality_dashboard(manager: QualityGateManager, output_dir: str) -> str:
    """Create HTML quality dashboard."""
    if not manager.execution_results:
        return ""
    
    report = manager.generate_quality_report()
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Quality Gates Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background: #007bff; color: white; padding: 15px; border-radius: 5px; }}
            .gate {{ margin: 15px 0; padding: 15px; border-radius: 5px; }}
            .passed {{ background: #d4edda; border: 1px solid #c3e6cb; }}
            .failed {{ background: #f8d7da; border: 1px solid #f5c6cb; }}
            .warning {{ background: #fff3cd; border: 1px solid #ffeaa7; }}
            .score {{ font-size: 24px; font-weight: bold; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Quality Gates Dashboard</h1>
            <p>Overall Status: <strong>{report['overall_status']}</strong></p>
            <p>Overall Score: <span class="score">{report['overall_score']:.1f}/100</span></p>
            <p>Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <h2>Summary</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Gates</td><td>{report['summary']['total_gates']}</td></tr>
            <tr><td>Passed</td><td>{report['summary']['passed']}</td></tr>
            <tr><td>Failed</td><td>{report['summary']['failed']}</td></tr>
            <tr><td>Warnings</td><td>{report['summary']['warnings']}</td></tr>
        </table>
        
        <h2>Quality Gates</h2>
    """
    
    for gate_name, gate_result in report['gates'].items():
        status = gate_result['status']
        css_class = status if status in ['passed', 'failed', 'warning'] else 'failed'
        
        html_content += f"""
        <div class="gate {css_class}">
            <h3>{gate_name.replace('_', ' ').title()}</h3>
            <p><strong>Status:</strong> {status.upper()}</p>
            <p><strong>Score:</strong> {gate_result['score']:.1f}/{gate_result['threshold']:.1f}</p>
            <p><strong>Message:</strong> {gate_result['message']}</p>
            <p><strong>Execution Time:</strong> {gate_result['execution_time']:.2f}s</p>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    output_path = Path(output_dir) / "quality_dashboard.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    return str(output_path)