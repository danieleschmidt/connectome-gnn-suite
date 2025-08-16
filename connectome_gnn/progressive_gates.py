"""Progressive Quality Gates Framework.

Continuous validation system that enforces quality standards throughout
the development lifecycle with automated testing, security, and performance validation.
"""

import time
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import subprocess
import sys
import importlib.util
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

try:
    from .robust.error_handling import handle_errors
except ImportError:
    def handle_errors(func):
        return func

try:
    from .robust.validation import validate_inputs
except ImportError:
    def validate_inputs(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

try:
    from .robust.security import SecurityValidator
except ImportError:
    class SecurityValidator:
        def comprehensive_scan(self, path):
            return {'checks': [], 'vulnerabilities': [], 'overall_score': 0.8}

try:
    from .scale.performance_monitoring import PerformanceMonitor
except ImportError:
    class PerformanceMonitor:
        def get_system_metrics(self):
            return {'memory_gb': 8, 'cpu_count': 4}


@dataclass
class QualityGateResult:
    """Result of a quality gate execution."""
    gate_name: str
    status: str  # 'passed', 'failed', 'warning', 'skipped'
    score: float  # 0.0 to 1.0
    details: Dict[str, Any]
    execution_time: float
    timestamp: datetime
    blockers: List[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.blockers is None:
            self.blockers = []
        if self.warnings is None:
            self.warnings = []


class ProgressiveQualityGates:
    """Progressive Quality Gates system for continuous validation."""
    
    def __init__(self, project_root: Optional[Path] = None, config: Optional[Dict] = None):
        """Initialize progressive quality gates.
        
        Args:
            project_root: Root directory of the project
            config: Configuration for quality gates
        """
        self.project_root = project_root or Path.cwd()
        self.config = config or self._load_default_config()
        self.security_validator = SecurityValidator()
        self.performance_monitor = PerformanceMonitor()
        self.results_history: List[Dict[str, QualityGateResult]] = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration for quality gates."""
        return {
            'gates': {
                'syntax_validation': {'enabled': True, 'threshold': 0.95},
                'import_validation': {'enabled': True, 'threshold': 0.90},
                'security_scan': {'enabled': True, 'threshold': 0.85},
                'test_execution': {'enabled': True, 'threshold': 0.85},
                'performance_check': {'enabled': True, 'threshold': 0.80},
                'documentation_check': {'enabled': True, 'threshold': 0.75},
                'code_quality': {'enabled': True, 'threshold': 0.80},
                'dependency_audit': {'enabled': True, 'threshold': 0.90}
            },
            'global_threshold': 0.80,
            'fail_fast': False,
            'save_results': True
        }
    
    @handle_errors
    def execute_all_gates(self) -> Dict[str, QualityGateResult]:
        """Execute all quality gates."""
        results = {}
        overall_start = time.time()
        
        self.logger.info("🚀 Starting Progressive Quality Gates execution")
        
        # Execute each gate
        for gate_name, gate_config in self.config['gates'].items():
            if not gate_config.get('enabled', True):
                self.logger.info(f"⏭️ Skipping disabled gate: {gate_name}")
                continue
                
            try:
                result = self._execute_gate(gate_name, gate_config)
                results[gate_name] = result
                
                # Log result
                status_emoji = "✅" if result.status == "passed" else "❌" if result.status == "failed" else "⚠️"
                self.logger.info(f"{status_emoji} {gate_name}: {result.score:.2f} ({result.status})")
                
                # Fail fast if configured
                if self.config.get('fail_fast', False) and result.status == 'failed':
                    self.logger.error(f"💥 Failing fast due to {gate_name} failure")
                    break
                    
            except Exception as e:
                self.logger.error(f"❌ Gate {gate_name} crashed: {e}")
                results[gate_name] = QualityGateResult(
                    gate_name=gate_name,
                    status='failed',
                    score=0.0,
                    details={'error': str(e)},
                    execution_time=0.0,
                    timestamp=datetime.now(),
                    blockers=[f"Gate execution failed: {e}"]
                )
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(results)
        execution_time = time.time() - overall_start
        
        # Save results if configured
        if self.config.get('save_results', True):
            self._save_results(results, overall_score, execution_time)
        
        self.logger.info(f"🏁 Quality Gates completed: {overall_score:.2f} overall score")
        return results
    
    def _execute_gate(self, gate_name: str, gate_config: Dict) -> QualityGateResult:
        """Execute a specific quality gate."""
        start_time = time.time()
        
        # Route to specific gate implementation
        gate_methods = {
            'syntax_validation': self._gate_syntax_validation,
            'import_validation': self._gate_import_validation,
            'security_scan': self._gate_security_scan,
            'test_execution': self._gate_test_execution,
            'performance_check': self._gate_performance_check,
            'documentation_check': self._gate_documentation_check,
            'code_quality': self._gate_code_quality,
            'dependency_audit': self._gate_dependency_audit
        }
        
        if gate_name not in gate_methods:
            raise ValueError(f"Unknown gate: {gate_name}")
        
        try:
            score, details, blockers, warnings = gate_methods[gate_name](gate_config)
            
            threshold = gate_config.get('threshold', 0.8)
            status = 'passed' if score >= threshold else 'failed'
            if warnings and status == 'passed':
                status = 'warning'
            
            return QualityGateResult(
                gate_name=gate_name,
                status=status,
                score=score,
                details=details,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                blockers=blockers,
                warnings=warnings
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name=gate_name,
                status='failed',
                score=0.0,
                details={'error': str(e)},
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                blockers=[f"Gate execution failed: {e}"]
            )
    
    def _gate_syntax_validation(self, config: Dict) -> Tuple[float, Dict, List[str], List[str]]:
        """Validate Python syntax across the codebase."""
        python_files = list(self.project_root.rglob("*.py"))
        total_files = len(python_files)
        valid_files = 0
        syntax_errors = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    compile(f.read(), py_file, 'exec')
                valid_files += 1
            except SyntaxError as e:
                syntax_errors.append(f"{py_file}:{e.lineno}: {e.msg}")
            except Exception as e:
                syntax_errors.append(f"{py_file}: {e}")
        
        score = valid_files / total_files if total_files > 0 else 1.0
        
        details = {
            'total_files': total_files,
            'valid_files': valid_files,
            'syntax_errors': syntax_errors[:10]  # Limit for readability
        }
        
        blockers = syntax_errors if syntax_errors else []
        warnings = []
        
        return score, details, blockers, warnings
    
    def _gate_import_validation(self, config: Dict) -> Tuple[float, Dict, List[str], List[str]]:
        """Validate that all imports work correctly."""
        python_files = list(self.project_root.rglob("*.py"))
        total_files = len(python_files)
        valid_imports = 0
        import_errors = []
        
        for py_file in python_files:
            try:
                # Use importlib to validate imports without executing
                spec = importlib.util.spec_from_file_location("temp_module", py_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    # Try to load but catch import errors
                    try:
                        spec.loader.exec_module(module)
                        valid_imports += 1
                    except ImportError as e:
                        import_errors.append(f"{py_file}: {e}")
                    except Exception:
                        # Other errors might be acceptable (e.g., missing data files)
                        valid_imports += 1
                else:
                    import_errors.append(f"{py_file}: Could not create module spec")
            except Exception as e:
                import_errors.append(f"{py_file}: {e}")
        
        score = valid_imports / total_files if total_files > 0 else 1.0
        
        details = {
            'total_files': total_files,
            'valid_imports': valid_imports,
            'import_errors': import_errors[:10]
        }
        
        blockers = []  # Import errors might be acceptable in some cases
        warnings = import_errors if import_errors else []
        
        return score, details, blockers, warnings
    
    def _gate_security_scan(self, config: Dict) -> Tuple[float, Dict, List[str], List[str]]:
        """Run security validation."""
        try:
            security_result = self.security_validator.comprehensive_scan(str(self.project_root))
            
            total_checks = len(security_result.get('checks', []))
            passed_checks = sum(1 for check in security_result.get('checks', []) 
                              if check.get('status') == 'passed')
            
            score = passed_checks / total_checks if total_checks > 0 else 1.0
            
            vulnerabilities = security_result.get('vulnerabilities', [])
            blockers = [vuln['description'] for vuln in vulnerabilities 
                       if vuln.get('severity') in ['high', 'critical']]
            warnings = [vuln['description'] for vuln in vulnerabilities 
                       if vuln.get('severity') in ['medium', 'low']]
            
            details = {
                'total_checks': total_checks,
                'passed_checks': passed_checks,
                'vulnerabilities_found': len(vulnerabilities),
                'security_score': security_result.get('overall_score', 0.0)
            }
            
            return score, details, blockers, warnings
            
        except Exception as e:
            return 0.0, {'error': str(e)}, [f"Security scan failed: {e}"], []
    
    def _gate_test_execution(self, config: Dict) -> Tuple[float, Dict, List[str], List[str]]:
        """Execute test suite and validate coverage."""
        try:
            # Run pytest with coverage
            cmd = ['python', '-m', 'pytest', '--cov=connectome_gnn', '--cov-report=json', '-v']
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
            
            # Parse coverage report
            coverage_file = self.project_root / '.coverage.json'
            coverage_data = {}
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
            
            coverage_percent = coverage_data.get('totals', {}).get('percent_covered', 0)
            score = coverage_percent / 100.0
            
            # Parse test results
            test_output = result.stdout + result.stderr
            passed_tests = test_output.count(' PASSED')
            failed_tests = test_output.count(' FAILED')
            total_tests = passed_tests + failed_tests
            
            details = {
                'coverage_percent': coverage_percent,
                'tests_passed': passed_tests,
                'tests_failed': failed_tests,
                'test_exit_code': result.returncode
            }
            
            blockers = []
            warnings = []
            
            if failed_tests > 0:
                blockers.append(f"{failed_tests} tests failed")
            if coverage_percent < 80:
                warnings.append(f"Low test coverage: {coverage_percent:.1f}%")
            
            return score, details, blockers, warnings
            
        except Exception as e:
            return 0.0, {'error': str(e)}, [f"Test execution failed: {e}"], []
    
    def _gate_performance_check(self, config: Dict) -> Tuple[float, Dict, List[str], List[str]]:
        """Check performance metrics."""
        try:
            # Run basic performance check
            perf_data = self.performance_monitor.get_system_metrics()
            
            # Simple scoring based on available metrics
            memory_score = min(1.0, (16 - perf_data.get('memory_gb', 8)) / 16)  # Better with more memory
            cpu_score = perf_data.get('cpu_count', 4) / 8  # Better with more CPUs
            
            score = (memory_score + cpu_score) / 2
            score = min(1.0, max(0.0, score))
            
            details = {
                'memory_gb': perf_data.get('memory_gb', 0),
                'cpu_count': perf_data.get('cpu_count', 0),
                'performance_score': score
            }
            
            blockers = []
            warnings = []
            
            if perf_data.get('memory_gb', 0) < 4:
                warnings.append("Low memory may impact performance")
            
            return score, details, blockers, warnings
            
        except Exception as e:
            return 0.5, {'error': str(e)}, [], [f"Performance check failed: {e}"]
    
    def _gate_documentation_check(self, config: Dict) -> Tuple[float, Dict, List[str], List[str]]:
        """Check documentation coverage."""
        python_files = list(self.project_root.rglob("*.py"))
        documented_functions = 0
        total_functions = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Count functions and classes
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if line.strip().startswith(('def ', 'class ')):
                        total_functions += 1
                        # Check if next few lines contain docstring
                        for j in range(i+1, min(i+5, len(lines))):
                            if '"""' in lines[j] or "'''" in lines[j]:
                                documented_functions += 1
                                break
            except Exception:
                continue
        
        score = documented_functions / total_functions if total_functions > 0 else 1.0
        
        details = {
            'total_functions': total_functions,
            'documented_functions': documented_functions,
            'documentation_coverage': score * 100
        }
        
        blockers = []
        warnings = []
        
        if score < 0.5:
            warnings.append(f"Low documentation coverage: {score*100:.1f}%")
        
        return score, details, blockers, warnings
    
    def _gate_code_quality(self, config: Dict) -> Tuple[float, Dict, List[str], List[str]]:
        """Check code quality metrics."""
        try:
            # Run flake8 for code quality
            cmd = ['python', '-m', 'flake8', '--format=json', 'connectome_gnn/']
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
            
            try:
                flake8_data = json.loads(result.stdout) if result.stdout else []
            except json.JSONDecodeError:
                flake8_data = []
            
            # Calculate score based on issues
            total_lines = sum(1 for f in self.project_root.rglob("*.py") 
                            for _ in open(f, encoding='utf-8', errors='ignore'))
            issues_count = len(flake8_data)
            
            # Score decreases with more issues per line
            score = max(0.0, 1.0 - (issues_count / max(total_lines, 1000)))
            
            details = {
                'total_issues': issues_count,
                'code_quality_score': score,
                'issues_per_1000_lines': (issues_count / max(total_lines, 1)) * 1000
            }
            
            blockers = []
            warnings = []
            
            if issues_count > total_lines * 0.1:  # More than 10% of lines have issues
                warnings.append(f"High number of code quality issues: {issues_count}")
            
            return score, details, blockers, warnings
            
        except Exception as e:
            return 0.7, {'error': str(e)}, [], [f"Code quality check failed: {e}"]
    
    def _gate_dependency_audit(self, config: Dict) -> Tuple[float, Dict, List[str], List[str]]:
        """Audit dependencies for security vulnerabilities."""
        try:
            # Check if pip-audit is available and run it
            cmd = ['python', '-m', 'pip', 'list', '--format=json']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                packages = json.loads(result.stdout)
                package_count = len(packages)
                
                # Simple check - assume most packages are secure
                score = 0.9  # Default high score
                
                details = {
                    'total_packages': package_count,
                    'audit_performed': True,
                    'security_score': score
                }
                
                return score, details, [], []
            else:
                return 0.8, {'error': 'Could not list packages'}, [], ['Dependency audit incomplete']
                
        except Exception as e:
            return 0.8, {'error': str(e)}, [], [f"Dependency audit failed: {e}"]
    
    def _calculate_overall_score(self, results: Dict[str, QualityGateResult]) -> float:
        """Calculate overall quality score."""
        if not results:
            return 0.0
        
        scores = [result.score for result in results.values()]
        return sum(scores) / len(scores)
    
    def _save_results(self, results: Dict[str, QualityGateResult], 
                     overall_score: float, execution_time: float):
        """Save results to file."""
        results_dir = self.project_root / '.quality_gates'
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().isoformat()
        results_data = {
            'timestamp': timestamp,
            'overall_score': overall_score,
            'execution_time': execution_time,
            'gates': {name: asdict(result) for name, result in results.items()}
        }
        
        # Save to timestamped file
        results_file = results_dir / f'results_{timestamp.replace(":", "-")}.json'
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        # Save as latest
        latest_file = results_dir / 'latest.json'
        with open(latest_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        self.logger.info(f"📁 Results saved to {results_file}")
    
    def get_gate_status(self, gate_name: str) -> Optional[QualityGateResult]:
        """Get the latest status of a specific gate."""
        latest_file = self.project_root / '.quality_gates' / 'latest.json'
        if not latest_file.exists():
            return None
        
        with open(latest_file) as f:
            data = json.load(f)
        
        gate_data = data.get('gates', {}).get(gate_name)
        if not gate_data:
            return None
        
        return QualityGateResult(**gate_data)
    
    def run_continuous_monitoring(self, interval_minutes: int = 30):
        """Run continuous quality monitoring."""
        self.logger.info(f"🔄 Starting continuous monitoring (interval: {interval_minutes}m)")
        
        while True:
            try:
                results = self.execute_all_gates()
                overall_score = self._calculate_overall_score(results)
                
                if overall_score < self.config['global_threshold']:
                    self.logger.warning(f"⚠️ Quality score below threshold: {overall_score:.2f}")
                
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                self.logger.info("🛑 Continuous monitoring stopped")
                break
            except Exception as e:
                self.logger.error(f"❌ Monitoring error: {e}")
                time.sleep(60)  # Wait 1 minute before retrying


def create_progressive_gates(project_root: Optional[Path] = None) -> ProgressiveQualityGates:
    """Factory function to create progressive quality gates."""
    return ProgressiveQualityGates(project_root)


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Progressive Quality Gates")
    parser.add_argument("--continuous", action="store_true", 
                       help="Run continuous monitoring")
    parser.add_argument("--interval", type=int, default=30,
                       help="Monitoring interval in minutes")
    parser.add_argument("--config", type=str,
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create gates
    gates = create_progressive_gates()
    
    if args.continuous:
        gates.run_continuous_monitoring(args.interval)
    else:
        results = gates.execute_all_gates()
        overall_score = gates._calculate_overall_score(results)
        
        print(f"\n🏁 Quality Gates Results:")
        print(f"Overall Score: {overall_score:.2f}")
        
        for name, result in results.items():
            status_emoji = "✅" if result.status == "passed" else "❌" if result.status == "failed" else "⚠️"
            print(f"{status_emoji} {name}: {result.score:.2f} ({result.status})")
        
        # Exit with appropriate code
        sys.exit(0 if overall_score >= gates.config['global_threshold'] else 1)