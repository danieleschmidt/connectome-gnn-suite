"""TERRAGON Integration Tests.

Comprehensive integration testing to validate that all TERRAGON components
work together correctly for autonomous SDLC execution.
"""

import asyncio
import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import logging

# Import TERRAGON components
from connectome_gnn.progressive_gates import ProgressiveQualityGates, create_progressive_gates
from connectome_gnn.autonomous_workflow import AutonomousWorkflowEngine, create_autonomous_workflow, WorkflowPhase, TaskPriority
from connectome_gnn.adaptive_learning import AdaptiveLearningSystem, create_adaptive_learning_system
from connectome_gnn.terragon_orchestrator import TERRAGONMasterOrchestrator, create_terragon_orchestrator, OrchestrationMode


class TestTERRAGONIntegration:
    """Integration tests for TERRAGON framework."""
    
    @pytest.fixture
    def temp_project(self):
        """Create temporary project directory for testing."""
        temp_dir = tempfile.mkdtemp()
        project_root = Path(temp_dir)
        
        # Create basic project structure
        (project_root / 'connectome_gnn').mkdir()
        (project_root / 'connectome_gnn' / '__init__.py').touch()
        (project_root / 'tests').mkdir()
        (project_root / 'tests' / '__init__.py').touch()
        
        # Create basic files
        (project_root / 'README.md').write_text("# Test Project")
        (project_root / 'pyproject.toml').write_text("""
[project]
name = "test-project"
version = "0.1.0"
""")
        
        yield project_root
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_progressive_gates_creation(self, temp_project):
        """Test Progressive Quality Gates creation and basic functionality."""
        gates = create_progressive_gates(temp_project)
        
        assert gates is not None
        assert gates.project_root == temp_project
        assert isinstance(gates.config, dict)
        
        # Test gate execution
        results = gates.execute_all_gates()
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Check that each result has required fields
        for gate_name, result in results.items():
            assert hasattr(result, 'gate_name')
            assert hasattr(result, 'status')
            assert hasattr(result, 'score')
            assert 0.0 <= result.score <= 1.0
    
    def test_autonomous_workflow_creation(self, temp_project):
        """Test Autonomous Workflow Engine creation and task management."""
        engine = create_autonomous_workflow(temp_project)
        
        assert engine is not None
        assert engine.project_root == temp_project
        
        # Test standard workflow definition
        tasks = engine.define_standard_workflow()
        assert isinstance(tasks, dict)
        assert len(tasks) > 0
        
        # Check task structure
        for task_id, task in tasks.items():
            assert hasattr(task, 'id')
            assert hasattr(task, 'name')
            assert hasattr(task, 'phase')
            assert hasattr(task, 'priority')
            assert isinstance(task.phase, WorkflowPhase)
            assert isinstance(task.priority, TaskPriority)
        
        # Test ready tasks detection
        ready_tasks = engine.get_ready_tasks()
        assert isinstance(ready_tasks, list)
        
        # Test task prioritization
        prioritized = engine.prioritize_tasks(ready_tasks)
        assert isinstance(prioritized, list)
    
    def test_adaptive_learning_creation(self, temp_project):
        """Test Adaptive Learning System creation and basic learning."""
        learning_system = create_adaptive_learning_system(temp_project)
        
        assert learning_system is not None
        assert learning_system.project_root == temp_project
        
        # Test execution recording
        execution_data = {
            'task_type': 'testing',
            'duration': 45.0,
            'success': True,
            'complexity': 'medium'
        }
        
        learning_system.record_execution(execution_data)
        assert len(learning_system.execution_history) == 1
        
        # Test learning insights
        insights = learning_system.get_learning_insights()
        assert isinstance(insights, dict)
        assert 'total_execution_records' in insights
        assert insights['total_execution_records'] >= 1
    
    def test_terragon_orchestrator_creation(self, temp_project):
        """Test TERRAGON Master Orchestrator creation and status."""
        orchestrator = create_terragon_orchestrator(temp_project)
        
        assert orchestrator is not None
        assert orchestrator.project_root == temp_project
        assert orchestrator.quality_gates is not None
        assert orchestrator.workflow_engine is not None
        assert orchestrator.learning_system is not None
        
        # Test status retrieval
        status = orchestrator.get_orchestration_status()
        assert isinstance(status, dict)
        assert 'is_running' in status
        assert 'state' in status
        assert 'components_status' in status
    
    def test_component_integration_quality_workflow(self, temp_project):
        """Test integration between Quality Gates and Workflow Engine."""
        gates = create_progressive_gates(temp_project)
        engine = create_autonomous_workflow(temp_project)
        
        # Execute quality gates
        quality_results = gates.execute_all_gates()
        overall_score = gates._calculate_overall_score(quality_results)
        
        # Define workflow
        engine.define_standard_workflow()
        
        # Test that workflow can use quality results
        assert isinstance(overall_score, float)
        assert 0.0 <= overall_score <= 1.0
        
        # Check that quality gates can inform workflow decisions
        ready_tasks = engine.get_ready_tasks()
        assert isinstance(ready_tasks, list)
    
    def test_component_integration_workflow_learning(self, temp_project):
        """Test integration between Workflow Engine and Adaptive Learning."""
        engine = create_autonomous_workflow(temp_project)
        learning_system = create_adaptive_learning_system(temp_project)
        
        # Define workflow
        tasks = engine.define_standard_workflow()
        
        # Simulate task execution data
        for task_id, task in list(tasks.items())[:3]:  # First 3 tasks
            execution_data = {
                'task_id': task_id,
                'task_type': task.metadata.get('type', 'default'),
                'phase': task.phase.value,
                'duration': task.estimated_duration + 10,  # Slightly longer than estimate
                'success': True,
                'complexity': 'medium'
            }
            learning_system.record_execution(execution_data)
        
        # Test duration prediction
        context = {
            'task_type': 'testing',
            'complexity': 'medium',
            'phase': 'testing'
        }
        
        predicted_duration = learning_system.predict_task_duration(context)
        assert isinstance(predicted_duration, float)
        assert predicted_duration > 0
    
    def test_component_integration_learning_gates(self, temp_project):
        """Test integration between Adaptive Learning and Quality Gates."""
        learning_system = create_adaptive_learning_system(temp_project)
        gates = create_progressive_gates(temp_project)
        
        # Record some execution data with quality scores
        for i in range(5):
            execution_data = {
                'task_type': 'testing',
                'duration': 30 + i * 5,  # Varying durations
                'success': i < 4,  # One failure
                'quality_score': 0.8 + i * 0.02,  # Improving quality
                'complexity': 'medium'
            }
            learning_system.record_execution(execution_data)
        
        # Learn patterns
        patterns = learning_system.learn_patterns()
        assert isinstance(patterns, dict)
        
        # Test error prediction
        error_context = {'task_type': 'testing', 'complexity': 'medium'}
        error_prediction = learning_system.predict_potential_errors(error_context)
        assert isinstance(error_prediction, dict)
        assert 'risk_level' in error_prediction
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization_phase(self, temp_project):
        """Test orchestrator initialization phase."""
        orchestrator = create_terragon_orchestrator(temp_project)
        
        # Mock some methods to avoid actual execution
        with patch.object(orchestrator, '_validate_project_structure', return_value=True):
            with patch.object(orchestrator.quality_gates, 'execute_all_gates', return_value={}):
                with patch.object(orchestrator.quality_gates, '_calculate_overall_score', return_value=0.85):
                    
                    # Test initialization phase
                    await orchestrator._phase_initialization()
                    
                    # Check state transition
                    assert orchestrator.state.phase.value == 'planning'
                    assert orchestrator.state.quality_score == 0.85
    
    @pytest.mark.asyncio
    async def test_orchestrator_planning_phase(self, temp_project):
        """Test orchestrator planning phase."""
        orchestrator = create_terragon_orchestrator(temp_project)
        orchestrator.state.phase = orchestrator.state.phase.__class__('planning')
        
        # Mock workflow optimization methods
        with patch.object(orchestrator, '_optimize_task_estimates') as mock_optimize:
            with patch.object(orchestrator, '_apply_workflow_optimizations') as mock_apply:
                mock_optimize.return_value = None
                mock_apply.return_value = None
                
                # Test planning phase
                await orchestrator._phase_planning()
                
                # Check that methods were called
                mock_optimize.assert_called_once()
                mock_apply.assert_called_once()
                
                # Check state transition
                assert orchestrator.state.phase.value == 'execution'
    
    def test_workflow_task_optimization_with_learning(self, temp_project):
        """Test workflow task optimization using learning data."""
        engine = create_autonomous_workflow(temp_project)
        learning_system = create_adaptive_learning_system(temp_project)
        
        # Define workflow
        engine.define_standard_workflow()
        
        # Add learning data for specific task types
        for i in range(10):
            execution_data = {
                'task_type': 'coding',
                'complexity': 'medium',
                'phase': 'implementation',
                'duration': 90 + i * 5,  # 90-135 minutes
                'success': True
            }
            learning_system.record_execution(execution_data)
        
        # Learn patterns
        learning_system.learn_patterns()
        
        # Test prediction
        context = {
            'task_type': 'coding',
            'complexity': 'medium',
            'phase': 'implementation'
        }
        
        predicted_duration = learning_system.predict_task_duration(context)
        
        # Should be close to the average of our test data (112.5 minutes)
        assert 100 <= predicted_duration <= 125
    
    def test_quality_gates_comprehensive_execution(self, temp_project):
        """Test comprehensive quality gates execution."""
        gates = create_progressive_gates(temp_project)
        
        # Create some Python files to test
        (temp_project / 'connectome_gnn' / 'test_module.py').write_text('''
"""Test module for quality gates."""

def test_function():
    """Test function with docstring."""
    return True

class TestClass:
    """Test class with docstring."""
    
    def __init__(self):
        self.value = 42
    
    def get_value(self):
        """Get the value."""
        return self.value
''')
        
        # Execute all gates
        results = gates.execute_all_gates()
        
        # Check that we have results for expected gates
        expected_gates = [
            'syntax_validation',
            'import_validation', 
            'security_scan',
            'test_execution',
            'performance_check',
            'documentation_check',
            'code_quality',
            'dependency_audit'
        ]
        
        for gate in expected_gates:
            assert gate in results
            result = results[gate]
            assert hasattr(result, 'score')
            assert hasattr(result, 'status')
            assert result.status in ['passed', 'failed', 'warning', 'skipped']
    
    @pytest.mark.asyncio
    async def test_full_terragon_workflow_simulation(self, temp_project):
        """Simulate a complete TERRAGON workflow execution."""
        orchestrator = create_terragon_orchestrator(temp_project)
        
        # Mock external dependencies and long-running operations
        with patch.object(orchestrator, '_validate_project_structure', return_value=True):
            with patch.object(orchestrator.quality_gates, 'execute_all_gates') as mock_gates:
                with patch.object(orchestrator.workflow_engine, 'run_workflow') as mock_workflow:
                    
                    # Configure mocks
                    mock_gates.return_value = {
                        'syntax_validation': Mock(score=0.95, status='passed'),
                        'security_scan': Mock(score=0.88, status='passed')
                    }
                    
                    mock_workflow_metrics = Mock()
                    mock_workflow_metrics.success_rate = 0.9
                    mock_workflow_metrics.total_duration = 120
                    mock_workflow.return_value = mock_workflow_metrics
                    
                    # Mock workflow status
                    orchestrator.workflow_engine.get_workflow_status = Mock(return_value={
                        'total_tasks': 10,
                        'task_status': {
                            'completed': 9,
                            'failed': 1,
                            'skipped': 0,
                            'pending': 0,
                            'running': 0
                        }
                    })
                    
                    # Run through phases manually (faster than full orchestration)
                    await orchestrator._phase_initialization()
                    assert orchestrator.state.phase.value == 'planning'
                    
                    await orchestrator._phase_planning()
                    assert orchestrator.state.phase.value == 'execution'
                    
                    # Simulate execution completion
                    orchestrator.state.total_tasks_completed = 9
                    orchestrator.state.success_rate = 0.9
                    
                    await orchestrator._phase_monitoring()
                    assert orchestrator.state.phase.value == 'optimization'
                    
                    await orchestrator._phase_optimization()
                    # Phase should transition to completion if quality is good
                    assert orchestrator.state.phase.value in ['completion', 'planning']
    
    def test_learning_system_pattern_recognition(self, temp_project):
        """Test learning system pattern recognition capabilities."""
        learning_system = create_adaptive_learning_system(temp_project)
        
        # Add diverse execution data
        test_scenarios = [
            # High-complexity tasks take longer
            {'task_type': 'coding', 'complexity': 'high', 'duration': 180, 'success': True},
            {'task_type': 'coding', 'complexity': 'high', 'duration': 190, 'success': True},
            {'task_type': 'coding', 'complexity': 'high', 'duration': 170, 'success': False},
            
            # Medium-complexity tasks
            {'task_type': 'coding', 'complexity': 'medium', 'duration': 90, 'success': True},
            {'task_type': 'coding', 'complexity': 'medium', 'duration': 95, 'success': True},
            
            # Testing tasks
            {'task_type': 'testing', 'complexity': 'medium', 'duration': 45, 'success': True},
            {'task_type': 'testing', 'complexity': 'medium', 'duration': 50, 'success': True},
            
            # Error patterns
            {'task_type': 'testing', 'complexity': 'high', 'duration': 60, 'success': False, 'error': 'ImportError: module not found'},
            {'task_type': 'testing', 'complexity': 'high', 'duration': 55, 'success': False, 'error': 'ImportError: package missing'},
        ]
        
        for scenario in test_scenarios:
            learning_system.record_execution(scenario)
        
        # Learn patterns
        patterns = learning_system.learn_patterns()
        
        # Test duration prediction for different complexities
        high_complexity_duration = learning_system.predict_task_duration({
            'task_type': 'coding',
            'complexity': 'high',
            'phase': 'implementation'
        })
        
        medium_complexity_duration = learning_system.predict_task_duration({
            'task_type': 'coding', 
            'complexity': 'medium',
            'phase': 'implementation'
        })
        
        # High complexity should predict longer duration
        assert high_complexity_duration > medium_complexity_duration
        
        # Test error prediction
        error_prediction = learning_system.predict_potential_errors({
            'task_type': 'testing',
            'complexity': 'high'
        })
        
        assert isinstance(error_prediction, dict)
        assert 'risk_level' in error_prediction
    
    @pytest.mark.asyncio
    async def test_metrics_collection_and_analysis(self, temp_project):
        """Test comprehensive metrics collection and analysis."""
        orchestrator = create_terragon_orchestrator(temp_project)
        
        # Mock component metrics
        with patch.object(orchestrator.quality_gates, 'execute_all_gates') as mock_gates:
            with patch.object(orchestrator.learning_system, 'get_learning_insights') as mock_insights:
                with patch.object(orchestrator.performance_monitor, 'get_system_metrics') as mock_perf:
                    
                    # Configure mocks
                    mock_gates.return_value = {
                        'syntax_validation': Mock(score=0.95),
                        'security_scan': Mock(score=0.88),
                        'test_execution': Mock(score=0.92)
                    }
                    
                    mock_insights.return_value = {
                        'total_execution_records': 25,
                        'total_learned_patterns': 8,
                        'recent_learning_activity': 3
                    }
                    
                    mock_perf.return_value = {
                        'memory_gb': 8,
                        'cpu_count': 4
                    }
                    
                    # Collect metrics
                    await orchestrator._collect_metrics()
                    
                    # Check that metrics were collected
                    assert len(orchestrator.metrics_history) > 0
                    
                    latest_metrics = orchestrator.metrics_history[-1]
                    assert hasattr(latest_metrics, 'overall_health')
                    assert hasattr(latest_metrics, 'autonomy_level')
                    assert hasattr(latest_metrics, 'efficiency_score')
                    assert hasattr(latest_metrics, 'learning_velocity')
                    
                    # Check that derived metrics are in valid range
                    assert 0.0 <= latest_metrics.overall_health <= 1.0
                    assert 0.0 <= latest_metrics.autonomy_level <= 1.0
                    assert 0.0 <= latest_metrics.efficiency_score <= 1.0
                    assert 0.0 <= latest_metrics.learning_velocity <= 1.0
    
    def test_configuration_management(self, temp_project):
        """Test configuration management across components."""
        # Test with custom config
        custom_config = {
            'quality_gates': {
                'gates': {
                    'syntax_validation': {'enabled': True, 'threshold': 0.95}
                },
                'global_threshold': 0.85
            },
            'workflow_engine': {
                'max_concurrent_tasks': 6,
                'task_timeout': 2400
            },
            'learning_system': {
                'max_history_size': 500,
                'confidence_threshold': 0.7
            }
        }
        
        orchestrator = TERRAGONMasterOrchestrator(temp_project, custom_config)
        
        # Check that config was applied
        assert orchestrator.config == custom_config
        assert orchestrator.quality_gates.config['global_threshold'] == 0.85
        assert orchestrator.workflow_engine.config['max_concurrent_tasks'] == 6
        assert orchestrator.learning_system.config['max_history_size'] == 500
    
    def test_error_handling_and_recovery(self, temp_project):
        """Test error handling and recovery mechanisms."""
        orchestrator = create_terragon_orchestrator(temp_project)
        
        # Test orchestration failure handling
        test_error = Exception("Test orchestration failure")
        
        # This should not raise but should handle gracefully
        asyncio.create_task(orchestrator._handle_orchestration_failure(test_error))
        
        # Check that error was recorded
        assert len(orchestrator.execution_events) > 0
        error_event = orchestrator.execution_events[-1]
        assert error_event['type'] == 'orchestration_failure'
        assert 'Test orchestration failure' in error_event['error']
    
    def test_component_isolation(self, temp_project):
        """Test that components can work independently."""
        # Test Quality Gates isolation
        gates = create_progressive_gates(temp_project)
        results = gates.execute_all_gates()
        assert isinstance(results, dict)
        
        # Test Workflow Engine isolation
        engine = create_autonomous_workflow(temp_project)
        tasks = engine.define_standard_workflow()
        assert isinstance(tasks, dict)
        
        # Test Learning System isolation
        learning = create_adaptive_learning_system(temp_project)
        insights = learning.get_learning_insights()
        assert isinstance(insights, dict)
        
        # Each component should work without the others
        assert len(results) > 0
        assert len(tasks) > 0
        assert 'total_execution_records' in insights


def test_terragon_integration_suite():
    """Run integration tests to validate TERRAGON framework."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🧪 Running TERRAGON Integration Tests...")
    
    # Run pytest programmatically
    import subprocess
    import sys
    
    result = subprocess.run([
        sys.executable, '-m', 'pytest', 
        __file__, 
        '-v',
        '--tb=short'
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        logger.info("✅ All TERRAGON integration tests passed!")
        return True
    else:
        logger.error("❌ Some TERRAGON integration tests failed:")
        logger.error(result.stdout)
        logger.error(result.stderr)
        return False


if __name__ == "__main__":
    # Run integration tests
    success = test_terragon_integration_suite()
    exit(0 if success else 1)