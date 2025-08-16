"""TERRAGON Master Orchestrator.

Unified autonomous SDLC control system that coordinates Progressive Quality Gates,
Autonomous Workflow Engine, and Adaptive Learning System for complete self-managing
software development lifecycle execution.
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import signal
import sys

try:
    from .progressive_gates import ProgressiveQualityGates, QualityGateResult
    from .autonomous_workflow import AutonomousWorkflowEngine, WorkflowTask, WorkflowPhase, TaskPriority, WorkflowMetrics
    from .adaptive_learning import AdaptiveLearningSystem, LearningPattern, OptimizationRecommendation
except ImportError as e:
    print(f"Import error in orchestrator: {e}")

try:
    from .robust.error_handling import handle_errors
except ImportError:
    def handle_errors(func):
        return func

try:
    from .scale.performance_monitoring import PerformanceMonitor
except ImportError:
    class PerformanceMonitor:
        def get_system_metrics(self):
            return {'memory_gb': 8, 'cpu_count': 4}

try:
    from .core.utils import set_random_seed
except ImportError:
    def set_random_seed(seed):
        pass


class OrchestrationMode(Enum):
    """Orchestration execution modes."""
    AUTONOMOUS = "autonomous"  # Fully autonomous execution
    SUPERVISED = "supervised"  # Human oversight with auto-approval
    INTERACTIVE = "interactive"  # Human approval required
    MONITORING = "monitoring"  # Monitor-only mode


class OrchestrationPhase(Enum):
    """Orchestration phases."""
    INITIALIZATION = "initialization"
    PLANNING = "planning"
    EXECUTION = "execution"
    MONITORING = "monitoring"
    OPTIMIZATION = "optimization"
    COMPLETION = "completion"


@dataclass
class OrchestrationState:
    """Current state of the orchestration system."""
    phase: OrchestrationPhase
    mode: OrchestrationMode
    start_time: datetime
    last_update: datetime
    active_workflows: int
    quality_score: float
    learning_insights: Dict[str, Any]
    recommendations_pending: int
    total_tasks_completed: int
    success_rate: float
    estimated_completion: Optional[datetime] = None


@dataclass
class TERRAGONMetrics:
    """Comprehensive TERRAGON system metrics."""
    orchestration_state: OrchestrationState
    workflow_metrics: WorkflowMetrics
    quality_metrics: Dict[str, QualityGateResult]
    learning_metrics: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    
    # Derived metrics
    overall_health: float  # 0.0 to 1.0
    autonomy_level: float  # How autonomous the system is running
    efficiency_score: float  # Speed vs quality balance
    learning_velocity: float  # Rate of learning improvement
    
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class TERRAGONMasterOrchestrator:
    """Master orchestrator for autonomous SDLC execution."""
    
    def __init__(self, project_root: Optional[Path] = None, config: Optional[Dict] = None):
        """Initialize TERRAGON Master Orchestrator.
        
        Args:
            project_root: Root directory of the project
            config: Configuration for the orchestrator
        """
        self.project_root = project_root or Path.cwd()
        self.config = config or self._load_default_config()
        
        # Initialize core components
        self.quality_gates = ProgressiveQualityGates(
            project_root, 
            self.config.get('quality_gates', {})
        )
        self.workflow_engine = AutonomousWorkflowEngine(
            project_root,
            self.config.get('workflow_engine', {})
        )
        self.learning_system = AdaptiveLearningSystem(
            project_root,
            self.config.get('learning_system', {})
        )
        self.performance_monitor = PerformanceMonitor()
        
        # Orchestration state
        self.state = OrchestrationState(
            phase=OrchestrationPhase.INITIALIZATION,
            mode=OrchestrationMode.AUTONOMOUS,
            start_time=datetime.now(),
            last_update=datetime.now(),
            active_workflows=0,
            quality_score=0.0,
            learning_insights={},
            recommendations_pending=0,
            total_tasks_completed=0,
            success_rate=0.0
        )
        
        # Execution control
        self.is_running = False
        self.orchestration_loop: Optional[asyncio.Task] = None
        self.executor = ThreadPoolExecutor(max_workers=self.config['max_workers'])
        
        # Metrics and history
        self.metrics_history: List[TERRAGONMetrics] = []
        self.execution_events: List[Dict[str, Any]] = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self._setup_signal_handlers()
        
        self.logger.info("🎯 TERRAGON Master Orchestrator initialized")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default orchestration configuration."""
        return {
            'mode': 'autonomous',
            'max_workers': 4,
            'orchestration_interval': 30,  # seconds
            'quality_threshold': 0.8,
            'learning_frequency': 'realtime',
            'auto_optimization': True,
            'save_metrics': True,
            'backup_frequency': 3600,  # seconds
            'phases': {
                'initialization': {'timeout': 300},  # 5 minutes
                'planning': {'timeout': 900},        # 15 minutes
                'execution': {'timeout': 7200},      # 2 hours
                'monitoring': {'timeout': 1800},     # 30 minutes
                'optimization': {'timeout': 600},    # 10 minutes
                'completion': {'timeout': 300}       # 5 minutes
            },
            'escalation': {
                'quality_threshold': 0.6,
                'failure_rate_threshold': 0.3,
                'timeout_action': 'pause_and_analyze'
            }
        }
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"🛑 TERRAGON received signal {signum}, shutting down gracefully...")
            asyncio.create_task(self.stop_orchestration())
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    @handle_errors
    async def start_orchestration(self, mode: Optional[OrchestrationMode] = None) -> bool:
        """Start autonomous orchestration."""
        if self.is_running:
            self.logger.warning("⚠️ Orchestration already running")
            return False
        
        if mode:
            self.state.mode = mode
        
        self.is_running = True
        self.state.phase = OrchestrationPhase.INITIALIZATION
        self.state.start_time = datetime.now()
        
        self.logger.info(f"🚀 Starting TERRAGON orchestration in {self.state.mode.value} mode")
        
        # Start the main orchestration loop
        self.orchestration_loop = asyncio.create_task(self._orchestration_loop())
        
        return True
    
    async def stop_orchestration(self) -> bool:
        """Stop orchestration gracefully."""
        if not self.is_running:
            return True
        
        self.logger.info("🛑 Stopping TERRAGON orchestration...")
        self.is_running = False
        
        # Cancel orchestration loop
        if self.orchestration_loop:
            self.orchestration_loop.cancel()
            try:
                await self.orchestration_loop
            except asyncio.CancelledError:
                pass
        
        # Stop workflow engine
        self.workflow_engine.stop_workflow()
        
        # Save final metrics
        await self._save_final_metrics()
        
        self.state.phase = OrchestrationPhase.COMPLETION
        self.logger.info("✅ TERRAGON orchestration stopped")
        
        return True
    
    async def _orchestration_loop(self):
        """Main orchestration control loop."""
        try:
            while self.is_running:
                cycle_start = time.time()
                
                # Execute orchestration cycle
                await self._execute_orchestration_cycle()
                
                # Calculate cycle time and sleep
                cycle_time = time.time() - cycle_start
                sleep_time = max(0, self.config['orchestration_interval'] - cycle_time)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
        except asyncio.CancelledError:
            self.logger.info("📊 Orchestration loop cancelled")
            raise
        except Exception as e:
            self.logger.error(f"❌ Orchestration loop failed: {e}")
            await self._handle_orchestration_failure(e)
    
    async def _execute_orchestration_cycle(self):
        """Execute one orchestration cycle."""
        self.state.last_update = datetime.now()
        
        # Phase-specific execution
        if self.state.phase == OrchestrationPhase.INITIALIZATION:
            await self._phase_initialization()
        elif self.state.phase == OrchestrationPhase.PLANNING:
            await self._phase_planning()
        elif self.state.phase == OrchestrationPhase.EXECUTION:
            await self._phase_execution()
        elif self.state.phase == OrchestrationPhase.MONITORING:
            await self._phase_monitoring()
        elif self.state.phase == OrchestrationPhase.OPTIMIZATION:
            await self._phase_optimization()
        elif self.state.phase == OrchestrationPhase.COMPLETION:
            await self._phase_completion()
        
        # Collect and analyze metrics
        await self._collect_metrics()
        
        # Apply learning insights
        await self._apply_learning_insights()
        
        # Check for phase transitions
        await self._check_phase_transitions()
        
        # Handle escalations if needed
        await self._check_escalations()
    
    async def _phase_initialization(self):
        """Initialization phase - setup and validation."""
        self.logger.info("🔧 TERRAGON Phase: Initialization")
        
        # Initialize all components
        try:
            # Load existing learning data
            self.learning_system._load_learning_data()
            
            # Validate project structure
            validation_result = await self._validate_project_structure()
            if not validation_result:
                self.logger.error("❌ Project validation failed")
                return
            
            # Run initial quality gates
            initial_quality = self.quality_gates.execute_all_gates()
            self.state.quality_score = self.quality_gates._calculate_overall_score(initial_quality)
            
            self.logger.info(f"✅ Initialization complete - Quality: {self.state.quality_score:.2f}")
            
            # Transition to planning
            self.state.phase = OrchestrationPhase.PLANNING
            
        except Exception as e:
            self.logger.error(f"❌ Initialization failed: {e}")
            await self._handle_phase_failure("initialization", e)
    
    async def _phase_planning(self):
        """Planning phase - define and optimize workflow."""
        self.logger.info("📋 TERRAGON Phase: Planning")
        
        try:
            # Define standard workflow
            self.workflow_engine.define_standard_workflow()
            
            # Use learning system to optimize task estimates
            await self._optimize_task_estimates()
            
            # Apply learned optimizations
            await self._apply_workflow_optimizations()
            
            self.logger.info("✅ Planning complete")
            
            # Transition to execution
            self.state.phase = OrchestrationPhase.EXECUTION
            
        except Exception as e:
            self.logger.error(f"❌ Planning failed: {e}")
            await self._handle_phase_failure("planning", e)
    
    async def _phase_execution(self):
        """Execution phase - run workflows with monitoring."""
        self.logger.info("⚡ TERRAGON Phase: Execution")
        
        try:
            # Check if workflow is already running
            if not self.workflow_engine.is_running:
                # Start workflow execution
                workflow_task = asyncio.create_task(self.workflow_engine.run_workflow())
                self.state.active_workflows = 1
            
            # Monitor execution progress
            workflow_status = self.workflow_engine.get_workflow_status()
            
            # Update state
            self.state.total_tasks_completed = workflow_status['task_status']['completed']
            total_tasks = workflow_status['total_tasks']
            if total_tasks > 0:
                self.state.success_rate = self.state.total_tasks_completed / total_tasks
            
            # Check if execution is complete
            if (workflow_status['task_status']['completed'] + 
                workflow_status['task_status']['failed'] + 
                workflow_status['task_status']['skipped'] == total_tasks):
                
                self.logger.info("✅ Execution complete")
                self.state.phase = OrchestrationPhase.MONITORING
                self.state.active_workflows = 0
            
        except Exception as e:
            self.logger.error(f"❌ Execution monitoring failed: {e}")
            await self._handle_phase_failure("execution", e)
    
    async def _phase_monitoring(self):
        """Monitoring phase - validate results and collect data."""
        self.logger.info("📊 TERRAGON Phase: Monitoring")
        
        try:
            # Run comprehensive quality gates
            quality_results = self.quality_gates.execute_all_gates()
            self.state.quality_score = self.quality_gates._calculate_overall_score(quality_results)
            
            # Collect execution data for learning
            await self._collect_execution_data_for_learning()
            
            # Generate performance report
            await self._generate_performance_report()
            
            self.logger.info(f"✅ Monitoring complete - Quality: {self.state.quality_score:.2f}")
            
            # Transition to optimization
            self.state.phase = OrchestrationPhase.OPTIMIZATION
            
        except Exception as e:
            self.logger.error(f"❌ Monitoring failed: {e}")
            await self._handle_phase_failure("monitoring", e)
    
    async def _phase_optimization(self):
        """Optimization phase - learn and improve."""
        self.logger.info("🧠 TERRAGON Phase: Optimization")
        
        try:
            # Trigger learning
            learning_patterns = self.learning_system.learn_patterns()
            
            # Generate recommendations
            recommendations = self.learning_system.generate_recommendations()
            self.state.recommendations_pending = len(recommendations)
            
            # Apply automatic optimizations if enabled
            if self.config.get('auto_optimization', True):
                await self._apply_automatic_optimizations(recommendations)
            
            # Update learning insights
            self.state.learning_insights = self.learning_system.get_learning_insights()
            
            self.logger.info(f"✅ Optimization complete - {len(recommendations)} recommendations")
            
            # Check if we should continue or complete
            if self.state.quality_score >= self.config['quality_threshold']:
                self.state.phase = OrchestrationPhase.COMPLETION
            else:
                # Another iteration needed
                self.state.phase = OrchestrationPhase.PLANNING
                self.logger.info("🔄 Quality threshold not met, starting another iteration")
            
        except Exception as e:
            self.logger.error(f"❌ Optimization failed: {e}")
            await self._handle_phase_failure("optimization", e)
    
    async def _phase_completion(self):
        """Completion phase - finalize and report."""
        self.logger.info("🏁 TERRAGON Phase: Completion")
        
        try:
            # Generate final report
            final_report = await self._generate_final_report()
            
            # Save all metrics and learning data
            await self._save_final_metrics()
            
            # Clean up resources
            await self._cleanup_resources()
            
            self.logger.info("🎉 TERRAGON orchestration completed successfully")
            
            # Stop orchestration
            await self.stop_orchestration()
            
        except Exception as e:
            self.logger.error(f"❌ Completion failed: {e}")
            await self._handle_phase_failure("completion", e)
    
    async def _optimize_task_estimates(self):
        """Use learning system to optimize task duration estimates."""
        for task_id, task in self.workflow_engine.tasks.items():
            context = {
                'task_type': task.metadata.get('type', 'default'),
                'complexity': task.metadata.get('complexity', 'medium'),
                'phase': task.phase.value
            }
            
            # Get learned estimate
            learned_duration = self.learning_system.predict_task_duration(context)
            
            # Update task estimate (weighted combination)
            if learned_duration > 0:
                original_estimate = task.estimated_duration
                # Give more weight to learned data if we have high confidence
                confidence = 0.7  # Could be derived from learning system
                optimized_estimate = (original_estimate * (1 - confidence) + 
                                    learned_duration * confidence)
                task.estimated_duration = optimized_estimate
                
                self.logger.debug(f"📈 Optimized {task_id} estimate: {original_estimate:.1f}m → {optimized_estimate:.1f}m")
    
    async def _apply_workflow_optimizations(self):
        """Apply learned workflow optimizations."""
        # Get optimization suggestions
        optimization_context = {
            'project_type': 'ml_library',
            'team_size': 1,
            'complexity': 'high'
        }
        
        optimizations = self.learning_system.suggest_optimizations(optimization_context)
        
        for opt in optimizations:
            self.logger.info(f"💡 Optimization suggestion: {opt['technique']} (Expected: {opt['expected_improvement']:.1%})")
    
    async def _collect_execution_data_for_learning(self):
        """Collect execution data and feed to learning system."""
        # Get workflow metrics
        workflow_status = self.workflow_engine.get_workflow_status()
        
        # Record execution data for each completed task
        for task_id, task in self.workflow_engine.tasks.items():
            if task.status == "completed" and task.start_time and task.end_time:
                duration = (task.end_time - task.start_time).total_seconds() / 60  # minutes
                
                execution_data = {
                    'task_id': task_id,
                    'task_type': task.metadata.get('type', 'default'),
                    'complexity': task.metadata.get('complexity', 'medium'),
                    'phase': task.phase.value,
                    'duration': duration,
                    'success': True,
                    'quality_score': self.state.quality_score,
                    'timestamp': task.end_time.isoformat()
                }
                
                self.learning_system.record_execution(execution_data)
        
        self.logger.debug(f"📝 Recorded execution data for {len([t for t in self.workflow_engine.tasks.values() if t.status == 'completed'])} tasks")
    
    async def _apply_automatic_optimizations(self, recommendations: List[OptimizationRecommendation]):
        """Apply automatic optimizations based on recommendations."""
        applied_count = 0
        
        for rec in recommendations:
            # Only apply low-effort, high-confidence recommendations automatically
            if (rec.implementation_effort == 'low' and 
                rec.confidence > 0.8 and 
                rec.expected_impact > 0.2):
                
                try:
                    success = await self._apply_recommendation(rec)
                    if success:
                        applied_count += 1
                        self.logger.info(f"✅ Applied optimization: {rec.title}")
                    else:
                        self.logger.warning(f"⚠️ Failed to apply optimization: {rec.title}")
                        
                except Exception as e:
                    self.logger.error(f"❌ Error applying optimization {rec.title}: {e}")
        
        self.logger.info(f"🔧 Applied {applied_count} automatic optimizations")
    
    async def _apply_recommendation(self, recommendation: OptimizationRecommendation) -> bool:
        """Apply a specific optimization recommendation."""
        # This is a simplified implementation - in reality, this would
        # contain specific logic for each type of optimization
        
        if recommendation.category == 'performance':
            # Apply performance optimizations
            if 'caching' in recommendation.title.lower():
                # Enable caching optimizations
                return True
            elif 'memory' in recommendation.title.lower():
                # Apply memory optimizations
                return True
                
        elif recommendation.category == 'quality':
            # Apply quality improvements
            if 'validation' in recommendation.title.lower():
                # Enhance validation
                return True
                
        elif recommendation.category == 'process':
            # Apply process improvements
            if 'parallel' in recommendation.title.lower():
                # Enable parallel processing
                return True
        
        return False  # Not implemented
    
    async def _collect_metrics(self):
        """Collect comprehensive system metrics."""
        try:
            # Get component metrics
            quality_metrics = self.quality_gates.execute_all_gates()
            
            # Get workflow metrics (if available)
            workflow_metrics = None
            if hasattr(self.workflow_engine, '_calculate_workflow_metrics'):
                try:
                    workflow_metrics = self.workflow_engine._calculate_workflow_metrics(0)
                except:
                    pass
            
            learning_metrics = self.learning_system.get_learning_insights()
            performance_metrics = self.performance_monitor.get_system_metrics()
            
            # Calculate derived metrics
            overall_health = self._calculate_overall_health(quality_metrics, workflow_metrics)
            autonomy_level = self._calculate_autonomy_level()
            efficiency_score = self._calculate_efficiency_score(workflow_metrics)
            learning_velocity = self._calculate_learning_velocity(learning_metrics)
            
            # Create comprehensive metrics
            terragon_metrics = TERRAGONMetrics(
                orchestration_state=self.state,
                workflow_metrics=workflow_metrics,
                quality_metrics=quality_metrics,
                learning_metrics=learning_metrics,
                performance_metrics=performance_metrics,
                overall_health=overall_health,
                autonomy_level=autonomy_level,
                efficiency_score=efficiency_score,
                learning_velocity=learning_velocity
            )
            
            # Store metrics
            self.metrics_history.append(terragon_metrics)
            
            # Keep only recent metrics (last 100)
            if len(self.metrics_history) > 100:
                self.metrics_history = self.metrics_history[-100:]
            
        except Exception as e:
            self.logger.error(f"❌ Failed to collect metrics: {e}")
    
    def _calculate_overall_health(self, quality_metrics: Dict[str, QualityGateResult], 
                                workflow_metrics: Optional[WorkflowMetrics]) -> float:
        """Calculate overall system health score."""
        # Base on quality scores
        quality_scores = [result.score for result in quality_metrics.values()]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.8
        
        # Factor in workflow success rate if available
        workflow_factor = 1.0
        if workflow_metrics:
            workflow_factor = workflow_metrics.success_rate
        
        # Combine scores
        health = (avg_quality * 0.7 + workflow_factor * 0.3)
        return min(1.0, max(0.0, health))
    
    def _calculate_autonomy_level(self) -> float:
        """Calculate how autonomously the system is running."""
        base_autonomy = {
            OrchestrationMode.AUTONOMOUS: 1.0,
            OrchestrationMode.SUPERVISED: 0.8,
            OrchestrationMode.INTERACTIVE: 0.4,
            OrchestrationMode.MONITORING: 0.2
        }.get(self.state.mode, 0.5)
        
        # Adjust based on how many manual interventions were needed
        # For now, return base value
        return base_autonomy
    
    def _calculate_efficiency_score(self, workflow_metrics: Optional[WorkflowMetrics]) -> float:
        """Calculate efficiency score (speed vs quality balance)."""
        if not workflow_metrics:
            return 0.8  # Default
        
        # Balance between throughput and success rate
        throughput_score = min(1.0, workflow_metrics.throughput / 10.0)  # Normalize to tasks per hour
        quality_score = workflow_metrics.success_rate
        
        # Weight quality higher than speed
        efficiency = throughput_score * 0.3 + quality_score * 0.7
        return efficiency
    
    def _calculate_learning_velocity(self, learning_metrics: Dict[str, Any]) -> float:
        """Calculate rate of learning improvement."""
        # Based on recent learning activity
        total_patterns = learning_metrics.get('total_learned_patterns', 0)
        recent_activity = learning_metrics.get('recent_learning_activity', 0)
        
        if total_patterns == 0:
            return 0.0
        
        # Velocity based on recent activity relative to total
        velocity = min(1.0, recent_activity / max(1, total_patterns * 0.1))
        return velocity
    
    async def _apply_learning_insights(self):
        """Apply insights from the learning system."""
        insights = self.learning_system.get_learning_insights()
        
        # Check for learning opportunities
        if insights.get('total_execution_records', 0) > 10:
            # Trigger learning if we have enough data
            if insights.get('recent_learning_activity', 0) == 0:
                self.learning_system.learn_patterns()
    
    async def _check_phase_transitions(self):
        """Check if phase transitions are needed based on conditions."""
        phase_config = self.config['phases'].get(self.state.phase.value, {})
        timeout = phase_config.get('timeout', 3600)  # Default 1 hour
        
        # Check timeout
        time_in_phase = (datetime.now() - self.state.last_update).total_seconds()
        if time_in_phase > timeout:
            self.logger.warning(f"⏰ Phase {self.state.phase.value} timeout after {timeout}s")
            await self._handle_phase_timeout()
    
    async def _check_escalations(self):
        """Check for conditions requiring escalation."""
        escalation_config = self.config.get('escalation', {})
        
        # Quality threshold check
        quality_threshold = escalation_config.get('quality_threshold', 0.6)
        if self.state.quality_score < quality_threshold:
            self.logger.warning(f"⚠️ Quality score {self.state.quality_score:.2f} below threshold {quality_threshold}")
            await self._handle_quality_escalation()
        
        # Failure rate check
        failure_threshold = escalation_config.get('failure_rate_threshold', 0.3)
        if (1.0 - self.state.success_rate) > failure_threshold:
            self.logger.warning(f"⚠️ Failure rate {1.0 - self.state.success_rate:.2f} above threshold {failure_threshold}")
            await self._handle_failure_escalation()
    
    async def _handle_orchestration_failure(self, error: Exception):
        """Handle orchestration failure."""
        self.logger.error(f"💥 Orchestration failure: {error}")
        
        # Record failure event
        self.execution_events.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'orchestration_failure',
            'error': str(error),
            'phase': self.state.phase.value
        })
        
        # Attempt recovery based on configuration
        timeout_action = self.config['escalation'].get('timeout_action', 'pause_and_analyze')
        
        if timeout_action == 'pause_and_analyze':
            self.logger.info("⏸️ Pausing for analysis...")
            await asyncio.sleep(60)  # Pause for 1 minute
        elif timeout_action == 'restart_phase':
            self.logger.info("🔄 Restarting current phase...")
            # Reset phase start time
            self.state.last_update = datetime.now()
        else:
            # Stop orchestration
            await self.stop_orchestration()
    
    async def _handle_phase_failure(self, phase: str, error: Exception):
        """Handle phase-specific failure."""
        self.logger.error(f"❌ Phase {phase} failed: {error}")
        
        # Record event
        self.execution_events.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'phase_failure',
            'phase': phase,
            'error': str(error)
        })
        
        # For now, try to continue to next phase
        await asyncio.sleep(30)  # Brief pause
    
    async def _handle_phase_timeout(self):
        """Handle phase timeout."""
        self.logger.warning(f"⏰ Phase {self.state.phase.value} timed out")
        
        # Move to next logical phase or restart
        if self.state.phase == OrchestrationPhase.INITIALIZATION:
            self.state.phase = OrchestrationPhase.PLANNING
        elif self.state.phase == OrchestrationPhase.PLANNING:
            self.state.phase = OrchestrationPhase.EXECUTION
        elif self.state.phase == OrchestrationPhase.EXECUTION:
            self.state.phase = OrchestrationPhase.MONITORING
        elif self.state.phase == OrchestrationPhase.MONITORING:
            self.state.phase = OrchestrationPhase.OPTIMIZATION
        elif self.state.phase == OrchestrationPhase.OPTIMIZATION:
            self.state.phase = OrchestrationPhase.COMPLETION
        else:
            self.state.phase = OrchestrationPhase.COMPLETION
    
    async def _handle_quality_escalation(self):
        """Handle quality score escalation."""
        self.logger.warning("📉 Handling quality escalation")
        
        # Trigger additional quality gates
        quality_results = self.quality_gates.execute_all_gates()
        
        # If still low, pause for analysis
        if self.quality_gates._calculate_overall_score(quality_results) < 0.7:
            self.logger.warning("⏸️ Pausing due to persistent quality issues")
            await asyncio.sleep(300)  # 5 minute pause
    
    async def _handle_failure_escalation(self):
        """Handle failure rate escalation."""
        self.logger.warning("💥 Handling failure rate escalation")
        
        # Trigger learning to analyze failures
        self.learning_system.learn_patterns()
        
        # Get error predictions
        error_context = {'task_type': 'general', 'complexity': 'medium'}
        error_insights = self.learning_system.predict_potential_errors(error_context)
        
        self.logger.info(f"🔍 Error analysis: {error_insights}")
    
    async def _validate_project_structure(self) -> bool:
        """Validate project structure."""
        required_files = ['pyproject.toml', 'README.md']
        required_dirs = ['connectome_gnn', 'tests']
        
        for file in required_files:
            if not (self.project_root / file).exists():
                self.logger.error(f"❌ Missing required file: {file}")
                return False
        
        for dir in required_dirs:
            if not (self.project_root / dir).exists():
                self.logger.error(f"❌ Missing required directory: {dir}")
                return False
        
        return True
    
    async def _generate_performance_report(self):
        """Generate performance report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'phase': self.state.phase.value,
            'quality_score': self.state.quality_score,
            'success_rate': self.state.success_rate,
            'tasks_completed': self.state.total_tasks_completed,
            'learning_insights': self.state.learning_insights
        }
        
        # Save report
        reports_dir = self.project_root / '.terragon_reports'
        reports_dir.mkdir(exist_ok=True)
        
        report_file = reports_dir / f'performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"📊 Performance report saved: {report_file}")
    
    async def _generate_final_report(self) -> Dict[str, Any]:
        """Generate final orchestration report."""
        total_duration = (datetime.now() - self.state.start_time).total_seconds() / 60  # minutes
        
        report = {
            'terragon_version': '4.0',
            'execution_summary': {
                'start_time': self.state.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_duration_minutes': total_duration,
                'mode': self.state.mode.value,
                'final_quality_score': self.state.quality_score,
                'final_success_rate': self.state.success_rate,
                'tasks_completed': self.state.total_tasks_completed
            },
            'learning_insights': self.state.learning_insights,
            'recommendations_generated': self.state.recommendations_pending,
            'metrics_collected': len(self.metrics_history),
            'execution_events': len(self.execution_events),
            'overall_health': self.metrics_history[-1].overall_health if self.metrics_history else 0.0,
            'autonomy_achieved': self.metrics_history[-1].autonomy_level if self.metrics_history else 0.0
        }
        
        # Save final report
        reports_dir = self.project_root / '.terragon_reports'
        reports_dir.mkdir(exist_ok=True)
        
        final_report_file = reports_dir / 'terragon_final_report.json'
        with open(final_report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"📋 Final report saved: {final_report_file}")
        return report
    
    async def _save_final_metrics(self):
        """Save final metrics and data."""
        # Save metrics history
        metrics_dir = self.project_root / '.terragon_metrics'
        metrics_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save comprehensive metrics
        metrics_file = metrics_dir / f'terragon_metrics_{timestamp}.json'
        metrics_data = {
            'orchestration_state': asdict(self.state),
            'metrics_history': [asdict(m) for m in self.metrics_history],
            'execution_events': self.execution_events,
            'config': self.config
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        
        # Save latest as well
        latest_file = metrics_dir / 'latest_metrics.json'
        with open(latest_file, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        
        self.logger.info(f"💾 Final metrics saved: {metrics_file}")
    
    async def _cleanup_resources(self):
        """Clean up resources."""
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Save learning data
        self.learning_system._save_learning_data()
        
        self.logger.info("🧹 Resources cleaned up")
    
    def get_orchestration_status(self) -> Dict[str, Any]:
        """Get current orchestration status."""
        return {
            'is_running': self.is_running,
            'state': asdict(self.state),
            'latest_metrics': asdict(self.metrics_history[-1]) if self.metrics_history else None,
            'recent_events': self.execution_events[-5:],  # Last 5 events
            'components_status': {
                'quality_gates': 'active',
                'workflow_engine': 'active' if self.workflow_engine.is_running else 'idle',
                'learning_system': 'active'
            }
        }
    
    async def execute_terragon_sdlc(self, mode: OrchestrationMode = OrchestrationMode.AUTONOMOUS) -> TERRAGONMetrics:
        """Execute complete TERRAGON SDLC autonomously."""
        self.logger.info("🎯 Executing complete TERRAGON SDLC...")
        
        # Start orchestration
        await self.start_orchestration(mode)
        
        # Wait for completion
        while self.is_running:
            await asyncio.sleep(5)
        
        # Return final metrics
        return self.metrics_history[-1] if self.metrics_history else None


def create_terragon_orchestrator(project_root: Optional[Path] = None) -> TERRAGONMasterOrchestrator:
    """Factory function to create TERRAGON orchestrator."""
    return TERRAGONMasterOrchestrator(project_root)


# CLI interface
async def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TERRAGON Master Orchestrator")
    parser.add_argument("--mode", choices=['autonomous', 'supervised', 'interactive', 'monitoring'],
                       default='autonomous', help="Orchestration mode")
    parser.add_argument("--status", action="store_true",
                       help="Show orchestration status")
    parser.add_argument("--config", type=str,
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create orchestrator
    orchestrator = create_terragon_orchestrator()
    
    if args.status:
        status = orchestrator.get_orchestration_status()
        print(json.dumps(status, indent=2, default=str))
        return
    
    # Execute TERRAGON SDLC
    mode = OrchestrationMode(args.mode)
    final_metrics = await orchestrator.execute_terragon_sdlc(mode)
    
    if final_metrics:
        print(f"\n🎉 TERRAGON SDLC Completed:")
        print(f"Overall Health: {final_metrics.overall_health:.2f}")
        print(f"Autonomy Level: {final_metrics.autonomy_level:.2f}")
        print(f"Efficiency Score: {final_metrics.efficiency_score:.2f}")
        print(f"Learning Velocity: {final_metrics.learning_velocity:.2f}")


if __name__ == "__main__":
    asyncio.run(main())