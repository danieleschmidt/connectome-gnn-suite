#!/usr/bin/env python3
"""TERRAGON CLI - Command Line Interface for Autonomous SDLC Execution.

Provides easy-to-use command line interface for executing TERRAGON autonomous
software development lifecycle with all components integrated.
"""

import asyncio
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import signal
import os

# Import TERRAGON components
try:
    from connectome_gnn.terragon_orchestrator import (
        TERRAGONMasterOrchestrator, 
        create_terragon_orchestrator,
        OrchestrationMode
    )
    from connectome_gnn.progressive_gates import create_progressive_gates
    from connectome_gnn.autonomous_workflow import create_autonomous_workflow
    from connectome_gnn.adaptive_learning import create_adaptive_learning_system
except ImportError as e:
    print(f"❌ Error importing TERRAGON components: {e}")
    print("Make sure you're running from the project root directory.")
    sys.exit(1)


class TERRAGONCLIRunner:
    """CLI runner for TERRAGON framework."""
    
    def __init__(self, project_root: Optional[Path] = None, verbose: bool = False):
        """Initialize CLI runner.
        
        Args:
            project_root: Root directory of the project
            verbose: Enable verbose logging
        """
        self.project_root = project_root or Path.cwd()
        self.verbose = verbose
        self.orchestrator: Optional[TERRAGONMasterOrchestrator] = None
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.DEBUG if self.verbose else logging.INFO
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        
        # File handler
        log_dir = self.project_root / '.terragon_logs'
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / f'terragon_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        
        # Suppress some verbose loggers
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"🛑 Received signal {signum}, shutting down...")
            if self.orchestrator:
                asyncio.create_task(self.orchestrator.stop_orchestration())
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def print_banner(self):
        """Print TERRAGON banner."""
        banner = """
╔════════════════════════════════════════════════════════════════════════════════╗
║                                                                                ║
║   ████████╗███████╗██████╗ ██████╗  █████╗  ██████╗  ██████╗ ███╗   ██╗       ║
║   ╚══██╔══╝██╔════╝██╔══██╗██╔══██╗██╔══██╗██╔════╝ ██╔═══██╗████╗  ██║       ║
║      ██║   █████╗  ██████╔╝██████╔╝███████║██║  ███╗██║   ██║██╔██╗ ██║       ║
║      ██║   ██╔══╝  ██╔══██╗██╔══██╗██╔══██║██║   ██║██║   ██║██║╚██╗██║       ║
║      ██║   ███████╗██║  ██║██║  ██║██║  ██║╚██████╔╝╚██████╔╝██║ ╚████║       ║
║      ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝       ║
║                                                                                ║
║                    🤖 Autonomous SDLC Master Prompt v4.0                      ║
║                         Intelligent • Progressive • Adaptive                  ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
"""
        print(banner)
        print(f"🚀 Project Root: {self.project_root}")
        print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    async def run_autonomous_sdlc(self, mode: str = 'autonomous', config_file: Optional[str] = None) -> bool:
        """Run complete autonomous SDLC."""
        self.logger.info("🎯 Starting TERRAGON Autonomous SDLC...")
        
        try:
            # Load configuration
            config = self._load_config(config_file) if config_file else None
            
            # Create orchestrator
            self.orchestrator = create_terragon_orchestrator(self.project_root)
            if config:
                self.orchestrator.config.update(config)
            
            # Convert mode string to enum
            orchestration_mode = OrchestrationMode(mode.lower())
            
            # Execute TERRAGON SDLC
            final_metrics = await self.orchestrator.execute_terragon_sdlc(orchestration_mode)
            
            # Print results
            if final_metrics:
                self._print_final_results(final_metrics)
                return True
            else:
                self.logger.error("❌ TERRAGON execution failed - no metrics returned")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ TERRAGON execution failed: {e}")
            return False
    
    def run_quality_gates(self, save_results: bool = True) -> bool:
        """Run quality gates only."""
        self.logger.info("🛡️ Running Progressive Quality Gates...")
        
        try:
            gates = create_progressive_gates(self.project_root)
            results = gates.execute_all_gates()
            overall_score = gates._calculate_overall_score(results)
            
            self._print_quality_results(results, overall_score)
            
            if save_results:
                self._save_quality_results(results, overall_score)
            
            return overall_score >= 0.8  # Success threshold
            
        except Exception as e:
            self.logger.error(f"❌ Quality gates failed: {e}")
            return False
    
    def run_workflow_planning(self) -> bool:
        """Run workflow planning only."""
        self.logger.info("📋 Running Autonomous Workflow Planning...")
        
        try:
            engine = create_autonomous_workflow(self.project_root)
            tasks = engine.define_standard_workflow()
            
            self._print_workflow_plan(tasks)
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Workflow planning failed: {e}")
            return False
    
    def run_learning_analysis(self) -> bool:
        """Run learning analysis only."""
        self.logger.info("🧠 Running Adaptive Learning Analysis...")
        
        try:
            learning_system = create_adaptive_learning_system(self.project_root)
            
            # Trigger learning
            patterns = learning_system.learn_patterns()
            insights = learning_system.get_learning_insights()
            recommendations = learning_system.recommendations
            
            self._print_learning_results(patterns, insights, recommendations)
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Learning analysis failed: {e}")
            return False
    
    def show_status(self) -> bool:
        """Show current TERRAGON status."""
        self.logger.info("📊 Checking TERRAGON Status...")
        
        try:
            # Check for existing orchestrator state
            status_file = self.project_root / '.terragon_metrics' / 'latest_metrics.json'
            
            if status_file.exists():
                with open(status_file, 'r') as f:
                    data = json.load(f)
                
                self._print_status_info(data)
            else:
                print("📋 No TERRAGON execution history found.")
                print("   Run 'terragon run' to start autonomous SDLC execution.")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Status check failed: {e}")
            return False
    
    def cleanup(self) -> bool:
        """Clean up TERRAGON artifacts."""
        self.logger.info("🧹 Cleaning up TERRAGON artifacts...")
        
        try:
            cleanup_dirs = [
                '.terragon_metrics',
                '.terragon_reports', 
                '.terragon_logs',
                '.quality_gates',
                '.workflow_metrics',
                '.adaptive_learning'
            ]
            
            cleaned_count = 0
            for dir_name in cleanup_dirs:
                dir_path = self.project_root / dir_name
                if dir_path.exists():
                    import shutil
                    shutil.rmtree(dir_path)
                    cleaned_count += 1
                    self.logger.info(f"🗑️ Removed {dir_name}")
            
            print(f"✅ Cleaned up {cleaned_count} TERRAGON directories")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Cleanup failed: {e}")
            return False
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from file."""
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() == '.json':
                return json.load(f)
            else:
                # Try to load as JSON anyway
                return json.load(f)
    
    def _print_final_results(self, metrics):
        """Print final TERRAGON results."""
        print("\n" + "="*80)
        print("🎉 TERRAGON AUTONOMOUS SDLC COMPLETED")
        print("="*80)
        
        print(f"📊 Overall Health:     {metrics.overall_health:.2%}")
        print(f"🤖 Autonomy Level:     {metrics.autonomy_level:.2%}") 
        print(f"⚡ Efficiency Score:   {metrics.efficiency_score:.2%}")
        print(f"🧠 Learning Velocity:  {metrics.learning_velocity:.2%}")
        
        if hasattr(metrics, 'orchestration_state'):
            state = metrics.orchestration_state
            print(f"✅ Tasks Completed:    {state.total_tasks_completed}")
            print(f"📈 Success Rate:       {state.success_rate:.2%}")
            print(f"🏆 Quality Score:      {state.quality_score:.2%}")
        
        print("\n📋 Detailed reports saved in:")
        print(f"   📁 {self.project_root / '.terragon_reports'}")
        print(f"   📁 {self.project_root / '.terragon_metrics'}")
        print("="*80)
    
    def _print_quality_results(self, results: Dict, overall_score: float):
        """Print quality gate results."""
        print("\n" + "="*60)
        print("🛡️ PROGRESSIVE QUALITY GATES RESULTS")
        print("="*60)
        
        for gate_name, result in results.items():
            status_emoji = "✅" if result.status == "passed" else "❌" if result.status == "failed" else "⚠️"
            print(f"{status_emoji} {gate_name:<25} {result.score:.2%} ({result.status})")
        
        print("-"*60)
        overall_emoji = "✅" if overall_score >= 0.8 else "⚠️" if overall_score >= 0.6 else "❌"
        print(f"{overall_emoji} Overall Score:           {overall_score:.2%}")
        print("="*60)
    
    def _print_workflow_plan(self, tasks: Dict):
        """Print workflow plan."""
        print("\n" + "="*60)
        print("📋 AUTONOMOUS WORKFLOW PLAN")
        print("="*60)
        
        # Group by phase
        phases = {}
        for task in tasks.values():
            phase = task.phase.value
            if phase not in phases:
                phases[phase] = []
            phases[phase].append(task)
        
        for phase_name, phase_tasks in phases.items():
            print(f"\n📁 {phase_name.upper()} PHASE:")
            for task in phase_tasks:
                priority_emoji = "🔴" if task.priority.value == "critical" else "🟡" if task.priority.value == "high" else "🟢"
                print(f"   {priority_emoji} {task.name} ({task.estimated_duration:.0f}m)")
        
        print("="*60)
    
    def _print_learning_results(self, patterns: Dict, insights: Dict, recommendations: list):
        """Print learning analysis results."""
        print("\n" + "="*60)
        print("🧠 ADAPTIVE LEARNING ANALYSIS")
        print("="*60)
        
        print(f"📊 Execution Records:   {insights.get('total_execution_records', 0)}")
        print(f"🎯 Learned Patterns:    {insights.get('total_learned_patterns', 0)}")
        print(f"⚡ Recent Activity:     {insights.get('recent_learning_activity', 0)}")
        
        if recommendations:
            print(f"\n💡 TOP RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"   {i}. {rec.title}")
                print(f"      Impact: {rec.expected_impact:.1%}, Priority: {rec.priority}")
        
        print("="*60)
    
    def _print_status_info(self, data: Dict):
        """Print status information."""
        print("\n" + "="*60)
        print("📊 TERRAGON STATUS")
        print("="*60)
        
        state = data.get('orchestration_state', {})
        print(f"📅 Last Run:           {state.get('last_update', 'Unknown')}")
        print(f"🎯 Phase:              {state.get('phase', 'Unknown')}")
        print(f"📈 Quality Score:      {state.get('quality_score', 0):.2%}")
        print(f"✅ Success Rate:       {state.get('success_rate', 0):.2%}")
        print(f"📋 Tasks Completed:    {state.get('total_tasks_completed', 0)}")
        
        latest_metrics = data.get('metrics_history', [])
        if latest_metrics:
            metrics = latest_metrics[-1]
            print(f"🏥 Overall Health:     {metrics.get('overall_health', 0):.2%}")
            print(f"🤖 Autonomy Level:     {metrics.get('autonomy_level', 0):.2%}")
        
        print("="*60)
    
    def _save_quality_results(self, results: Dict, overall_score: float):
        """Save quality results to file."""
        results_dir = self.project_root / '.quality_reports'
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f'quality_results_{timestamp}.json'
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'overall_score': overall_score,
            'results': {name: {
                'score': result.score,
                'status': result.status,
                'execution_time': result.execution_time
            } for name, result in results.items()}
        }
        
        with open(results_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"💾 Quality results saved: {results_file}")


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        prog='terragon',
        description='TERRAGON - Autonomous SDLC Master Prompt v4.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    terragon run                          # Run full autonomous SDLC
    terragon run --mode supervised        # Run with human oversight  
    terragon quality                      # Run quality gates only
    terragon workflow                     # Plan workflow only
    terragon learn                        # Run learning analysis
    terragon status                       # Show current status
    terragon cleanup                      # Clean up artifacts
    
For more information, visit: https://github.com/danieleschmidt/connectome-gnn-suite
        """
    )
    
    # Global options
    parser.add_argument('--project-root', type=str, default='.',
                       help='Project root directory (default: current directory)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--config', type=str,
                       help='Configuration file path')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run autonomous SDLC')
    run_parser.add_argument('--mode', choices=['autonomous', 'supervised', 'interactive', 'monitoring'],
                           default='autonomous', help='Orchestration mode (default: autonomous)')
    
    # Quality command
    quality_parser = subparsers.add_parser('quality', help='Run quality gates')
    quality_parser.add_argument('--save', action='store_true', default=True,
                               help='Save results to file (default: True)')
    
    # Workflow command
    workflow_parser = subparsers.add_parser('workflow', help='Plan workflow')
    
    # Learn command
    learn_parser = subparsers.add_parser('learn', help='Run learning analysis')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show status')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up artifacts')
    
    return parser


async def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Create CLI runner
    project_root = Path(args.project_root).resolve()
    runner = TERRAGONCLIRunner(project_root, args.verbose)
    
    # Print banner
    runner.print_banner()
    
    # Handle commands
    success = False
    
    try:
        if args.command == 'run':
            success = await runner.run_autonomous_sdlc(args.mode, args.config)
        
        elif args.command == 'quality':
            success = runner.run_quality_gates(args.save)
        
        elif args.command == 'workflow':
            success = runner.run_workflow_planning()
        
        elif args.command == 'learn':
            success = runner.run_learning_analysis()
        
        elif args.command == 'status':
            success = runner.show_status()
        
        elif args.command == 'cleanup':
            success = runner.cleanup()
        
        else:
            # No command specified, run full SDLC by default
            success = await runner.run_autonomous_sdlc('autonomous', args.config)
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        success = False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        success = False
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    # Check if we're in an event loop already
    try:
        loop = asyncio.get_running_loop()
        # If we get here, we're already in an event loop
        print("Error: Cannot run TERRAGON CLI from within an async context")
        sys.exit(1)
    except RuntimeError:
        # Not in an event loop, we can run normally
        asyncio.run(main())