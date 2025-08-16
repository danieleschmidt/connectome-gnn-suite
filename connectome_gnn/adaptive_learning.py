"""Adaptive Learning System.

AI-driven continuous improvement system that learns from execution patterns,
optimizes development processes, and provides intelligent recommendations
for autonomous software development lifecycle management.
"""

import numpy as np
import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import pickle
import hashlib
from abc import ABC, abstractmethod
import math

try:
    from .robust.error_handling import handle_errors
except ImportError:
    def handle_errors(func):
        return func

try:
    from .core.utils import set_random_seed
except ImportError:
    def set_random_seed(seed):
        pass


@dataclass
class LearningPattern:
    """Represents a learned pattern from execution data."""
    pattern_id: str
    pattern_type: str  # 'duration', 'success_rate', 'error_pattern', 'optimization'
    conditions: Dict[str, Any]  # Conditions when pattern applies
    prediction: Union[float, str, Dict]  # What the pattern predicts
    confidence: float  # 0.0 to 1.0
    observations: int  # Number of observations supporting this pattern
    last_updated: datetime
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass 
class OptimizationRecommendation:
    """Represents an optimization recommendation."""
    recommendation_id: str
    category: str  # 'performance', 'quality', 'process', 'resource'
    title: str
    description: str
    expected_impact: float  # 0.0 to 1.0
    implementation_effort: str  # 'low', 'medium', 'high'
    priority: int  # 1-10
    conditions: Dict[str, Any]
    actions: List[str]
    confidence: float
    created_at: datetime


class LearningAlgorithm(ABC):
    """Abstract base class for learning algorithms."""
    
    @abstractmethod
    def learn(self, data: List[Dict[str, Any]]) -> List[LearningPattern]:
        """Learn patterns from data."""
        pass
    
    @abstractmethod
    def predict(self, context: Dict[str, Any], patterns: List[LearningPattern]) -> Any:
        """Make predictions based on learned patterns."""
        pass


class DurationLearningAlgorithm(LearningAlgorithm):
    """Learn task duration patterns."""
    
    def learn(self, data: List[Dict[str, Any]]) -> List[LearningPattern]:
        """Learn duration patterns from task execution data."""
        patterns = []
        
        # Group by task type and context
        grouped_data = defaultdict(list)
        for entry in data:
            key = (
                entry.get('task_type', 'default'),
                entry.get('complexity', 'medium'),
                entry.get('phase', 'implementation')
            )
            if 'duration' in entry and entry['duration'] > 0:
                grouped_data[key].append(entry['duration'])
        
        # Create patterns for each group
        for (task_type, complexity, phase), durations in grouped_data.items():
            if len(durations) >= 3:  # Need minimum observations
                mean_duration = np.mean(durations)
                std_duration = np.std(durations)
                confidence = min(1.0, len(durations) / 10.0)  # Higher confidence with more data
                
                pattern = LearningPattern(
                    pattern_id=f"duration_{task_type}_{complexity}_{phase}",
                    pattern_type="duration",
                    conditions={
                        'task_type': task_type,
                        'complexity': complexity,
                        'phase': phase
                    },
                    prediction={
                        'mean': mean_duration,
                        'std': std_duration,
                        'range': [max(0, mean_duration - std_duration), 
                                 mean_duration + std_duration]
                    },
                    confidence=confidence,
                    observations=len(durations),
                    last_updated=datetime.now()
                )
                patterns.append(pattern)
        
        return patterns
    
    def predict(self, context: Dict[str, Any], patterns: List[LearningPattern]) -> float:
        """Predict task duration based on context."""
        # Find matching patterns
        matching_patterns = []
        for pattern in patterns:
            if pattern.pattern_type != "duration":
                continue
            
            match_score = 0
            total_conditions = len(pattern.conditions)
            
            for key, value in pattern.conditions.items():
                if context.get(key) == value:
                    match_score += 1
            
            if match_score > 0:
                match_ratio = match_score / total_conditions
                matching_patterns.append((pattern, match_ratio))
        
        if not matching_patterns:
            return 60.0  # Default estimate in minutes
        
        # Weight predictions by match ratio and confidence
        weighted_predictions = []
        for pattern, match_ratio in matching_patterns:
            weight = match_ratio * pattern.confidence
            prediction = pattern.prediction['mean']
            weighted_predictions.append((prediction, weight))
        
        # Calculate weighted average
        total_weight = sum(weight for _, weight in weighted_predictions)
        if total_weight > 0:
            weighted_avg = sum(pred * weight for pred, weight in weighted_predictions) / total_weight
            return weighted_avg
        
        return 60.0  # Default


class SuccessRateLearningAlgorithm(LearningAlgorithm):
    """Learn success rate patterns."""
    
    def learn(self, data: List[Dict[str, Any]]) -> List[LearningPattern]:
        """Learn success rate patterns."""
        patterns = []
        
        # Group by conditions that might affect success
        grouped_data = defaultdict(list)
        for entry in data:
            key = (
                entry.get('task_type', 'default'),
                entry.get('complexity', 'medium'),
                entry.get('team_size', 1),
                entry.get('time_of_day', 'day')
            )
            success = entry.get('success', True)
            grouped_data[key].append(1 if success else 0)
        
        # Create patterns
        for (task_type, complexity, team_size, time_of_day), successes in grouped_data.items():
            if len(successes) >= 5:  # Need minimum observations
                success_rate = np.mean(successes)
                confidence = min(1.0, len(successes) / 20.0)
                
                pattern = LearningPattern(
                    pattern_id=f"success_{task_type}_{complexity}_{team_size}_{time_of_day}",
                    pattern_type="success_rate",
                    conditions={
                        'task_type': task_type,
                        'complexity': complexity,
                        'team_size': team_size,
                        'time_of_day': time_of_day
                    },
                    prediction=success_rate,
                    confidence=confidence,
                    observations=len(successes),
                    last_updated=datetime.now()
                )
                patterns.append(pattern)
        
        return patterns
    
    def predict(self, context: Dict[str, Any], patterns: List[LearningPattern]) -> float:
        """Predict success rate."""
        matching_patterns = []
        for pattern in patterns:
            if pattern.pattern_type != "success_rate":
                continue
            
            match_score = 0
            for key, value in pattern.conditions.items():
                if context.get(key) == value:
                    match_score += 1
            
            if match_score > 0:
                match_ratio = match_score / len(pattern.conditions)
                matching_patterns.append((pattern, match_ratio))
        
        if not matching_patterns:
            return 0.8  # Default success rate
        
        # Weight by match ratio and confidence
        weighted_sum = 0
        total_weight = 0
        for pattern, match_ratio in matching_patterns:
            weight = match_ratio * pattern.confidence
            weighted_sum += pattern.prediction * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.8


class ErrorPatternLearningAlgorithm(LearningAlgorithm):
    """Learn error pattern recognition."""
    
    def learn(self, data: List[Dict[str, Any]]) -> List[LearningPattern]:
        """Learn error patterns."""
        patterns = []
        
        # Collect error data
        error_data = defaultdict(list)
        for entry in data:
            if entry.get('error') or entry.get('failed', False):
                error_type = self._categorize_error(entry.get('error', ''))
                context = (
                    entry.get('task_type', 'default'),
                    entry.get('complexity', 'medium'),
                    error_type
                )
                error_data[context].append(entry)
        
        # Create error patterns
        for (task_type, complexity, error_type), errors in error_data.items():
            if len(errors) >= 3:
                # Common causes
                causes = [e.get('cause', 'unknown') for e in errors]
                most_common_cause = max(set(causes), key=causes.count)
                
                # Solutions that worked
                solutions = [e.get('solution', '') for e in errors if e.get('resolved', False)]
                
                pattern = LearningPattern(
                    pattern_id=f"error_{task_type}_{complexity}_{error_type}",
                    pattern_type="error_pattern",
                    conditions={
                        'task_type': task_type,
                        'complexity': complexity,
                        'error_type': error_type
                    },
                    prediction={
                        'likely_cause': most_common_cause,
                        'frequency': len(errors),
                        'common_solutions': list(set(solutions))
                    },
                    confidence=min(1.0, len(errors) / 10.0),
                    observations=len(errors),
                    last_updated=datetime.now()
                )
                patterns.append(pattern)
        
        return patterns
    
    def _categorize_error(self, error_message: str) -> str:
        """Categorize error by type."""
        error_lower = error_message.lower()
        
        if any(word in error_lower for word in ['import', 'module', 'package']):
            return 'import_error'
        elif any(word in error_lower for word in ['memory', 'oom', 'allocation']):
            return 'memory_error'
        elif any(word in error_lower for word in ['permission', 'access', 'denied']):
            return 'permission_error'
        elif any(word in error_lower for word in ['timeout', 'connection', 'network']):
            return 'network_error'
        elif any(word in error_lower for word in ['syntax', 'invalid', 'parse']):
            return 'syntax_error'
        else:
            return 'other_error'
    
    def predict(self, context: Dict[str, Any], patterns: List[LearningPattern]) -> Dict[str, Any]:
        """Predict likely errors and solutions."""
        matching_patterns = []
        for pattern in patterns:
            if pattern.pattern_type != "error_pattern":
                continue
            
            match_score = 0
            for key, value in pattern.conditions.items():
                if context.get(key) == value:
                    match_score += 1
            
            if match_score > 0:
                matching_patterns.append((pattern, match_score))
        
        if not matching_patterns:
            return {'risk_level': 'low', 'recommendations': []}
        
        # Sort by relevance
        matching_patterns.sort(key=lambda x: x[1], reverse=True)
        
        top_patterns = matching_patterns[:3]  # Top 3 most relevant
        
        recommendations = []
        risk_level = 'low'
        
        for pattern, _ in top_patterns:
            prediction = pattern.prediction
            if prediction['frequency'] > 5:
                risk_level = 'high' if prediction['frequency'] > 10 else 'medium'
            
            recommendations.extend(prediction.get('common_solutions', []))
        
        return {
            'risk_level': risk_level,
            'recommendations': list(set(recommendations)),
            'likely_errors': [p.conditions['error_type'] for p, _ in top_patterns]
        }


class OptimizationLearningAlgorithm(LearningAlgorithm):
    """Learn optimization opportunities."""
    
    def learn(self, data: List[Dict[str, Any]]) -> List[LearningPattern]:
        """Learn optimization patterns."""
        patterns = []
        
        # Analyze performance improvements
        improvement_data = []
        for entry in data:
            if 'before_metric' in entry and 'after_metric' in entry:
                improvement = (entry['after_metric'] - entry['before_metric']) / entry['before_metric']
                if improvement > 0.1:  # Significant improvement
                    improvement_data.append({
                        'improvement': improvement,
                        'optimization_type': entry.get('optimization_type', 'unknown'),
                        'context': entry.get('context', {}),
                        'technique': entry.get('technique', 'unknown')
                    })
        
        # Group by optimization type and technique
        grouped_improvements = defaultdict(list)
        for item in improvement_data:
            key = (item['optimization_type'], item['technique'])
            grouped_improvements[key].append(item['improvement'])
        
        # Create optimization patterns
        for (opt_type, technique), improvements in grouped_improvements.items():
            if len(improvements) >= 3:
                avg_improvement = np.mean(improvements)
                
                pattern = LearningPattern(
                    pattern_id=f"optimization_{opt_type}_{technique}",
                    pattern_type="optimization",
                    conditions={
                        'optimization_type': opt_type,
                        'technique': technique
                    },
                    prediction={
                        'expected_improvement': avg_improvement,
                        'success_rate': len(improvements) / (len(improvements) + 1),  # Bayesian estimate
                        'technique': technique
                    },
                    confidence=min(1.0, len(improvements) / 10.0),
                    observations=len(improvements),
                    last_updated=datetime.now()
                )
                patterns.append(pattern)
        
        return patterns
    
    def predict(self, context: Dict[str, Any], patterns: List[LearningPattern]) -> List[Dict[str, Any]]:
        """Predict optimization opportunities."""
        optimization_suggestions = []
        
        for pattern in patterns:
            if pattern.pattern_type != "optimization":
                continue
            
            # Check if optimization is applicable
            if pattern.confidence > 0.5 and pattern.prediction['expected_improvement'] > 0.2:
                optimization_suggestions.append({
                    'type': pattern.conditions['optimization_type'],
                    'technique': pattern.prediction['technique'],
                    'expected_improvement': pattern.prediction['expected_improvement'],
                    'confidence': pattern.confidence
                })
        
        # Sort by expected impact
        optimization_suggestions.sort(
            key=lambda x: x['expected_improvement'] * x['confidence'], 
            reverse=True
        )
        
        return optimization_suggestions[:5]  # Top 5 suggestions


class AdaptiveLearningSystem:
    """Main adaptive learning system coordinating all learning algorithms."""
    
    def __init__(self, project_root: Optional[Path] = None, config: Optional[Dict] = None):
        """Initialize adaptive learning system.
        
        Args:
            project_root: Root directory of the project
            config: Configuration for learning system
        """
        self.project_root = project_root or Path.cwd()
        self.config = config or self._load_default_config()
        
        # Learning algorithms
        self.algorithms = {
            'duration': DurationLearningAlgorithm(),
            'success_rate': SuccessRateLearningAlgorithm(),
            'error_pattern': ErrorPatternLearningAlgorithm(),
            'optimization': OptimizationLearningAlgorithm()
        }
        
        # Learning data storage
        self.execution_history: deque = deque(maxlen=self.config['max_history_size'])
        self.learned_patterns: Dict[str, List[LearningPattern]] = defaultdict(list)
        self.recommendations: List[OptimizationRecommendation] = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Load existing data
        self._load_learning_data()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            'max_history_size': 1000,
            'learning_frequency': 'daily',  # 'realtime', 'hourly', 'daily'
            'confidence_threshold': 0.6,
            'pattern_expiry_days': 30,
            'auto_recommendations': True,
            'save_patterns': True
        }
    
    @handle_errors
    def record_execution(self, execution_data: Dict[str, Any]):
        """Record execution data for learning."""
        # Add timestamp if not present
        if 'timestamp' not in execution_data:
            execution_data['timestamp'] = datetime.now().isoformat()
        
        # Add to history
        self.execution_history.append(execution_data)
        
        # Trigger learning if realtime mode
        if self.config['learning_frequency'] == 'realtime':
            self.learn_patterns()
        
        self.logger.debug(f"📝 Recorded execution data: {execution_data.get('task_type', 'unknown')}")
    
    def learn_patterns(self) -> Dict[str, List[LearningPattern]]:
        """Execute learning across all algorithms."""
        self.logger.info("🧠 Learning patterns from execution data...")
        
        # Convert deque to list for processing
        data_list = list(self.execution_history)
        
        # Learn patterns with each algorithm
        new_patterns = {}
        for algo_name, algorithm in self.algorithms.items():
            try:
                patterns = algorithm.learn(data_list)
                
                # Filter patterns by confidence
                confident_patterns = [
                    p for p in patterns 
                    if p.confidence >= self.config['confidence_threshold']
                ]
                
                # Update learned patterns
                self.learned_patterns[algo_name] = confident_patterns
                new_patterns[algo_name] = confident_patterns
                
                self.logger.info(f"🎯 Learned {len(confident_patterns)} {algo_name} patterns")
                
            except Exception as e:
                self.logger.error(f"❌ Learning failed for {algo_name}: {e}")
                new_patterns[algo_name] = []
        
        # Generate recommendations
        if self.config.get('auto_recommendations', True):
            self.generate_recommendations()
        
        # Save patterns
        if self.config.get('save_patterns', True):
            self._save_learning_data()
        
        return new_patterns
    
    def predict_task_duration(self, context: Dict[str, Any]) -> float:
        """Predict task duration based on learned patterns."""
        duration_patterns = self.learned_patterns.get('duration', [])
        
        if not duration_patterns:
            return 60.0  # Default 1 hour
        
        duration_algo = self.algorithms['duration']
        prediction = duration_algo.predict(context, duration_patterns)
        
        self.logger.debug(f"🔮 Predicted duration: {prediction:.1f}m for {context}")
        return prediction
    
    def predict_success_rate(self, context: Dict[str, Any]) -> float:
        """Predict task success rate."""
        success_patterns = self.learned_patterns.get('success_rate', [])
        
        if not success_patterns:
            return 0.8  # Default 80%
        
        success_algo = self.algorithms['success_rate']
        prediction = success_algo.predict(context, success_patterns)
        
        return prediction
    
    def predict_potential_errors(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Predict potential errors and solutions."""
        error_patterns = self.learned_patterns.get('error_pattern', [])
        
        if not error_patterns:
            return {'risk_level': 'low', 'recommendations': []}
        
        error_algo = self.algorithms['error_pattern']
        prediction = error_algo.predict(context, error_patterns)
        
        return prediction
    
    def suggest_optimizations(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest optimization opportunities."""
        optimization_patterns = self.learned_patterns.get('optimization', [])
        
        if not optimization_patterns:
            return []
        
        optimization_algo = self.algorithms['optimization']
        suggestions = optimization_algo.predict(context, optimization_patterns)
        
        return suggestions
    
    def generate_recommendations(self) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on learned patterns."""
        recommendations = []
        
        # Analyze patterns to generate recommendations
        for pattern_type, patterns in self.learned_patterns.items():
            for pattern in patterns:
                if pattern.confidence < 0.7:  # Only high-confidence patterns
                    continue
                
                rec = self._pattern_to_recommendation(pattern)
                if rec:
                    recommendations.append(rec)
        
        # Sort by expected impact and priority
        recommendations.sort(key=lambda x: (x.priority, x.expected_impact), reverse=True)
        
        # Keep top recommendations
        self.recommendations = recommendations[:20]
        
        self.logger.info(f"💡 Generated {len(self.recommendations)} recommendations")
        return self.recommendations
    
    def _pattern_to_recommendation(self, pattern: LearningPattern) -> Optional[OptimizationRecommendation]:
        """Convert a learned pattern to an optimization recommendation."""
        if pattern.pattern_type == 'duration':
            pred = pattern.prediction
            if isinstance(pred, dict) and pred.get('mean', 0) > 120:  # Tasks taking > 2 hours
                return OptimizationRecommendation(
                    recommendation_id=f"rec_{pattern.pattern_id}_{datetime.now().timestamp()}",
                    category='performance',
                    title='Optimize Long-Running Tasks',
                    description=f"Tasks of type '{pattern.conditions.get('task_type')}' take an average of {pred.get('mean', 0):.1f} minutes. Consider breaking into smaller tasks or optimizing the process.",
                    expected_impact=0.3,
                    implementation_effort='medium',
                    priority=7,
                    conditions=pattern.conditions,
                    actions=[
                        'Break task into smaller subtasks',
                        'Identify bottlenecks',
                        'Consider parallel execution'
                    ],
                    confidence=pattern.confidence,
                    created_at=datetime.now()
                )
        
        elif pattern.pattern_type == 'success_rate':
            if pattern.prediction < 0.7:  # Low success rate
                return OptimizationRecommendation(
                    recommendation_id=f"rec_{pattern.pattern_id}_{datetime.now().timestamp()}",
                    category='quality',
                    title='Improve Task Success Rate',
                    description=f"Tasks matching conditions {pattern.conditions} have only {pattern.prediction:.1%} success rate. Consider additional validation or training.",
                    expected_impact=0.4,
                    implementation_effort='medium',
                    priority=8,
                    conditions=pattern.conditions,
                    actions=[
                        'Add pre-execution validation',
                        'Improve error handling',
                        'Provide additional training/documentation'
                    ],
                    confidence=pattern.confidence,
                    created_at=datetime.now()
                )
        
        elif pattern.pattern_type == 'error_pattern':
            pred = pattern.prediction
            if isinstance(pred, dict) and pred.get('frequency', 0) > 5:
                return OptimizationRecommendation(
                    recommendation_id=f"rec_{pattern.pattern_id}_{datetime.now().timestamp()}",
                    category='quality',
                    title='Address Recurring Error Pattern',
                    description=f"Error type '{pattern.conditions.get('error_type')}' occurs frequently. Common cause: {pred.get('likely_cause', 'unknown')}.",
                    expected_impact=0.5,
                    implementation_effort='low',
                    priority=9,
                    conditions=pattern.conditions,
                    actions=pred.get('common_solutions', ['Investigate and fix root cause']),
                    confidence=pattern.confidence,
                    created_at=datetime.now()
                )
        
        elif pattern.pattern_type == 'optimization':
            pred = pattern.prediction
            if isinstance(pred, dict) and pred.get('expected_improvement', 0) > 0.2:
                return OptimizationRecommendation(
                    recommendation_id=f"rec_{pattern.pattern_id}_{datetime.now().timestamp()}",
                    category='performance',
                    title=f"Apply {pred.get('technique', 'Unknown')} Optimization",
                    description=f"Applying {pred.get('technique')} to {pattern.conditions.get('optimization_type')} has shown {pred.get('expected_improvement', 0):.1%} improvement.",
                    expected_impact=pred.get('expected_improvement', 0),
                    implementation_effort='medium',
                    priority=6,
                    conditions=pattern.conditions,
                    actions=[f"Implement {pred.get('technique')} optimization"],
                    confidence=pattern.confidence,
                    created_at=datetime.now()
                )
        
        return None
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from the learning system."""
        total_patterns = sum(len(patterns) for patterns in self.learned_patterns.values())
        
        # Pattern confidence distribution
        all_patterns = []
        for patterns in self.learned_patterns.values():
            all_patterns.extend(patterns)
        
        confidence_dist = {
            'high': len([p for p in all_patterns if p.confidence >= 0.8]),
            'medium': len([p for p in all_patterns if 0.6 <= p.confidence < 0.8]),
            'low': len([p for p in all_patterns if p.confidence < 0.6])
        }
        
        # Recent learning activity
        recent_patterns = [
            p for p in all_patterns 
            if (datetime.now() - p.last_updated).days <= 7
        ]
        
        insights = {
            'total_execution_records': len(self.execution_history),
            'total_learned_patterns': total_patterns,
            'pattern_breakdown': {
                name: len(patterns) for name, patterns in self.learned_patterns.items()
            },
            'confidence_distribution': confidence_dist,
            'recent_learning_activity': len(recent_patterns),
            'active_recommendations': len(self.recommendations),
            'learning_data_age_days': (datetime.now() - datetime.fromisoformat(
                list(self.execution_history)[0]['timestamp']
            )).days if self.execution_history else 0
        }
        
        return insights
    
    def _load_learning_data(self):
        """Load previously saved learning data."""
        learning_file = self.project_root / '.adaptive_learning' / 'learning_data.json'
        
        if learning_file.exists():
            try:
                with open(learning_file, 'r') as f:
                    data = json.load(f)
                
                # Load execution history
                history_data = data.get('execution_history', [])
                self.execution_history.extend(history_data)
                
                # Load patterns
                patterns_data = data.get('learned_patterns', {})
                for pattern_type, pattern_list in patterns_data.items():
                    self.learned_patterns[pattern_type] = [
                        LearningPattern(**p) for p in pattern_list
                    ]
                
                # Load recommendations
                rec_data = data.get('recommendations', [])
                self.recommendations = [
                    OptimizationRecommendation(**r) for r in rec_data
                ]
                
                self.logger.info(f"📚 Loaded learning data: {len(self.execution_history)} records")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to load learning data: {e}")
    
    def _save_learning_data(self):
        """Save learning data to file."""
        learning_dir = self.project_root / '.adaptive_learning'
        learning_dir.mkdir(exist_ok=True)
        
        # Prepare data for serialization
        data = {
            'timestamp': datetime.now().isoformat(),
            'execution_history': list(self.execution_history),
            'learned_patterns': {
                pattern_type: [asdict(p) for p in patterns]
                for pattern_type, patterns in self.learned_patterns.items()
            },
            'recommendations': [asdict(r) for r in self.recommendations],
            'config': self.config
        }
        
        # Save to file
        learning_file = learning_dir / 'learning_data.json'
        with open(learning_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        # Also save a backup with timestamp
        backup_file = learning_dir / f'backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(backup_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        self.logger.info(f"💾 Saved learning data to {learning_file}")
    
    def export_patterns(self, format: str = 'json') -> str:
        """Export learned patterns for analysis or sharing."""
        if format == 'json':
            export_data = {
                'exported_at': datetime.now().isoformat(),
                'total_patterns': sum(len(patterns) for patterns in self.learned_patterns.values()),
                'patterns': {
                    pattern_type: [asdict(p) for p in patterns]
                    for pattern_type, patterns in self.learned_patterns.items()
                },
                'recommendations': [asdict(r) for r in self.recommendations],
                'insights': self.get_learning_insights()
            }
            return json.dumps(export_data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def reset_learning(self, keep_history: bool = True):
        """Reset learned patterns while optionally keeping execution history."""
        self.learned_patterns.clear()
        self.recommendations.clear()
        
        if not keep_history:
            self.execution_history.clear()
        
        self.logger.info("🔄 Reset learning data")


def create_adaptive_learning_system(project_root: Optional[Path] = None) -> AdaptiveLearningSystem:
    """Factory function to create adaptive learning system."""
    return AdaptiveLearningSystem(project_root)


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Adaptive Learning System")
    parser.add_argument("--learn", action="store_true", 
                       help="Trigger learning from execution data")
    parser.add_argument("--insights", action="store_true",
                       help="Show learning insights")
    parser.add_argument("--recommendations", action="store_true",
                       help="Show optimization recommendations")
    parser.add_argument("--export", type=str, choices=['json'],
                       help="Export learned patterns")
    parser.add_argument("--reset", action="store_true",
                       help="Reset learning data")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create learning system
    learning_system = create_adaptive_learning_system()
    
    if args.learn:
        patterns = learning_system.learn_patterns()
        total_patterns = sum(len(p) for p in patterns.values())
        print(f"🧠 Learned {total_patterns} patterns")
    
    elif args.insights:
        insights = learning_system.get_learning_insights()
        print(json.dumps(insights, indent=2))
    
    elif args.recommendations:
        recommendations = learning_system.recommendations
        print(f"💡 {len(recommendations)} active recommendations:")
        for rec in recommendations[:10]:  # Show top 10
            print(f"  • {rec.title} (Impact: {rec.expected_impact:.1%}, Priority: {rec.priority})")
    
    elif args.export:
        export_data = learning_system.export_patterns(args.export)
        print(export_data)
    
    elif args.reset:
        learning_system.reset_learning()
        print("🔄 Learning data reset")
    
    else:
        print("Use --learn, --insights, --recommendations, --export, or --reset")