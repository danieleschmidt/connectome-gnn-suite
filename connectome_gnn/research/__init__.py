"""Research framework for connectome GNN analysis and publication."""

from .experimental import (
    ExperimentalFramework,
    BaselineComparison,
    ExperimentConfig,
    ExperimentResult
)

from .benchmarking import (
    ConnectomeBenchmarkSuite,
    PerformanceProfiler,
    PerformanceMetrics,
    ScalabilityResults
)

from .statistical_validation import (
    StatisticalValidator,
    ValidationReport,
    StatisticalTest,
    ResearchReproducibility
)

from .advanced_visualization import (
    PublicationVisualizer,
    InterpretabilityAnalyzer,
    VisualizationConfig
)

from .interpretability import (
    ConnectomeExplainer,
    BrainAttentionAnalyzer
)

from .novel_architectures import (
    GraphWaveletGNN,
    AttentionPoolingGNN,
    NeuroGraphTransformer
)

from .publication import (
    PublicationPreparation,
    ManuscriptGenerator,
    FigureGenerator,
    ResultsFormatter
)


__all__ = [
    # Experimental framework
    'ExperimentalFramework',
    'BaselineComparison', 
    'ExperimentConfig',
    'ExperimentResult',
    
    # Benchmarking
    'ConnectomeBenchmarkSuite',
    'PerformanceProfiler',
    'PerformanceMetrics',
    'ScalabilityResults',
    
    # Statistical validation
    'StatisticalValidator',
    'ValidationReport',
    'StatisticalTest',
    'ResearchReproducibility',
    
    # Visualization
    'PublicationVisualizer',
    'InterpretabilityAnalyzer',
    'VisualizationConfig',
    
    # Interpretability
    'ConnectomeExplainer',
    'BrainAttentionAnalyzer',
    
    # Novel architectures
    'GraphWaveletGNN',
    'AttentionPoolingGNN',
    'NeuroGraphTransformer',
    
    # Publication tools
    'PublicationPreparation',
    'ManuscriptGenerator',
    'FigureGenerator',
    'ResultsFormatter'
]


# Research workflow convenience functions
def create_research_pipeline(
    experiment_config: dict,
    dataset,
    output_dir: str = "./research_output"
) -> dict:
    """Create a complete research pipeline for connectome GNN analysis.
    
    Args:
        experiment_config: Experiment configuration
        dataset: Dataset for analysis
        output_dir: Output directory for results
        
    Returns:
        Dictionary containing all results and artifacts
    """
    
    from pathlib import Path
    import json
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        'config': experiment_config,
        'artifacts': {},
        'analysis': {}
    }
    
    # 1. Run experimental framework
    exp_framework = ExperimentalFramework(output_dir=output_path / "experiments")
    
    exp_config = ExperimentConfig(
        experiment_name=experiment_config['name'],
        description=experiment_config.get('description', ''),
        model_class=experiment_config['model']['class'],
        model_params=experiment_config['model']['params'],
        task_class=experiment_config['task']['class'],
        task_params=experiment_config['task']['params'],
        training_params=experiment_config.get('training', {}),
        evaluation_metrics=experiment_config.get('metrics', ['accuracy'])
    )
    
    exp_result = exp_framework.run_experiment(exp_config, dataset)
    results['analysis']['experiment'] = exp_result
    
    # 2. Statistical validation
    if experiment_config.get('statistical_validation', True):
        validator = StatisticalValidator()
        
        # Convert experiment results to format expected by validator
        validation_results = {
            'novel_model': exp_result.cv_results[exp_config.evaluation_metrics[0]]
        }
        
        validation_report = validator.validate_experimental_results(
            validation_results, experiment_config
        )
        results['analysis']['statistical'] = validation_report
        
        # Generate report
        report_path = output_path / "statistical_validation_report.md"
        validator.generate_report(validation_report, report_path)
        results['artifacts']['statistical_report'] = str(report_path)
    
    # 3. Visualization
    if experiment_config.get('visualization', True):
        visualizer = PublicationVisualizer(output_dir=output_path / "figures")
        
        # Model comparison (if multiple models)
        if 'baseline_models' in experiment_config:
            comparison_data = {
                experiment_config['name']: {
                    metric: exp_result.test_results.get(metric, 0)
                    for metric in exp_config.evaluation_metrics
                }
            }
            
            for baseline_name, baseline_result in experiment_config['baseline_models'].items():
                comparison_data[baseline_name] = baseline_result
            
            fig_path = visualizer.create_model_comparison_figure(
                comparison_data, exp_config.evaluation_metrics
            )
            results['artifacts']['comparison_figure'] = fig_path
    
    # 4. Save consolidated results
    results_file = output_path / "research_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    results['artifacts']['results_file'] = str(results_file)
    
    return results


def validate_research_reproducibility(
    experiment_config: dict,
    code_path: str = None
) -> dict:
    """Validate research reproducibility checklist.
    
    Args:
        experiment_config: Experiment configuration
        code_path: Path to code repository
        
    Returns:
        Reproducibility assessment
    """
    
    reproducibility = ResearchReproducibility()
    return reproducibility.check_reproducibility(experiment_config, code_path)