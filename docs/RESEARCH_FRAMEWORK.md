# Advanced Research Framework for Connectome-GNN-Suite

## Overview

The Connectome-GNN-Suite includes a comprehensive research framework designed for conducting rigorous, publication-ready neuroscience research using graph neural networks. This framework implements the **TERRAGON SDLC** methodology with advanced experimental design, statistical validation, and publication tools.

## 🔬 Research Components

### 1. Experimental Framework (`connectome_gnn.research.experimental`)

**Purpose**: Conduct reproducible, statistically rigorous experiments with proper cross-validation and baseline comparisons.

**Key Features**:
- Automated cross-validation with multiple metrics
- Baseline model comparisons 
- Statistical significance testing
- Confidence interval computation
- Experiment registry and caching

**Example Usage**:
```python
from connectome_gnn.research import ExperimentalFramework, ExperimentConfig

# Configure experiment
config = ExperimentConfig(
    experiment_name="hierarchical_vs_baseline",
    description="Comparing hierarchical brain GNN against baselines",
    model_class="HierarchicalBrainGNN",
    model_params={
        "node_features": 100,
        "hidden_dim": 256,
        "num_levels": 4,
        "attention_type": "dense"
    },
    task_class="CognitiveScorePrediction",
    task_params={"target": "fluid_intelligence"},
    evaluation_metrics=["mae", "r2", "accuracy"],
    cv_folds=5,
    random_seed=42
)

# Run experiment
framework = ExperimentalFramework("./experiments")
results = framework.run_experiment(config, dataset)

print(f"CV Score: {results.mean_cv_score:.4f} ± {results.std_cv_score:.4f}")
print(f"Test Score: {results.test_results['mae']:.4f}")
```

### 2. Statistical Validation (`connectome_gnn.research.statistical_validation`)

**Purpose**: Comprehensive statistical analysis with assumption checking, effect size computation, and multiple comparison correction.

**Key Features**:
- Normality and homogeneity testing
- Parametric and non-parametric test selection
- Effect size calculations (Cohen's d, eta-squared, etc.)
- Multiple comparison corrections (Holm, Bonferroni, FDR)
- Power analysis
- Reproducibility assessment

**Example Usage**:
```python
from connectome_gnn.research import StatisticalValidator

validator = StatisticalValidator(alpha=0.05, correction_method="holm")

# Compare multiple models
results = {
    "novel_model": [0.85, 0.87, 0.84, 0.88, 0.86],
    "baseline_gcn": [0.78, 0.79, 0.77, 0.80, 0.76],
    "simple_mlp": [0.72, 0.74, 0.71, 0.73, 0.70]
}

report = validator.validate_experimental_results(
    results, 
    experiment_config={'experiment_name': 'model_comparison'},
    comparison_type="between_groups"
)

print(f"Validity Score: {report.validity_score:.2f}/1.00")
print(f"Effect Size (Cohen's d): {report.effect_sizes.get('cohens_d', 'N/A')}")

# Generate publication-ready report
validator.generate_report(report, "statistical_report.md")
```

### 3. Advanced Visualization (`connectome_gnn.research.advanced_visualization`)

**Purpose**: Create publication-quality figures with advanced brain connectivity visualizations and interpretability analysis.

**Key Features**:
- Publication-style matplotlib configuration
- Statistical results visualization
- Model comparison figures
- Training dynamics plots
- Brain connectivity visualizations
- Attention weight analysis
- Interactive Plotly dashboards
- Embedding visualizations (UMAP, t-SNE, PCA)

**Example Usage**:
```python
from connectome_gnn.research import PublicationVisualizer, VisualizationConfig

# Configure for publication
config = VisualizationConfig(
    figure_size=(12, 8),
    dpi=300,
    style="publication",
    save_format="pdf"
)

visualizer = PublicationVisualizer(config, "./figures")

# Create model comparison figure
model_results = {
    "HierarchicalGNN": {"accuracy": 0.87, "f1": 0.85},
    "TemporalGNN": {"accuracy": 0.84, "f1": 0.82},
    "BaselineGCN": {"accuracy": 0.78, "f1": 0.76}
}

fig_path = visualizer.create_model_comparison_figure(
    model_results, 
    metrics=["accuracy", "f1"]
)

# Create brain connectivity visualization
connectivity_fig = visualizer.create_brain_connectivity_figure(
    connectivity_matrix,
    coordinates=brain_coordinates,
    node_labels=region_names
)

# Generate interactive dashboard
dashboard = visualizer.create_interactive_dashboard({
    'model_performance': model_results,
    'training_history': training_logs,
    'attention_weights': attention_matrices,
    'connectivity_matrix': connectivity_data
})
```

### 4. Benchmarking Suite (`connectome_gnn.research.benchmarking`)

**Purpose**: Comprehensive performance evaluation including scalability, robustness, and computational efficiency analysis.

**Key Features**:
- Performance metrics (accuracy, speed, memory)
- Scalability analysis across dataset sizes
- Robustness testing (noise, dropout, perturbations)
- Memory profiling and optimization
- FLOP estimation
- Time complexity analysis

**Example Usage**:
```python
from connectome_gnn.research import ConnectomeBenchmarkSuite

benchmark = ConnectomeBenchmarkSuite("./benchmarks")

results = benchmark.run_comprehensive_benchmark(
    model=trained_model,
    task=prediction_task,
    dataset=test_dataset,
    benchmark_name="hierarchical_gnn_benchmark",
    include_scalability=True,
    include_robustness=True
)

print(f"Test Accuracy: {results['performance']['test_accuracy']:.4f}")
print(f"Training Time: {results['performance']['training_time']:.2f}s")
print(f"Memory Usage: {results['performance']['memory_usage']:.2f}GB")
print(f"Scalability Score: {results['scalability']['scalability_score']:.3f}")
```

### 5. Interpretability Analysis (`connectome_gnn.research.advanced_visualization`)

**Purpose**: Advanced model interpretability using gradient-based methods and attention analysis.

**Key Features**:
- Integrated gradients attribution
- Saliency map computation
- Feature importance ranking
- Attention weight extraction
- Node embedding analysis
- Brain region importance scoring

**Example Usage**:
```python
from connectome_gnn.research import InterpretabilityAnalyzer

analyzer = InterpretabilityAnalyzer(trained_model)

# Analyze feature importance
importance = analyzer.analyze_feature_importance(
    data_sample,
    method="integrated_gradients"
)

# Extract attention weights
attention = analyzer.extract_attention_weights(data_sample)

# Visualize important brain regions
visualizer.create_attention_visualization(
    attention,
    node_labels=brain_regions
)
```

## 🔬 Research Workflow

### Complete Research Pipeline

The framework provides a high-level interface for running complete research studies:

```python
from connectome_gnn.research import create_research_pipeline

# Define experiment configuration
experiment_config = {
    'name': 'novel_architecture_study',
    'description': 'Evaluating novel hierarchical GNN for cognitive prediction',
    'model': {
        'class': 'HierarchicalBrainGNN',
        'params': {'hidden_dim': 256, 'num_levels': 4}
    },
    'task': {
        'class': 'CognitiveScorePrediction',
        'params': {'target': 'fluid_intelligence'}
    },
    'training': {'batch_size': 16, 'learning_rate': 1e-3},
    'metrics': ['mae', 'r2'],
    'statistical_validation': True,
    'visualization': True,
    'baseline_models': {
        'SimpleGCN': {'mae': 0.15, 'r2': 0.72},
        'MLP': {'mae': 0.18, 'r2': 0.68}
    }
}

# Run complete pipeline
results = create_research_pipeline(
    experiment_config, 
    dataset, 
    output_dir="./research_output"
)

# Results include:
# - Experimental results with cross-validation
# - Statistical validation report
# - Publication-quality figures
# - Reproducibility assessment
```

### Reproducibility Validation

```python
from connectome_gnn.research import validate_research_reproducibility

reproducibility = validate_research_reproducibility(
    experiment_config,
    code_path="./connectome_gnn"
)

print(f"Reproducibility Score: {reproducibility['reproducibility_percentage']:.1f}%")
for component, status in reproducibility['checklist_results'].items():
    print(f"- {component}: {'✓' if status else '✗'}")
```

## 📊 Statistical Methods

### Supported Statistical Tests

**Parametric Tests**:
- Independent samples t-test
- Paired samples t-test  
- One-way ANOVA
- Repeated measures ANOVA

**Non-parametric Tests**:
- Mann-Whitney U test
- Wilcoxon signed-rank test
- Kruskal-Wallis test
- Friedman test

**Effect Size Measures**:
- Cohen's d
- Hedges' g
- Glass's delta
- Eta-squared
- Omega-squared

**Multiple Comparison Corrections**:
- Bonferroni
- Holm-Bonferroni
- False Discovery Rate (FDR)
- Sidak

### Assumption Checking

The framework automatically checks statistical assumptions:

- **Normality**: Shapiro-Wilk, Anderson-Darling, Kolmogorov-Smirnov
- **Homogeneity of variance**: Levene's test
- **Outlier detection**: IQR method
- **Independence**: Through experimental design

## 🎯 Publication Tools

### Manuscript Generation

```python
from connectome_gnn.research import PublicationPreparation

pub_prep = PublicationPreparation()

# Generate manuscript draft
manuscript = pub_prep.generate_manuscript_draft(
    research_results,
    output_path="./publication/manuscript.md"
)

# Generate supplementary materials
supplementary = pub_prep.generate_supplementary_materials(
    research_results,
    output_path="./publication/supplementary"
)
```

### Figure Standards

All visualizations follow academic publication standards:

- **Resolution**: 300+ DPI for print quality
- **Fonts**: Arial/DejaVu Sans family
- **Color schemes**: Colorblind-friendly palettes
- **Error bars**: 95% confidence intervals or ±1 SEM
- **Statistical annotations**: Significance levels marked
- **Multi-panel layouts**: Labeled (A), (B), (C)...

## 🏗️ Architecture Integration

### Novel Architecture Support

The framework seamlessly integrates with novel architectures:

```python
# Custom architecture with research framework
class MyNovelGNN(BaseConnectomeModel):
    def __init__(self, **kwargs):
        super().__init__()
        # Architecture implementation
        
    def get_attention_weights(self):
        # Return attention weights for interpretability
        return self.attention_weights
        
    def get_embeddings(self):
        # Return node embeddings for visualization
        return self.node_embeddings

# Use with research framework
results = framework.run_experiment(
    config_with_novel_architecture,
    dataset
)
```

### Multi-Modal Analysis

```python
# Multi-modal connectome analysis
config = ExperimentConfig(
    model_class="MultiModalBrainGNN",
    model_params={
        "structural_features": 100,
        "functional_features": 200,
        "fusion_method": "adaptive"
    }
)
```

## 🔒 Quality Assurance

### Validation Checklist

- ✅ **Statistical rigor**: Proper test selection and assumption checking
- ✅ **Effect size reporting**: Clinical and practical significance
- ✅ **Multiple comparison correction**: Family-wise error control  
- ✅ **Confidence intervals**: Uncertainty quantification
- ✅ **Power analysis**: Adequate sample size determination
- ✅ **Reproducibility**: Seed setting and environment documentation
- ✅ **Cross-validation**: Unbiased performance estimation
- ✅ **Baseline comparisons**: Context for novel contributions

### Research Standards Compliance

The framework ensures compliance with:

- **CONSORT guidelines** for randomized studies
- **STROBE guidelines** for observational studies  
- **TRIPOD guidelines** for prediction models
- **PRISMA guidelines** for systematic reviews
- **APA guidelines** for statistical reporting

## 📈 Success Metrics

### Research Impact Indicators

- **Statistical significance**: p < 0.05 with multiple comparison correction
- **Effect size**: Cohen's d > 0.5 for meaningful differences
- **Reproducibility score**: > 80% for publication readiness
- **Validity score**: > 0.8 for methodological rigor
- **Power**: > 0.8 for adequate sample size

### Publication Readiness

The framework generates publication-ready materials:

- 📄 **Manuscript draft** with methods and results
- 📊 **High-quality figures** at publication resolution
- 📋 **Statistical reports** with detailed analysis
- 🔍 **Supplementary materials** with additional details
- ✅ **Reproducibility checklist** for peer review

## 🚀 Getting Started

### Quick Start Example

```python
# 1. Import research framework
from connectome_gnn.research import *

# 2. Load your data
dataset = ConnectomeDataset("data/hcp", resolution="7mm")

# 3. Configure experiment
config = {
    'name': 'my_research_study',
    'model': {'class': 'HierarchicalBrainGNN', 'params': {}},
    'task': {'class': 'CognitiveScorePrediction', 'params': {}},
    'metrics': ['mae', 'r2']
}

# 4. Run complete research pipeline
results = create_research_pipeline(config, dataset)

# 5. Generate publication materials
materials = generate_publication_materials(results)

print("Research study complete!")
print(f"Results: {results['artifacts']['results_file']}")
print(f"Figures: {materials['figures_dir']}")
print(f"Manuscript: {materials['manuscript']}")
```

This research framework transforms the Connectome-GNN-Suite into a comprehensive platform for conducting rigorous, publication-ready neuroscience research with state-of-the-art statistical validation and visualization capabilities.