"""Advanced Statistical Validation Framework for Novel GNN Architectures.

Implements comprehensive statistical validation including permutation tests,
bootstrap confidence intervals, Bayesian model comparison, and publication-ready
statistical analysis for research papers.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple, Optional, Callable
from dataclasses import dataclass
import json
import warnings
from pathlib import Path

# Statistical libraries
from scipy import stats
from scipy.stats import (
    ttest_rel, wilcoxon, mannwhitneyu, kruskal, friedmanchisquare,
    bootstrap, permutation_test
)
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd

# Bayesian analysis
try:
    import pymc as pm
    import arviz as az
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    warnings.warn("PyMC not available. Bayesian analysis will be limited.")


@dataclass
class ValidationConfig:
    """Configuration for statistical validation procedures."""
    
    # Significance testing
    alpha: float = 0.05
    bonferroni_correction: bool = True
    fdr_correction: str = 'benjamini_hochberg'  # or 'benjamini_yekutieli'
    
    # Bootstrap parameters
    n_bootstrap: int = 10000
    bootstrap_confidence: float = 0.95
    bootstrap_method: str = 'percentile'  # or 'bias_corrected'
    
    # Permutation testing
    n_permutations: int = 10000
    permutation_seed: int = 42
    
    # Cross-validation
    cv_folds: int = 10
    cv_stratified: bool = True
    cv_repeats: int = 5
    
    # Effect size calculations
    effect_size_measures: List[str] = None
    
    # Bayesian analysis (if available)
    mcmc_samples: int = 2000
    mcmc_tune: int = 1000
    mcmc_chains: int = 4
    
    def __post_init__(self):
        if self.effect_size_measures is None:
            self.effect_size_measures = ['cohens_d', 'hedges_g', 'cliff_delta']


@dataclass 
class ValidationResult:
    """Results from statistical validation."""
    
    # Basic statistics
    mean_performance: float
    std_performance: float
    median_performance: float
    
    # Confidence intervals
    ci_lower: float
    ci_upper: float
    ci_method: str
    
    # Significance tests
    p_values: Dict[str, float]
    test_statistics: Dict[str, float]
    significant_tests: List[str]
    
    # Effect sizes
    effect_sizes: Dict[str, float]
    effect_size_interpretations: Dict[str, str]
    
    # Multiple comparison correction
    corrected_p_values: Dict[str, float] 
    significant_after_correction: List[str]
    
    # Cross-validation results
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    
    # Bayesian results (optional)
    bayesian_results: Optional[Dict[str, Any]] = None
    
    # Meta-analysis
    power_analysis: Optional[Dict[str, float]] = None
    sample_size_recommendation: Optional[int] = None


class AdvancedStatisticalValidator:
    """Comprehensive statistical validation for novel GNN architectures."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        np.random.seed(config.permutation_seed)
        
    def validate_model_comparison(
        self,
        baseline_results: np.ndarray,
        novel_results: np.ndarray,
        baseline_name: str = "Baseline",
        novel_name: str = "Novel",
        metric_name: str = "Performance"
    ) -> ValidationResult:
        """Comprehensive statistical validation of model comparison.
        
        Args:
            baseline_results: Performance results from baseline model
            novel_results: Performance results from novel model
            baseline_name: Name of baseline model
            novel_name: Name of novel model
            metric_name: Name of performance metric
            
        Returns:
            ValidationResult with comprehensive statistical analysis
        """
        
        # Basic descriptive statistics
        novel_mean = np.mean(novel_results)
        novel_std = np.std(novel_results, ddof=1)
        novel_median = np.median(novel_results)
        
        # Bootstrap confidence intervals
        ci_lower, ci_upper = self._bootstrap_confidence_interval(novel_results)
        
        # Significance tests
        p_values, test_stats = self._perform_significance_tests(
            baseline_results, novel_results
        )
        
        # Effect size calculations
        effect_sizes = self._calculate_effect_sizes(baseline_results, novel_results)
        effect_interpretations = self._interpret_effect_sizes(effect_sizes)
        
        # Multiple comparison correction
        corrected_p_values = self._apply_multiple_comparison_correction(p_values)
        
        # Determine significant tests
        significant_tests = [
            test for test, p in p_values.items() 
            if p < self.config.alpha
        ]
        significant_after_correction = [
            test for test, p in corrected_p_values.items()
            if p < self.config.alpha
        ]
        
        # Cross-validation analysis
        cv_scores, cv_mean, cv_std = self._cross_validation_analysis(
            novel_results
        )
        
        # Bayesian analysis if available
        bayesian_results = None
        if BAYESIAN_AVAILABLE:
            bayesian_results = self._bayesian_model_comparison(
                baseline_results, novel_results
            )
        
        # Power analysis
        power_analysis = self._power_analysis(baseline_results, novel_results)
        
        return ValidationResult(
            mean_performance=novel_mean,
            std_performance=novel_std,
            median_performance=novel_median,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_method=self.config.bootstrap_method,
            p_values=p_values,
            test_statistics=test_stats,
            significant_tests=significant_tests,
            effect_sizes=effect_sizes,
            effect_size_interpretations=effect_interpretations,
            corrected_p_values=corrected_p_values,
            significant_after_correction=significant_after_correction,
            cv_scores=cv_scores,
            cv_mean=cv_mean,
            cv_std=cv_std,
            bayesian_results=bayesian_results,
            power_analysis=power_analysis
        )
    
    def _bootstrap_confidence_interval(
        self, 
        data: np.ndarray
    ) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval."""
        
        def mean_statistic(x, axis):
            return np.mean(x, axis=axis)
        
        # Use scipy's bootstrap function
        rng = np.random.default_rng(seed=self.config.permutation_seed)
        bootstrap_result = bootstrap(
            (data,), 
            mean_statistic,
            n_resamples=self.config.n_bootstrap,
            confidence_level=self.config.bootstrap_confidence,
            method=self.config.bootstrap_method,
            random_state=rng
        )
        
        return bootstrap_result.confidence_interval.low, bootstrap_result.confidence_interval.high
    
    def _perform_significance_tests(
        self, 
        baseline: np.ndarray, 
        novel: np.ndarray
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Perform multiple significance tests."""
        
        p_values = {}
        test_stats = {}
        
        # Paired t-test (assumes paired data)
        if len(baseline) == len(novel):
            t_stat, t_p = ttest_rel(novel, baseline)
            p_values['paired_t_test'] = t_p
            test_stats['paired_t_test'] = t_stat
        
        # Wilcoxon signed-rank test (non-parametric paired)
        if len(baseline) == len(novel):
            try:
                w_stat, w_p = wilcoxon(novel, baseline, zero_method='wilcox')
                p_values['wilcoxon_signed_rank'] = w_p
                test_stats['wilcoxon_signed_rank'] = w_stat
            except ValueError:
                # Handle case where differences are all zero
                p_values['wilcoxon_signed_rank'] = 1.0
                test_stats['wilcoxon_signed_rank'] = 0.0
        
        # Mann-Whitney U test (independent samples)
        u_stat, u_p = mannwhitneyu(novel, baseline, alternative='two-sided')
        p_values['mann_whitney_u'] = u_p
        test_stats['mann_whitney_u'] = u_stat
        
        # Permutation test
        def test_statistic(x, y):
            return np.mean(x) - np.mean(y)
        
        perm_result = permutation_test(
            (novel, baseline),
            test_statistic,
            n_resamples=self.config.n_permutations,
            alternative='two-sided',
            random_state=self.config.permutation_seed
        )
        p_values['permutation_test'] = perm_result.pvalue
        test_stats['permutation_test'] = test_statistic(novel, baseline)
        
        return p_values, test_stats
    
    def _calculate_effect_sizes(
        self, 
        baseline: np.ndarray, 
        novel: np.ndarray
    ) -> Dict[str, float]:
        """Calculate various effect size measures."""
        
        effect_sizes = {}
        
        # Cohen's d
        pooled_std = np.sqrt(
            ((len(baseline) - 1) * np.var(baseline, ddof=1) + 
             (len(novel) - 1) * np.var(novel, ddof=1)) /
            (len(baseline) + len(novel) - 2)
        )
        if pooled_std > 0:
            cohens_d = (np.mean(novel) - np.mean(baseline)) / pooled_std
            effect_sizes['cohens_d'] = cohens_d
        else:
            effect_sizes['cohens_d'] = 0.0
        
        # Hedges' g (bias-corrected Cohen's d)
        if pooled_std > 0:
            n = len(baseline) + len(novel)
            j = 1 - (3 / (4 * n - 9))  # Bias correction factor
            hedges_g = effect_sizes['cohens_d'] * j
            effect_sizes['hedges_g'] = hedges_g
        else:
            effect_sizes['hedges_g'] = 0.0
        
        # Cliff's Delta (non-parametric effect size)
        cliff_delta = self._calculate_cliff_delta(baseline, novel)
        effect_sizes['cliff_delta'] = cliff_delta
        
        # Glass's Delta
        if np.std(baseline, ddof=1) > 0:
            glass_delta = (np.mean(novel) - np.mean(baseline)) / np.std(baseline, ddof=1)
            effect_sizes['glass_delta'] = glass_delta
        else:
            effect_sizes['glass_delta'] = 0.0
        
        return effect_sizes
    
    def _calculate_cliff_delta(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cliff's Delta effect size."""
        n1, n2 = len(group1), len(group2)
        
        # Count pairs where group2 > group1, group2 < group1
        greater = np.sum(group2.reshape(-1, 1) > group1.reshape(1, -1))
        less = np.sum(group2.reshape(-1, 1) < group1.reshape(1, -1))
        
        cliff_delta = (greater - less) / (n1 * n2)
        return cliff_delta
    
    def _interpret_effect_sizes(self, effect_sizes: Dict[str, float]) -> Dict[str, str]:
        """Interpret effect size magnitudes."""
        
        interpretations = {}
        
        # Cohen's d interpretation
        if 'cohens_d' in effect_sizes:
            d = abs(effect_sizes['cohens_d'])
            if d < 0.2:
                interpretations['cohens_d'] = 'negligible'
            elif d < 0.5:
                interpretations['cohens_d'] = 'small'
            elif d < 0.8:
                interpretations['cohens_d'] = 'medium'
            else:
                interpretations['cohens_d'] = 'large'
        
        # Cliff's Delta interpretation
        if 'cliff_delta' in effect_sizes:
            delta = abs(effect_sizes['cliff_delta'])
            if delta < 0.147:
                interpretations['cliff_delta'] = 'negligible'
            elif delta < 0.33:
                interpretations['cliff_delta'] = 'small'
            elif delta < 0.474:
                interpretations['cliff_delta'] = 'medium'
            else:
                interpretations['cliff_delta'] = 'large'
        
        return interpretations
    
    def _apply_multiple_comparison_correction(
        self, 
        p_values: Dict[str, float]
    ) -> Dict[str, float]:
        """Apply multiple comparison corrections."""
        
        test_names = list(p_values.keys())
        p_vals = list(p_values.values())
        
        # Apply FDR correction
        if self.config.fdr_correction:
            rejected, corrected_p, _, _ = multipletests(
                p_vals, 
                alpha=self.config.alpha,
                method=self.config.fdr_correction
            )
            corrected_p_values = dict(zip(test_names, corrected_p))
        else:
            corrected_p_values = p_values.copy()
        
        # Apply Bonferroni if requested
        if self.config.bonferroni_correction:
            bonferroni_p = [min(p * len(p_vals), 1.0) for p in p_vals]
            bonferroni_corrected = dict(zip(test_names, bonferroni_p))
            
            # Use the most conservative correction
            for test in test_names:
                corrected_p_values[test] = max(
                    corrected_p_values[test],
                    bonferroni_corrected[test]
                )
        
        return corrected_p_values
    
    def _cross_validation_analysis(self, results: np.ndarray) -> Tuple[List[float], float, float]:
        """Analyze cross-validation stability."""
        
        # If results represent CV folds, return as-is
        if len(results) >= self.config.cv_folds:
            cv_scores = results.tolist()
            cv_mean = np.mean(results)
            cv_std = np.std(results, ddof=1)
        else:
            # Simulate CV analysis (placeholder)
            cv_scores = results.tolist()
            cv_mean = np.mean(results)
            cv_std = np.std(results, ddof=1)
        
        return cv_scores, cv_mean, cv_std
    
    def _bayesian_model_comparison(
        self, 
        baseline: np.ndarray, 
        novel: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """Perform Bayesian model comparison."""
        
        if not BAYESIAN_AVAILABLE:
            return None
        
        try:
            with pm.Model() as model:
                # Priors for means and standard deviations
                mu_baseline = pm.Normal('mu_baseline', mu=0, sigma=10)
                mu_novel = pm.Normal('mu_novel', mu=0, sigma=10)
                
                sigma_baseline = pm.HalfNormal('sigma_baseline', sigma=10)
                sigma_novel = pm.HalfNormal('sigma_novel', sigma=10)
                
                # Likelihoods
                obs_baseline = pm.Normal('obs_baseline', mu=mu_baseline, 
                                       sigma=sigma_baseline, observed=baseline)
                obs_novel = pm.Normal('obs_novel', mu=mu_novel,
                                    sigma=sigma_novel, observed=novel)
                
                # Difference in means
                diff = pm.Deterministic('difference', mu_novel - mu_baseline)
                
                # Sample from posterior
                trace = pm.sample(
                    draws=self.config.mcmc_samples,
                    tune=self.config.mcmc_tune,
                    chains=self.config.mcmc_chains,
                    return_inferencedata=True,
                    progressbar=False
                )
            
            # Extract results
            posterior = trace.posterior
            diff_samples = posterior['difference'].values.flatten()
            
            # Calculate probabilities
            prob_novel_better = np.mean(diff_samples > 0)
            prob_significant = np.mean(np.abs(diff_samples) > 0.1)  # Practical significance
            
            # Credible interval for difference
            ci_diff = np.percentile(diff_samples, [2.5, 97.5])
            
            # Model comparison using WAIC
            waic_baseline = az.waic(trace, var_name='obs_baseline')
            waic_novel = az.waic(trace, var_name='obs_novel')
            
            return {
                'probability_novel_better': prob_novel_better,
                'probability_significant_difference': prob_significant,
                'difference_credible_interval': ci_diff.tolist(),
                'waic_baseline': waic_baseline.waic,
                'waic_novel': waic_novel.waic,
                'bayes_factor_approximation': np.exp(
                    (waic_baseline.waic - waic_novel.waic) / 2
                )
            }
            
        except Exception as e:
            warnings.warn(f"Bayesian analysis failed: {str(e)}")
            return None
    
    def _power_analysis(
        self, 
        baseline: np.ndarray, 
        novel: np.ndarray
    ) -> Dict[str, float]:
        """Perform statistical power analysis."""
        
        # Calculate observed effect size
        effect_size = self._calculate_effect_sizes(baseline, novel)['cohens_d']
        
        # Estimate power for current sample size
        n = min(len(baseline), len(novel))
        
        # Simplified power calculation (for t-test)
        # This is an approximation - proper power analysis would use specialized libraries
        alpha = self.config.alpha
        df = 2 * n - 2
        
        # Non-centrality parameter
        ncp = abs(effect_size) * np.sqrt(n / 2)
        
        # Critical t-value
        t_crit = stats.t.ppf(1 - alpha/2, df)
        
        # Power approximation
        power = 1 - stats.t.cdf(t_crit, df, ncp) + stats.t.cdf(-t_crit, df, ncp)
        
        # Sample size recommendation for 80% power
        target_power = 0.8
        if abs(effect_size) > 0:
            # Simplified calculation
            recommended_n = int(2 * (
                (stats.t.ppf(1 - alpha/2, float('inf')) + 
                 stats.t.ppf(target_power, float('inf'))) ** 2
            ) / (effect_size ** 2))
        else:
            recommended_n = n
        
        return {
            'current_power': power,
            'target_power': target_power,
            'recommended_sample_size': recommended_n,
            'current_sample_size': n,
            'observed_effect_size': effect_size
        }
    
    def generate_publication_report(
        self, 
        validation_result: ValidationResult,
        output_path: Optional[Path] = None
    ) -> str:
        """Generate publication-ready statistical report."""
        
        report = []
        
        # Header
        report.append("# Statistical Validation Report")
        report.append("## Model Performance Analysis")
        report.append("")
        
        # Descriptive statistics
        report.append("### Descriptive Statistics")
        report.append(f"Mean performance: {validation_result.mean_performance:.4f}")
        report.append(f"Standard deviation: {validation_result.std_performance:.4f}")
        report.append(f"Median performance: {validation_result.median_performance:.4f}")
        report.append(f"95% Confidence interval: [{validation_result.ci_lower:.4f}, {validation_result.ci_upper:.4f}]")
        report.append("")
        
        # Significance tests
        report.append("### Significance Testing")
        for test, p_value in validation_result.p_values.items():
            significance = "significant" if p_value < self.config.alpha else "not significant"
            report.append(f"{test}: p = {p_value:.4f} ({significance})")
        
        report.append("")
        report.append("### Multiple Comparison Correction")
        for test, p_value in validation_result.corrected_p_values.items():
            significance = "significant" if p_value < self.config.alpha else "not significant"
            report.append(f"{test} (corrected): p = {p_value:.4f} ({significance})")
        
        report.append("")
        
        # Effect sizes
        report.append("### Effect Sizes")
        for effect, size in validation_result.effect_sizes.items():
            interpretation = validation_result.effect_size_interpretations.get(effect, "unknown")
            report.append(f"{effect}: {size:.4f} ({interpretation})")
        
        report.append("")
        
        # Cross-validation
        report.append("### Cross-Validation Analysis")
        report.append(f"CV Mean: {validation_result.cv_mean:.4f}")
        report.append(f"CV Standard deviation: {validation_result.cv_std:.4f}")
        report.append("")
        
        # Power analysis
        if validation_result.power_analysis:
            power = validation_result.power_analysis
            report.append("### Power Analysis")
            report.append(f"Current power: {power['current_power']:.4f}")
            report.append(f"Recommended sample size for 80% power: {power['recommended_sample_size']}")
            report.append("")
        
        # Bayesian results
        if validation_result.bayesian_results:
            bayes = validation_result.bayesian_results
            report.append("### Bayesian Analysis")
            report.append(f"Probability novel model is better: {bayes['probability_novel_better']:.4f}")
            report.append(f"Bayes Factor (approximation): {bayes['bayes_factor_approximation']:.4f}")
            report.append("")
        
        # Conclusions
        report.append("### Statistical Conclusions")
        if validation_result.significant_after_correction:
            report.append("The novel model shows statistically significant improvement after multiple comparison correction.")
            report.append(f"Significant tests: {', '.join(validation_result.significant_after_correction)}")
        else:
            report.append("The novel model does not show statistically significant improvement after multiple comparison correction.")
        
        # Effect size interpretation
        cohens_d = validation_result.effect_sizes.get('cohens_d', 0)
        if abs(cohens_d) >= 0.8:
            report.append("The effect size indicates a large practical significance.")
        elif abs(cohens_d) >= 0.5:
            report.append("The effect size indicates a medium practical significance.")
        elif abs(cohens_d) >= 0.2:
            report.append("The effect size indicates a small practical significance.")
        else:
            report.append("The effect size indicates negligible practical significance.")
        
        report_text = "\n".join(report)
        
        # Save to file if path provided
        if output_path:
            output_path.write_text(report_text)
        
        return report_text


class ModelComparator:
    """High-level interface for comparing multiple models."""
    
    def __init__(self, config: ValidationConfig):
        self.validator = AdvancedStatisticalValidator(config)
        self.config = config
    
    def compare_models(
        self,
        results: Dict[str, np.ndarray],
        baseline_name: str = None
    ) -> Dict[str, ValidationResult]:
        """Compare multiple models against a baseline.
        
        Args:
            results: Dictionary mapping model names to performance arrays
            baseline_name: Name of baseline model (if None, uses first model)
            
        Returns:
            Dictionary mapping comparisons to ValidationResult objects
        """
        
        model_names = list(results.keys())
        if baseline_name is None:
            baseline_name = model_names[0]
        
        if baseline_name not in results:
            raise ValueError(f"Baseline model '{baseline_name}' not found in results")
        
        baseline_results = results[baseline_name]
        comparisons = {}
        
        for model_name, model_results in results.items():
            if model_name == baseline_name:
                continue
                
            comparison_name = f"{model_name}_vs_{baseline_name}"
            validation_result = self.validator.validate_model_comparison(
                baseline_results,
                model_results,
                baseline_name,
                model_name
            )
            comparisons[comparison_name] = validation_result
        
        return comparisons
    
    def rank_models(
        self, 
        results: Dict[str, np.ndarray],
        metric_name: str = "Performance"
    ) -> List[Tuple[str, float, float]]:
        """Rank models by performance with statistical significance.
        
        Returns:
            List of tuples (model_name, mean_performance, confidence_interval_width)
        """
        
        model_stats = []
        
        for model_name, model_results in results.items():
            mean_perf = np.mean(model_results)
            ci_lower, ci_upper = self.validator._bootstrap_confidence_interval(model_results)
            ci_width = ci_upper - ci_lower
            
            model_stats.append((model_name, mean_perf, ci_width))
        
        # Sort by mean performance (descending)
        model_stats.sort(key=lambda x: x[1], reverse=True)
        
        return model_stats