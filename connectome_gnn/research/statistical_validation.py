"""Advanced statistical validation framework for connectome GNN research."""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import json

from scipy import stats
from scipy.stats import (
    ttest_ind, wilcoxon, mannwhitneyu, kruskal, friedmanchisquare,
    chi2_contingency, fisher_exact, pearsonr, spearmanr,
    shapiro, levene, anderson, kstest
)
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import ttest_power
from statsmodels.stats.effect_size import cohens_d
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class StatisticalTest:
    """Statistical test configuration and results."""
    
    test_name: str
    test_type: str  # 'parametric', 'non_parametric', 'correlation', 'effect_size'
    assumptions: List[str]
    statistic: float
    p_value: float
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    power: Optional[float] = None
    interpretation: str = ""
    assumptions_met: Dict[str, bool] = None


@dataclass
class ValidationReport:
    """Comprehensive validation report for research results."""
    
    experiment_name: str
    sample_sizes: Dict[str, int]
    descriptive_stats: Dict[str, Dict[str, float]]
    statistical_tests: List[StatisticalTest]
    multiple_comparison_correction: Dict[str, Any]
    effect_sizes: Dict[str, float]
    power_analysis: Dict[str, float]
    recommendations: List[str]
    validity_score: float


class StatisticalValidator:
    """Advanced statistical validation for connectome GNN research."""
    
    def __init__(self, alpha: float = 0.05, correction_method: str = "holm"):
        """Initialize statistical validator.
        
        Args:
            alpha: Significance level
            correction_method: Multiple comparison correction method
        """
        self.alpha = alpha
        self.correction_method = correction_method
        self.results_cache = {}
        
        # Test registry
        self.parametric_tests = {
            'independent_ttest': self._independent_ttest,
            'paired_ttest': self._paired_ttest,
            'one_way_anova': self._one_way_anova,
            'repeated_measures_anova': self._repeated_measures_anova
        }
        
        self.nonparametric_tests = {
            'mann_whitney': self._mann_whitney,
            'wilcoxon': self._wilcoxon,
            'kruskal_wallis': self._kruskal_wallis,
            'friedman': self._friedman
        }
        
        self.correlation_tests = {
            'pearson': self._pearson_correlation,
            'spearman': self._spearman_correlation,
            'kendall': self._kendall_correlation
        }
    
    def validate_experimental_results(
        self,
        results: Dict[str, List[float]],
        experiment_config: Dict[str, Any],
        comparison_type: str = "between_groups"
    ) -> ValidationReport:
        """Validate experimental results with comprehensive statistical analysis.
        
        Args:
            results: Dictionary of group_name -> list of scores
            experiment_config: Experimental configuration
            comparison_type: Type of comparison ('between_groups', 'within_subjects')
            
        Returns:
            Comprehensive validation report
        """
        
        # Descriptive statistics
        descriptive_stats = self._compute_descriptive_statistics(results)
        
        # Check assumptions
        assumption_checks = self._check_statistical_assumptions(results)
        
        # Select appropriate tests
        selected_tests = self._select_statistical_tests(
            results, comparison_type, assumption_checks
        )
        
        # Run statistical tests
        test_results = []
        for test_config in selected_tests:
            test_result = self._run_statistical_test(results, test_config)
            test_results.append(test_result)
        
        # Multiple comparison correction
        p_values = [test.p_value for test in test_results]
        corrected_results = self._apply_multiple_comparison_correction(p_values)
        
        # Update test results with corrected p-values
        for i, test in enumerate(test_results):
            test.p_value = corrected_results['corrected_p_values'][i]
        
        # Effect size analysis
        effect_sizes = self._compute_effect_sizes(results, comparison_type)
        
        # Power analysis
        power_analysis = self._conduct_power_analysis(results, test_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            test_results, effect_sizes, power_analysis, assumption_checks
        )
        
        # Compute validity score
        validity_score = self._compute_validity_score(
            test_results, assumption_checks, effect_sizes, power_analysis
        )
        
        # Create report
        report = ValidationReport(
            experiment_name=experiment_config.get('experiment_name', 'Unknown'),
            sample_sizes={group: len(scores) for group, scores in results.items()},
            descriptive_stats=descriptive_stats,
            statistical_tests=test_results,
            multiple_comparison_correction=corrected_results,
            effect_sizes=effect_sizes,
            power_analysis=power_analysis,
            recommendations=recommendations,
            validity_score=validity_score
        )
        
        return report
    
    def _compute_descriptive_statistics(
        self, 
        results: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, float]]:
        """Compute comprehensive descriptive statistics."""
        
        descriptive_stats = {}
        
        for group_name, scores in results.items():
            scores_array = np.array(scores)
            
            stats_dict = {
                'n': len(scores),
                'mean': np.mean(scores_array),
                'std': np.std(scores_array, ddof=1),
                'sem': stats.sem(scores_array),
                'median': np.median(scores_array),
                'q25': np.percentile(scores_array, 25),
                'q75': np.percentile(scores_array, 75),
                'min': np.min(scores_array),
                'max': np.max(scores_array),
                'range': np.ptp(scores_array),
                'iqr': np.percentile(scores_array, 75) - np.percentile(scores_array, 25),
                'skewness': stats.skew(scores_array),
                'kurtosis': stats.kurtosis(scores_array),
                'variance': np.var(scores_array, ddof=1)
            }
            
            # Confidence intervals
            ci_95 = stats.t.interval(
                0.95, len(scores) - 1, 
                loc=stats_dict['mean'], 
                scale=stats_dict['sem']
            )
            stats_dict['ci_95_lower'] = ci_95[0]
            stats_dict['ci_95_upper'] = ci_95[1]
            
            descriptive_stats[group_name] = stats_dict
        
        return descriptive_stats
    
    def _check_statistical_assumptions(
        self, 
        results: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, bool]]:
        """Check statistical assumptions for parametric tests."""
        
        assumption_checks = {}
        
        for group_name, scores in results.items():
            scores_array = np.array(scores)
            
            checks = {}
            
            # Normality tests
            if len(scores) >= 3:
                try:
                    # Shapiro-Wilk test (n < 50)
                    if len(scores) < 50:
                        shapiro_stat, shapiro_p = shapiro(scores_array)
                        checks['normality_shapiro'] = shapiro_p > self.alpha
                    
                    # Anderson-Darling test
                    anderson_result = anderson(scores_array, dist='norm')
                    checks['normality_anderson'] = anderson_result.statistic < anderson_result.critical_values[2]  # 5% level
                    
                    # Kolmogorov-Smirnov test
                    ks_stat, ks_p = kstest(scores_array, 'norm', args=(np.mean(scores_array), np.std(scores_array)))
                    checks['normality_ks'] = ks_p > self.alpha
                    
                except:
                    checks['normality_shapiro'] = False
                    checks['normality_anderson'] = False
                    checks['normality_ks'] = False
            else:
                checks['normality_shapiro'] = False
                checks['normality_anderson'] = False
                checks['normality_ks'] = False
            
            # Outlier detection (IQR method)
            q1, q3 = np.percentile(scores_array, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = np.sum((scores_array < lower_bound) | (scores_array > upper_bound))
            checks['no_extreme_outliers'] = outliers <= 0.05 * len(scores)  # < 5% outliers
            
            assumption_checks[group_name] = checks
        
        # Homogeneity of variance (Levene's test)
        if len(results) >= 2:
            all_scores = list(results.values())
            if all(len(scores) >= 2 for scores in all_scores):
                try:
                    levene_stat, levene_p = levene(*all_scores)
                    assumption_checks['homogeneity_of_variance'] = {'levene_test': levene_p > self.alpha}
                except:
                    assumption_checks['homogeneity_of_variance'] = {'levene_test': False}
        
        return assumption_checks
    
    def _select_statistical_tests(
        self,
        results: Dict[str, List[float]],
        comparison_type: str,
        assumption_checks: Dict[str, Dict[str, bool]]
    ) -> List[Dict[str, Any]]:
        """Select appropriate statistical tests based on data and assumptions."""
        
        selected_tests = []
        n_groups = len(results)
        
        # Check if parametric assumptions are met
        normality_met = self._check_overall_normality(assumption_checks)
        homogeneity_met = assumption_checks.get('homogeneity_of_variance', {}).get('levene_test', False)
        
        if comparison_type == "between_groups":
            if n_groups == 2:
                # Two-group comparison
                if normality_met and homogeneity_met:
                    selected_tests.append({
                        'test_name': 'independent_ttest',
                        'test_type': 'parametric',
                        'assumptions': ['normality', 'homogeneity_of_variance', 'independence']
                    })
                else:
                    selected_tests.append({
                        'test_name': 'mann_whitney',
                        'test_type': 'non_parametric',
                        'assumptions': ['independence']
                    })
            elif n_groups > 2:
                # Multi-group comparison
                if normality_met and homogeneity_met:
                    selected_tests.append({
                        'test_name': 'one_way_anova',
                        'test_type': 'parametric',
                        'assumptions': ['normality', 'homogeneity_of_variance', 'independence']
                    })
                else:
                    selected_tests.append({
                        'test_name': 'kruskal_wallis',
                        'test_type': 'non_parametric',
                        'assumptions': ['independence']
                    })
        
        elif comparison_type == "within_subjects":
            if n_groups == 2:
                # Paired comparison
                if normality_met:
                    selected_tests.append({
                        'test_name': 'paired_ttest',
                        'test_type': 'parametric',
                        'assumptions': ['normality_of_differences']
                    })
                else:
                    selected_tests.append({
                        'test_name': 'wilcoxon',
                        'test_type': 'non_parametric',
                        'assumptions': ['symmetry_of_differences']
                    })
            elif n_groups > 2:
                # Repeated measures
                if normality_met:
                    selected_tests.append({
                        'test_name': 'repeated_measures_anova',
                        'test_type': 'parametric',
                        'assumptions': ['normality', 'sphericity']
                    })
                else:
                    selected_tests.append({
                        'test_name': 'friedman',
                        'test_type': 'non_parametric',
                        'assumptions': ['block_design']
                    })
        
        return selected_tests
    
    def _check_overall_normality(self, assumption_checks: Dict[str, Dict[str, bool]]) -> bool:
        """Check if normality assumption is met across all groups."""
        
        normality_results = []
        
        for group_name, checks in assumption_checks.items():
            if isinstance(checks, dict):
                # Use majority rule for normality tests
                normality_tests = [
                    checks.get('normality_shapiro', False),
                    checks.get('normality_anderson', False),
                    checks.get('normality_ks', False)
                ]
                group_normality = sum(normality_tests) >= 2  # Majority rule
                normality_results.append(group_normality)
        
        # All groups must pass normality
        return all(normality_results) if normality_results else False
    
    def _run_statistical_test(
        self,
        results: Dict[str, List[float]],
        test_config: Dict[str, Any]
    ) -> StatisticalTest:
        """Run a specific statistical test."""
        
        test_name = test_config['test_name']
        test_type = test_config['test_type']
        assumptions = test_config['assumptions']
        
        # Get test function
        if test_type == 'parametric':
            test_func = self.parametric_tests.get(test_name)
        elif test_type == 'non_parametric':
            test_func = self.nonparametric_tests.get(test_name)
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        if test_func is None:
            raise ValueError(f"Unknown test: {test_name}")
        
        # Run test
        test_result = test_func(results)
        
        # Create StatisticalTest object
        statistical_test = StatisticalTest(
            test_name=test_name,
            test_type=test_type,
            assumptions=assumptions,
            statistic=test_result['statistic'],
            p_value=test_result['p_value'],
            effect_size=test_result.get('effect_size'),
            confidence_interval=test_result.get('confidence_interval'),
            power=test_result.get('power'),
            interpretation=self._interpret_result(test_result['p_value'], test_name)
        )
        
        return statistical_test
    
    def _independent_ttest(self, results: Dict[str, List[float]]) -> Dict[str, Any]:
        """Independent samples t-test."""
        
        groups = list(results.values())
        if len(groups) != 2:
            raise ValueError("Independent t-test requires exactly 2 groups")
        
        group1, group2 = groups
        statistic, p_value = ttest_ind(group1, group2)
        
        # Effect size (Cohen's d)
        effect_size = cohens_d(group1, group2)
        
        # Confidence interval for difference in means
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        se_diff = pooled_std * np.sqrt(1/n1 + 1/n2)
        
        t_critical = stats.t.ppf(1 - self.alpha/2, n1 + n2 - 2)
        mean_diff = mean1 - mean2
        margin_error = t_critical * se_diff
        
        ci = (mean_diff - margin_error, mean_diff + margin_error)
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': effect_size,
            'confidence_interval': ci
        }
    
    def _paired_ttest(self, results: Dict[str, List[float]]) -> Dict[str, Any]:
        """Paired samples t-test."""
        
        groups = list(results.values())
        if len(groups) != 2:
            raise ValueError("Paired t-test requires exactly 2 groups")
        
        group1, group2 = groups
        if len(group1) != len(group2):
            raise ValueError("Paired t-test requires equal sample sizes")
        
        statistic, p_value = ttest_rel(group1, group2)
        
        # Effect size for paired data
        differences = np.array(group1) - np.array(group2)
        effect_size = np.mean(differences) / np.std(differences, ddof=1)
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': effect_size
        }
    
    def _mann_whitney(self, results: Dict[str, List[float]]) -> Dict[str, Any]:
        """Mann-Whitney U test."""
        
        groups = list(results.values())
        if len(groups) != 2:
            raise ValueError("Mann-Whitney test requires exactly 2 groups")
        
        group1, group2 = groups
        statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
        
        # Effect size (rank biserial correlation)
        n1, n2 = len(group1), len(group2)
        effect_size = 1 - (2 * statistic) / (n1 * n2)
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': effect_size
        }
    
    def _wilcoxon(self, results: Dict[str, List[float]]) -> Dict[str, Any]:
        """Wilcoxon signed-rank test."""
        
        groups = list(results.values())
        if len(groups) != 2:
            raise ValueError("Wilcoxon test requires exactly 2 groups")
        
        group1, group2 = groups
        if len(group1) != len(group2):
            raise ValueError("Wilcoxon test requires equal sample sizes")
        
        statistic, p_value = wilcoxon(group1, group2)
        
        # Effect size (r = Z / sqrt(N))
        n = len(group1)
        z_score = stats.norm.ppf(1 - p_value/2)  # Approximation
        effect_size = abs(z_score) / np.sqrt(n)
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': effect_size
        }
    
    def _one_way_anova(self, results: Dict[str, List[float]]) -> Dict[str, Any]:
        """One-way ANOVA."""
        
        groups = list(results.values())
        statistic, p_value = stats.f_oneway(*groups)
        
        # Effect size (eta-squared)
        all_scores = np.concatenate(groups)
        group_labels = np.concatenate([np.full(len(group), i) for i, group in enumerate(groups)])
        
        ss_between = sum(len(group) * (np.mean(group) - np.mean(all_scores))**2 for group in groups)
        ss_total = sum((score - np.mean(all_scores))**2 for score in all_scores)
        
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': eta_squared
        }
    
    def _kruskal_wallis(self, results: Dict[str, List[float]]) -> Dict[str, Any]:
        """Kruskal-Wallis test."""
        
        groups = list(results.values())
        statistic, p_value = kruskal(*groups)
        
        # Effect size (epsilon-squared)
        n_total = sum(len(group) for group in groups)
        effect_size = (statistic - len(groups) + 1) / (n_total - len(groups))
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': effect_size
        }
    
    def _repeated_measures_anova(self, results: Dict[str, List[float]]) -> Dict[str, Any]:
        """Repeated measures ANOVA (simplified implementation)."""
        
        # Note: This is a simplified implementation
        # For full repeated measures ANOVA, consider using statsmodels
        
        groups = list(results.values())
        statistic, p_value = stats.f_oneway(*groups)
        
        # Placeholder effect size
        effect_size = 0.0  # Would need proper implementation
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': effect_size
        }
    
    def _friedman(self, results: Dict[str, List[float]]) -> Dict[str, Any]:
        """Friedman test."""
        
        groups = list(results.values())
        
        # Check equal sample sizes
        if not all(len(group) == len(groups[0]) for group in groups):
            raise ValueError("Friedman test requires equal sample sizes")
        
        statistic, p_value = friedmanchisquare(*groups)
        
        # Effect size (Kendall's W)
        n = len(groups[0])
        k = len(groups)
        effect_size = statistic / (n * (k - 1))
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': effect_size
        }
    
    def _apply_multiple_comparison_correction(
        self, 
        p_values: List[float]
    ) -> Dict[str, Any]:
        """Apply multiple comparison correction."""
        
        if not p_values:
            return {
                'method': self.correction_method,
                'original_p_values': p_values,
                'corrected_p_values': p_values,
                'rejected': [],
                'adjusted_alpha': self.alpha
            }
        
        rejected, corrected_p_values, alpha_sidak, alpha_bonf = multipletests(
            p_values, alpha=self.alpha, method=self.correction_method
        )
        
        return {
            'method': self.correction_method,
            'original_p_values': p_values,
            'corrected_p_values': corrected_p_values.tolist(),
            'rejected': rejected.tolist(),
            'adjusted_alpha': alpha_bonf if self.correction_method == 'bonferroni' else alpha_sidak
        }
    
    def _compute_effect_sizes(
        self,
        results: Dict[str, List[float]],
        comparison_type: str
    ) -> Dict[str, float]:
        """Compute various effect size measures."""
        
        effect_sizes = {}
        groups = list(results.values())
        
        if len(groups) == 2:
            # Two-group comparisons
            group1, group2 = groups
            
            # Cohen's d
            cohens_d_value = cohens_d(group1, group2)
            effect_sizes['cohens_d'] = cohens_d_value
            
            # Glass's delta
            pooled_std = np.sqrt((np.var(group1, ddof=1) + np.var(group2, ddof=1)) / 2)
            if pooled_std > 0:
                glass_delta = (np.mean(group1) - np.mean(group2)) / np.std(group2, ddof=1)
                effect_sizes['glass_delta'] = glass_delta
            
            # Hedges' g (bias-corrected Cohen's d)
            n1, n2 = len(group1), len(group2)
            correction_factor = 1 - (3 / (4 * (n1 + n2) - 9))
            hedges_g = cohens_d_value * correction_factor
            effect_sizes['hedges_g'] = hedges_g
            
        elif len(groups) > 2:
            # Multi-group comparisons
            all_scores = np.concatenate(groups)
            
            # Eta-squared
            ss_between = sum(len(group) * (np.mean(group) - np.mean(all_scores))**2 for group in groups)
            ss_total = sum((score - np.mean(all_scores))**2 for score in all_scores)
            
            if ss_total > 0:
                eta_squared = ss_between / ss_total
                effect_sizes['eta_squared'] = eta_squared
                
                # Omega-squared (less biased estimate)
                k = len(groups)
                n_total = len(all_scores)
                mse = (ss_total - ss_between) / (n_total - k)
                omega_squared = (ss_between - (k - 1) * mse) / (ss_total + mse)
                effect_sizes['omega_squared'] = max(0, omega_squared)
        
        return effect_sizes
    
    def _conduct_power_analysis(
        self,
        results: Dict[str, List[float]],
        test_results: List[StatisticalTest]
    ) -> Dict[str, float]:
        """Conduct power analysis for the tests."""
        
        power_analysis = {}
        
        for test in test_results:
            if test.effect_size is not None:
                # Sample size for each group
                group_sizes = [len(scores) for scores in results.values()]
                
                if test.test_name in ['independent_ttest', 'mann_whitney']:
                    # Two-group power analysis
                    n1, n2 = group_sizes
                    
                    # Power for t-test
                    try:
                        power = ttest_power(
                            effect_size=abs(test.effect_size),
                            nobs=min(n1, n2),
                            alpha=self.alpha,
                            alternative='two-sided'
                        )
                        power_analysis[f'{test.test_name}_power'] = power
                        
                        # Required sample size for 80% power
                        from statsmodels.stats.power import ttest_power
                        # This would require iterative calculation
                        power_analysis[f'{test.test_name}_n_for_80_power'] = 'Calculate separately'
                        
                    except:
                        power_analysis[f'{test.test_name}_power'] = 'Unable to calculate'
                
                elif test.test_name in ['one_way_anova', 'kruskal_wallis']:
                    # ANOVA power analysis (simplified)
                    power_analysis[f'{test.test_name}_power'] = 'Use specialized ANOVA power calculation'
        
        return power_analysis
    
    def _generate_recommendations(
        self,
        test_results: List[StatisticalTest],
        effect_sizes: Dict[str, float],
        power_analysis: Dict[str, float],
        assumption_checks: Dict[str, Dict[str, bool]]
    ) -> List[str]:
        """Generate recommendations based on statistical analysis."""
        
        recommendations = []
        
        # Check significance
        significant_tests = [test for test in test_results if test.p_value < self.alpha]
        
        if not significant_tests:
            recommendations.append(
                "No statistically significant differences found. Consider: "
                "(1) Increasing sample size, (2) Checking for systematic errors, "
                "(3) Reevaluating hypotheses."
            )
        
        # Check effect sizes
        for effect_name, effect_value in effect_sizes.items():
            if effect_name == 'cohens_d':
                if abs(effect_value) < 0.2:
                    recommendations.append(
                        f"Small effect size (Cohen's d = {effect_value:.3f}). "
                        "Consider practical significance alongside statistical significance."
                    )
                elif abs(effect_value) > 0.8:
                    recommendations.append(
                        f"Large effect size (Cohen's d = {effect_value:.3f}). "
                        "Strong practical significance indicated."
                    )
        
        # Check assumptions
        for group_name, checks in assumption_checks.items():
            if isinstance(checks, dict):
                failed_assumptions = [
                    assumption for assumption, passed in checks.items() 
                    if not passed and assumption != 'no_extreme_outliers'
                ]
                
                if failed_assumptions:
                    recommendations.append(
                        f"Group '{group_name}' violates assumptions: {', '.join(failed_assumptions)}. "
                        "Consider non-parametric alternatives or data transformation."
                    )
        
        # Power analysis recommendations
        for test_name, power_value in power_analysis.items():
            if isinstance(power_value, float) and power_value < 0.8:
                recommendations.append(
                    f"Low statistical power ({power_value:.3f}) for {test_name}. "
                    "Consider increasing sample size for adequate power (≥0.8)."
                )
        
        return recommendations
    
    def _compute_validity_score(
        self,
        test_results: List[StatisticalTest],
        assumption_checks: Dict[str, Dict[str, bool]],
        effect_sizes: Dict[str, float],
        power_analysis: Dict[str, float]
    ) -> float:
        """Compute overall validity score for the analysis."""
        
        validity_components = []
        
        # Assumption compliance (0-1)
        assumption_scores = []
        for group_name, checks in assumption_checks.items():
            if isinstance(checks, dict):
                group_score = sum(checks.values()) / len(checks)
                assumption_scores.append(group_score)
        
        if assumption_scores:
            assumption_compliance = np.mean(assumption_scores)
            validity_components.append(assumption_compliance * 0.3)  # 30% weight
        
        # Effect size adequacy (0-1)
        effect_adequacy = 0.5  # Default moderate
        if 'cohens_d' in effect_sizes:
            d_value = abs(effect_sizes['cohens_d'])
            if d_value >= 0.8:
                effect_adequacy = 1.0
            elif d_value >= 0.5:
                effect_adequacy = 0.8
            elif d_value >= 0.2:
                effect_adequacy = 0.6
            else:
                effect_adequacy = 0.3
        
        validity_components.append(effect_adequacy * 0.25)  # 25% weight
        
        # Statistical power (0-1)
        power_scores = []
        for test_name, power_value in power_analysis.items():
            if isinstance(power_value, float):
                power_scores.append(min(power_value, 1.0))
        
        if power_scores:
            avg_power = np.mean(power_scores)
            validity_components.append(avg_power * 0.25)  # 25% weight
        else:
            validity_components.append(0.5 * 0.25)  # Default moderate
        
        # Methodological rigor (0-1)
        rigor_score = 0.7  # Base score
        
        # Bonus for multiple comparison correction
        if len(test_results) > 1:
            rigor_score += 0.1
        
        # Bonus for appropriate test selection
        rigor_score += 0.1
        
        validity_components.append(rigor_score * 0.2)  # 20% weight
        
        # Compute final score
        final_score = sum(validity_components)
        
        return min(final_score, 1.0)  # Cap at 1.0
    
    def _interpret_result(self, p_value: float, test_name: str) -> str:
        """Interpret statistical test result."""
        
        if p_value < 0.001:
            significance = "highly significant (p < 0.001)"
        elif p_value < 0.01:
            significance = "very significant (p < 0.01)"
        elif p_value < 0.05:
            significance = "significant (p < 0.05)"
        elif p_value < 0.10:
            significance = "marginally significant (p < 0.10)"
        else:
            significance = "not significant (p ≥ 0.10)"
        
        return f"Result is {significance}"
    
    def generate_report(self, report: ValidationReport, output_path: Optional[str] = None) -> str:
        """Generate a comprehensive statistical validation report."""
        
        report_text = f"""
# Statistical Validation Report

## Experiment: {report.experiment_name}

### Sample Sizes
"""
        
        for group, size in report.sample_sizes.items():
            report_text += f"- **{group}**: n = {size}\n"
        
        report_text += "\n### Descriptive Statistics\n\n"
        
        for group, stats in report.descriptive_stats.items():
            report_text += f"**{group}**:\n"
            report_text += f"- Mean ± SD: {stats['mean']:.4f} ± {stats['std']:.4f}\n"
            report_text += f"- Median [IQR]: {stats['median']:.4f} [{stats['q25']:.4f}, {stats['q75']:.4f}]\n"
            report_text += f"- 95% CI: [{stats['ci_95_lower']:.4f}, {stats['ci_95_upper']:.4f}]\n\n"
        
        report_text += "### Statistical Tests\n\n"
        
        for test in report.statistical_tests:
            report_text += f"**{test.test_name.replace('_', ' ').title()}**:\n"
            report_text += f"- Statistic: {test.statistic:.4f}\n"
            report_text += f"- p-value: {test.p_value:.6f}\n"
            
            if test.effect_size is not None:
                report_text += f"- Effect size: {test.effect_size:.4f}\n"
            
            report_text += f"- Interpretation: {test.interpretation}\n"
            report_text += f"- Assumptions: {', '.join(test.assumptions)}\n\n"
        
        report_text += "### Multiple Comparison Correction\n\n"
        
        correction = report.multiple_comparison_correction
        report_text += f"- Method: {correction['method']}\n"
        report_text += f"- Number of tests: {len(correction['original_p_values'])}\n"
        report_text += f"- Significant after correction: {sum(correction['rejected'])}\n\n"
        
        report_text += "### Effect Sizes\n\n"
        
        for effect_name, effect_value in report.effect_sizes.items():
            report_text += f"- **{effect_name.replace('_', ' ').title()}**: {effect_value:.4f}\n"
        
        report_text += "\n### Recommendations\n\n"
        
        for i, recommendation in enumerate(report.recommendations, 1):
            report_text += f"{i}. {recommendation}\n"
        
        report_text += f"\n### Validity Score: {report.validity_score:.2f}/1.00\n"
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
        
        return report_text


class ResearchReproducibility:
    """Framework for ensuring research reproducibility."""
    
    def __init__(self):
        """Initialize reproducibility framework."""
        self.reproducibility_checklist = {
            'data_preprocessing': False,
            'model_architecture': False,
            'hyperparameters': False,
            'random_seeds': False,
            'computational_environment': False,
            'statistical_methods': False,
            'code_availability': False,
            'data_availability': False
        }
    
    def check_reproducibility(
        self,
        experiment_config: Dict[str, Any],
        code_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Check reproducibility of an experiment."""
        
        reproducibility_score = 0
        checklist_results = {}
        
        # Check each component
        for component, _ in self.reproducibility_checklist.items():
            if component in experiment_config:
                checklist_results[component] = True
                reproducibility_score += 1
            else:
                checklist_results[component] = False
        
        total_components = len(self.reproducibility_checklist)
        reproducibility_percentage = (reproducibility_score / total_components) * 100
        
        return {
            'reproducibility_score': reproducibility_score,
            'total_components': total_components,
            'reproducibility_percentage': reproducibility_percentage,
            'checklist_results': checklist_results,
            'recommendations': self._generate_reproducibility_recommendations(checklist_results)
        }
    
    def _generate_reproducibility_recommendations(
        self, 
        checklist_results: Dict[str, bool]
    ) -> List[str]:
        """Generate recommendations for improving reproducibility."""
        
        recommendations = []
        
        for component, passed in checklist_results.items():
            if not passed:
                if component == 'data_preprocessing':
                    recommendations.append(
                        "Document all data preprocessing steps including normalization, "
                        "filtering, and transformation procedures."
                    )
                elif component == 'model_architecture':
                    recommendations.append(
                        "Provide complete model architecture specifications including "
                        "layer dimensions, activation functions, and initialization methods."
                    )
                elif component == 'hyperparameters':
                    recommendations.append(
                        "Document all hyperparameters including learning rates, "
                        "batch sizes, regularization parameters, and optimization settings."
                    )
                elif component == 'random_seeds':
                    recommendations.append(
                        "Set and document random seeds for all stochastic components "
                        "including data splitting, weight initialization, and training."
                    )
                elif component == 'computational_environment':
                    recommendations.append(
                        "Document computational environment including hardware specifications, "
                        "software versions, and dependency requirements."
                    )
                elif component == 'statistical_methods':
                    recommendations.append(
                        "Provide detailed description of statistical methods including "
                        "significance testing, multiple comparison corrections, and effect sizes."
                    )
                elif component == 'code_availability':
                    recommendations.append(
                        "Make source code publicly available with clear documentation "
                        "and installation instructions."
                    )
                elif component == 'data_availability':
                    recommendations.append(
                        "Ensure data availability or provide clear instructions for "
                        "obtaining equivalent datasets, respecting privacy constraints."
                    )
        
        return recommendations