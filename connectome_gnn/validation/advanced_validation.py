"""Advanced validation framework for Connectome-GNN-Suite."""

import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
import time

from ..robust.error_handling import ConnectomeError, ValidationError
from ..robust.logging_config import get_logger


@dataclass
class ValidationResult:
    """Validation result container."""
    passed: bool
    score: float
    details: Dict[str, Any]
    timestamp: float
    validator_name: str
    warnings: List[str]
    errors: List[str]


class BaseValidator(ABC):
    """Base class for all validators."""
    
    def __init__(self, name: str, threshold: float = 0.8):
        self.name = name
        self.threshold = threshold
        self.logger = get_logger(__name__)
        
    @abstractmethod
    def validate(self, data: Any) -> ValidationResult:
        """Perform validation on data."""
        pass
        
    def _create_result(
        self,
        passed: bool,
        score: float,
        details: Dict[str, Any],
        warnings: List[str] = None,
        errors: List[str] = None
    ) -> ValidationResult:
        """Create validation result."""
        return ValidationResult(
            passed=passed,
            score=score,
            details=details,
            timestamp=time.time(),
            validator_name=self.name,
            warnings=warnings or [],
            errors=errors or []
        )


class ConnectomeDataValidator(BaseValidator):
    """Validator for connectome data integrity and quality."""
    
    def __init__(self, threshold: float = 0.8):
        super().__init__("ConnectomeDataValidator", threshold)
        
    def validate(self, data: Any) -> ValidationResult:
        """Validate connectome data."""
        errors = []
        warnings = []
        scores = {}
        
        try:
            # Check data type and structure
            if not hasattr(data, 'x') or not hasattr(data, 'edge_index'):
                errors.append("Missing required attributes: x, edge_index")
                return self._create_result(False, 0.0, {}, warnings, errors)
                
            # Validate node features
            node_score = self._validate_node_features(data.x, warnings, errors)
            scores['node_features'] = node_score
            
            # Validate edge connectivity
            edge_score = self._validate_edges(data.edge_index, data.x.shape[0], warnings, errors)
            scores['edge_connectivity'] = edge_score
            
            # Validate edge attributes if present
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                edge_attr_score = self._validate_edge_attributes(data.edge_attr, warnings, errors)
                scores['edge_attributes'] = edge_attr_score
            else:
                scores['edge_attributes'] = 1.0
                
            # Check for anatomical consistency
            anatomy_score = self._validate_anatomical_consistency(data, warnings, errors)
            scores['anatomical_consistency'] = anatomy_score
            
            # Overall score
            overall_score = np.mean(list(scores.values()))
            passed = overall_score >= self.threshold and len(errors) == 0
            
            return self._create_result(
                passed=passed,
                score=overall_score,
                details=scores,
                warnings=warnings,
                errors=errors
            )
            
        except Exception as e:
            errors.append(f"Validation failed with exception: {str(e)}")
            return self._create_result(False, 0.0, {}, warnings, errors)
            
    def _validate_node_features(
        self,
        node_features: torch.Tensor,
        warnings: List[str],
        errors: List[str]
    ) -> float:
        """Validate node features."""
        if node_features is None:
            errors.append("Node features are None")
            return 0.0
            
        if len(node_features.shape) != 2:
            errors.append(f"Node features should be 2D, got shape: {node_features.shape}")
            return 0.0
            
        # Check for NaN or infinite values
        if torch.isnan(node_features).any():
            errors.append("Node features contain NaN values")
            return 0.0
            
        if torch.isinf(node_features).any():
            errors.append("Node features contain infinite values")
            return 0.0
            
        # Check feature value ranges
        if node_features.min() < -100 or node_features.max() > 100:
            warnings.append("Node features have extreme values")
            
        # Check for constant features
        std_devs = torch.std(node_features, dim=0)
        constant_features = (std_devs == 0).sum().item()
        if constant_features > 0:
            warnings.append(f"Found {constant_features} constant features")
            
        return 1.0 - (constant_features / node_features.shape[1])
        
    def _validate_edges(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        warnings: List[str],
        errors: List[str]
    ) -> float:
        """Validate edge connectivity."""
        if edge_index is None:
            errors.append("Edge index is None")
            return 0.0
            
        if len(edge_index.shape) != 2 or edge_index.shape[0] != 2:
            errors.append(f"Edge index should be 2xN, got shape: {edge_index.shape}")
            return 0.0
            
        # Check edge indices are within valid range
        if edge_index.min() < 0 or edge_index.max() >= num_nodes:
            errors.append("Edge indices out of valid range")
            return 0.0
            
        # Check for self-loops
        self_loops = (edge_index[0] == edge_index[1]).sum().item()
        if self_loops > 0:
            warnings.append(f"Found {self_loops} self-loops")
            
        # Check connectivity
        num_edges = edge_index.shape[1]
        max_edges = num_nodes * (num_nodes - 1)
        connectivity_ratio = num_edges / max_edges
        
        if connectivity_ratio < 0.01:
            warnings.append(f"Very sparse connectivity: {connectivity_ratio:.4f}")
        elif connectivity_ratio > 0.5:
            warnings.append(f"Very dense connectivity: {connectivity_ratio:.4f}")
            
        return min(1.0, connectivity_ratio * 10)  # Scale connectivity score
        
    def _validate_edge_attributes(
        self,
        edge_attr: torch.Tensor,
        warnings: List[str],
        errors: List[str]
    ) -> float:
        """Validate edge attributes."""
        if torch.isnan(edge_attr).any():
            errors.append("Edge attributes contain NaN values")
            return 0.0
            
        if torch.isinf(edge_attr).any():
            errors.append("Edge attributes contain infinite values")
            return 0.0
            
        # Check for negative weights in connectivity matrix
        if (edge_attr < 0).any():
            warnings.append("Found negative edge weights")
            
        # Check weight distribution
        if edge_attr.std() == 0:
            warnings.append("All edge weights are identical")
            return 0.5
            
        return 1.0
        
    def _validate_anatomical_consistency(
        self,
        data: Any,
        warnings: List[str],
        errors: List[str]
    ) -> float:
        """Validate anatomical consistency of connectome."""
        # Check if node count matches standard brain atlases
        num_nodes = data.x.shape[0]
        
        # Common brain atlas sizes
        standard_sizes = [90, 116, 200, 400, 1000]
        
        if num_nodes not in standard_sizes:
            warnings.append(f"Non-standard brain atlas size: {num_nodes} nodes")
            
        # Check hemispheric symmetry (simplified)
        if num_nodes % 2 == 0:
            half_nodes = num_nodes // 2
            # Assuming first half is left hemisphere, second half is right
            left_features = data.x[:half_nodes]
            right_features = data.x[half_nodes:]
            
            # Check if hemispheres are roughly similar
            correlation = torch.corrcoef(
                torch.stack([left_features.mean(dim=1), right_features.mean(dim=1)])
            )[0, 1]
            
            if correlation < 0.5:
                warnings.append("Low hemispheric similarity in features")
                return 0.7
                
        return 1.0


class ModelPerformanceValidator(BaseValidator):
    """Validator for model performance and behavior."""
    
    def __init__(self, threshold: float = 0.8):
        super().__init__("ModelPerformanceValidator", threshold)
        
    def validate(self, model_data: Dict[str, Any]) -> ValidationResult:
        """Validate model performance metrics."""
        errors = []
        warnings = []
        scores = {}
        
        try:
            # Validate training metrics
            if 'training_metrics' in model_data:
                train_score = self._validate_training_metrics(
                    model_data['training_metrics'], warnings, errors
                )
                scores['training'] = train_score
                
            # Validate validation metrics
            if 'validation_metrics' in model_data:
                val_score = self._validate_validation_metrics(
                    model_data['validation_metrics'], warnings, errors
                )
                scores['validation'] = val_score
                
            # Check for overfitting
            if 'training_metrics' in model_data and 'validation_metrics' in model_data:
                overfit_score = self._check_overfitting(
                    model_data['training_metrics'],
                    model_data['validation_metrics'],
                    warnings, errors
                )
                scores['overfitting'] = overfit_score
                
            # Validate model robustness
            if 'robustness_metrics' in model_data:
                robust_score = self._validate_robustness(
                    model_data['robustness_metrics'], warnings, errors
                )
                scores['robustness'] = robust_score
                
            overall_score = np.mean(list(scores.values())) if scores else 0.0
            passed = overall_score >= self.threshold and len(errors) == 0
            
            return self._create_result(
                passed=passed,
                score=overall_score,
                details=scores,
                warnings=warnings,
                errors=errors
            )
            
        except Exception as e:
            errors.append(f"Model validation failed: {str(e)}")
            return self._create_result(False, 0.0, {}, warnings, errors)
            
    def _validate_training_metrics(
        self,
        metrics: Dict[str, List[float]],
        warnings: List[str],
        errors: List[str]
    ) -> float:
        """Validate training metrics."""
        if 'loss' not in metrics:
            errors.append("Missing training loss")
            return 0.0
            
        losses = metrics['loss']
        
        # Check for NaN losses
        if any(np.isnan(loss) for loss in losses):
            errors.append("Training loss contains NaN values")
            return 0.0
            
        # Check for decreasing loss trend
        if len(losses) > 5:
            recent_trend = np.polyfit(range(len(losses[-10:])), losses[-10:], 1)[0]
            if recent_trend > 0:
                warnings.append("Training loss is increasing")
                return 0.6
                
        # Check convergence
        if len(losses) > 10:
            recent_variance = np.var(losses[-10:])
            if recent_variance > 0.1:
                warnings.append("Training loss not converged")
                return 0.8
                
        return 1.0
        
    def _validate_validation_metrics(
        self,
        metrics: Dict[str, List[float]],
        warnings: List[str],
        errors: List[str]
    ) -> float:
        """Validate validation metrics."""
        if 'accuracy' in metrics:
            accuracies = metrics['accuracy']
            final_accuracy = accuracies[-1] if accuracies else 0.0
            
            if final_accuracy < 0.5:
                warnings.append(f"Low validation accuracy: {final_accuracy:.3f}")
                return final_accuracy
            elif final_accuracy < 0.7:
                warnings.append(f"Moderate validation accuracy: {final_accuracy:.3f}")
                return 0.8
                
        return 1.0
        
    def _check_overfitting(
        self,
        train_metrics: Dict[str, List[float]],
        val_metrics: Dict[str, List[float]],
        warnings: List[str],
        errors: List[str]
    ) -> float:
        """Check for overfitting."""
        if 'loss' in train_metrics and 'loss' in val_metrics:
            train_loss = train_metrics['loss'][-1] if train_metrics['loss'] else float('inf')
            val_loss = val_metrics['loss'][-1] if val_metrics['loss'] else float('inf')
            
            gap = val_loss - train_loss
            if gap > 0.5:
                warnings.append(f"Potential overfitting detected (gap: {gap:.3f})")
                return 0.5
            elif gap > 0.2:
                warnings.append(f"Some overfitting detected (gap: {gap:.3f})")
                return 0.8
                
        return 1.0
        
    def _validate_robustness(
        self,
        robustness_metrics: Dict[str, float],
        warnings: List[str],
        errors: List[str]
    ) -> float:
        """Validate model robustness metrics."""
        scores = []
        
        if 'adversarial_accuracy' in robustness_metrics:
            adv_acc = robustness_metrics['adversarial_accuracy']
            if adv_acc < 0.3:
                warnings.append(f"Low adversarial robustness: {adv_acc:.3f}")
            scores.append(min(1.0, adv_acc * 2))  # Scale adversarial accuracy
            
        if 'noise_robustness' in robustness_metrics:
            noise_rob = robustness_metrics['noise_robustness']
            if noise_rob < 0.5:
                warnings.append(f"Low noise robustness: {noise_rob:.3f}")
            scores.append(noise_rob)
            
        return np.mean(scores) if scores else 1.0


class SecurityValidator(BaseValidator):
    """Validator for security-related aspects."""
    
    def __init__(self, threshold: float = 0.9):
        super().__init__("SecurityValidator", threshold)
        
    def validate(self, security_data: Dict[str, Any]) -> ValidationResult:
        """Validate security aspects."""
        errors = []
        warnings = []
        scores = {}
        
        # Check privacy preservation
        if 'privacy_metrics' in security_data:
            privacy_score = self._validate_privacy(
                security_data['privacy_metrics'], warnings, errors
            )
            scores['privacy'] = privacy_score
            
        # Check access controls
        if 'access_logs' in security_data:
            access_score = self._validate_access_controls(
                security_data['access_logs'], warnings, errors
            )
            scores['access_control'] = access_score
            
        # Check data integrity
        if 'integrity_checks' in security_data:
            integrity_score = self._validate_data_integrity(
                security_data['integrity_checks'], warnings, errors
            )
            scores['data_integrity'] = integrity_score
            
        overall_score = np.mean(list(scores.values())) if scores else 0.0
        passed = overall_score >= self.threshold and len(errors) == 0
        
        return self._create_result(
            passed=passed,
            score=overall_score,
            details=scores,
            warnings=warnings,
            errors=errors
        )
        
    def _validate_privacy(
        self,
        privacy_metrics: Dict[str, float],
        warnings: List[str],
        errors: List[str]
    ) -> float:
        """Validate privacy preservation."""
        if 'epsilon' in privacy_metrics:
            epsilon = privacy_metrics['epsilon']
            if epsilon > 10.0:
                warnings.append(f"High privacy budget spent: ε={epsilon:.2f}")
                return 0.5
            elif epsilon > 5.0:
                warnings.append(f"Moderate privacy budget spent: ε={epsilon:.2f}")
                return 0.8
                
        return 1.0
        
    def _validate_access_controls(
        self,
        access_logs: List[Dict[str, Any]],
        warnings: List[str],
        errors: List[str]
    ) -> float:
        """Validate access control mechanisms."""
        failed_attempts = sum(
            1 for log in access_logs 
            if log.get('status') == 'failed'
        )
        
        total_attempts = len(access_logs)
        if total_attempts > 0:
            failure_rate = failed_attempts / total_attempts
            if failure_rate > 0.1:
                warnings.append(f"High access failure rate: {failure_rate:.2%}")
                return 0.7
                
        return 1.0
        
    def _validate_data_integrity(
        self,
        integrity_checks: Dict[str, bool],
        warnings: List[str],
        errors: List[str]
    ) -> float:
        """Validate data integrity checks."""
        passed_checks = sum(integrity_checks.values())
        total_checks = len(integrity_checks)
        
        if total_checks == 0:
            warnings.append("No integrity checks performed")
            return 0.5
            
        integrity_score = passed_checks / total_checks
        if integrity_score < 1.0:
            errors.append(f"Some integrity checks failed: {integrity_score:.2%}")
            
        return integrity_score


class ValidationPipeline:
    """Validation pipeline for orchestrating multiple validators."""
    
    def __init__(self):
        self.validators: List[BaseValidator] = []
        self.logger = get_logger(__name__)
        
    def add_validator(self, validator: BaseValidator):
        """Add validator to pipeline."""
        self.validators.append(validator)
        
    def validate_all(self, data: Dict[str, Any]) -> Dict[str, ValidationResult]:
        """Run all validators on provided data."""
        results = {}
        
        for validator in self.validators:
            try:
                # Extract relevant data for each validator
                validator_data = self._extract_validator_data(validator, data)
                result = validator.validate(validator_data)
                results[validator.name] = result
                
                if not result.passed:
                    self.logger.warning(
                        f"Validation failed: {validator.name} "
                        f"(score: {result.score:.3f})"
                    )
                    
            except Exception as e:
                self.logger.error(f"Validator {validator.name} failed: {e}")
                results[validator.name] = ValidationResult(
                    passed=False,
                    score=0.0,
                    details={'error': str(e)},
                    timestamp=time.time(),
                    validator_name=validator.name,
                    warnings=[],
                    errors=[str(e)]
                )
                
        return results
        
    def _extract_validator_data(
        self,
        validator: BaseValidator,
        data: Dict[str, Any]
    ) -> Any:
        """Extract relevant data for specific validator."""
        if isinstance(validator, ConnectomeDataValidator):
            return data.get('connectome_data')
        elif isinstance(validator, ModelPerformanceValidator):
            return data.get('model_data', {})
        elif isinstance(validator, SecurityValidator):
            return data.get('security_data', {})
        else:
            return data
            
    def get_overall_status(self, results: Dict[str, ValidationResult]) -> bool:
        """Get overall validation status."""
        return all(result.passed for result in results.values())
        
    def get_summary(self, results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """Get validation summary."""
        passed_count = sum(1 for result in results.values() if result.passed)
        total_count = len(results)
        
        all_warnings = []
        all_errors = []
        
        for result in results.values():
            all_warnings.extend(result.warnings)
            all_errors.extend(result.errors)
            
        return {
            'overall_passed': self.get_overall_status(results),
            'passed_validators': passed_count,
            'total_validators': total_count,
            'pass_rate': passed_count / total_count if total_count > 0 else 0.0,
            'total_warnings': len(all_warnings),
            'total_errors': len(all_errors),
            'warnings': all_warnings[:10],  # Limit to first 10
            'errors': all_errors[:10],      # Limit to first 10
            'timestamp': time.time()
        }