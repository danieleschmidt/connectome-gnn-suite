"""Core data types and structures."""

from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import json


@dataclass
class ConnectomeData:
    """Core data structure for connectome information."""
    subject_id: str
    connectivity_matrix: Optional[Any] = None  # Will be tensor when available
    node_features: Optional[Any] = None
    edge_features: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'subject_id': self.subject_id,
            'has_connectivity': self.connectivity_matrix is not None,
            'has_node_features': self.node_features is not None,
            'has_edge_features': self.edge_features is not None,
            'metadata': self.metadata or {}
        }


@dataclass 
class ModelConfig:
    """Configuration for GNN models."""
    model_type: str
    node_features: int
    hidden_dim: int = 256
    output_dim: int = 1
    num_layers: int = 3
    dropout: float = 0.1
    activation: str = "relu"
    residual: bool = True
    batch_norm: bool = True
    
    # Model-specific parameters
    extra_params: Optional[Dict[str, Any]] = None
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        if self.node_features <= 0:
            raise ValueError("node_features must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0 <= self.dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")
            
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_type': self.model_type,
            'node_features': self.node_features,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'activation': self.activation,
            'residual': self.residual,
            'batch_norm': self.batch_norm,
            'extra_params': self.extra_params or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """Create from dictionary."""
        return cls(
            model_type=data['model_type'],
            node_features=data['node_features'],
            hidden_dim=data.get('hidden_dim', 256),
            output_dim=data.get('output_dim', 1),
            num_layers=data.get('num_layers', 3),
            dropout=data.get('dropout', 0.1),
            activation=data.get('activation', 'relu'),
            residual=data.get('residual', True),
            batch_norm=data.get('batch_norm', True),
            extra_params=data.get('extra_params')
        )


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    optimizer: str = "adam"
    scheduler: Optional[str] = None
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-4
    
    # Validation
    validation_split: float = 0.2
    test_split: float = 0.1
    
    # Logging and checkpointing
    log_interval: int = 10
    save_interval: int = 50
    checkpoint_dir: Optional[str] = None
    
    # Hardware
    device: str = "auto"
    num_workers: int = 4
    
    def validate(self) -> bool:
        """Validate training configuration."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if not 0 <= self.validation_split <= 1:
            raise ValueError("validation_split must be between 0 and 1")
        if not 0 <= self.test_split <= 1:
            raise ValueError("test_split must be between 0 and 1")
        if self.validation_split + self.test_split >= 1:
            raise ValueError("validation_split + test_split must be < 1")
            
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'early_stopping': self.early_stopping,
            'patience': self.patience,
            'min_delta': self.min_delta,
            'validation_split': self.validation_split,
            'test_split': self.test_split,
            'log_interval': self.log_interval,
            'save_interval': self.save_interval,
            'checkpoint_dir': self.checkpoint_dir,
            'device': self.device,
            'num_workers': self.num_workers
        }


class ConfigManager:
    """Manages configuration loading and saving."""
    
    @staticmethod
    def save_config(config: Union[ModelConfig, TrainingConfig], filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
    
    @staticmethod
    def load_model_config(filepath: str) -> ModelConfig:
        """Load model configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return ModelConfig.from_dict(data)
    
    @staticmethod
    def load_training_config(filepath: str) -> TrainingConfig:
        """Load training configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return TrainingConfig(**data)


class MetricsTracker:
    """Tracks training and validation metrics."""
    
    def __init__(self):
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'epochs': []
        }
    
    def update(self, epoch: int, train_loss: float, val_loss: float = None, 
               train_metrics: Dict = None, val_metrics: Dict = None):
        """Update metrics for an epoch."""
        self.metrics['epochs'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        
        if val_loss is not None:
            self.metrics['val_loss'].append(val_loss)
        
        if train_metrics:
            self.metrics['train_metrics'].append(train_metrics)
            
        if val_metrics:
            self.metrics['val_metrics'].append(val_metrics)
    
    def get_best_epoch(self, metric: str = 'val_loss', minimize: bool = True) -> int:
        """Get epoch with best metric value."""
        if metric not in self.metrics or not self.metrics[metric]:
            return 0
            
        values = self.metrics[metric]
        if minimize:
            best_idx = values.index(min(values))
        else:
            best_idx = values.index(max(values))
            
        return self.metrics['epochs'][best_idx]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return self.metrics.copy()
    
    def save(self, filepath: str):
        """Save metrics to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


class ConfigurationError(Exception):
    """Raised when configuration is invalid.""" 
    pass


class DataError(Exception):
    """Raised when data processing fails."""
    pass