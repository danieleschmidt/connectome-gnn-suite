"""Configuration management for production deployment."""

import os
import json
import yaml
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import logging

from ..robust.logging_config import get_logger
from ..robust.security import SecurityManager


class Environment(Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class DeploymentConfig:
    """Configuration for deployment."""
    
    # Environment settings
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    log_level: str = "INFO"
    
    # Application settings
    app_name: str = "connectome-gnn"
    app_version: str = "0.1.0"
    port: int = 8000
    host: str = "0.0.0.0"
    workers: int = 4
    
    # Database settings
    database_url: Optional[str] = None
    redis_url: Optional[str] = None
    
    # Model settings
    model_cache_size: int = 5
    max_batch_size: int = 32
    inference_timeout: float = 30.0
    model_dir: str = "/app/models"
    
    # Security settings
    api_key_required: bool = True
    rate_limit: str = "100/hour"
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    
    # Performance settings
    memory_limit_mb: float = 2048.0
    cpu_limit: float = 2.0
    auto_scaling: bool = True
    min_replicas: int = 1
    max_replicas: int = 10
    
    # Monitoring settings
    metrics_enabled: bool = True
    health_check_interval: int = 30
    prometheus_port: int = 9090
    
    # Storage settings
    storage_type: str = "local"  # local, s3, gcs
    storage_bucket: Optional[str] = None
    storage_region: Optional[str] = None
    
    # Custom settings
    custom_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        config_dict = {}
        
        for key, value in self.__dict__.items():
            if isinstance(value, Environment):
                config_dict[key] = value.value
            elif isinstance(value, (list, dict)):
                config_dict[key] = value.copy()
            else:
                config_dict[key] = value
        
        return config_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeploymentConfig':
        """Create from dictionary."""
        # Handle environment enum
        if 'environment' in data:
            if isinstance(data['environment'], str):
                data['environment'] = Environment(data['environment'])
        
        return cls(**data)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Port validation
        if not (1 <= self.port <= 65535):
            issues.append(f"Invalid port: {self.port}")
        
        # Worker validation
        if self.workers < 1:
            issues.append(f"Workers must be >= 1: {self.workers}")
        
        # Memory validation
        if self.memory_limit_mb < 512:
            issues.append(f"Memory limit too low: {self.memory_limit_mb}MB")
        
        # Replica validation
        if self.min_replicas < 1:
            issues.append(f"Min replicas must be >= 1: {self.min_replicas}")
        
        if self.max_replicas < self.min_replicas:
            issues.append(f"Max replicas must be >= min replicas")
        
        # URL validation
        if self.database_url and not self.database_url.startswith(('postgresql://', 'mysql://', 'sqlite://')):
            issues.append(f"Invalid database URL format")
        
        return issues
    
    def get_environment_specific_config(self) -> Dict[str, Any]:
        """Get configuration specific to the environment."""
        base_config = self.to_dict()
        
        if self.environment == Environment.PRODUCTION:
            # Production overrides
            base_config.update({
                'debug': False,
                'log_level': 'INFO',
                'api_key_required': True,
                'cors_origins': [],  # Restrict CORS in production
                'metrics_enabled': True
            })
        elif self.environment == Environment.DEVELOPMENT:
            # Development overrides
            base_config.update({
                'debug': True,
                'log_level': 'DEBUG',
                'api_key_required': False,
                'cors_origins': ['*'],
                'auto_scaling': False,
                'min_replicas': 1,
                'max_replicas': 1
            })
        elif self.environment == Environment.TESTING:
            # Testing overrides
            base_config.update({
                'debug': True,
                'log_level': 'DEBUG',
                'api_key_required': False,
                'auto_scaling': False,
                'min_replicas': 1,
                'max_replicas': 1,
                'health_check_interval': 5
            })
        
        return base_config


class EnvironmentManager:
    """Manages environment-specific configurations."""
    
    def __init__(self, config_dir: str = "config",
                 logger: Optional[logging.Logger] = None):
        self.config_dir = Path(config_dir)
        self.logger = logger or get_logger("environment_manager")
        self.security_manager = SecurityManager(logger=logger)
        
        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Environment configurations
        self.configs = {}
        self.current_environment = None
        
        # Load configurations
        self._load_configurations()
    
    def _load_configurations(self):
        """Load all environment configurations."""
        for env in Environment:
            config_file = self.config_dir / f"{env.value}.yaml"
            
            if config_file.exists():
                try:
                    config_data = self._load_config_file(config_file)
                    self.configs[env] = DeploymentConfig.from_dict(config_data)
                    self.logger.info(f"Loaded configuration for {env.value}")
                except Exception as e:
                    self.logger.error(f"Failed to load config for {env.value}: {e}")
                    # Create default config
                    self.configs[env] = DeploymentConfig(environment=env)
            else:
                # Create default config
                self.configs[env] = DeploymentConfig(environment=env)
                self._save_config(env, self.configs[env])
    
    def _load_config_file(self, file_path: Path) -> Dict[str, Any]:
        """Load configuration from file."""
        if file_path.suffix.lower() == '.json':
            return self.security_manager.secure_json_load(str(file_path)) or {}
        elif file_path.suffix.lower() in ['.yaml', '.yml']:
            try:
                with open(file_path, 'r') as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                self.logger.error(f"Failed to load YAML config: {e}")
                return {}
        else:
            raise ValueError(f"Unsupported config file format: {file_path.suffix}")
    
    def _save_config(self, environment: Environment, config: DeploymentConfig):
        """Save configuration to file."""
        config_file = self.config_dir / f"{environment.value}.yaml"
        
        try:
            with open(config_file, 'w') as f:
                yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)
            self.logger.info(f"Saved configuration for {environment.value}")
        except Exception as e:
            self.logger.error(f"Failed to save config for {environment.value}: {e}")
    
    def get_config(self, environment: Environment = None) -> DeploymentConfig:
        """Get configuration for specified environment."""
        env = environment or self.current_environment or Environment.DEVELOPMENT
        
        if env not in self.configs:
            self.logger.warning(f"No configuration found for {env.value}, using default")
            return DeploymentConfig(environment=env)
        
        return self.configs[env]
    
    def set_environment(self, environment: Environment):
        """Set current environment."""
        self.current_environment = environment
        self.logger.info(f"Set current environment to {environment.value}")
    
    def update_config(self, environment: Environment, updates: Dict[str, Any]):
        """Update configuration for an environment."""
        if environment not in self.configs:
            self.configs[environment] = DeploymentConfig(environment=environment)
        
        config = self.configs[environment]
        config_dict = config.to_dict()
        config_dict.update(updates)
        
        self.configs[environment] = DeploymentConfig.from_dict(config_dict)
        self._save_config(environment, self.configs[environment])
        
        self.logger.info(f"Updated configuration for {environment.value}")
    
    def validate_all_configs(self) -> Dict[Environment, List[str]]:
        """Validate all configurations."""
        validation_results = {}
        
        for env, config in self.configs.items():
            issues = config.validate()
            validation_results[env] = issues
            
            if issues:
                self.logger.warning(f"Configuration issues for {env.value}: {issues}")
            else:
                self.logger.info(f"Configuration valid for {env.value}")
        
        return validation_results
    
    def generate_docker_env(self, environment: Environment) -> str:
        """Generate Docker environment file."""
        config = self.get_config(environment)
        env_config = config.get_environment_specific_config()
        
        env_lines = [
            f"# Environment configuration for {environment.value}",
            f"ENVIRONMENT={environment.value}",
            f"APP_NAME={env_config['app_name']}",
            f"APP_VERSION={env_config['app_version']}",
            f"DEBUG={str(env_config['debug']).lower()}",
            f"LOG_LEVEL={env_config['log_level']}",
            f"PORT={env_config['port']}",
            f"HOST={env_config['host']}",
            f"WORKERS={env_config['workers']}",
            f"MODEL_CACHE_SIZE={env_config['model_cache_size']}",
            f"MAX_BATCH_SIZE={env_config['max_batch_size']}",
            f"INFERENCE_TIMEOUT={env_config['inference_timeout']}",
            f"MEMORY_LIMIT_MB={env_config['memory_limit_mb']}",
            f"METRICS_ENABLED={str(env_config['metrics_enabled']).lower()}",
            f"HEALTH_CHECK_INTERVAL={env_config['health_check_interval']}"
        ]
        
        # Add optional settings
        if env_config.get('database_url'):
            env_lines.append(f"DATABASE_URL={env_config['database_url']}")
        
        if env_config.get('redis_url'):
            env_lines.append(f"REDIS_URL={env_config['redis_url']}")
        
        return '\n'.join(env_lines)
    
    def generate_kubernetes_config(self, environment: Environment) -> Dict[str, Any]:
        """Generate Kubernetes configuration."""
        config = self.get_config(environment)
        env_config = config.get_environment_specific_config()
        
        k8s_config = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': f"{env_config['app_name']}-{environment.value}",
                'labels': {
                    'app': env_config['app_name'],
                    'environment': environment.value,
                    'version': env_config['app_version']
                }
            },
            'spec': {
                'replicas': env_config['min_replicas'],
                'selector': {
                    'matchLabels': {
                        'app': env_config['app_name'],
                        'environment': environment.value
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': env_config['app_name'],
                            'environment': environment.value,
                            'version': env_config['app_version']
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': env_config['app_name'],
                            'image': f"{env_config['app_name']}:{env_config['app_version']}",
                            'ports': [{
                                'containerPort': env_config['port'],
                                'name': 'http'
                            }],
                            'env': [
                                {'name': 'ENVIRONMENT', 'value': environment.value},
                                {'name': 'PORT', 'value': str(env_config['port'])},
                                {'name': 'WORKERS', 'value': str(env_config['workers'])},
                                {'name': 'LOG_LEVEL', 'value': env_config['log_level']},
                                {'name': 'DEBUG', 'value': str(env_config['debug']).lower()}
                            ],
                            'resources': {
                                'requests': {
                                    'memory': f"{int(env_config['memory_limit_mb'] * 0.8)}Mi",
                                    'cpu': f"{env_config['cpu_limit'] * 0.5}"
                                },
                                'limits': {
                                    'memory': f"{int(env_config['memory_limit_mb'])}Mi",
                                    'cpu': str(env_config['cpu_limit'])
                                }
                            },
                            'livenessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': env_config['port']
                                },
                                'initialDelaySeconds': 30,
                                'periodSeconds': env_config['health_check_interval']
                            },
                            'readinessProbe': {
                                'httpGet': {
                                    'path': '/ready',
                                    'port': env_config['port']
                                },
                                'initialDelaySeconds': 5,
                                'periodSeconds': 5
                            }
                        }]
                    }
                }
            }
        }
        
        return k8s_config
    
    def export_configs(self, output_dir: str):
        """Export all configurations to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for env in Environment:
            # YAML config
            config_file = output_path / f"{env.value}.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(self.get_config(env).to_dict(), f, default_flow_style=False, indent=2)
            
            # Docker env file
            env_file = output_path / f"{env.value}.env"
            with open(env_file, 'w') as f:
                f.write(self.generate_docker_env(env))
            
            # Kubernetes config
            k8s_file = output_path / f"{env.value}-k8s.yaml"
            with open(k8s_file, 'w') as f:
                yaml.dump(self.generate_kubernetes_config(env), f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Exported configurations to {output_dir}")
    
    def load_from_environment_variables(self) -> DeploymentConfig:
        """Load configuration from environment variables."""
        env_config = {}
        
        # Map environment variables to config fields
        env_mapping = {
            'ENVIRONMENT': 'environment',
            'DEBUG': 'debug', 
            'LOG_LEVEL': 'log_level',
            'APP_NAME': 'app_name',
            'APP_VERSION': 'app_version',
            'PORT': 'port',
            'HOST': 'host',
            'WORKERS': 'workers',
            'DATABASE_URL': 'database_url',
            'REDIS_URL': 'redis_url',
            'MODEL_CACHE_SIZE': 'model_cache_size',
            'MAX_BATCH_SIZE': 'max_batch_size',
            'INFERENCE_TIMEOUT': 'inference_timeout',
            'MEMORY_LIMIT_MB': 'memory_limit_mb',
            'METRICS_ENABLED': 'metrics_enabled'
        }
        
        for env_var, config_key in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                # Type conversion
                if config_key in ['debug', 'metrics_enabled', 'auto_scaling']:
                    env_config[config_key] = value.lower() in ('true', '1', 'yes', 'on')
                elif config_key in ['port', 'workers', 'model_cache_size', 'max_batch_size']:
                    env_config[config_key] = int(value)
                elif config_key in ['inference_timeout', 'memory_limit_mb', 'cpu_limit']:
                    env_config[config_key] = float(value)
                elif config_key == 'environment':
                    env_config[config_key] = Environment(value)
                else:
                    env_config[config_key] = value
        
        # Create config with current environment as fallback
        current_env = env_config.get('environment', Environment.DEVELOPMENT)
        base_config = self.get_config(current_env).to_dict()
        base_config.update(env_config)
        
        return DeploymentConfig.from_dict(base_config)


def create_default_configs(output_dir: str):
    """Create default configuration files for all environments."""
    manager = EnvironmentManager()
    
    # Customize configs for each environment
    manager.update_config(Environment.DEVELOPMENT, {
        'debug': True,
        'log_level': 'DEBUG',
        'workers': 1,
        'auto_scaling': False
    })
    
    manager.update_config(Environment.STAGING, {
        'debug': False,
        'log_level': 'INFO',
        'workers': 2,
        'min_replicas': 2,
        'max_replicas': 4
    })
    
    manager.update_config(Environment.PRODUCTION, {
        'debug': False,
        'log_level': 'WARNING',
        'workers': 4,
        'min_replicas': 3,
        'max_replicas': 10,
        'api_key_required': True,
        'cors_origins': []
    })
    
    # Export all configurations
    manager.export_configs(output_dir)
    
    return manager