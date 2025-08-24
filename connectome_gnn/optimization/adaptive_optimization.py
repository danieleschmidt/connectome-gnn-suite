"""Adaptive optimization algorithms and strategies for Connectome-GNN-Suite."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
import time
import math
from abc import ABC, abstractmethod

from ..robust.logging_config import get_logger


@dataclass
class OptimizationConfig:
    """Configuration for adaptive optimization."""
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    amsgrad: bool = False
    adapt_lr: bool = True
    warmup_steps: int = 1000
    lr_schedule: str = "cosine"  # linear, cosine, exponential
    patience: int = 10
    factor: float = 0.5
    min_lr: float = 1e-6


class AdaptiveLearningRateScheduler:
    """Adaptive learning rate scheduler with multiple strategies."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: OptimizationConfig,
        total_steps: Optional[int] = None
    ):
        self.optimizer = optimizer
        self.config = config
        self.total_steps = total_steps
        
        self.step_count = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        self.logger = get_logger(__name__)
        
    def step(self, loss: Optional[float] = None):
        """Update learning rate based on current step and loss."""
        self.step_count += 1
        
        # Warmup phase
        if self.step_count <= self.config.warmup_steps:
            self._apply_warmup()
        else:
            # Main scheduling
            if self.config.lr_schedule == "cosine":
                self._apply_cosine_schedule()
            elif self.config.lr_schedule == "linear":
                self._apply_linear_schedule()
            elif self.config.lr_schedule == "exponential":
                self._apply_exponential_schedule()
                
        # Loss-based adaptation
        if loss is not None and self.config.adapt_lr:
            self._adapt_to_loss(loss)
            
    def _apply_warmup(self):
        """Apply warmup learning rate schedule."""
        warmup_factor = self.step_count / self.config.warmup_steps
        
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.base_lrs[i] * warmup_factor
            
    def _apply_cosine_schedule(self):
        """Apply cosine annealing schedule."""
        if self.total_steps is None:
            return
            
        effective_step = self.step_count - self.config.warmup_steps
        effective_total = self.total_steps - self.config.warmup_steps
        
        cosine_factor = 0.5 * (1 + math.cos(math.pi * effective_step / effective_total))
        
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.config.min_lr + (
                self.base_lrs[i] - self.config.min_lr
            ) * cosine_factor
            
    def _apply_linear_schedule(self):
        """Apply linear decay schedule."""
        if self.total_steps is None:
            return
            
        effective_step = self.step_count - self.config.warmup_steps
        effective_total = self.total_steps - self.config.warmup_steps
        
        decay_factor = max(0, 1 - effective_step / effective_total)
        
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.config.min_lr + (
                self.base_lrs[i] - self.config.min_lr
            ) * decay_factor
            
    def _apply_exponential_schedule(self):
        """Apply exponential decay schedule."""
        decay_factor = self.config.factor ** (
            self.step_count / max(1, self.total_steps or 10000)
        )
        
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = max(
                self.config.min_lr,
                self.base_lrs[i] * decay_factor
            )
            
    def _adapt_to_loss(self, loss: float):
        """Adapt learning rate based on loss progression."""
        if loss < self.best_loss:
            self.best_loss = loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        # Reduce learning rate on plateau
        if self.patience_counter >= self.config.patience:
            self._reduce_lr()
            self.patience_counter = 0
            
    def _reduce_lr(self):
        """Reduce learning rate."""
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(self.config.min_lr, old_lr * self.config.factor)
            param_group['lr'] = new_lr
            
            if old_lr != new_lr:
                self.logger.info(f"Reducing learning rate: {old_lr:.2e} -> {new_lr:.2e}")


class AdaptiveOptimizer(torch.optim.Optimizer):
    """Adaptive optimizer that combines multiple optimization strategies."""
    
    def __init__(
        self,
        params,
        config: OptimizationConfig,
        use_lookahead: bool = True,
        use_radam: bool = True
    ):
        self.config = config
        self.use_lookahead = use_lookahead
        self.use_radam = use_radam
        
        # Initialize base optimizer
        if use_radam:
            self.base_optimizer = self._create_radam_optimizer(params)
        else:
            self.base_optimizer = optim.Adam(
                params,
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=config.epsilon,
                weight_decay=config.weight_decay,
                amsgrad=config.amsgrad
            )
            
        # Add Lookahead wrapper if enabled
        if use_lookahead:
            self.optimizer = self._wrap_with_lookahead(self.base_optimizer)
        else:
            self.optimizer = self.base_optimizer
            
        super().__init__(params, {})
        
        self.logger = get_logger(__name__)
        
    def _create_radam_optimizer(self, params):
        """Create RAdam optimizer."""
        return RAdam(
            params,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.epsilon,
            weight_decay=self.config.weight_decay
        )
        
    def _wrap_with_lookahead(self, base_optimizer):
        """Wrap optimizer with Lookahead."""
        return Lookahead(base_optimizer, k=5, alpha=0.5)
        
    def step(self, closure=None):
        """Perform optimization step."""
        return self.optimizer.step(closure)
        
    def zero_grad(self):
        """Clear gradients."""
        self.optimizer.zero_grad()
        
    @property
    def param_groups(self):
        """Get parameter groups."""
        return self.optimizer.param_groups


class RAdam(torch.optim.Optimizer):
    """RAdam optimizer implementation."""
    
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(RAdam, self).__init__(params, defaults)
        
    def step(self, closure=None):
        """Perform optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')
                    
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Variance rectification
                rho_inf = 2 / (1 - beta2) - 1
                rho = rho_inf - 2 * state['step'] * beta2 ** state['step'] / bias_correction2
                
                if rho >= 5:
                    # Adaptive momentum
                    var_correction = math.sqrt(
                        (rho - 4) * (rho - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho)
                    )
                    
                    denominator = (exp_avg_sq / bias_correction2).sqrt() + group['eps']
                    step_size = group['lr'] * var_correction / bias_correction1
                    
                    p.data.add_(exp_avg / denominator, alpha=-step_size)
                else:
                    # Fallback to bias-corrected first moment
                    step_size = group['lr'] / bias_correction1
                    p.data.add_(exp_avg, alpha=-step_size)
                    
                # Weight decay
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                    
        return loss


class Lookahead(torch.optim.Optimizer):
    """Lookahead optimizer wrapper."""
    
    def __init__(self, base_optimizer, k=5, alpha=0.5):
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.base_optimizer.param_groups
        self.state = {}
        
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state.setdefault(p, {})
                param_state['slow_buffer'] = torch.zeros_like(p.data)
                param_state['slow_buffer'].copy_(p.data)
                
    def step(self, closure=None):
        """Perform optimization step."""
        loss = self.base_optimizer.step(closure)
        
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                
                if 'step' not in param_state:
                    param_state['step'] = 0
                    
                param_state['step'] += 1
                
                # Update slow weights
                if param_state['step'] % self.k == 0:
                    slow_buffer = param_state['slow_buffer']
                    
                    # Interpolate between fast and slow weights
                    slow_buffer.add_(p.data - slow_buffer, alpha=self.alpha)
                    
                    # Copy slow weights to fast weights
                    p.data.copy_(slow_buffer)
                    
        return loss
        
    def zero_grad(self):
        """Clear gradients."""
        self.base_optimizer.zero_grad()


class GradientClippingManager:
    """Advanced gradient clipping strategies."""
    
    def __init__(
        self,
        clip_type: str = "norm",  # norm, value, adaptive
        clip_value: float = 1.0,
        adaptive_threshold: float = 10.0
    ):
        self.clip_type = clip_type
        self.clip_value = clip_value
        self.adaptive_threshold = adaptive_threshold
        
        self.gradient_history = []
        self.logger = get_logger(__name__)
        
    def clip_gradients(
        self,
        parameters,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> float:
        """Clip gradients using specified strategy."""
        if self.clip_type == "norm":
            return self._clip_by_norm(parameters)
        elif self.clip_type == "value":
            return self._clip_by_value(parameters)
        elif self.clip_type == "adaptive":
            return self._clip_adaptive(parameters)
        else:
            raise ValueError(f"Unknown clipping type: {self.clip_type}")
            
    def _clip_by_norm(self, parameters) -> float:
        """Clip gradients by global norm."""
        total_norm = torch.nn.utils.clip_grad_norm_(parameters, self.clip_value)
        return float(total_norm)
        
    def _clip_by_value(self, parameters) -> float:
        """Clip gradients by value."""
        total_norm = 0.0
        
        for param in parameters:
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                
                # Clip by value
                param.grad.data.clamp_(-self.clip_value, self.clip_value)
                
        return math.sqrt(total_norm)
        
    def _clip_adaptive(self, parameters) -> float:
        """Adaptive gradient clipping based on gradient history."""
        # Compute current gradient norm
        total_norm = 0.0
        for param in parameters:
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                
        total_norm = math.sqrt(total_norm)
        self.gradient_history.append(total_norm)
        
        # Keep only recent history
        if len(self.gradient_history) > 100:
            self.gradient_history = self.gradient_history[-100:]
            
        # Compute adaptive threshold
        if len(self.gradient_history) > 10:
            mean_norm = np.mean(self.gradient_history)
            std_norm = np.std(self.gradient_history)
            adaptive_clip = mean_norm + self.adaptive_threshold * std_norm
        else:
            adaptive_clip = self.clip_value
            
        # Apply clipping
        if total_norm > adaptive_clip:
            clip_coef = adaptive_clip / (total_norm + 1e-6)
            for param in parameters:
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
                    
        return total_norm


class OptimizerFactory:
    """Factory for creating optimizers with different configurations."""
    
    @staticmethod
    def create_optimizer(
        model: nn.Module,
        optimizer_name: str,
        config: OptimizationConfig
    ) -> torch.optim.Optimizer:
        """Create optimizer based on configuration."""
        parameters = model.parameters()
        
        if optimizer_name.lower() == "adam":
            return optim.Adam(
                parameters,
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=config.epsilon,
                weight_decay=config.weight_decay,
                amsgrad=config.amsgrad
            )
        elif optimizer_name.lower() == "adamw":
            return optim.AdamW(
                parameters,
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=config.epsilon,
                weight_decay=config.weight_decay,
                amsgrad=config.amsgrad
            )
        elif optimizer_name.lower() == "sgd":
            return optim.SGD(
                parameters,
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay
            )
        elif optimizer_name.lower() == "radam":
            return RAdam(
                parameters,
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=config.epsilon,
                weight_decay=config.weight_decay
            )
        elif optimizer_name.lower() == "adaptive":
            return AdaptiveOptimizer(parameters, config)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
            
    @staticmethod
    def create_scheduler(
        optimizer: torch.optim.Optimizer,
        scheduler_name: str,
        config: OptimizationConfig,
        total_steps: Optional[int] = None
    ):
        """Create learning rate scheduler."""
        if scheduler_name.lower() == "adaptive":
            return AdaptiveLearningRateScheduler(optimizer, config, total_steps)
        elif scheduler_name.lower() == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps or 1000,
                eta_min=config.min_lr
            )
        elif scheduler_name.lower() == "step":
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.patience,
                gamma=config.factor
            )
        elif scheduler_name.lower() == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=config.factor,
                patience=config.patience,
                min_lr=config.min_lr
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")


class OptimizationPipeline:
    """Complete optimization pipeline with monitoring."""
    
    def __init__(
        self,
        model: nn.Module,
        config: OptimizationConfig,
        optimizer_name: str = "adaptive",
        scheduler_name: str = "adaptive"
    ):
        self.model = model
        self.config = config
        
        # Create optimizer and scheduler
        self.optimizer = OptimizerFactory.create_optimizer(
            model, optimizer_name, config
        )
        self.scheduler = OptimizerFactory.create_scheduler(
            self.optimizer, scheduler_name, config
        )
        
        # Gradient clipping
        self.grad_clipper = GradientClippingManager(
            clip_type="adaptive",
            clip_value=1.0
        )
        
        # Monitoring
        self.optimization_history = []
        self.logger = get_logger(__name__)
        
    def step(self, loss: torch.Tensor) -> Dict[str, float]:
        """Perform optimization step with monitoring."""
        # Backward pass
        loss.backward()
        
        # Clip gradients
        grad_norm = self.grad_clipper.clip_gradients(self.model.parameters())
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Scheduler step
        if hasattr(self.scheduler, 'step'):
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(loss.item())
            else:
                self.scheduler.step(loss.item() if hasattr(self.scheduler, '_adapt_to_loss') else None)
                
        # Get current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # Record metrics
        metrics = {
            'loss': loss.item(),
            'learning_rate': current_lr,
            'gradient_norm': grad_norm
        }
        
        self.optimization_history.append(metrics)
        
        return metrics
        
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        if not self.optimization_history:
            return {}
            
        recent_history = self.optimization_history[-100:]  # Last 100 steps
        
        return {
            'total_steps': len(self.optimization_history),
            'recent_avg_loss': np.mean([h['loss'] for h in recent_history]),
            'recent_avg_lr': np.mean([h['learning_rate'] for h in recent_history]),
            'recent_avg_grad_norm': np.mean([h['gradient_norm'] for h in recent_history]),
            'loss_trend': self._compute_loss_trend(),
            'lr_stability': self._compute_lr_stability()
        }
        
    def _compute_loss_trend(self) -> float:
        """Compute loss trend (negative means decreasing)."""
        if len(self.optimization_history) < 10:
            return 0.0
            
        recent_losses = [h['loss'] for h in self.optimization_history[-50:]]
        x = np.arange(len(recent_losses))
        trend = np.polyfit(x, recent_losses, 1)[0]
        
        return float(trend)
        
    def _compute_lr_stability(self) -> float:
        """Compute learning rate stability (lower is more stable)."""
        if len(self.optimization_history) < 10:
            return 0.0
            
        recent_lrs = [h['learning_rate'] for h in self.optimization_history[-50:]]
        return float(np.std(recent_lrs) / (np.mean(recent_lrs) + 1e-8))