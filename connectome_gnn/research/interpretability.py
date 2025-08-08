"""Interpretability and explainability tools for connectome GNNs."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from ..models.base import BaseConnectomeModel
from ..utils import get_brain_region_names


@dataclass
class AttentionAnalysis:
    """Results from attention analysis."""
    
    attention_weights: torch.Tensor
    top_connections: List[Tuple[int, int, float]]
    region_importance: torch.Tensor
    network_interactions: Dict[str, float]
    attention_entropy: float


@dataclass
class ConnectomeExplanation:
    """Explanation of model prediction for a single connectome."""
    
    subject_id: str
    prediction: float
    target: Optional[float]
    
    # Node-level explanations
    node_importance: torch.Tensor
    node_contributions: torch.Tensor
    important_regions: List[Tuple[str, float]]
    
    # Edge-level explanations
    edge_importance: torch.Tensor
    important_connections: List[Tuple[str, str, float]]
    
    # Network-level explanations
    network_contributions: Dict[str, float]
    
    # Global explanations
    feature_importance: torch.Tensor
    explanation_fidelity: float


class ConnectomeExplainer:
    """Explainer for connectome GNN predictions."""
    
    def __init__(
        self,
        model: BaseConnectomeModel,
        parcellation: str = "AAL",
        device: Optional[torch.device] = None
    ):
        """Initialize connectome explainer.
        
        Args:
            model: Trained connectome model
            parcellation: Brain parcellation used
            device: Device for computation
        """
        self.model = model
        self.parcellation = parcellation
        self.device = device or torch.device('cpu')
        
        self.region_names = get_brain_region_names(parcellation)
        self.num_regions = len(self.region_names)
        
        # Brain network definitions
        self.brain_networks = self._define_brain_networks()
        
        # Set model to evaluation mode
        self.model.eval()
    
    def explain_prediction(
        self,
        data,
        method: str = "integrated_gradients",
        baseline: Optional[torch.Tensor] = None,
        steps: int = 50
    ) -> ConnectomeExplanation:
        """Explain a single prediction.
        
        Args:
            data: Input connectome data
            method: Explanation method (integrated_gradients, lime, shap)
            baseline: Baseline for integrated gradients
            steps: Number of integration steps
            
        Returns:
            Connectome explanation
        """
        subject_id = getattr(data, 'subject_id', 'unknown')
        
        # Get prediction
        with torch.no_grad():
            prediction = self.model(data).item()
            target = data.y.item() if hasattr(data, 'y') and data.y is not None else None
        
        # Compute explanations based on method
        if method == "integrated_gradients":
            explanations = self._integrated_gradients(data, baseline, steps)
        elif method == "lime":
            explanations = self._lime_explanation(data)
        elif method == "shap":
            explanations = self._shap_explanation(data)
        elif method == "gradient":
            explanations = self._gradient_explanation(data)
        else:
            raise ValueError(f"Unknown explanation method: {method}")
        
        # Process explanations
        node_importance = explanations['node_importance']
        edge_importance = explanations.get('edge_importance', torch.zeros(data.edge_index.size(1)))
        
        # Get important regions
        region_scores = node_importance.abs()
        top_k = min(10, len(region_scores))
        top_indices = torch.topk(region_scores, top_k).indices
        
        important_regions = [
            (self.region_names[idx], region_scores[idx].item())
            for idx in top_indices
        ]
        
        # Get important connections
        if data.edge_index.size(1) > 0:
            top_edge_k = min(10, data.edge_index.size(1))
            top_edge_indices = torch.topk(edge_importance.abs(), top_edge_k).indices
            
            important_connections = []
            for edge_idx in top_edge_indices:
                source_idx = data.edge_index[0, edge_idx].item()
                target_idx = data.edge_index[1, edge_idx].item()
                edge_score = edge_importance[edge_idx].item()
                
                source_name = self.region_names[source_idx] if source_idx < len(self.region_names) else f"Region_{source_idx}"
                target_name = self.region_names[target_idx] if target_idx < len(self.region_names) else f"Region_{target_idx}"
                
                important_connections.append((source_name, target_name, edge_score))
        else:
            important_connections = []
        
        # Network-level contributions
        network_contributions = self._compute_network_contributions(node_importance)
        
        # Feature importance (if applicable)
        feature_importance = explanations.get('feature_importance', torch.zeros(data.x.size(1)))
        
        # Compute explanation fidelity
        fidelity = self._compute_explanation_fidelity(data, explanations)
        
        return ConnectomeExplanation(
            subject_id=subject_id,
            prediction=prediction,
            target=target,
            node_importance=node_importance,
            node_contributions=node_importance,  # For now, same as importance
            important_regions=important_regions,
            edge_importance=edge_importance,
            important_connections=important_connections,
            network_contributions=network_contributions,
            feature_importance=feature_importance,
            explanation_fidelity=fidelity
        )
    
    def _integrated_gradients(
        self,
        data,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 50
    ) -> Dict[str, torch.Tensor]:
        """Compute integrated gradients explanation."""
        
        # Create baseline if not provided
        if baseline is None:
            baseline = torch.zeros_like(data.x)
        
        # Prepare for gradient computation
        data.x.requires_grad_(True)
        
        # Compute integrated gradients
        integrated_grads = torch.zeros_like(data.x)
        
        for step in range(steps):
            alpha = step / (steps - 1) if steps > 1 else 1.0
            
            # Interpolate between baseline and input
            interpolated_x = baseline + alpha * (data.x - baseline)
            
            # Create new data object with interpolated features
            interpolated_data = data.clone()
            interpolated_data.x = interpolated_x
            interpolated_data.x.requires_grad_(True)
            
            # Forward pass
            output = self.model(interpolated_data)
            
            # Compute gradients
            gradients = torch.autograd.grad(
                outputs=output.sum(),
                inputs=interpolated_data.x,
                create_graph=False,
                retain_graph=False
            )[0]
            
            integrated_grads += gradients
        
        # Average and scale by input difference
        integrated_grads = integrated_grads / steps
        integrated_grads = integrated_grads * (data.x - baseline)
        
        # Node importance (sum across features)
        node_importance = integrated_grads.sum(dim=1)
        
        # Feature importance (sum across nodes)
        feature_importance = integrated_grads.sum(dim=0)
        
        return {
            'node_importance': node_importance,
            'feature_importance': feature_importance,
            'integrated_gradients': integrated_grads
        }
    
    def _gradient_explanation(self, data) -> Dict[str, torch.Tensor]:
        """Compute gradient-based explanation."""
        
        data.x.requires_grad_(True)
        
        # Forward pass
        output = self.model(data)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=output.sum(),
            inputs=data.x,
            create_graph=False,
            retain_graph=False
        )[0]
        
        # Node importance
        node_importance = gradients.sum(dim=1)
        
        # Feature importance
        feature_importance = gradients.sum(dim=0)
        
        return {
            'node_importance': node_importance,
            'feature_importance': feature_importance,
            'gradients': gradients
        }
    
    def _lime_explanation(self, data) -> Dict[str, torch.Tensor]:
        """LIME-based explanation (simplified implementation)."""
        
        # Generate perturbations
        num_samples = 1000
        perturbations = []
        predictions = []
        
        original_prediction = self.model(data).item()
        
        for _ in range(num_samples):
            # Perturb features
            perturbed_data = data.clone()
            
            # Random masking of features
            mask = torch.rand_like(data.x) > 0.3  # Keep 70% of features
            perturbed_data.x = data.x * mask.float()
            
            # Get prediction
            with torch.no_grad():
                pred = self.model(perturbed_data).item()
            
            perturbations.append(mask.float().flatten())
            predictions.append(pred)
        
        # Convert to numpy for linear regression
        X = torch.stack(perturbations).numpy()
        y = np.array(predictions)
        
        # Fit linear model
        from sklearn.linear_model import LinearRegression
        
        lr = LinearRegression()
        lr.fit(X, y)
        
        # Feature importance from linear model coefficients
        feature_importance = torch.tensor(lr.coef_[:data.x.size(1)])
        node_importance = torch.tensor(lr.coef_[data.x.size(1):]) if len(lr.coef_) > data.x.size(1) else torch.zeros(data.x.size(0))
        
        return {
            'node_importance': node_importance,
            'feature_importance': feature_importance
        }
    
    def _shap_explanation(self, data) -> Dict[str, torch.Tensor]:
        """SHAP-based explanation (simplified implementation)."""
        
        # Simplified SHAP using sampling
        num_samples = 500
        baseline_prediction = self.model(self._create_baseline(data)).item()
        
        node_contributions = torch.zeros(data.x.size(0))
        feature_contributions = torch.zeros(data.x.size(1))
        
        # Sample coalitions
        for _ in range(num_samples):
            # Random coalition of features
            coalition_mask = torch.rand_like(data.x) > 0.5
            
            # Compute marginal contribution
            with_data = data.clone()
            with_data.x = data.x * coalition_mask.float()
            
            without_data = self._create_baseline(data)
            
            with_pred = self.model(with_data).item()
            without_pred = self.model(without_data).item()
            
            marginal_contribution = with_pred - without_pred
            
            # Attribute to nodes and features
            node_mask = coalition_mask.any(dim=1).float()
            feature_mask = coalition_mask.any(dim=0).float()
            
            node_contributions += marginal_contribution * node_mask
            feature_contributions += marginal_contribution * feature_mask
        
        # Average contributions
        node_importance = node_contributions / num_samples
        feature_importance = feature_contributions / num_samples
        
        return {
            'node_importance': node_importance,
            'feature_importance': feature_importance
        }
    
    def _create_baseline(self, data):
        """Create baseline data for SHAP."""
        baseline_data = data.clone()
        baseline_data.x = torch.zeros_like(data.x)
        return baseline_data
    
    def _compute_network_contributions(self, node_importance: torch.Tensor) -> Dict[str, float]:
        """Compute network-level contributions."""
        
        network_contributions = {}
        
        for network_name, region_indices in self.brain_networks.items():
            # Filter indices that exist in our parcellation
            valid_indices = [idx for idx in region_indices if idx < len(node_importance)]
            
            if valid_indices:
                network_contribution = node_importance[valid_indices].sum().item()
                network_contributions[network_name] = network_contribution
            else:
                network_contributions[network_name] = 0.0
        
        return network_contributions
    
    def _compute_explanation_fidelity(self, data, explanations: Dict[str, torch.Tensor]) -> float:
        """Compute fidelity of explanation."""
        
        # Original prediction
        original_pred = self.model(data).item()
        
        # Prediction with top features only
        node_importance = explanations['node_importance']
        top_k = min(10, len(node_importance))
        top_indices = torch.topk(node_importance.abs(), top_k).indices
        
        # Create masked data
        masked_data = data.clone()
        mask = torch.zeros_like(data.x)
        mask[top_indices] = 1.0
        masked_data.x = data.x * mask
        
        masked_pred = self.model(masked_data).item()
        
        # Fidelity as correlation between original and masked prediction
        fidelity = abs(original_pred - masked_pred) / (abs(original_pred) + 1e-8)
        
        return 1.0 - fidelity  # Higher is better
    
    def _define_brain_networks(self) -> Dict[str, List[int]]:
        """Define brain networks based on parcellation."""
        
        networks = {}
        
        if self.parcellation == "AAL":
            # Simplified network definitions for AAL
            networks = {
                'frontal': [i for i, name in enumerate(self.region_names) 
                           if any(term in name.lower() for term in ['frontal', 'precentral', 'motor'])],
                'parietal': [i for i, name in enumerate(self.region_names)
                            if any(term in name.lower() for term in ['parietal', 'postcentral', 'precuneus'])],
                'temporal': [i for i, name in enumerate(self.region_names)
                            if 'temporal' in name.lower()],
                'occipital': [i for i, name in enumerate(self.region_names)
                             if any(term in name.lower() for term in ['occipital', 'calcarine', 'cuneus'])],
                'subcortical': [i for i, name in enumerate(self.region_names)
                               if any(term in name.lower() for term in ['hippocampus', 'amygdala', 'caudate', 'putamen', 'thalamus'])],
                'cerebellar': [i for i, name in enumerate(self.region_names)
                              if any(term in name.lower() for term in ['cerebelum', 'vermis'])]
            }
        else:
            # Generic networks
            num_regions = len(self.region_names)
            network_size = num_regions // 6
            
            networks = {
                'network_1': list(range(0, network_size)),
                'network_2': list(range(network_size, 2*network_size)),
                'network_3': list(range(2*network_size, 3*network_size)),
                'network_4': list(range(3*network_size, 4*network_size)),
                'network_5': list(range(4*network_size, 5*network_size)),
                'network_6': list(range(5*network_size, num_regions))
            }
        
        return networks
    
    def visualize_explanation(
        self,
        explanation: ConnectomeExplanation,
        save_path: Optional[str] = None,
        show_top_k: int = 10
    ):
        """Visualize connectome explanation."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Top important regions
        ax1 = axes[0, 0]
        regions, scores = zip(*explanation.important_regions[:show_top_k])
        
        colors = ['red' if score > 0 else 'blue' for score in scores]
        bars = ax1.barh(range(len(regions)), scores, color=colors)
        ax1.set_yticks(range(len(regions)))
        ax1.set_yticklabels(regions)
        ax1.set_xlabel('Importance Score')
        ax1.set_title(f'Top {show_top_k} Important Brain Regions')
        ax1.grid(True, alpha=0.3)
        
        # 2. Network contributions
        ax2 = axes[0, 1]
        networks = list(explanation.network_contributions.keys())
        contributions = list(explanation.network_contributions.values())
        
        colors = ['red' if contrib > 0 else 'blue' for contrib in contributions]
        ax2.bar(networks, contributions, color=colors)
        ax2.set_xlabel('Brain Networks')
        ax2.set_ylabel('Contribution')
        ax2.set_title('Network-Level Contributions')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. Feature importance heatmap
        ax3 = axes[1, 0]
        feature_importance = explanation.feature_importance.detach().numpy()
        
        # Reshape for visualization if needed
        if len(feature_importance) > 20:
            # Group features for visualization
            group_size = len(feature_importance) // 20
            grouped_features = []
            for i in range(0, len(feature_importance), group_size):
                group = feature_importance[i:i+group_size]
                grouped_features.append(np.mean(group))
            feature_importance = np.array(grouped_features)
        
        im = ax3.imshow(feature_importance.reshape(1, -1), cmap='RdBu_r', aspect='auto')
        ax3.set_title('Feature Importance')
        ax3.set_xlabel('Feature Index')
        ax3.set_yticks([])
        plt.colorbar(im, ax=ax3)
        
        # 4. Prediction summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = f"""
        Subject: {explanation.subject_id}
        Prediction: {explanation.prediction:.4f}
        Target: {explanation.target:.4f if explanation.target is not None else 'N/A'}
        Fidelity: {explanation.explanation_fidelity:.4f}
        
        Top Contributing Regions:
        {chr(10).join([f"• {region}: {score:.3f}" for region, score in explanation.important_regions[:5]])}
        """
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()


class BrainAttentionAnalyzer:
    """Analyzer for attention patterns in brain GNNs."""
    
    def __init__(self, model: BaseConnectomeModel, parcellation: str = "AAL"):
        """Initialize attention analyzer.
        
        Args:
            model: Model with attention mechanisms
            parcellation: Brain parcellation
        """
        self.model = model
        self.parcellation = parcellation
        self.region_names = get_brain_region_names(parcellation)
    
    def analyze_attention_patterns(self, data) -> AttentionAnalysis:
        """Analyze attention patterns in the model.
        
        Args:
            data: Input connectome data
            
        Returns:
            Attention analysis results
        """
        
        # Get attention weights from model
        attention_weights = self.model.get_attention_weights(data)
        
        if not attention_weights:
            raise ValueError("Model does not support attention weight extraction")
        
        # Analyze first attention layer
        attn_matrix = attention_weights[0]  # [num_heads, num_nodes, num_nodes]
        
        # Average across heads
        avg_attention = torch.mean(attn_matrix, dim=0)  # [num_nodes, num_nodes]
        
        # Find top connections
        num_nodes = avg_attention.size(0)
        top_connections = []
        
        # Get upper triangle (avoid duplicate connections)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                weight = avg_attention[i, j].item()
                top_connections.append((i, j, weight))
        
        # Sort by attention weight
        top_connections.sort(key=lambda x: abs(x[2]), reverse=True)
        top_connections = top_connections[:20]  # Top 20 connections
        
        # Compute region importance (sum of attention weights)
        region_importance = torch.sum(avg_attention, dim=1)
        
        # Network interactions
        network_interactions = self._compute_network_interactions(avg_attention)
        
        # Attention entropy
        attention_entropy = self._compute_attention_entropy(avg_attention)
        
        return AttentionAnalysis(
            attention_weights=avg_attention,
            top_connections=top_connections,
            region_importance=region_importance,
            network_interactions=network_interactions,
            attention_entropy=attention_entropy
        )
    
    def _compute_network_interactions(self, attention_matrix: torch.Tensor) -> Dict[str, float]:
        """Compute interactions between brain networks."""
        
        # Define networks (simplified)
        networks = {
            'frontal': list(range(0, 20)),
            'parietal': list(range(20, 40)),
            'temporal': list(range(40, 60)),
            'occipital': list(range(60, 80)),
            'subcortical': list(range(80, 90))
        }
        
        interactions = {}
        
        for net1_name, net1_regions in networks.items():
            for net2_name, net2_regions in networks.items():
                if net1_name <= net2_name:  # Avoid duplicates
                    # Filter regions that exist in attention matrix
                    valid_net1 = [r for r in net1_regions if r < attention_matrix.size(0)]
                    valid_net2 = [r for r in net2_regions if r < attention_matrix.size(1)]
                    
                    if valid_net1 and valid_net2:
                        interaction_strength = attention_matrix[valid_net1][:, valid_net2].mean().item()
                        interactions[f"{net1_name}-{net2_name}"] = interaction_strength
        
        return interactions
    
    def _compute_attention_entropy(self, attention_matrix: torch.Tensor) -> float:
        """Compute entropy of attention distribution."""
        
        # Flatten attention matrix and compute entropy
        attention_flat = attention_matrix.flatten()
        attention_flat = F.softmax(attention_flat, dim=0)
        
        # Compute entropy
        entropy = -torch.sum(attention_flat * torch.log(attention_flat + 1e-8))
        
        return entropy.item()
    
    def visualize_attention_matrix(
        self,
        analysis: AttentionAnalysis,
        save_path: Optional[str] = None
    ):
        """Visualize attention matrix."""
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Attention matrix heatmap
        ax1 = axes[0]
        attention_np = analysis.attention_weights.detach().numpy()
        
        im1 = ax1.imshow(attention_np, cmap='Blues', aspect='auto')
        ax1.set_title('Attention Matrix')
        ax1.set_xlabel('Target Regions')
        ax1.set_ylabel('Source Regions')
        plt.colorbar(im1, ax=ax1)
        
        # Region importance
        ax2 = axes[1]
        importance_np = analysis.region_importance.detach().numpy()
        
        # Show top regions only
        top_k = min(20, len(importance_np))
        top_indices = np.argsort(importance_np)[-top_k:]
        
        top_regions = [self.region_names[i] if i < len(self.region_names) else f"Region_{i}" 
                      for i in top_indices]
        top_scores = importance_np[top_indices]
        
        ax2.barh(range(len(top_regions)), top_scores)
        ax2.set_yticks(range(len(top_regions)))
        ax2.set_yticklabels(top_regions, fontsize=8)
        ax2.set_xlabel('Attention Importance')
        ax2.set_title(f'Top {top_k} Regions by Attention')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()