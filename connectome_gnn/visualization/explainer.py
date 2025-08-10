"""Model explainability for brain GNNs."""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from pathlib import Path
from captum.attr import (
    IntegratedGradients,
    GradientShap,
    Saliency,
    InputXGradient,
    GuidedBackprop
)
from captum.attr import LayerIntegratedGradients, LayerConductance
import networkx as nx
from sklearn.metrics import roc_auc_score
import pandas as pd

from ..models.base import BaseConnectomeModel


class BrainGNNExplainer:
    """Comprehensive explainer for brain GNN models.
    
    Provides multiple explanation methods:
    - Gradient-based methods (Saliency, Integrated Gradients, etc.)
    - Attention visualization
    - Feature importance
    - Subgraph explanation
    - Counterfactual analysis
    """
    
    def __init__(
        self,
        model: BaseConnectomeModel,
        device: torch.device = None
    ):
        self.model = model
        self.device = device or torch.device('cpu')
        self.model.to(self.device)
        
        # Initialize attribution methods
        self.attribution_methods = {
            'saliency': Saliency(self.model),
            'integrated_gradients': IntegratedGradients(self.model),
            'gradient_shap': GradientShap(self.model),
            'input_x_gradient': InputXGradient(self.model),
            'guided_backprop': GuidedBackprop(self.model)
        }
        
        # Layer attribution methods (for intermediate layers)
        self.layer_methods = {}
        
        # Store explanations
        self.explanations = {}
    
    def explain_prediction(
        self,
        data,
        target_class: Optional[int] = None,
        method: str = 'integrated_gradients',
        baseline: str = 'zero',
        n_steps: int = 50
    ) -> Dict[str, torch.Tensor]:
        """Generate explanation for a single prediction.
        
        Args:
            data: Input data (PyTorch Geometric Data object)
            target_class: Target class for explanation (None for regression)
            method: Attribution method to use
            baseline: Baseline for integrated gradients ('zero', 'random', 'mean')
            n_steps: Number of steps for integrated gradients
            
        Returns:
            Dictionary containing node and edge attributions
        """
        self.model.eval()
        
        # Prepare input
        x = data.x.to(self.device).requires_grad_(True)
        edge_index = data.edge_index.to(self.device)
        edge_attr = data.edge_attr.to(self.device).requires_grad_(True) if data.edge_attr is not None else None
        
        # Create baseline
        if method in ['integrated_gradients', 'gradient_shap']:
            if baseline == 'zero':
                x_baseline = torch.zeros_like(x)
                edge_baseline = torch.zeros_like(edge_attr) if edge_attr is not None else None
            elif baseline == 'random':
                x_baseline = torch.randn_like(x)
                edge_baseline = torch.randn_like(edge_attr) if edge_attr is not None else None
            elif baseline == 'mean':
                x_baseline = torch.mean(x, dim=0, keepdim=True).expand_as(x)
                edge_baseline = torch.mean(edge_attr, dim=0, keepdim=True).expand_as(edge_attr) if edge_attr is not None else None
        else:
            x_baseline = edge_baseline = None
        
        # Forward function for attribution
        def forward_func(*inputs):
            x_input, edge_attr_input = inputs[0], inputs[1] if len(inputs) > 1 else None
            
            # Create temporary data object
            temp_data = type(data)()
            temp_data.x = x_input
            temp_data.edge_index = edge_index
            temp_data.edge_attr = edge_attr_input
            
            # Get prediction
            outputs = self.model(temp_data)
            
            if isinstance(outputs, dict):
                predictions = outputs['predictions']
            else:
                predictions = outputs
            
            if target_class is not None:
                return predictions[:, target_class] if predictions.dim() > 1 else predictions
            else:
                return predictions.squeeze() if predictions.dim() > 1 else predictions
        
        # Compute attributions
        attr_method = self.attribution_methods[method]
        
        if method == 'integrated_gradients':
            if edge_attr is not None:
                node_attr, edge_attr_attr = attr_method.attribute(
                    (x, edge_attr),
                    baselines=(x_baseline, edge_baseline),
                    n_steps=n_steps,
                    return_convergence_delta=False
                )
            else:
                node_attr = attr_method.attribute(
                    x,
                    baselines=x_baseline,
                    n_steps=n_steps
                )
                edge_attr_attr = None
        
        elif method == 'gradient_shap':
            # Generate multiple baselines for GradientShap
            n_samples = 10
            if baseline == 'zero':
                baselines = [torch.zeros_like(x) for _ in range(n_samples)]
                edge_baselines = [torch.zeros_like(edge_attr) for _ in range(n_samples)] if edge_attr is not None else None
            else:
                baselines = [torch.randn_like(x) for _ in range(n_samples)]
                edge_baselines = [torch.randn_like(edge_attr) for _ in range(n_samples)] if edge_attr is not None else None
            
            if edge_attr is not None:
                node_attr, edge_attr_attr = attr_method.attribute(
                    (x, edge_attr),
                    baselines=(baselines, edge_baselines),
                    n_samples=n_samples
                )
            else:
                node_attr = attr_method.attribute(
                    x,
                    baselines=baselines,
                    n_samples=n_samples
                )
                edge_attr_attr = None
        
        else:
            # Simple gradient-based methods
            if edge_attr is not None:
                node_attr, edge_attr_attr = attr_method.attribute((x, edge_attr))
            else:
                node_attr = attr_method.attribute(x)
                edge_attr_attr = None
        
        explanations = {
            'node_attributions': node_attr.detach(),
            'edge_attributions': edge_attr_attr.detach() if edge_attr_attr is not None else None,
            'method': method,
            'target_class': target_class
        }
        
        return explanations
    
    def explain_edges(
        self,
        data,
        target_class: Optional[Union[str, int]] = None,
        method: str = 'integrated_gradients',
        top_k: int = 20
    ) -> Dict[str, Any]:
        """Explain important edges for prediction.
        
        Args:
            data: Input connectome data
            target_class: Target class or description
            method: Attribution method
            top_k: Number of top edges to return
            
        Returns:
            Dictionary with edge explanations
        """
        # Get basic explanations
        explanations = self.explain_prediction(data, target_class, method)
        
        edge_index = data.edge_index.numpy()
        edge_attributions = explanations['edge_attributions']
        
        if edge_attributions is None:
            # If no edge attributions, use connectivity strengths
            edge_attributions = data.edge_attr.abs() if data.edge_attr is not None else torch.ones(edge_index.shape[1])
        
        # Compute edge importance scores
        if edge_attributions.dim() > 1:
            edge_scores = edge_attributions.abs().mean(dim=1)
        else:
            edge_scores = edge_attributions.abs()
        
        # Get top-k edges
        top_k_indices = torch.topk(edge_scores, min(top_k, len(edge_scores)))[1]
        
        important_edges = []
        for idx in top_k_indices:
            src, tgt = edge_index[:, idx]
            score = edge_scores[idx].item()
            
            important_edges.append({
                'source': int(src),
                'target': int(tgt),
                'importance_score': score,
                'edge_index': int(idx)
            })
        
        return {
            'important_edges': important_edges,
            'edge_scores': edge_scores,
            'method': method,
            'target_class': target_class
        }
    
    def explain_nodes(
        self,
        data,
        target_class: Optional[Union[str, int]] = None,
        method: str = 'integrated_gradients',
        top_k: int = 10
    ) -> Dict[str, Any]:
        """Explain important brain regions for prediction.
        
        Args:
            data: Input connectome data
            target_class: Target class or description
            method: Attribution method
            top_k: Number of top regions to return
            
        Returns:
            Dictionary with node explanations
        """
        # Get explanations
        explanations = self.explain_prediction(data, target_class, method)
        
        node_attributions = explanations['node_attributions']
        
        # Compute node importance scores
        if node_attributions.dim() > 1:
            node_scores = node_attributions.abs().mean(dim=1)
        else:
            node_scores = node_attributions.abs()
        
        # Get top-k nodes
        top_k_indices = torch.topk(node_scores, min(top_k, len(node_scores)))[1]
        
        important_regions = []
        for idx in top_k_indices:
            score = node_scores[idx].item()
            
            important_regions.append({
                'region_index': int(idx),
                'importance_score': score,
                'attribution_vector': node_attributions[idx].tolist()
            })
        
        return {
            'important_regions': important_regions,
            'node_scores': node_scores,
            'method': method,
            'target_class': target_class
        }
    
    def attention_explanation(
        self,
        data,
        layer_name: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """Extract attention weights if model has attention mechanism."""
        self.model.eval()
        
        with torch.no_grad():
            data = data.to(self.device)
            outputs = self.model(data)
            
            # Try to extract attention weights
            attention_weights = None
            
            if isinstance(outputs, dict) and 'attention_weights' in outputs:
                attention_weights = outputs['attention_weights']
            elif hasattr(self.model, 'get_attention_weights'):
                attention_weights = self.model.get_attention_weights(data)
            
            # Try to get attention from specific layers
            if attention_weights is None:
                for module_name, module in self.model.named_modules():
                    if hasattr(module, 'last_attention') and module.last_attention is not None:
                        attention_weights = module.last_attention
                        break
        
        return {
            'attention_weights': attention_weights,
            'layer_name': layer_name
        }
    
    def counterfactual_analysis(
        self,
        data,
        target_regions: List[int],
        perturbation_strength: float = 0.1,
        n_samples: int = 100
    ) -> Dict[str, Any]:
        """Perform counterfactual analysis.
        
        Args:
            data: Original data
            target_regions: Regions to perturb
            perturbation_strength: Strength of perturbation
            n_samples: Number of counterfactual samples
            
        Returns:
            Counterfactual analysis results
        """
        self.model.eval()
        
        # Get original prediction
        with torch.no_grad():
            original_data = data.to(self.device)
            original_output = self.model(original_data)
            if isinstance(original_output, dict):
                original_pred = original_output['predictions']
            else:
                original_pred = original_output
        
        # Generate counterfactuals
        counterfactual_preds = []
        
        for _ in range(n_samples):
            # Create perturbed data
            perturbed_data = original_data.clone()
            
            # Add noise to target regions
            noise = torch.randn_like(perturbed_data.x[target_regions]) * perturbation_strength
            perturbed_data.x[target_regions] += noise
            
            # Get prediction
            with torch.no_grad():
                perturbed_output = self.model(perturbed_data)
                if isinstance(perturbed_output, dict):
                    perturbed_pred = perturbed_output['predictions']
                else:
                    perturbed_pred = perturbed_output
                
                counterfactual_preds.append(perturbed_pred.cpu())
        
        # Analyze changes
        counterfactual_preds = torch.stack(counterfactual_preds)
        pred_changes = counterfactual_preds - original_pred.cpu()
        
        return {
            'original_prediction': original_pred.cpu(),
            'counterfactual_predictions': counterfactual_preds,
            'prediction_changes': pred_changes,
            'mean_change': pred_changes.mean(dim=0),
            'std_change': pred_changes.std(dim=0),
            'perturbation_strength': perturbation_strength,
            'target_regions': target_regions
        }
    
    def visualize_explanation(
        self,
        explanations: Dict[str, Any],
        coordinates: np.ndarray,
        region_names: Optional[List[str]] = None,
        brain_template: str = "MNI152",
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """Visualize explanations on brain.
        
        Args:
            explanations: Explanation results
            coordinates: 3D brain coordinates
            region_names: Names of brain regions
            brain_template: Brain template name
            save_path: Path to save visualization
            
        Returns:
            Matplotlib figure
        """
        if 'important_regions' in explanations:
            return self._visualize_node_explanations(
                explanations, coordinates, region_names, save_path
            )
        elif 'important_edges' in explanations:
            return self._visualize_edge_explanations(
                explanations, coordinates, region_names, save_path
            )
        else:
            raise ValueError("Unknown explanation type")
    
    def _visualize_node_explanations(
        self,
        explanations: Dict[str, Any],
        coordinates: np.ndarray,
        region_names: Optional[List[str]],
        save_path: Optional[Path]
    ) -> plt.Figure:
        """Visualize node importance."""
        important_regions = explanations['important_regions']
        node_scores = explanations['node_scores'].numpy()
        
        fig = plt.figure(figsize=(15, 5))
        
        # 3D brain plot
        ax1 = fig.add_subplot(131, projection='3d')
        
        # Plot all regions
        scatter = ax1.scatter(
            coordinates[:, 0],
            coordinates[:, 1], 
            coordinates[:, 2],
            c=node_scores,
            s=50,
            cmap='viridis',
            alpha=0.6
        )
        
        # Highlight important regions
        important_indices = [r['region_index'] for r in important_regions]
        ax1.scatter(
            coordinates[important_indices, 0],
            coordinates[important_indices, 1],
            coordinates[important_indices, 2],
            c='red',
            s=100,
            alpha=0.8
        )
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('Brain Regions Importance')
        plt.colorbar(scatter, ax=ax1, label='Importance Score')
        
        # Bar plot of top regions
        ax2 = fig.add_subplot(132)
        
        top_scores = [r['importance_score'] for r in important_regions[:10]]
        top_indices = [r['region_index'] for r in important_regions[:10]]
        
        if region_names:
            labels = [region_names[i] for i in top_indices]
        else:
            labels = [f'Region {i}' for i in top_indices]
        
        bars = ax2.barh(range(len(top_scores)), top_scores)
        ax2.set_yticks(range(len(top_scores)))
        ax2.set_yticklabels(labels)
        ax2.set_xlabel('Importance Score')
        ax2.set_title('Top Important Regions')
        
        # Color bars by importance
        colors = plt.cm.viridis([s/max(top_scores) for s in top_scores])
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Histogram of all scores
        ax3 = fig.add_subplot(133)
        ax3.hist(node_scores, bins=30, alpha=0.7, color='skyblue')
        ax3.axvline(np.mean(node_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(node_scores):.3f}')
        ax3.set_xlabel('Importance Score')
        ax3.set_ylabel('Number of Regions')
        ax3.set_title('Distribution of Importance Scores')
        ax3.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _visualize_edge_explanations(
        self,
        explanations: Dict[str, Any],
        coordinates: np.ndarray,
        region_names: Optional[List[str]],
        save_path: Optional[Path]
    ) -> plt.Figure:
        """Visualize edge importance."""
        important_edges = explanations['important_edges']
        
        fig = plt.figure(figsize=(12, 8))
        
        # 3D brain plot with edges
        ax1 = fig.add_subplot(121, projection='3d')
        
        # Plot all regions
        ax1.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            coordinates[:, 2],
            c='lightblue',
            s=30,
            alpha=0.6
        )
        
        # Plot important edges
        max_score = max([e['importance_score'] for e in important_edges])
        
        for edge in important_edges[:10]:  # Top 10 edges
            src, tgt = edge['source'], edge['target']
            score = edge['importance_score']
            
            # Draw edge
            ax1.plot(
                [coordinates[src, 0], coordinates[tgt, 0]],
                [coordinates[src, 1], coordinates[tgt, 1]],
                [coordinates[src, 2], coordinates[tgt, 2]],
                'r-',
                alpha=score/max_score,
                linewidth=2
            )
            
            # Highlight connected nodes
            ax1.scatter(
                [coordinates[src, 0], coordinates[tgt, 0]],
                [coordinates[src, 1], coordinates[tgt, 1]],
                [coordinates[src, 2], coordinates[tgt, 2]],
                c='red',
                s=60
            )
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('Important Brain Connections')
        
        # Bar plot of edge importance
        ax2 = fig.add_subplot(122)
        
        top_scores = [e['importance_score'] for e in important_edges[:10]]
        edge_labels = []
        
        for e in important_edges[:10]:
            src, tgt = e['source'], e['target']
            if region_names:
                label = f"{region_names[src][:10]}\n-> {region_names[tgt][:10]}"
            else:
                label = f"R{src}\n-> R{tgt}"
            edge_labels.append(label)
        
        bars = ax2.barh(range(len(top_scores)), top_scores)
        ax2.set_yticks(range(len(top_scores)))
        ax2.set_yticklabels(edge_labels, fontsize=8)
        ax2.set_xlabel('Importance Score')
        ax2.set_title('Top Important Connections')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class ConnectomeExplainer(BrainGNNExplainer):
    """Specialized explainer for connectome data with neuroscience-specific methods."""
    
    def __init__(self, model: BaseConnectomeModel, atlas: str = "AAL"):
        super().__init__(model)
        self.atlas = atlas
        
        # Neuroscience-specific explanation methods
        self.network_analyzer = NetworkAnalyzer()
    
    def network_analysis(
        self,
        data,
        explanations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze explanations from network neuroscience perspective."""
        # Build graph from important connections
        G = nx.Graph()
        
        if 'important_edges' in explanations:
            for edge in explanations['important_edges']:
                G.add_edge(edge['source'], edge['target'], 
                          weight=edge['importance_score'])
        
        # Compute network metrics
        metrics = {
            'clustering': nx.average_clustering(G),
            'path_length': nx.average_shortest_path_length(G) if nx.is_connected(G) else float('inf'),
            'modularity': self._compute_modularity(G),
            'small_world': self._compute_small_world_coefficient(G),
            'rich_club': self._compute_rich_club(G)
        }
        
        return {
            'network_metrics': metrics,
            'graph': G,
            'explanation_network': explanations
        }
    
    def _compute_modularity(self, G: nx.Graph) -> float:
        """Compute modularity of the network."""
        try:
            communities = nx.community.greedy_modularity_communities(G)
            return nx.community.modularity(G, communities)
        except:
            return 0.0
    
    def _compute_small_world_coefficient(self, G: nx.Graph) -> float:
        """Compute small-world coefficient."""
        try:
            return nx.sigma(G)
        except:
            return 0.0
    
    def _compute_rich_club(self, G: nx.Graph) -> Dict[int, float]:
        """Compute rich club coefficient."""
        try:
            return nx.rich_club_coefficient(G)
        except:
            return {}


class NetworkAnalyzer:
    """Network analysis utilities for brain explanations."""
    
    def analyze_connectivity_patterns(
        self,
        connectivity_matrix: np.ndarray,
        importance_scores: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze connectivity patterns with importance weighting."""
        
        # Weight connectivity by importance
        weighted_connectivity = connectivity_matrix * importance_scores.reshape(-1, 1)
        
        # Create graph
        G = nx.from_numpy_array(weighted_connectivity)
        
        # Compute network metrics
        metrics = {
            'degree_centrality': nx.degree_centrality(G),
            'betweenness_centrality': nx.betweenness_centrality(G),
            'eigenvector_centrality': nx.eigenvector_centrality(G, max_iter=1000),
            'pagerank': nx.pagerank(G),
            'clustering': nx.clustering(G)
        }
        
        return {
            'network_metrics': metrics,
            'weighted_connectivity': weighted_connectivity,
            'graph': G
        }