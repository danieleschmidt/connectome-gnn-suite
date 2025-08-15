"""Advanced visualization tools for connectome GNN research publication."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import json

from sklearn.manifold import TSNE, UMAP
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import gaussian_kde
from scipy.spatial.distance import pdist, squareform
import networkx as nx

try:
    import nibabel as nib
    from nilearn import plotting, datasets
    NILEARN_AVAILABLE = True
except ImportError:
    NILEARN_AVAILABLE = False

from ..models.base import BaseConnectomeModel
from ..utils import get_device


@dataclass
class VisualizationConfig:
    """Configuration for research visualizations."""
    
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    style: str = "publication"  # publication, presentation, web
    color_palette: str = "viridis"
    font_size: int = 12
    save_format: str = "png"  # png, pdf, svg
    interactive: bool = False


class PublicationVisualizer:
    """Advanced visualization suite for research publications."""
    
    def __init__(
        self,
        config: Optional[VisualizationConfig] = None,
        output_dir: str = "./visualizations"
    ):
        """Initialize publication visualizer.
        
        Args:
            config: Visualization configuration
            output_dir: Directory to save visualizations
        """
        self.config = config or VisualizationConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set publication style
        self._setup_publication_style()
        
        # Color palettes for different contexts
        self.color_palettes = {
            'sequential': ['viridis', 'plasma', 'inferno', 'magma'],
            'diverging': ['RdBu', 'RdYlBu', 'spectral', 'coolwarm'],
            'qualitative': ['Set1', 'Set2', 'tab10', 'Dark2'],
            'publication': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        }
    
    def _setup_publication_style(self):
        """Setup matplotlib style for publication-quality figures."""
        
        plt.style.use('default')
        
        # Publication parameters
        pub_params = {
            'figure.figsize': self.config.figure_size,
            'figure.dpi': self.config.dpi,
            'savefig.dpi': self.config.dpi,
            'font.size': self.config.font_size,
            'axes.titlesize': self.config.font_size + 2,
            'axes.labelsize': self.config.font_size,
            'xtick.labelsize': self.config.font_size - 1,
            'ytick.labelsize': self.config.font_size - 1,
            'legend.fontsize': self.config.font_size - 1,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
            'axes.linewidth': 1.0,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.axisbelow': True
        }
        
        plt.rcParams.update(pub_params)
    
    def create_model_comparison_figure(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str],
        save_path: Optional[str] = None
    ) -> str:
        """Create comprehensive model comparison figure.
        
        Args:
            results: Dictionary of model_name -> metrics
            metrics: List of metrics to compare
            save_path: Path to save figure
            
        Returns:
            Path to saved figure
        """
        
        # Create figure with subplots
        n_metrics = len(metrics)
        fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=(15, 10))
        axes = axes.flatten() if n_metrics > 1 else [axes]
        
        model_names = list(results.keys())
        colors = plt.cm.Set1(np.linspace(0, 1, len(model_names)))
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Extract metric values
            values = [results[model][metric] for model in model_names]
            
            # Bar plot with error bars (if available)
            bars = ax.bar(model_names, values, color=colors, alpha=0.7, edgecolor='black')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
            ax.set_ylabel('Score')
            ax.tick_params(axis='x', rotation=45)
            
            # Highlight best performer
            best_idx = np.argmax(values) if 'accuracy' in metric.lower() or 'f1' in metric.lower() else np.argmin(values)
            bars[best_idx].set_color('#FF6B6B')
            bars[best_idx].set_edgecolor('darkred')
            bars[best_idx].set_linewidth(2)
        
        # Remove empty subplots
        for i in range(n_metrics, len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / f"model_comparison.{self.config.save_format}"
        
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def create_training_dynamics_figure(
        self,
        training_histories: Dict[str, Dict[str, List[float]]],
        save_path: Optional[str] = None
    ) -> str:
        """Create training dynamics visualization.
        
        Args:
            training_histories: Dictionary of model_name -> history
            save_path: Path to save figure
            
        Returns:
            Path to saved figure
        """
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Colors for different models
        colors = plt.cm.tab10(np.linspace(0, 1, len(training_histories)))
        
        for i, (model_name, history) in enumerate(training_histories.items()):
            color = colors[i]
            
            # Training loss
            if 'train_loss' in history:
                axes[0, 0].plot(history['train_loss'], label=model_name, color=color, linewidth=2)
            
            # Validation loss
            if 'val_loss' in history:
                axes[0, 1].plot(history['val_loss'], label=model_name, color=color, linewidth=2)
            
            # Training accuracy/metric
            train_metric = next((k for k in history.keys() if 'train' in k and 'acc' in k), None)
            if train_metric:
                axes[1, 0].plot(history[train_metric], label=model_name, color=color, linewidth=2)
            
            # Validation accuracy/metric
            val_metric = next((k for k in history.keys() if 'val' in k and 'acc' in k), None)
            if val_metric:
                axes[1, 1].plot(history[val_metric], label=model_name, color=color, linewidth=2)
        
        # Configure subplots
        axes[0, 0].set_title('Training Loss', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Validation Loss', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('Training Performance', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title('Validation Performance', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Training Dynamics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / f"training_dynamics.{self.config.save_format}"
        
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def create_statistical_results_figure(
        self,
        statistical_results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> str:
        """Create statistical analysis results visualization.
        
        Args:
            statistical_results: Results from statistical validation
            save_path: Path to save figure
            
        Returns:
            Path to saved figure
        """
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. P-values heatmap
        if 'pairwise_comparisons' in statistical_results:
            comparisons = statistical_results['pairwise_comparisons']
            n_models = len(set([c['experiment_pair'][0] for c in comparisons] + 
                             [c['experiment_pair'][1] for c in comparisons]))
            
            p_matrix = np.ones((n_models, n_models))
            model_names = sorted(set([c['experiment_pair'][0] for c in comparisons] + 
                                   [c['experiment_pair'][1] for c in comparisons]))
            
            for comp in comparisons:
                i = model_names.index(comp['experiment_pair'][0])
                j = model_names.index(comp['experiment_pair'][1])
                p_matrix[i, j] = comp['p_value']
                p_matrix[j, i] = comp['p_value']
            
            sns.heatmap(
                p_matrix,
                annot=True,
                fmt='.3f',
                xticklabels=model_names,
                yticklabels=model_names,
                cmap='RdYlGn_r',
                center=0.05,
                ax=axes[0, 0]
            )
            axes[0, 0].set_title('Pairwise P-values', fontweight='bold')
        
        # 2. Effect sizes
        if 'effect_sizes' in statistical_results:
            effect_sizes = statistical_results['effect_sizes']
            effect_names = list(effect_sizes.keys())
            effect_values = list(effect_sizes.values())
            
            bars = axes[0, 1].bar(effect_names, effect_values, color='skyblue', edgecolor='navy')
            axes[0, 1].set_title('Effect Sizes', fontweight='bold')
            axes[0, 1].set_ylabel('Effect Size')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Add effect size interpretation lines
            axes[0, 1].axhline(y=0.2, color='green', linestyle='--', alpha=0.7, label='Small')
            axes[0, 1].axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Medium')
            axes[0, 1].axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Large')
            axes[0, 1].legend()
        
        # 3. Power analysis
        if 'power_analysis' in statistical_results:
            power_data = statistical_results['power_analysis']
            power_tests = [k for k in power_data.keys() if 'power' in k]
            power_values = [power_data[test] for test in power_tests if isinstance(power_data[test], (int, float))]
            
            if power_values:
                axes[0, 2].bar(range(len(power_values)), power_values, color='lightcoral', edgecolor='darkred')
                axes[0, 2].set_title('Statistical Power', fontweight='bold')
                axes[0, 2].set_ylabel('Power')
                axes[0, 2].axhline(y=0.8, color='blue', linestyle='--', label='Adequate Power')
                axes[0, 2].set_xticks(range(len(power_tests)))
                axes[0, 2].set_xticklabels([t.replace('_power', '') for t in power_tests], rotation=45)
                axes[0, 2].legend()
        
        # 4. Distribution comparison
        if 'descriptive_stats' in statistical_results:
            desc_stats = statistical_results['descriptive_stats']
            groups = list(desc_stats.keys())
            means = [desc_stats[group]['mean'] for group in groups]
            stds = [desc_stats[group]['std'] for group in groups]
            
            x_pos = np.arange(len(groups))
            axes[1, 0].bar(x_pos, means, yerr=stds, capsize=5, color='lightgreen', 
                          edgecolor='darkgreen', alpha=0.7)
            axes[1, 0].set_title('Group Means ± SD', fontweight='bold')
            axes[1, 0].set_xticks(x_pos)
            axes[1, 0].set_xticklabels(groups, rotation=45)
            axes[1, 0].set_ylabel('Mean Value')
        
        # 5. Confidence intervals
        if 'descriptive_stats' in statistical_results:
            desc_stats = statistical_results['descriptive_stats']
            groups = list(desc_stats.keys())
            
            for i, group in enumerate(groups):
                stats = desc_stats[group]
                ci_lower = stats.get('ci_95_lower', stats['mean'] - 1.96 * stats['sem'])
                ci_upper = stats.get('ci_95_upper', stats['mean'] + 1.96 * stats['sem'])
                
                axes[1, 1].errorbar(i, stats['mean'], 
                                   yerr=[[stats['mean'] - ci_lower], [ci_upper - stats['mean']]], 
                                   fmt='o', capsize=5, capthick=2, markersize=8)
            
            axes[1, 1].set_title('95% Confidence Intervals', fontweight='bold')
            axes[1, 1].set_xticks(range(len(groups)))
            axes[1, 1].set_xticklabels(groups, rotation=45)
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Assumption checks
        if 'assumption_checks' in statistical_results:
            # This would be implemented based on the structure of assumption_checks
            axes[1, 2].text(0.5, 0.5, 'Assumption\nChecks\n(Implementation\nPending)', 
                           ha='center', va='center', transform=axes[1, 2].transAxes,
                           fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
            axes[1, 2].set_title('Assumption Validation', fontweight='bold')
            axes[1, 2].set_xticks([])
            axes[1, 2].set_yticks([])
        
        plt.suptitle('Statistical Analysis Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / f"statistical_results.{self.config.save_format}"
        
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def create_embeddings_visualization(
        self,
        embeddings: np.ndarray,
        labels: Optional[np.ndarray] = None,
        method: str = "umap",
        save_path: Optional[str] = None
    ) -> str:
        """Create embeddings visualization using dimensionality reduction.
        
        Args:
            embeddings: High-dimensional embeddings
            labels: Optional labels for coloring points
            method: Dimensionality reduction method ('umap', 'tsne', 'pca')
            save_path: Path to save figure
            
        Returns:
            Path to saved figure
        """
        
        # Dimensionality reduction
        if method.lower() == 'umap':
            try:
                reducer = UMAP(n_components=2, random_state=42)
                embedding_2d = reducer.fit_transform(embeddings)
                title_suffix = "UMAP"
            except ImportError:
                method = 'pca'
        
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
            embedding_2d = reducer.fit_transform(embeddings)
            title_suffix = "t-SNE"
        
        if method.lower() == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            embedding_2d = reducer.fit_transform(embeddings)
            title_suffix = "PCA"
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 2D scatter plot
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                axes[0].scatter(embedding_2d[mask, 0], embedding_2d[mask, 1], 
                              c=[colors[i]], label=f'Class {label}', alpha=0.7, s=50)
            
            axes[0].legend()
        else:
            axes[0].scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.7, s=50)
        
        axes[0].set_title(f'Embeddings Visualization ({title_suffix})', fontweight='bold')
        axes[0].set_xlabel(f'{title_suffix} 1')
        axes[0].set_ylabel(f'{title_suffix} 2')
        axes[0].grid(True, alpha=0.3)
        
        # Density plot
        if len(embedding_2d) > 10:
            try:
                kde = gaussian_kde(embedding_2d.T)
                x_min, x_max = embedding_2d[:, 0].min(), embedding_2d[:, 0].max()
                y_min, y_max = embedding_2d[:, 1].min(), embedding_2d[:, 1].max()
                
                xx, yy = np.mgrid[x_min:x_max:.1*(x_max-x_min), 
                                 y_min:y_max:.1*(y_max-y_min)]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                density = np.reshape(kde(positions).T, xx.shape)
                
                axes[1].contour(xx, yy, density, colors='black', alpha=0.5)
                axes[1].contourf(xx, yy, density, alpha=0.7, cmap='viridis')
                axes[1].scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                              c='white', s=20, alpha=0.8, edgecolors='black')
                
                axes[1].set_title(f'Density Plot ({title_suffix})', fontweight='bold')
                axes[1].set_xlabel(f'{title_suffix} 1')
                axes[1].set_ylabel(f'{title_suffix} 2')
            except:
                axes[1].text(0.5, 0.5, 'Density plot\nunavailable', 
                           ha='center', va='center', transform=axes[1].transAxes)
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / f"embeddings_{method.lower()}.{self.config.save_format}"
        
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def create_attention_visualization(
        self,
        attention_weights: np.ndarray,
        node_labels: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> str:
        """Create attention weights visualization.
        
        Args:
            attention_weights: Attention weights matrix [nodes, nodes]
            node_labels: Optional labels for nodes
            save_path: Path to save figure
            
        Returns:
            Path to saved figure
        """
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # 1. Attention heatmap
        sns.heatmap(
            attention_weights,
            cmap='viridis',
            center=0,
            square=True,
            ax=axes[0]
        )
        axes[0].set_title('Attention Weights Heatmap', fontweight='bold')
        
        if node_labels and len(node_labels) == attention_weights.shape[0]:
            axes[0].set_xticklabels(node_labels, rotation=45)
            axes[0].set_yticklabels(node_labels, rotation=0)
        
        # 2. Attention distribution
        flat_attention = attention_weights.flatten()
        axes[1].hist(flat_attention, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1].set_title('Attention Weights Distribution', fontweight='bold')
        axes[1].set_xlabel('Attention Weight')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)
        
        # Add statistics
        mean_attention = np.mean(flat_attention)
        std_attention = np.std(flat_attention)
        axes[1].axvline(mean_attention, color='red', linestyle='--', label=f'Mean: {mean_attention:.3f}')
        axes[1].axvline(mean_attention + std_attention, color='orange', linestyle='--', alpha=0.7)
        axes[1].axvline(mean_attention - std_attention, color='orange', linestyle='--', alpha=0.7)
        axes[1].legend()
        
        # 3. Top attention connections
        # Find top k% connections
        k = 5  # top 5%
        threshold = np.percentile(flat_attention, 100 - k)
        
        # Create network graph
        G = nx.Graph()
        n_nodes = attention_weights.shape[0]
        
        # Add nodes
        for i in range(n_nodes):
            G.add_node(i)
        
        # Add edges for top connections
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if attention_weights[i, j] >= threshold:
                    G.add_edge(i, j, weight=attention_weights[i, j])
        
        # Draw network
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, 
            node_color='lightblue',
            node_size=300,
            ax=axes[2]
        )
        
        # Draw edges with thickness proportional to attention
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        
        if weights:
            max_weight = max(weights)
            edge_widths = [5 * w / max_weight for w in weights]
            
            nx.draw_networkx_edges(
                G, pos,
                width=edge_widths,
                alpha=0.6,
                edge_color='gray',
                ax=axes[2]
            )
        
        # Draw labels
        if node_labels and len(node_labels) == n_nodes:
            labels_dict = {i: label for i, label in enumerate(node_labels)}
            nx.draw_networkx_labels(G, pos, labels_dict, font_size=8, ax=axes[2])
        
        axes[2].set_title(f'Top {k}% Attention Connections', fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / f"attention_visualization.{self.config.save_format}"
        
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def create_brain_connectivity_figure(
        self,
        connectivity_matrix: np.ndarray,
        coordinates: Optional[np.ndarray] = None,
        node_labels: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> str:
        """Create brain connectivity visualization.
        
        Args:
            connectivity_matrix: Connectivity matrix [regions, regions]
            coordinates: 3D coordinates of brain regions [regions, 3]
            node_labels: Labels for brain regions
            save_path: Path to save figure
            
        Returns:
            Path to saved figure
        """
        
        if not NILEARN_AVAILABLE:
            # Fallback to basic network visualization
            return self._create_network_connectivity_figure(
                connectivity_matrix, node_labels, save_path
            )
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Connectivity matrix
        sns.heatmap(
            connectivity_matrix,
            cmap='RdBu_r',
            center=0,
            square=True,
            ax=axes[0, 0]
        )
        axes[0, 0].set_title('Connectivity Matrix', fontweight='bold')
        
        # 2. Connectivity distribution
        flat_conn = connectivity_matrix[np.triu_indices_from(connectivity_matrix, k=1)]
        axes[0, 1].hist(flat_conn, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].set_title('Connectivity Strength Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('Connectivity Strength')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Degree distribution
        degrees = np.sum(np.abs(connectivity_matrix), axis=1)
        axes[1, 0].bar(range(len(degrees)), degrees, color='orange', alpha=0.7)
        axes[1, 0].set_title('Node Degree Distribution', fontweight='bold')
        axes[1, 0].set_xlabel('Node Index')
        axes[1, 0].set_ylabel('Degree')
        
        if node_labels and len(node_labels) <= 20:  # Only show labels if not too many
            axes[1, 0].set_xticks(range(len(node_labels)))
            axes[1, 0].set_xticklabels(node_labels, rotation=45)
        
        # 4. Network properties
        # Convert to NetworkX graph for analysis
        G = nx.from_numpy_array(np.abs(connectivity_matrix))
        
        network_props = {
            'Nodes': G.number_of_nodes(),
            'Edges': G.number_of_edges(),
            'Density': nx.density(G),
            'Clustering': nx.average_clustering(G),
            'Avg Path Length': nx.average_shortest_path_length(G) if nx.is_connected(G) else 'Disconnected'
        }
        
        # Display network properties as text
        props_text = '\n'.join([f'{k}: {v:.3f}' if isinstance(v, float) else f'{k}: {v}' 
                               for k, v in network_props.items()])
        
        axes[1, 1].text(0.1, 0.7, props_text, transform=axes[1, 1].transAxes,
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[1, 1].set_title('Network Properties', fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / f"brain_connectivity.{self.config.save_format}"
        
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def _create_network_connectivity_figure(
        self,
        connectivity_matrix: np.ndarray,
        node_labels: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> str:
        """Fallback network connectivity visualization."""
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Connectivity matrix
        sns.heatmap(
            connectivity_matrix,
            cmap='RdBu_r',
            center=0,
            square=True,
            ax=axes[0]
        )
        axes[0].set_title('Connectivity Matrix', fontweight='bold')
        
        # 2. Network graph
        G = nx.from_numpy_array(np.abs(connectivity_matrix))
        
        # Remove weak connections for clarity
        threshold = np.percentile(np.abs(connectivity_matrix), 90)
        edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) 
                          if d['weight'] < threshold]
        G.remove_edges_from(edges_to_remove)
        
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=300, ax=axes[1])
        nx.draw_networkx_edges(G, pos, alpha=0.5, ax=axes[1])
        
        if node_labels and len(node_labels) == len(G.nodes()):
            labels_dict = {i: label for i, label in enumerate(node_labels)}
            nx.draw_networkx_labels(G, pos, labels_dict, font_size=8, ax=axes[1])
        
        axes[1].set_title('Network Visualization (Top 10% Connections)', fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / f"network_connectivity.{self.config.save_format}"
        
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def create_interactive_dashboard(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> str:
        """Create interactive dashboard using Plotly.
        
        Args:
            results: Dictionary containing various results
            save_path: Path to save HTML dashboard
            
        Returns:
            Path to saved dashboard
        """
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Model Performance', 'Training Loss',
                'Attention Weights', 'Connectivity Matrix',
                'Embedding Visualization', 'Statistical Results'
            ),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "heatmap"}, {"type": "heatmap"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Add traces based on available data
        if 'model_performance' in results:
            perf_data = results['model_performance']
            models = list(perf_data.keys())
            scores = list(perf_data.values())
            
            fig.add_trace(
                go.Bar(x=models, y=scores, name='Performance'),
                row=1, col=1
            )
        
        if 'training_history' in results:
            history = results['training_history']
            epochs = list(range(len(history.get('train_loss', []))))
            
            if 'train_loss' in history:
                fig.add_trace(
                    go.Scatter(x=epochs, y=history['train_loss'], 
                             mode='lines', name='Train Loss'),
                    row=1, col=2
                )
            
            if 'val_loss' in history:
                fig.add_trace(
                    go.Scatter(x=epochs, y=history['val_loss'], 
                             mode='lines', name='Val Loss'),
                    row=1, col=2
                )
        
        if 'attention_weights' in results:
            attention = results['attention_weights']
            fig.add_trace(
                go.Heatmap(z=attention, colorscale='Viridis'),
                row=2, col=1
            )
        
        if 'connectivity_matrix' in results:
            connectivity = results['connectivity_matrix']
            fig.add_trace(
                go.Heatmap(z=connectivity, colorscale='RdBu'),
                row=2, col=2
            )
        
        if 'embeddings' in results:
            embeddings = results['embeddings']
            labels = results.get('labels', None)
            
            fig.add_trace(
                go.Scatter(
                    x=embeddings[:, 0], 
                    y=embeddings[:, 1],
                    mode='markers',
                    marker=dict(
                        color=labels if labels is not None else 'blue',
                        colorscale='Viridis'
                    ),
                    name='Embeddings'
                ),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            title_text="Interactive Research Dashboard",
            showlegend=True,
            height=1000
        )
        
        # Save dashboard
        if save_path is None:
            save_path = self.output_dir / "interactive_dashboard.html"
        
        fig.write_html(str(save_path))
        
        return str(save_path)
    
    def create_publication_figure_panel(
        self,
        panels: Dict[str, Dict[str, Any]],
        layout: Tuple[int, int] = (2, 3),
        save_path: Optional[str] = None
    ) -> str:
        """Create multi-panel figure for publication.
        
        Args:
            panels: Dictionary of panel_name -> panel_data
            layout: (rows, cols) for subplot layout
            save_path: Path to save figure
            
        Returns:
            Path to saved figure
        """
        
        rows, cols = layout
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        axes = axes.flatten() if rows * cols > 1 else [axes]
        
        panel_names = list(panels.keys())
        
        for i, (panel_name, panel_data) in enumerate(panels.items()):
            if i >= len(axes):
                break
            
            ax = axes[i]
            panel_type = panel_data.get('type', 'bar')
            
            if panel_type == 'bar':
                x_data = panel_data.get('x', [])
                y_data = panel_data.get('y', [])
                ax.bar(x_data, y_data, color=panel_data.get('color', 'blue'))
                
            elif panel_type == 'line':
                x_data = panel_data.get('x', [])
                y_data = panel_data.get('y', [])
                ax.plot(x_data, y_data, color=panel_data.get('color', 'blue'))
                
            elif panel_type == 'heatmap':
                data = panel_data.get('data', np.random.randn(10, 10))
                im = ax.imshow(data, cmap=panel_data.get('cmap', 'viridis'))
                plt.colorbar(im, ax=ax)
                
            elif panel_type == 'scatter':
                x_data = panel_data.get('x', [])
                y_data = panel_data.get('y', [])
                ax.scatter(x_data, y_data, color=panel_data.get('color', 'blue'))
            
            # Set labels and title
            ax.set_title(f"({chr(65+i)}) {panel_name}", fontweight='bold', fontsize=14)
            ax.set_xlabel(panel_data.get('xlabel', ''))
            ax.set_ylabel(panel_data.get('ylabel', ''))
            
            if panel_data.get('grid', True):
                ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(len(panel_names), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / f"publication_figure.{self.config.save_format}"
        
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return str(save_path)


class InterpretabilityAnalyzer:
    """Advanced interpretability analysis for connectome GNNs."""
    
    def __init__(self, model: BaseConnectomeModel, device: str = "auto"):
        """Initialize interpretability analyzer.
        
        Args:
            model: Trained connectome GNN model
            device: Device for computation
        """
        self.model = model
        self.device = get_device(device)
        self.model.to(self.device)
        self.model.eval()
    
    def analyze_feature_importance(
        self,
        data_sample,
        method: str = "integrated_gradients",
        baseline: Optional[torch.Tensor] = None
    ) -> Dict[str, np.ndarray]:
        """Analyze feature importance using gradient-based methods.
        
        Args:
            data_sample: Input data sample
            method: Attribution method to use
            baseline: Baseline for integrated gradients
            
        Returns:
            Feature importance scores
        """
        
        self.model.requires_grad_(True)
        data_sample = data_sample.to(self.device)
        
        if method == "gradients":
            return self._compute_gradients(data_sample)
        elif method == "integrated_gradients":
            return self._compute_integrated_gradients(data_sample, baseline)
        elif method == "saliency":
            return self._compute_saliency(data_sample)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _compute_gradients(self, data_sample) -> Dict[str, np.ndarray]:
        """Compute gradient-based attribution."""
        
        data_sample.x.requires_grad_(True)
        
        output = self.model(data_sample)
        
        # For classification, use max class; for regression, use output value
        if output.dim() > 1 and output.size(1) > 1:
            target_output = output.max(dim=1)[0]
        else:
            target_output = output.squeeze()
        
        if target_output.dim() == 0:
            target_output = target_output.unsqueeze(0)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=target_output.sum(),
            inputs=data_sample.x,
            create_graph=False,
            retain_graph=False
        )[0]
        
        return {
            'node_importance': gradients.detach().cpu().numpy(),
            'feature_importance': gradients.abs().mean(dim=0).detach().cpu().numpy()
        }
    
    def _compute_integrated_gradients(
        self, 
        data_sample, 
        baseline: Optional[torch.Tensor] = None,
        steps: int = 50
    ) -> Dict[str, np.ndarray]:
        """Compute integrated gradients attribution."""
        
        if baseline is None:
            baseline = torch.zeros_like(data_sample.x)
        
        # Generate path from baseline to input
        alphas = torch.linspace(0, 1, steps).to(self.device)
        
        gradients_list = []
        
        for alpha in alphas:
            # Interpolated input
            interpolated_x = baseline + alpha * (data_sample.x - baseline)
            
            # Create interpolated data sample
            interpolated_sample = data_sample.clone()
            interpolated_sample.x = interpolated_x
            interpolated_sample.x.requires_grad_(True)
            
            # Forward pass
            output = self.model(interpolated_sample)
            
            # Target output
            if output.dim() > 1 and output.size(1) > 1:
                target_output = output.max(dim=1)[0]
            else:
                target_output = output.squeeze()
            
            if target_output.dim() == 0:
                target_output = target_output.unsqueeze(0)
            
            # Compute gradients
            gradients = torch.autograd.grad(
                outputs=target_output.sum(),
                inputs=interpolated_sample.x,
                create_graph=False,
                retain_graph=False
            )[0]
            
            gradients_list.append(gradients)
        
        # Average gradients and multiply by input difference
        avg_gradients = torch.stack(gradients_list).mean(dim=0)
        integrated_gradients = avg_gradients * (data_sample.x - baseline)
        
        return {
            'node_importance': integrated_gradients.detach().cpu().numpy(),
            'feature_importance': integrated_gradients.abs().mean(dim=0).detach().cpu().numpy()
        }
    
    def _compute_saliency(self, data_sample) -> Dict[str, np.ndarray]:
        """Compute saliency maps."""
        
        data_sample.x.requires_grad_(True)
        
        output = self.model(data_sample)
        
        # For classification, use max class; for regression, use output value
        if output.dim() > 1 and output.size(1) > 1:
            target_output = output.max(dim=1)[0]
        else:
            target_output = output.squeeze()
        
        if target_output.dim() == 0:
            target_output = target_output.unsqueeze(0)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=target_output.sum(),
            inputs=data_sample.x,
            create_graph=False,
            retain_graph=False
        )[0]
        
        # Saliency is absolute value of gradients
        saliency = gradients.abs()
        
        return {
            'node_importance': saliency.detach().cpu().numpy(),
            'feature_importance': saliency.mean(dim=0).detach().cpu().numpy()
        }
    
    def extract_attention_weights(self, data_sample) -> Optional[np.ndarray]:
        """Extract attention weights if model supports it."""
        
        if not hasattr(self.model, 'get_attention_weights'):
            return None
        
        data_sample = data_sample.to(self.device)
        
        with torch.no_grad():
            _ = self.model(data_sample)
            attention_weights = self.model.get_attention_weights()
        
        if attention_weights is not None:
            return attention_weights.cpu().numpy()
        
        return None
    
    def extract_node_embeddings(self, data_sample) -> Optional[np.ndarray]:
        """Extract node embeddings if model supports it."""
        
        if not hasattr(self.model, 'get_embeddings'):
            return None
        
        data_sample = data_sample.to(self.device)
        
        with torch.no_grad():
            _ = self.model(data_sample)
            embeddings = self.model.get_embeddings()
        
        if embeddings is not None:
            return embeddings.cpu().numpy()
        
        return None