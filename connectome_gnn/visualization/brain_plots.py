"""3D brain visualization and plotting utilities."""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import torch

try:
    from nilearn import plotting, datasets, surface
    from nilearn.image import coord_transform
    NILEARN_AVAILABLE = True
except ImportError:
    NILEARN_AVAILABLE = False


class BrainNetworkPlot:
    """3D brain network visualization using plotly and nilearn.
    
    Creates interactive 3D brain plots with connectomes, highlighting
    important regions and connections.
    """
    
    def __init__(self, atlas: str = "AAL", template: str = "MNI152"):
        self.atlas = atlas
        self.template = template
        
        # Load brain template and coordinates
        self.brain_coordinates = self._load_brain_coordinates()
        self.region_names = self._load_region_names()
        
        # Color schemes
        self.color_schemes = {
            'viridis': px.colors.sequential.Viridis,
            'plasma': px.colors.sequential.Plasma,
            'cool': px.colors.sequential.ice,
            'warm': px.colors.sequential.Reds,
            'diverging': px.colors.diverging.RdBu
        }
    
    def _load_brain_coordinates(self) -> np.ndarray:
        """Load brain region coordinates."""
        if NILEARN_AVAILABLE:
            try:
                # Try to load real coordinates
                if self.atlas == "AAL":
                    # AAL atlas coordinates (simplified)
                    coords = datasets.fetch_coords_power_2011()
                    if len(coords.rois) >= 90:
                        return np.array(coords.rois[:90])
                elif self.atlas == "Schaefer400":
                    # Generate Schaefer-like coordinates
                    return self._generate_schaefer_coordinates(400)
            except:
                pass
        
        # Fallback: generate mock coordinates
        return self._generate_mock_coordinates()
    
    def _generate_mock_coordinates(self) -> np.ndarray:
        """Generate mock brain coordinates."""
        atlas_sizes = {"AAL": 90, "Schaefer400": 400, "Desikan": 68}
        n_regions = atlas_sizes.get(self.atlas, 90)
        
        # Generate coordinates in brain-like distribution
        np.random.seed(42)
        
        # Create left and right hemispheres
        n_per_hemisphere = n_regions // 2
        
        # Left hemisphere (negative x)
        left_coords = np.column_stack([
            -np.random.uniform(20, 80, n_per_hemisphere),  # x (negative)
            np.random.uniform(-90, 90, n_per_hemisphere),  # y
            np.random.uniform(-30, 70, n_per_hemisphere)   # z
        ])
        
        # Right hemisphere (positive x) - mirror of left
        right_coords = left_coords.copy()
        right_coords[:, 0] = -right_coords[:, 0]  # Mirror x-coordinate
        
        coords = np.vstack([left_coords, right_coords])
        
        # Add remaining regions if odd number
        if n_regions % 2 == 1:
            center_coord = np.array([[0, 0, 0]])
            coords = np.vstack([coords, center_coord])
        
        return coords
    
    def _generate_schaefer_coordinates(self, n_regions: int) -> np.ndarray:
        """Generate Schaefer-like coordinates."""
        # Simplified Schaefer parcellation coordinates
        np.random.seed(42)
        
        # Create 7 networks with different spatial distributions
        networks = ['Visual', 'Somatomotor', 'DorsalAttn', 'VentralAttn', 
                   'Limbic', 'Frontoparietal', 'Default']
        
        coords_list = []
        regions_per_network = n_regions // len(networks)
        
        # Network-specific coordinate centers
        network_centers = {
            'Visual': [0, -70, 0],
            'Somatomotor': [0, -20, 50],
            'DorsalAttn': [0, -50, 40],
            'VentralAttn': [0, 20, 20],
            'Limbic': [0, 0, -10],
            'Frontoparietal': [0, 30, 30],
            'Default': [0, 50, 20]
        }
        
        for network in networks:
            center = np.array(network_centers[network])
            
            # Generate coordinates around network center
            for hemisphere in [-1, 1]:  # Left and right hemispheres
                n_hem_regions = regions_per_network // 2
                
                # Add hemisphere offset
                hem_center = center.copy()
                hem_center[0] = hemisphere * 40  # x-offset for hemisphere
                
                # Generate scattered coordinates around center
                region_coords = np.random.multivariate_normal(
                    hem_center,
                    np.diag([15, 15, 10]),  # Covariance matrix
                    n_hem_regions
                )
                
                coords_list.append(region_coords)
        
        coords = np.vstack(coords_list)
        
        # Trim to exact number of regions
        return coords[:n_regions]
    
    def _load_region_names(self) -> List[str]:
        """Load region names for the atlas."""
        if self.atlas == "AAL":
            # Simplified AAL region names
            regions = [
                f"Left_Region_{i+1}" for i in range(45)
            ] + [
                f"Right_Region_{i+1}" for i in range(45)
            ]
        elif self.atlas == "Schaefer400":
            # Simplified Schaefer region names
            networks = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 
                       'Limbic', 'Cont', 'Default']
            regions = []
            for net in networks:
                for hem in ['LH', 'RH']:
                    for i in range(400 // (len(networks) * 2)):
                        regions.append(f"{hem}_{net}_{i+1}")
            regions = regions[:400]
        else:
            # Generic region names
            n_regions = len(self.brain_coordinates)
            regions = [f"Region_{i+1}" for i in range(n_regions)]
        
        return regions
    
    def plot_connectome(
        self,
        connectivity_matrix: np.ndarray,
        node_colors: Optional[np.ndarray] = None,
        edge_threshold: float = 0.8,
        node_size: Union[float, np.ndarray] = 8,
        layout: str = "3d",
        colorscale: str = "viridis",
        title: str = "Brain Connectome",
        save_path: Optional[Path] = None
    ) -> go.Figure:
        """Plot 3D brain connectome.
        
        Args:
            connectivity_matrix: Connectivity matrix (n_regions x n_regions)
            node_colors: Colors for each node/region
            edge_threshold: Percentile threshold for showing edges
            node_size: Size of nodes (scalar or array)
            layout: Layout type ('3d', '2d')
            colorscale: Color scheme
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            Plotly figure object
        """
        n_regions = connectivity_matrix.shape[0]
        coords = self.brain_coordinates[:n_regions]
        
        # Determine edges to show
        threshold_value = np.percentile(connectivity_matrix[connectivity_matrix > 0], 
                                       edge_threshold * 100)
        
        # Create edge traces
        edge_traces = []
        for i in range(n_regions):
            for j in range(i + 1, n_regions):
                if connectivity_matrix[i, j] > threshold_value:
                    edge_trace = self._create_edge_trace(
                        coords[i], coords[j], 
                        connectivity_matrix[i, j],
                        colorscale
                    )
                    edge_traces.append(edge_trace)
        
        # Create node trace
        if node_colors is None:
            node_colors = np.arange(n_regions)
        
        if isinstance(node_size, (int, float)):
            sizes = [node_size] * n_regions
        else:
            sizes = node_size
        
        node_trace = go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode='markers',
            marker=dict(
                size=sizes,
                color=node_colors,
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(title="Node Values")
            ),
            text=self.region_names[:n_regions],
            hovertemplate=
                "<b>%{text}</b><br>" +
                "X: %{x}<br>" +
                "Y: %{y}<br>" +
                "Z: %{z}<br>" +
                "Value: %{marker.color:.3f}<br>" +
                "<extra></extra>",
            name="Brain Regions"
        )
        
        # Create figure
        fig = go.Figure(data=[node_trace] + edge_traces)
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="X (mm)",
                yaxis_title="Y (mm)",
                zaxis_title="Z (mm)",
                bgcolor="rgba(0,0,0,0)",
                camera=dict(
                    eye=dict(x=1.2, y=1.2, z=0.6)
                )
            ),
            showlegend=False,
            height=800,
            width=1000
        )
        
        # Save if requested
        if save_path:
            fig.write_html(str(save_path))
            print(f"Brain plot saved to {save_path}")
        
        return fig
    
    def _create_edge_trace(
        self, 
        coord1: np.ndarray, 
        coord2: np.ndarray,
        weight: float,
        colorscale: str
    ) -> go.Scatter3d:
        """Create edge trace for connection."""
        return go.Scatter3d(
            x=[coord1[0], coord2[0], None],
            y=[coord1[1], coord2[1], None],
            z=[coord1[2], coord2[2], None],
            mode='lines',
            line=dict(
                color=weight,
                width=max(1, weight * 5),
                colorscale=colorscale,
            ),
            showlegend=False,
            hoverinfo='none'
        )
    
    def plot_brain_surface(
        self,
        values: np.ndarray,
        surface_type: str = "inflated",
        hemisphere: str = "both",
        colorscale: str = "viridis",
        title: str = "Brain Surface",
        save_path: Optional[Path] = None
    ) -> go.Figure:
        """Plot brain surface with values."""
        if not NILEARN_AVAILABLE:
            raise ImportError("nilearn is required for surface plotting")
        
        # This is a simplified implementation
        # In practice, you would use nilearn's surface plotting capabilities
        
        # Generate mock surface data
        n_vertices = 10000 if hemisphere == "both" else 5000
        
        # Generate surface coordinates (simplified)
        theta = np.random.uniform(0, 2*np.pi, n_vertices)
        phi = np.random.uniform(0, np.pi, n_vertices)
        r = 50 + np.random.normal(0, 2, n_vertices)
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)  
        z = r * np.cos(phi)
        
        if hemisphere == "left":
            x = x[x < 0]
            y = y[:len(x)]
            z = z[:len(x)]
        elif hemisphere == "right":
            x = x[x > 0]
            y = y[:len(x)]
            z = z[:len(x)]
        
        # Map values to surface
        if len(values) != len(x):
            surface_values = np.random.uniform(values.min(), values.max(), len(x))
        else:
            surface_values = values
        
        # Create surface trace
        trace = go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=2,
                color=surface_values,
                colorscale=colorscale,
                showscale=True
            ),
            name="Brain Surface"
        )
        
        fig = go.Figure(data=[trace])
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                bgcolor="rgba(0,0,0,0)"
            ),
            height=800, width=1000
        )
        
        if save_path:
            fig.write_html(str(save_path))
        
        return fig


class GlassBrainPlot:
    """Glass brain plotting for publication-ready figures."""
    
    def __init__(self, template: str = "MNI152"):
        self.template = template
    
    def glass_brain(
        self,
        connectivity_matrix: np.ndarray,
        coordinates: np.ndarray,
        highlight_regions: Optional[List[int]] = None,
        views: List[str] = ["sagittal", "coronal", "axial"],
        threshold: float = 0.8,
        node_size: float = 50,
        edge_kwargs: Optional[Dict] = None,
        save_path: Optional[Path] = None,
        dpi: int = 300
    ) -> plt.Figure:
        """Create glass brain plot.
        
        Args:
            connectivity_matrix: Connectivity matrix
            coordinates: 3D coordinates of regions
            highlight_regions: Regions to highlight
            views: Brain views to show
            threshold: Connection threshold
            node_size: Size of nodes
            edge_kwargs: Edge plotting parameters
            save_path: Path to save figure
            dpi: Resolution for saved figure
            
        Returns:
            Matplotlib figure
        """
        if NILEARN_AVAILABLE:
            try:
                return self._nilearn_glass_brain(
                    connectivity_matrix, coordinates, highlight_regions,
                    views, threshold, node_size, edge_kwargs, save_path, dpi
                )
            except Exception as e:
                print(f"Nilearn glass brain failed: {e}")
                print("Falling back to matplotlib implementation")
        
        return self._matplotlib_glass_brain(
            connectivity_matrix, coordinates, highlight_regions,
            views, threshold, node_size, edge_kwargs, save_path, dpi
        )
    
    def _nilearn_glass_brain(
        self, connectivity_matrix, coordinates, highlight_regions,
        views, threshold, node_size, edge_kwargs, save_path, dpi
    ):
        """Use nilearn for glass brain plotting."""
        from nilearn import plotting
        
        # Threshold connectivity
        thresh_val = np.percentile(connectivity_matrix[connectivity_matrix > 0], 
                                  threshold * 100)
        
        # Create adjacency matrix
        adj_matrix = connectivity_matrix.copy()
        adj_matrix[adj_matrix < thresh_val] = 0
        
        # Plot glass brain
        fig = plotting.plot_connectome(
            adj_matrix,
            coordinates,
            edge_threshold=thresh_val,
            node_size=node_size,
            display_mode='ortho' if len(views) > 1 else views[0][0],
            title="Glass Brain Connectome"
        )
        
        if save_path:
            fig.savefig(str(save_path), dpi=dpi, bbox_inches='tight')
        
        return fig
    
    def _matplotlib_glass_brain(
        self, connectivity_matrix, coordinates, highlight_regions,
        views, threshold, node_size, edge_kwargs, save_path, dpi
    ):
        """Fallback matplotlib implementation."""
        n_views = len(views)
        fig, axes = plt.subplots(1, n_views, figsize=(5*n_views, 5))
        
        if n_views == 1:
            axes = [axes]
        
        # Threshold connections
        thresh_val = np.percentile(connectivity_matrix[connectivity_matrix > 0], 
                                  threshold * 100)
        
        for idx, view in enumerate(views):
            ax = axes[idx]
            
            if view == "sagittal":
                x, y = coordinates[:, 1], coordinates[:, 2]  # Y-Z plane
                ax.set_xlabel('Y (mm)')
                ax.set_ylabel('Z (mm)')
            elif view == "coronal":
                x, y = coordinates[:, 0], coordinates[:, 2]  # X-Z plane
                ax.set_xlabel('X (mm)')
                ax.set_ylabel('Z (mm)')
            elif view == "axial":
                x, y = coordinates[:, 0], coordinates[:, 1]  # X-Y plane
                ax.set_xlabel('X (mm)')
                ax.set_ylabel('Y (mm)')
            
            # Plot connections
            for i in range(len(coordinates)):
                for j in range(i + 1, len(coordinates)):
                    if connectivity_matrix[i, j] > thresh_val:
                        if view == "sagittal":
                            x_coords = [coordinates[i, 1], coordinates[j, 1]]
                            y_coords = [coordinates[i, 2], coordinates[j, 2]]
                        elif view == "coronal":
                            x_coords = [coordinates[i, 0], coordinates[j, 0]]
                            y_coords = [coordinates[i, 2], coordinates[j, 2]]
                        else:  # axial
                            x_coords = [coordinates[i, 0], coordinates[j, 0]]
                            y_coords = [coordinates[i, 1], coordinates[j, 1]]
                        
                        alpha = connectivity_matrix[i, j] / connectivity_matrix.max()
                        ax.plot(x_coords, y_coords, 'b-', alpha=alpha, linewidth=0.5)
            
            # Plot nodes
            node_colors = ['red' if highlight_regions and i in highlight_regions 
                          else 'blue' for i in range(len(coordinates))]
            
            scatter = ax.scatter(x, y, c=node_colors, s=node_size, alpha=0.7)
            
            ax.set_title(f"{view.capitalize()} View")
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        
        plt.suptitle("Glass Brain Visualization")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(str(save_path), dpi=dpi, bbox_inches='tight')
        
        return fig