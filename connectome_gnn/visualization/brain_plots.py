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
        
        for network in networks:\n            center = np.array(network_centers[network])\n            \n            # Generate coordinates around network center\n            for hemisphere in [-1, 1]:  # Left and right hemispheres\n                n_hem_regions = regions_per_network // 2\n                \n                # Add hemisphere offset\n                hem_center = center.copy()\n                hem_center[0] = hemisphere * 40  # x-offset for hemisphere\n                \n                # Generate scattered coordinates around center\n                region_coords = np.random.multivariate_normal(\n                    hem_center,\n                    np.diag([15, 15, 10]),  # Covariance matrix\n                    n_hem_regions\n                )\n                \n                coords_list.append(region_coords)\n        \n        coords = np.vstack(coords_list)\n        \n        # Trim to exact number of regions\n        return coords[:n_regions]\n    \n    def _load_region_names(self) -> List[str]:\n        \"\"\"Load region names for the atlas.\"\"\"\n        if self.atlas == \"AAL\":\n            # Simplified AAL region names\n            regions = [\n                f\"Left_Region_{i+1}\" for i in range(45)\n            ] + [\n                f\"Right_Region_{i+1}\" for i in range(45)\n            ]\n        elif self.atlas == \"Schaefer400\":\n            # Simplified Schaefer region names\n            networks = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', \n                       'Limbic', 'Cont', 'Default']\n            regions = []\n            for net in networks:\n                for hem in ['LH', 'RH']:\n                    for i in range(400 // (len(networks) * 2)):\n                        regions.append(f\"{hem}_{net}_{i+1}\")\n            regions = regions[:400]\n        else:\n            # Generic region names\n            n_regions = len(self.brain_coordinates)\n            regions = [f\"Region_{i+1}\" for i in range(n_regions)]\n        \n        return regions\n    \n    def plot_connectome(\n        self,\n        connectivity_matrix: np.ndarray,\n        node_colors: Optional[np.ndarray] = None,\n        edge_threshold: float = 0.8,\n        node_size: Union[float, np.ndarray] = 8,\n        layout: str = \"3d\",\n        colorscale: str = \"viridis\",\n        title: str = \"Brain Connectome\",\n        save_path: Optional[Path] = None\n    ) -> go.Figure:\n        \"\"\"Plot 3D brain connectome.\n        \n        Args:\n            connectivity_matrix: Connectivity matrix (n_regions x n_regions)\n            node_colors: Colors for each node/region\n            edge_threshold: Percentile threshold for showing edges\n            node_size: Size of nodes (scalar or array)\n            layout: Layout type ('3d', '2d')\n            colorscale: Color scheme\n            title: Plot title\n            save_path: Path to save plot\n            \n        Returns:\n            Plotly figure object\n        \"\"\"\n        n_regions = connectivity_matrix.shape[0]\n        coords = self.brain_coordinates[:n_regions]\n        \n        # Determine edges to show\n        threshold_value = np.percentile(connectivity_matrix[connectivity_matrix > 0], \n                                       edge_threshold * 100)\n        \n        # Create edge traces\n        edge_traces = []\n        for i in range(n_regions):\n            for j in range(i + 1, n_regions):\n                if connectivity_matrix[i, j] > threshold_value:\n                    edge_trace = self._create_edge_trace(\n                        coords[i], coords[j], \n                        connectivity_matrix[i, j],\n                        colorscale\n                    )\n                    edge_traces.append(edge_trace)\n        \n        # Create node trace\n        if node_colors is None:\n            node_colors = np.arange(n_regions)\n        \n        if isinstance(node_size, (int, float)):\n            sizes = [node_size] * n_regions\n        else:\n            sizes = node_size\n        \n        node_trace = go.Scatter3d(\n            x=coords[:, 0],\n            y=coords[:, 1],\n            z=coords[:, 2],\n            mode='markers',\n            marker=dict(\n                size=sizes,\n                color=node_colors,\n                colorscale=colorscale,\n                showscale=True,\n                colorbar=dict(title=\"Node Values\")\n            ),\n            text=self.region_names[:n_regions],\n            hovertemplate=\n                \"<b>%{text}</b><br>\" +\n                \"X: %{x}<br>\" +\n                \"Y: %{y}<br>\" +\n                \"Z: %{z}<br>\" +\n                \"Value: %{marker.color:.3f}<br>\" +\n                \"<extra></extra>\",\n            name=\"Brain Regions\"\n        )\n        \n        # Create figure\n        fig = go.Figure(data=[node_trace] + edge_traces)\n        \n        # Update layout\n        fig.update_layout(\n            title=title,\n            scene=dict(\n                xaxis_title=\"X (mm)\",\n                yaxis_title=\"Y (mm)\",\n                zaxis_title=\"Z (mm)\",\n                bgcolor=\"rgba(0,0,0,0)\",\n                camera=dict(\n                    eye=dict(x=1.2, y=1.2, z=0.6)\n                )\n            ),\n            showlegend=False,\n            height=800,\n            width=1000\n        )\n        \n        # Save if requested\n        if save_path:\n            fig.write_html(str(save_path))\n            print(f\"Brain plot saved to {save_path}\")\n        \n        return fig\n    \n    def _create_edge_trace(\n        self, \n        coord1: np.ndarray, \n        coord2: np.ndarray,\n        weight: float,\n        colorscale: str\n    ) -> go.Scatter3d:\n        \"\"\"Create edge trace for connection.\"\"\"\n        return go.Scatter3d(\n            x=[coord1[0], coord2[0], None],\n            y=[coord1[1], coord2[1], None],\n            z=[coord1[2], coord2[2], None],\n            mode='lines',\n            line=dict(\n                color=weight,\n                width=max(1, weight * 5),\n                colorscale=colorscale,\n            ),\n            showlegend=False,\n            hoverinfo='none'\n        )\n    \n    def plot_brain_surface(\n        self,\n        values: np.ndarray,\n        surface_type: str = \"inflated\",\n        hemisphere: str = \"both\",\n        colorscale: str = \"viridis\",\n        title: str = \"Brain Surface\",\n        save_path: Optional[Path] = None\n    ) -> go.Figure:\n        \"\"\"Plot brain surface with values.\"\"\"\n        if not NILEARN_AVAILABLE:\n            raise ImportError(\"nilearn is required for surface plotting\")\n        \n        # This is a simplified implementation\n        # In practice, you would use nilearn's surface plotting capabilities\n        \n        # Generate mock surface data\n        n_vertices = 10000 if hemisphere == \"both\" else 5000\n        \n        # Generate surface coordinates (simplified)\n        theta = np.random.uniform(0, 2*np.pi, n_vertices)\n        phi = np.random.uniform(0, np.pi, n_vertices)\n        r = 50 + np.random.normal(0, 2, n_vertices)\n        \n        x = r * np.sin(phi) * np.cos(theta)\n        y = r * np.sin(phi) * np.sin(theta)  \n        z = r * np.cos(phi)\n        \n        if hemisphere == \"left\":\n            x = x[x < 0]\n            y = y[:len(x)]\n            z = z[:len(x)]\n        elif hemisphere == \"right\":\n            x = x[x > 0]\n            y = y[:len(x)]\n            z = z[:len(x)]\n        \n        # Map values to surface\n        if len(values) != len(x):\n            surface_values = np.random.uniform(values.min(), values.max(), len(x))\n        else:\n            surface_values = values\n        \n        # Create surface trace\n        trace = go.Scatter3d(\n            x=x, y=y, z=z,\n            mode='markers',\n            marker=dict(\n                size=2,\n                color=surface_values,\n                colorscale=colorscale,\n                showscale=True\n            ),\n            name=\"Brain Surface\"\n        )\n        \n        fig = go.Figure(data=[trace])\n        fig.update_layout(\n            title=title,\n            scene=dict(\n                xaxis_title=\"X\", yaxis_title=\"Y\", zaxis_title=\"Z\",\n                bgcolor=\"rgba(0,0,0,0)\"\n            ),\n            height=800, width=1000\n        )\n        \n        if save_path:\n            fig.write_html(str(save_path))\n        \n        return fig\n\n\nclass GlassBrainPlot:\n    \"\"\"Glass brain plotting for publication-ready figures.\"\"\"\n    \n    def __init__(self, template: str = \"MNI152\"):\n        self.template = template\n    \n    def glass_brain(\n        self,\n        connectivity_matrix: np.ndarray,\n        coordinates: np.ndarray,\n        highlight_regions: Optional[List[int]] = None,\n        views: List[str] = [\"sagittal\", \"coronal\", \"axial\"],\n        threshold: float = 0.8,\n        node_size: float = 50,\n        edge_kwargs: Optional[Dict] = None,\n        save_path: Optional[Path] = None,\n        dpi: int = 300\n    ) -> plt.Figure:\n        \"\"\"Create glass brain plot.\n        \n        Args:\n            connectivity_matrix: Connectivity matrix\n            coordinates: 3D coordinates of regions\n            highlight_regions: Regions to highlight\n            views: Brain views to show\n            threshold: Connection threshold\n            node_size: Size of nodes\n            edge_kwargs: Edge plotting parameters\n            save_path: Path to save figure\n            dpi: Resolution for saved figure\n            \n        Returns:\n            Matplotlib figure\n        \"\"\"\n        if NILEARN_AVAILABLE:\n            try:\n                return self._nilearn_glass_brain(\n                    connectivity_matrix, coordinates, highlight_regions,\n                    views, threshold, node_size, edge_kwargs, save_path, dpi\n                )\n            except Exception as e:\n                print(f\"Nilearn glass brain failed: {e}\")\n                print(\"Falling back to matplotlib implementation\")\n        \n        return self._matplotlib_glass_brain(\n            connectivity_matrix, coordinates, highlight_regions,\n            views, threshold, node_size, edge_kwargs, save_path, dpi\n        )\n    \n    def _nilearn_glass_brain(\n        self, connectivity_matrix, coordinates, highlight_regions,\n        views, threshold, node_size, edge_kwargs, save_path, dpi\n    ):\n        \"\"\"Use nilearn for glass brain plotting.\"\"\"\n        from nilearn import plotting\n        \n        # Threshold connectivity\n        thresh_val = np.percentile(connectivity_matrix[connectivity_matrix > 0], \n                                  threshold * 100)\n        \n        # Create adjacency matrix\n        adj_matrix = connectivity_matrix.copy()\n        adj_matrix[adj_matrix < thresh_val] = 0\n        \n        # Plot glass brain\n        fig = plotting.plot_connectome(\n            adj_matrix,\n            coordinates,\n            edge_threshold=thresh_val,\n            node_size=node_size,\n            display_mode='ortho' if len(views) > 1 else views[0][0],\n            title=\"Glass Brain Connectome\"\n        )\n        \n        if save_path:\n            fig.savefig(str(save_path), dpi=dpi, bbox_inches='tight')\n        \n        return fig\n    \n    def _matplotlib_glass_brain(\n        self, connectivity_matrix, coordinates, highlight_regions,\n        views, threshold, node_size, edge_kwargs, save_path, dpi\n    ):\n        \"\"\"Fallback matplotlib implementation.\"\"\"\n        n_views = len(views)\n        fig, axes = plt.subplots(1, n_views, figsize=(5*n_views, 5))\n        \n        if n_views == 1:\n            axes = [axes]\n        \n        # Threshold connections\n        thresh_val = np.percentile(connectivity_matrix[connectivity_matrix > 0], \n                                  threshold * 100)\n        \n        for idx, view in enumerate(views):\n            ax = axes[idx]\n            \n            if view == \"sagittal\":\n                x, y = coordinates[:, 1], coordinates[:, 2]  # Y-Z plane\n                ax.set_xlabel('Y (mm)')\n                ax.set_ylabel('Z (mm)')\n            elif view == \"coronal\":\n                x, y = coordinates[:, 0], coordinates[:, 2]  # X-Z plane\n                ax.set_xlabel('X (mm)')\n                ax.set_ylabel('Z (mm)')\n            elif view == \"axial\":\n                x, y = coordinates[:, 0], coordinates[:, 1]  # X-Y plane\n                ax.set_xlabel('X (mm)')\n                ax.set_ylabel('Y (mm)')\n            \n            # Plot connections\n            for i in range(len(coordinates)):\n                for j in range(i + 1, len(coordinates)):\n                    if connectivity_matrix[i, j] > thresh_val:\n                        if view == \"sagittal\":\n                            x_coords = [coordinates[i, 1], coordinates[j, 1]]\n                            y_coords = [coordinates[i, 2], coordinates[j, 2]]\n                        elif view == \"coronal\":\n                            x_coords = [coordinates[i, 0], coordinates[j, 0]]\n                            y_coords = [coordinates[i, 2], coordinates[j, 2]]\n                        else:  # axial\n                            x_coords = [coordinates[i, 0], coordinates[j, 0]]\n                            y_coords = [coordinates[i, 1], coordinates[j, 1]]\n                        \n                        alpha = connectivity_matrix[i, j] / connectivity_matrix.max()\n                        ax.plot(x_coords, y_coords, 'b-', alpha=alpha, linewidth=0.5)\n            \n            # Plot nodes\n            node_colors = ['red' if highlight_regions and i in highlight_regions \n                          else 'blue' for i in range(len(coordinates))]\n            \n            scatter = ax.scatter(x, y, c=node_colors, s=node_size, alpha=0.7)\n            \n            ax.set_title(f\"{view.capitalize()} View\")\n            ax.grid(True, alpha=0.3)\n            ax.set_aspect('equal')\n        \n        plt.suptitle(\"Glass Brain Visualization\")\n        plt.tight_layout()\n        \n        if save_path:\n            plt.savefig(str(save_path), dpi=dpi, bbox_inches='tight')\n        \n        return fig