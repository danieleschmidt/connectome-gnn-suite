#!/bin/bash

# Post-create script for Connectome-GNN-Suite development container

echo "🚀 Setting up Connectome-GNN-Suite development environment..."

# Install the package in development mode
echo "📦 Installing connectome-gnn-suite in development mode..."
pip install -e ".[dev,viz,full]"

# Install pre-commit hooks
echo "🔗 Setting up pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Create necessary directories
echo "📁 Creating development directories..."
mkdir -p data/hcp
mkdir -p data/cache
mkdir -p logs
mkdir -p outputs
mkdir -p notebooks/tutorials
mkdir -p notebooks/experiments

# Set up git configuration if not already set
if [ -z "$(git config --global user.name)" ]; then
    echo "⚙️  Setting up git configuration..."
    git config --global user.name "Developer"
    git config --global user.email "developer@connectome-gnn-suite.local"
fi

# Initialize Jupyter Lab configuration
echo "🪐 Configuring Jupyter Lab..."
jupyter lab --generate-config
cat >> ~/.jupyter/jupyter_lab_config.py << 'EOF'
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.allow_root = True
c.ServerApp.token = ''
c.ServerApp.password = ''
EOF

# Set up environment variables
echo "🌍 Setting up environment variables..."
cat >> ~/.bashrc << 'EOF'
# Connectome-GNN-Suite environment
export CONNECTOME_GNN_DATA_DIR="/workspace/data"
export CONNECTOME_GNN_CACHE_DIR="/workspace/data/cache"
export CONNECTOME_GNN_LOG_LEVEL="INFO"
export PYTHONPATH="/workspace:${PYTHONPATH}"

# CUDA environment (if available)
export CUDA_VISIBLE_DEVICES="0"
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"

# Jupyter environment
export JUPYTER_ENABLE_LAB=yes

# Development aliases
alias cgnn-test="python -m pytest tests/ -v"
alias cgnn-lint="flake8 connectome_gnn tests"
alias cgnn-format="black connectome_gnn tests && isort connectome_gnn tests"
alias cgnn-typecheck="mypy connectome_gnn"
alias cgnn-docs="cd docs && make html"
alias cgnn-clean="find . -type d -name __pycache__ -delete && find . -name '*.pyc' -delete"

# Quick development commands
alias lab="jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser"
alias nb="jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser"

echo "🧠 Connectome-GNN-Suite development environment ready!"
EOF

# Source the updated bashrc
source ~/.bashrc

# Create a welcome notebook
echo "📓 Creating welcome notebook..."
cat > notebooks/Welcome.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Welcome to Connectome-GNN-Suite Development!\n",
    "\n",
    "This notebook helps you get started with the development environment.\n",
    "\n",
    "## Quick Setup Check\n",
    "\n",
    "Run the cells below to verify your environment is working correctly."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Check imports\n",
    "import torch\n",
    "import torch_geometric\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import plotly.graph_objects as go\n",
    "\n",
    "print(f\"PyTorch version: {torch.__version__}\")\n",
    "print(f\"PyTorch Geometric version: {torch_geometric.__version__}\")\n",
    "print(f\"CUDA available: {torch.cuda.is_available()}\")\n",
    "if torch.cuda.is_available():\n",
    "    print(f\"CUDA device: {torch.cuda.get_device_name(0)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test basic functionality\n",
    "try:\n",
    "    import connectome_gnn\n",
    "    print(\"✅ connectome_gnn package imported successfully!\")\n",
    "except ImportError as e:\n",
    "    print(f\"❌ Import error: {e}\")\n",
    "    print(\"Run: pip install -e '.[dev,viz,full]' in the terminal\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create a simple test graph\n",
    "from torch_geometric.data import Data\n",
    "import torch\n",
    "\n",
    "# Simple brain-like graph (4 regions)\n",
    "edge_index = torch.tensor([\n",
    "    [0, 1, 1, 2, 2, 3, 3, 0],\n",
    "    [1, 0, 2, 1, 3, 2, 0, 3]\n",
    "], dtype=torch.long)\n",
    "\n",
    "x = torch.randn(4, 16)  # 4 nodes, 16 features each\n",
    "edge_attr = torch.randn(edge_index.size(1), 1)  # Edge weights\n",
    "\n",
    "data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)\n",
    "print(f\"Test graph: {data}\")\n",
    "print(f\"Nodes: {data.num_nodes}, Edges: {data.num_edges}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test visualization\n",
    "import networkx as nx\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Convert to NetworkX for visualization\n",
    "G = nx.Graph()\n",
    "edges = data.edge_index.t().numpy()\n",
    "weights = data.edge_attr.squeeze().numpy()\n",
    "\n",
    "for i, (u, v) in enumerate(edges):\n",
    "    G.add_edge(u, v, weight=abs(weights[i]))\n",
    "\n",
    "plt.figure(figsize=(8, 6))\n",
    "pos = nx.spring_layout(G)\n",
    "nx.draw(G, pos, with_labels=True, node_color='lightblue', \n",
    "        node_size=500, font_size=16, font_weight='bold')\n",
    "plt.title('Test Brain Graph (4 regions)')\n",
    "plt.show()\n",
    "\n",
    "print(\"✅ Visualization working!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Development Commands\n",
    "\n",
    "Useful commands for development:\n",
    "\n",
    "```bash\n",
    "# Run tests\n",
    "cgnn-test\n",
    "\n",
    "# Format code\n",
    "cgnn-format\n",
    "\n",
    "# Type checking\n",
    "cgnn-typecheck\n",
    "\n",
    "# Lint code\n",
    "cgnn-lint\n",
    "\n",
    "# Clean cache files\n",
    "cgnn-clean\n",
    "```\n",
    "\n",
    "## Next Steps\n",
    "\n",
    "1. Explore the `connectome_gnn/` directory\n",
    "2. Check out the example notebooks in `notebooks/tutorials/`\n",
    "3. Read the documentation in `docs/`\n",
    "4. Start developing!\n",
    "\n",
    "Happy coding! 🧠🚀"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

echo "🎉 Development environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Run 'pip install -e .[dev,viz,full]' to install the package"
echo "2. Run 'lab' to start Jupyter Lab"
echo "3. Open the Welcome.ipynb notebook to verify everything works"
echo ""
echo "Happy coding! 🧠🚀"