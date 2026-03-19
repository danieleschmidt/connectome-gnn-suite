"""
connectome-gnn-suite
====================
Graph neural networks for brain connectome analysis.

Quick start
-----------
>>> from connectome_gnn.synthetic import generate_dataset
>>> from connectome_gnn.graph import ConnectomeDataLoader
>>> from connectome_gnn.models import GCNConnectome
>>> from connectome_gnn.train import Trainer
>>> import torch
>>>
>>> graphs = generate_dataset(num_subjects=200, seed=42)
>>> loader = ConnectomeDataLoader(graphs, batch_size=16)
>>> model = GCNConnectome(in_channels=5, hidden_dim=64, num_classes=2)
>>> trainer = Trainer(model, torch.optim.Adam(model.parameters(), lr=1e-3))
>>> history = trainer.fit(loader, loader, num_epochs=10, verbose=False)
"""

__version__ = "0.2.0"
__author__ = "Daniel Schmidt"

from connectome_gnn.graph import ConnectomeGraph, ConnectomeBatch, ConnectomeDataLoader, collate_graphs
from connectome_gnn.synthetic import generate_connectome, generate_dataset, REGION_NAMES
from connectome_gnn.models import GCNConnectome, GraphSAGEConnectome
from connectome_gnn.train import Trainer

__all__ = [
    "ConnectomeGraph",
    "ConnectomeBatch",
    "ConnectomeDataLoader",
    "collate_graphs",
    "generate_connectome",
    "generate_dataset",
    "REGION_NAMES",
    "GCNConnectome",
    "GraphSAGEConnectome",
    "Trainer",
]
