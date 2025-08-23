#!/usr/bin/env python3
"""Simple demo without PyG InMemoryDataset complexity."""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np
import sys
sys.path.insert(0, '.')

import connectome_gnn
from connectome_gnn.models import HierarchicalBrainGNN
from connectome_gnn.tasks import CognitiveScorePrediction


def create_simple_connectome_data(num_samples=50, num_nodes=100):
    """Create simple synthetic connectome data."""
    print("Creating synthetic connectome data...")
    
    data_list = []
    
    for i in range(num_samples):
        # Generate random connectivity matrix
        connectivity = np.random.rand(num_nodes, num_nodes)
        connectivity = (connectivity + connectivity.T) / 2  # Make symmetric
        np.fill_diagonal(connectivity, 0)
        
        # Threshold to create sparse connections
        connectivity[connectivity < 0.7] = 0
        
        # Convert to edge format
        edge_indices = np.nonzero(connectivity)
        edge_weights = connectivity[edge_indices]
        
        # Create edge index tensor
        edge_index = torch.tensor([edge_indices[0], edge_indices[1]], dtype=torch.long)
        edge_attr = torch.tensor(edge_weights, dtype=torch.float)
        
        # Node features (degree as simple feature)
        node_degrees = np.sum(connectivity > 0, axis=1)
        x = torch.tensor(node_degrees, dtype=torch.float).unsqueeze(1)
        
        # Random target
        y = torch.tensor([np.random.rand()], dtype=torch.float)
        
        # Create data object
        data = Data(
            x=x,
            edge_index=edge_index, 
            edge_attr=edge_attr,
            y=y,
            num_nodes=num_nodes
        )
        
        data_list.append(data)
    
    return data_list


def simple_demo():
    """Run a simple demo."""
    print("🧠 Simple Connectome-GNN Demo")
    print("=" * 40)
    
    # Check components
    components = connectome_gnn.get_available_components()
    print(f"Available: {components}")
    
    # Create synthetic data
    data_list = create_simple_connectome_data(num_samples=20, num_nodes=50)
    print(f"Created {len(data_list)} samples")
    
    # Examine first sample
    sample = data_list[0]
    print(f"Sample: {sample.num_nodes} nodes, {sample.edge_index.size(1)} edges")
    print(f"Features: {sample.x.shape}, Target: {sample.y.item():.3f}")
    
    # Create simple model
    model = HierarchicalBrainGNN(
        node_features=1,  # Just degree
        hidden_dim=32,
        num_levels=2,
        num_classes=1,
        dropout=0.1
    )
    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        prediction = model(sample)
        print(f"Prediction: {prediction.item():.3f}")
    
    # Create data loader
    loader = DataLoader(data_list, batch_size=4, shuffle=True)
    
    # Simple training loop 
    print("\nSimple training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    
    for epoch in range(5):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            
            # Forward pass
            out = model(batch)
            
            # Loss
            loss = F.mse_loss(out.squeeze(), batch.y.squeeze())
            
            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    print("\n✅ Simple demo completed successfully!")
    print("🎉 The core Connectome-GNN functionality is working!")


if __name__ == "__main__":
    simple_demo()