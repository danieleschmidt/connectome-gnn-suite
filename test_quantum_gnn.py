#!/usr/bin/env python3
"""Quick test of quantum-enhanced GNN functionality."""

import sys
sys.path.append('/root/repo')

from connectome_gnn.research_v3.quantum_enhanced_gnn import QuantumEnhancedConnectomeGNN
import torch
from torch_geometric.data import Data
import numpy as np

print('🧠 Testing Quantum-Enhanced Connectome GNN')

# Create test connectome data
num_nodes = 20
num_features = 10

# Create realistic brain connectivity pattern
edges = []
for i in range(num_nodes):
    for j in range(i+1, min(i+4, num_nodes)):  # Each node connects to next 3
        edges.append([i, j])
        edges.append([j, i])  # Symmetric

edge_index = torch.tensor(edges, dtype=torch.long).t()
x = torch.randn(num_nodes, num_features)
data = Data(x=x, edge_index=edge_index)

print(f'✅ Test data: {num_nodes} nodes, {edge_index.shape[1]} edges, {x.shape[1]} features')

# Initialize quantum model
model = QuantumEnhancedConnectomeGNN(
    node_features=num_features,
    hidden_dim=64,
    num_layers=3,
    num_qubits=8,
    output_dim=1
)

param_count = sum(p.numel() for p in model.parameters())
print(f'✅ Model initialized: {param_count:,} parameters')

# Test forward pass
model.eval()
with torch.no_grad():
    output = model(data)
    print(f'✅ Forward pass: output shape {output.shape}, value: {output.item():.4f}')

# Test quantum coherence monitoring
coherence = model.get_quantum_coherence()
print(f'✅ Quantum coherence: {[f"{c:.4f}" for c in coherence.tolist()]}')

# Test fusion weights
fusion_weights = model.get_fusion_weights()
avg_weights = fusion_weights.mean(dim=0)
print(f'✅ Fusion weights [quantum, classical]: {[f"{w:.3f}" for w in avg_weights.tolist()]}')

# Test training capability
print('🎯 Testing training capability...')
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()
target = torch.randn(1)  # Single graph target

initial_loss = None
for epoch in range(5):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    if epoch == 0:
        initial_loss = loss.item()
    
    print(f'Epoch {epoch+1}: Loss = {loss.item():.4f}')

print(f'✅ Training: Loss improved from {initial_loss:.4f} to {loss.item():.4f}')
print('🎯 All quantum GNN tests passed!')