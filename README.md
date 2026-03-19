# connectome-gnn-suite

Graph neural networks for brain connectome analysis — a clean, dependency-light toolkit built around Human Connectome Project (HCP)-style data.

## What is a brain connectome?

The human brain contains ~86 billion neurons organised into anatomically distinct regions (ROIs — regions of interest). Neuroimaging studies like the [Human Connectome Project](https://www.humanconnectome.org/) map the *structural* connections between these regions (via diffusion MRI tractography) and the *functional* correlations (via resting-state fMRI).

The result is a **weighted undirected graph**:
- **Nodes** = brain regions (e.g. 84 regions from the Desikan-Killiany parcellation)
- **Edges** = connectivity strength between region pairs (fractional anisotropy, correlation coefficient, etc.)
- **Graph label** = cognitive phenotype of the subject (e.g. fluid intelligence, working memory score)

Graph neural networks are a natural fit: they respect the non-Euclidean structure of the connectome, allow information to flow along anatomical pathways, and can learn which connectivity patterns predict cognitive traits.

## Architecture

```
Node features [N × 5]          Edge weights [E]
       │                               │
  ┌────▼───────────────────────────────▼────┐
  │          Message-Passing Layer           │  ×3
  │  (GCN: sym-norm aggregation)            │
  │  (GraphSAGE: concat+project)            │
  └─────────────────┬───────────────────────┘
                    │ [N × hidden]
             Mean Pool over nodes
                    │ [B × hidden]
              MLP Classifier
                    │ [B × num_classes]
```

Two baseline models are implemented in **pure PyTorch** (no PyTorch Geometric required):

### GCN (Kipf & Welling, 2017)
Symmetric normalised convolution on the weighted adjacency matrix:

```
H' = D̂^{-1/2} Â D̂^{-1/2} H W
```

where `Â = A + I` (self-loops), `D̂` is the weighted degree matrix.

### GraphSAGE (Hamilton et al., 2017)
Inductive mean aggregation:

```
h_v = ReLU( W · concat( h_v, mean_{u∈N(v)} w_vu · h_u ) )
```

Both models use:
- 3 message-passing layers
- BatchNorm + ReLU + Dropout after each layer
- Mean-pool readout → 2-layer MLP classifier

## Synthetic data generator

Since raw HCP data requires credentialed access, this toolkit ships a realistic **synthetic connectome generator** based on the Watts-Strogatz small-world model [1]:

- High clustering coefficient (local integration, like real brains)
- Short characteristic path length (global efficiency, like real brains)
- Sparse connectivity (~4% density, like real structural connectomes)

Node features (5 dimensions per region):
1. Weighted degree (normalised)
2. Local clustering coefficient
3. Regional volume proxy (log-normal)
4. Mean resting-state activation
5. Cortical thickness proxy

Labels are generated as a noisy linear function of graph statistics, mimicking the weak but real brain-behaviour correlations (r ~ 0.2–0.3) observed in neuroimaging.

## Installation

```bash
git clone https://github.com/danieleschmidt/connectome-gnn-suite
cd connectome-gnn-suite
pip install -e .
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.0, NumPy. No PyTorch Geometric.

## Quick start

```python
from connectome_gnn.synthetic import generate_dataset
from connectome_gnn.graph import ConnectomeDataLoader
from connectome_gnn.models import GCNConnectome, GraphSAGEConnectome
from connectome_gnn.train import Trainer
import torch

# Generate 200 synthetic subjects
graphs = generate_dataset(num_subjects=200, seed=42)
train, val = graphs[:160], graphs[160:]

train_loader = ConnectomeDataLoader(train, batch_size=16)
val_loader   = ConnectomeDataLoader(val,   batch_size=16, shuffle=False)

# Train GCN
model = GCNConnectome(in_channels=5, hidden_dim=64, num_classes=2)
trainer = Trainer(model, torch.optim.Adam(model.parameters(), lr=1e-3))
history = trainer.fit(train_loader, val_loader, num_epochs=50, patience=10)

# Evaluate
metrics = trainer.evaluate(val_loader)
print(f"Val accuracy: {metrics['accuracy']:.3f}")
```

## Full demo

```bash
python examples/demo.py
```

Trains both GCN and GraphSAGE on 300 synthetic subjects and prints a comparison table. Expected accuracy: **55–70%** (realistic for noisy brain-behaviour data).

## Tests

```bash
python -m pytest tests/ -v
```

43 tests covering graph data structures, synthetic generation, model forward passes, gradient flow, and the training loop.

## Repository structure

```
connectome_gnn/
  __init__.py    – public API
  graph.py       – ConnectomeGraph, ConnectomeBatch, DataLoader
  synthetic.py   – Watts-Strogatz connectome generator
  models.py      – GCNConnectome, GraphSAGEConnectome
  train.py       – Trainer with early stopping

tests/
  test_graph.py      – data structure tests
  test_synthetic.py  – generator tests
  test_models.py     – forward pass & gradient tests
  test_training.py   – training loop tests

examples/
  demo.py            – end-to-end training demo
```

## Extending to real HCP data

To use real HCP structural connectivity matrices:

1. Download processed connectivity matrices from [HCP ConnectomeDB](https://db.humanconnectome.org/)
2. Load as NumPy arrays (84×84 or 360×360 depending on parcellation)
3. Convert to `ConnectomeGraph` objects:

```python
import numpy as np
import torch
from connectome_gnn.graph import ConnectomeGraph

def hcp_matrix_to_graph(connectivity_matrix: np.ndarray, label: int) -> ConnectomeGraph:
    A = torch.tensor(connectivity_matrix, dtype=torch.float32)
    # Threshold: keep top 10% of connections
    threshold = A.flatten().quantile(0.90)
    A_thresh = (A > threshold).float() * A
    src, dst = torch.where(A_thresh > 0)
    edge_index = torch.stack([
        torch.cat([src, dst]),
        torch.cat([dst, src]),
    ])
    weights = A_thresh[src]
    edge_weight = torch.cat([weights, weights])
    # Node features: degree + threshold statistics
    deg = A_thresh.sum(dim=1, keepdim=True)
    node_features = deg / (deg.max() + 1e-8)  # add more features as needed
    return ConnectomeGraph(
        node_features=node_features,
        edge_index=edge_index,
        edge_weight=edge_weight,
        label=torch.tensor(label, dtype=torch.long),
    )
```

## References

[1] Watts, D. J. & Strogatz, S. H. (1998). Collective dynamics of 'small-world' networks. *Nature*, 393(6684), 440–442.

[2] Kipf, T. N. & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *ICLR 2017*. [arXiv:1609.02907](https://arxiv.org/abs/1609.02907)

[3] Hamilton, W., Ying, R. & Leskovec, J. (2017). Inductive representation learning on large graphs. *NeurIPS 2017*. [arXiv:1706.02216](https://arxiv.org/abs/1706.02216)

[4] Rubinov, M. & Sporns, O. (2010). Complex network measures of brain connectivity: uses and interpretations. *NeuroImage*, 52(3), 1059–1069.

[5] Van Essen, D. C. et al. (2013). The WU-Minn Human Connectome Project: an overview. *NeuroImage*, 80, 62–79.

## License

MIT
