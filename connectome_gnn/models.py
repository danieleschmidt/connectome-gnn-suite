"""
GNN models for brain connectome classification.

Both models operate on the ConnectomeBatch produced by collate_graphs().
No PyTorch Geometric required — message passing is implemented via
scatter operations on plain PyTorch tensors.

Models
------
GCNConnectome     – Graph Convolutional Network (Kipf & Welling, 2017)
GraphSAGEConnectome – GraphSAGE (Hamilton et al., 2017)

Both architectures share the same graph-level readout:
  mean-pool over node embeddings → MLP classifier

References
----------
Kipf, T. & Welling, M. (2017). Semi-supervised classification with
  graph convolutional networks. ICLR 2017.
  https://arxiv.org/abs/1609.02907

Hamilton, W., Ying, R. & Leskovec, J. (2017). Inductive representation
  learning on large graphs. NeurIPS 2017.
  https://arxiv.org/abs/1706.02216
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from connectome_gnn.graph import ConnectomeBatch


# ---------------------------------------------------------------------------
# Utility: sparse message passing via scatter
# ---------------------------------------------------------------------------

def _scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Scatter-mean: for each target node, average incoming messages."""
    out = torch.zeros(dim_size, src.shape[1], device=src.device, dtype=src.dtype)
    count = torch.zeros(dim_size, 1, device=src.device, dtype=src.dtype)
    idx = index.unsqueeze(1).expand_as(src)
    out.scatter_add_(0, idx, src)
    count.scatter_add_(0, index.unsqueeze(1), torch.ones(index.shape[0], 1, device=src.device))
    return out / (count + 1e-8)


def _scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros(dim_size, src.shape[1], device=src.device, dtype=src.dtype)
    idx = index.unsqueeze(1).expand_as(src)
    out.scatter_add_(0, idx, src)
    return out


def _graph_mean_pool(node_emb: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """Average pool node embeddings per graph → [num_graphs, hidden_dim]."""
    return _scatter_mean(node_emb, batch, dim_size=num_graphs)


# ---------------------------------------------------------------------------
# GCN layer
# ---------------------------------------------------------------------------

class GCNLayer(nn.Module):
    """
    Single GCN layer.

    Forward pass implements the symmetric normalised convolution:
        H' = D^{-1/2} A_hat D^{-1/2} H W
    where A_hat = A + I (self-loops added).

    For edge-weighted graphs we use the weighted adjacency A_w and
    re-derive the symmetric normalisation from weighted degrees.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(
        self,
        x: torch.Tensor,           # [N, in_channels]
        edge_index: torch.Tensor,  # [2, E]
        edge_weight: torch.Tensor, # [E]
    ) -> torch.Tensor:
        N = x.shape[0]
        src, dst = edge_index

        # Add self-loops (weight = 1)
        self_idx = torch.arange(N, device=x.device)
        sl_src = self_idx
        sl_dst = self_idx
        sl_w = torch.ones(N, device=x.device)
        src_aug = torch.cat([src, sl_src], dim=0)
        dst_aug = torch.cat([dst, sl_dst], dim=0)
        w_aug = torch.cat([edge_weight, sl_w], dim=0)

        # Weighted degree D_hat
        deg = torch.zeros(N, device=x.device)
        deg.scatter_add_(0, src_aug, w_aug)
        deg_inv_sqrt = (deg + 1e-8).pow(-0.5)  # [N]

        # Normalise: w_norm_ij = d_i^{-1/2} * w_ij * d_j^{-1/2}
        w_norm = deg_inv_sqrt[src_aug] * w_aug * deg_inv_sqrt[dst_aug]

        # Aggregate: sum_j w_norm_ij * (x_j W)
        x_transformed = self.linear(x)  # [N, out]
        msg = x_transformed[src_aug] * w_norm.unsqueeze(1)  # [E, out]
        out = _scatter_sum(msg, dst_aug, dim_size=N)
        return out + self.bias


# ---------------------------------------------------------------------------
# GraphSAGE layer (mean aggregator)
# ---------------------------------------------------------------------------

class SAGELayer(nn.Module):
    """
    GraphSAGE layer with mean neighbourhood aggregation.

    h_v = ReLU( W_self * h_v || W_neigh * mean_{u in N(v)} h_u )

    where || denotes concatenation then projection via a single linear.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Concat(self, agg) → out
        self.linear = nn.Linear(in_channels * 2, out_channels)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        N = x.shape[0]
        src, dst = edge_index

        # Weighted mean of neighbours (weight by edge_weight)
        msg = x[src] * edge_weight.unsqueeze(1)
        w_sum = torch.zeros(N, 1, device=x.device)
        w_sum.scatter_add_(0, dst.unsqueeze(1), edge_weight.unsqueeze(1))
        agg = _scatter_sum(msg, dst, dim_size=N) / (w_sum + 1e-8)

        combined = torch.cat([x, agg], dim=1)  # [N, 2*in]
        return F.relu(self.linear(combined))


# ---------------------------------------------------------------------------
# Full models
# ---------------------------------------------------------------------------

class GCNConnectome(nn.Module):
    """
    3-layer GCN for connectome graph classification.

    Architecture
    ------------
    Input (node features) → GCN×3 → mean-pool → MLP → class logits

    Parameters
    ----------
    in_channels   : input node feature dimension
    hidden_dim    : hidden layer width
    num_classes   : number of output classes
    num_layers    : number of GCN message-passing layers (default 3)
    dropout       : dropout rate applied after each hidden layer
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        num_classes: int = 2,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        dims = [in_channels] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList(
            [GCNLayer(dims[i], dims[i + 1]) for i in range(num_layers)]
        )
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)]
        )

        # MLP classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def encode(self, batch: ConnectomeBatch) -> torch.Tensor:
        """Return graph-level embeddings [B, hidden_dim]."""
        x = batch.node_features
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, batch.edge_index, batch.edge_weight)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return _graph_mean_pool(x, batch.batch, batch.num_graphs)

    def forward(self, batch: ConnectomeBatch) -> torch.Tensor:
        """Return class logits [B, num_classes]."""
        graph_emb = self.encode(batch)
        return self.classifier(graph_emb)


class GraphSAGEConnectome(nn.Module):
    """
    3-layer GraphSAGE for connectome graph classification.

    Architecture
    ------------
    Input → SAGE×3 → mean-pool → MLP → class logits

    Parameters mirror GCNConnectome.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        num_classes: int = 2,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        dims = [in_channels] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList(
            [SAGELayer(dims[i], dims[i + 1]) for i in range(num_layers)]
        )
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)]
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def encode(self, batch: ConnectomeBatch) -> torch.Tensor:
        x = batch.node_features
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, batch.edge_index, batch.edge_weight)
            x = bn(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return _graph_mean_pool(x, batch.batch, batch.num_graphs)

    def forward(self, batch: ConnectomeBatch) -> torch.Tensor:
        graph_emb = self.encode(batch)
        return self.classifier(graph_emb)
