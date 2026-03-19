"""
Connectome graph data structures.

A brain connectome is modeled as a weighted undirected graph:
  - Nodes  = brain regions (ROIs), each with optional feature vector
  - Edges  = structural or functional connectivity between regions
  - Graph-level label = cognitive trait / diagnostic category

We deliberately avoid PyG/DGL to keep dependencies minimal; everything
runs on plain PyTorch tensors plus a lightweight ConnectomeGraph dataclass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core data structure
# ---------------------------------------------------------------------------

@dataclass
class ConnectomeGraph:
    """
    A single subject's brain connectivity graph.

    Attributes
    ----------
    node_features : Tensor[N, F]
        Per-region feature matrix.  F can be 1 (just the degree) or richer
        (e.g. regional volume, mean activation, etc.)
    edge_index : Tensor[2, E]
        COO-format edge list.  Edges are stored once per direction for
        undirected graphs (i.e. both (i,j) and (j,i) are present).
    edge_weight : Tensor[E]
        Connectivity weight for each edge (e.g. FA, correlation, coherence).
    label : Optional[Tensor[]]
        Graph-level scalar label (e.g. cognitive score category).
    num_nodes : int
        Number of brain regions.
    subject_id : str
        Identifier string for the subject.
    """

    node_features: torch.Tensor          # [N, F]
    edge_index: torch.Tensor             # [2, E]
    edge_weight: torch.Tensor            # [E]
    label: Optional[torch.Tensor] = None # scalar long or float
    subject_id: str = "unknown"

    @property
    def num_nodes(self) -> int:
        return self.node_features.shape[0]

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

    @property
    def num_features(self) -> int:
        return self.node_features.shape[1]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def adjacency_matrix(self) -> torch.Tensor:
        """Return dense [N, N] weighted adjacency matrix."""
        N = self.num_nodes
        A = torch.zeros(N, N, dtype=self.edge_weight.dtype)
        src, dst = self.edge_index
        A[src, dst] = self.edge_weight
        return A

    def degree(self) -> torch.Tensor:
        """Return degree vector [N] (sum of outgoing edge weights)."""
        N = self.num_nodes
        deg = torch.zeros(N, dtype=self.edge_weight.dtype)
        deg.scatter_add_(0, self.edge_index[0], self.edge_weight)
        return deg

    def to(self, device: torch.device) -> "ConnectomeGraph":
        return ConnectomeGraph(
            node_features=self.node_features.to(device),
            edge_index=self.edge_index.to(device),
            edge_weight=self.edge_weight.to(device),
            label=self.label.to(device) if self.label is not None else None,
            subject_id=self.subject_id,
        )


# ---------------------------------------------------------------------------
# Batch collation
# ---------------------------------------------------------------------------

@dataclass
class ConnectomeBatch:
    """
    A mini-batch of ConnectomeGraph objects, packed into block-diagonal
    tensors for efficient message-passing.

    Attributes
    ----------
    node_features : Tensor[sum(N_i), F]
    edge_index    : Tensor[2, sum(E_i)]   – node indices are offset per graph
    edge_weight   : Tensor[sum(E_i)]
    batch         : Tensor[sum(N_i)]      – graph index for each node
    labels        : Tensor[B]            – graph-level labels
    ptr           : Tensor[B+1]          – cumulative node count pointer
    """

    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    batch: torch.Tensor
    labels: Optional[torch.Tensor]
    ptr: torch.Tensor

    @property
    def num_graphs(self) -> int:
        return int(self.ptr.shape[0]) - 1

    @property
    def num_nodes(self) -> int:
        return int(self.node_features.shape[0])

    def to(self, device: torch.device) -> "ConnectomeBatch":
        return ConnectomeBatch(
            node_features=self.node_features.to(device),
            edge_index=self.edge_index.to(device),
            edge_weight=self.edge_weight.to(device),
            batch=self.batch.to(device),
            labels=self.labels.to(device) if self.labels is not None else None,
            ptr=self.ptr.to(device),
        )


def collate_graphs(graphs: list[ConnectomeGraph]) -> ConnectomeBatch:
    """Collate a list of ConnectomeGraph into a ConnectomeBatch."""
    node_feats, edge_idxs, edge_wts, batches, labels = [], [], [], [], []
    ptr = [0]
    offset = 0

    for g_idx, g in enumerate(graphs):
        N = g.num_nodes
        node_feats.append(g.node_features)
        edge_idxs.append(g.edge_index + offset)
        edge_wts.append(g.edge_weight)
        batches.append(torch.full((N,), g_idx, dtype=torch.long))
        if g.label is not None:
            labels.append(g.label)
        offset += N
        ptr.append(offset)

    return ConnectomeBatch(
        node_features=torch.cat(node_feats, dim=0),
        edge_index=torch.cat(edge_idxs, dim=1),
        edge_weight=torch.cat(edge_wts, dim=0),
        batch=torch.cat(batches, dim=0),
        labels=torch.stack(labels) if labels else None,
        ptr=torch.tensor(ptr, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Simple DataLoader
# ---------------------------------------------------------------------------

class ConnectomeDataLoader:
    """Minimal DataLoader that batches ConnectomeGraph objects."""

    def __init__(
        self,
        dataset: list[ConnectomeGraph],
        batch_size: int = 16,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            perm = torch.randperm(len(indices)).tolist()
            indices = [indices[i] for i in perm]
        for start in range(0, len(indices), self.batch_size):
            chunk = indices[start : start + self.batch_size]
            yield collate_graphs([self.dataset[i] for i in chunk])
