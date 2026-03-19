"""
Synthetic connectome generator.

Generates realistic brain connectivity graphs using the Watts-Strogatz
small-world model, which is known to capture key properties of real
brain networks:

  - High clustering coefficient (local integration)
  - Short characteristic path length (global efficiency)
  - Sparse connectivity (~5-15% of possible edges in structural data)

The cognitive trait labels are generated via a linear combination of
graph-theoretic features plus Gaussian noise — mimicking the real (weak)
brain-behaviour correlation typically observed in neuroimaging studies.

References
----------
Watts & Strogatz (1998). Collective dynamics of 'small-world' networks.
  Nature, 393(6684), 440-442.
Rubinov & Sporns (2010). Complex network measures of brain connectivity.
  NeuroImage, 52(3), 1059-1069.
"""

from __future__ import annotations

import random
import math
from typing import Optional

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Parcellation atlas (abbreviated Desikan-Killiany, 84 ROIs)
# ---------------------------------------------------------------------------

REGION_NAMES: list[str] = [
    # Frontal
    "ctx-lh-superiorfrontal", "ctx-rh-superiorfrontal",
    "ctx-lh-rostralmiddlefrontal", "ctx-rh-rostralmiddlefrontal",
    "ctx-lh-caudalmiddlefrontal", "ctx-rh-caudalmiddlefrontal",
    "ctx-lh-parsopercularis", "ctx-rh-parsopercularis",
    "ctx-lh-parsorbitalis", "ctx-rh-parsorbitalis",
    "ctx-lh-parstriangularis", "ctx-rh-parstriangularis",
    "ctx-lh-lateralorbitofrontal", "ctx-rh-lateralorbitofrontal",
    "ctx-lh-medialorbitofrontal", "ctx-rh-medialorbitofrontal",
    "ctx-lh-precentral", "ctx-rh-precentral",
    # Parietal
    "ctx-lh-superiorparietal", "ctx-rh-superiorparietal",
    "ctx-lh-inferiorparietal", "ctx-rh-inferiorparietal",
    "ctx-lh-supramarginal", "ctx-rh-supramarginal",
    "ctx-lh-postcentral", "ctx-rh-postcentral",
    "ctx-lh-precuneus", "ctx-rh-precuneus",
    "ctx-lh-posteriorcingulate", "ctx-rh-posteriorcingulate",
    "ctx-lh-isthmuscingulate", "ctx-rh-isthmuscingulate",
    # Temporal
    "ctx-lh-superiortemporal", "ctx-rh-superiortemporal",
    "ctx-lh-middletemporal", "ctx-rh-middletemporal",
    "ctx-lh-inferiortemporal", "ctx-rh-inferiortemporal",
    "ctx-lh-fusiform", "ctx-rh-fusiform",
    "ctx-lh-entorhinal", "ctx-rh-entorhinal",
    "ctx-lh-parahippocampal", "ctx-rh-parahippocampal",
    "ctx-lh-transversetemporal", "ctx-rh-transversetemporal",
    # Occipital
    "ctx-lh-lateraloccipital", "ctx-rh-lateraloccipital",
    "ctx-lh-lingual", "ctx-rh-lingual",
    "ctx-lh-cuneus", "ctx-rh-cuneus",
    "ctx-lh-pericalcarine", "ctx-rh-pericalcarine",
    # Cingulate / Limbic
    "ctx-lh-rostralanteriorcingulate", "ctx-rh-rostralanteriorcingulate",
    "ctx-lh-caudalanteriorcingulate", "ctx-rh-caudalanteriorcingulate",
    "ctx-lh-paracingulate", "ctx-rh-paracingulate",
    # Subcortical
    "Left-Thalamus", "Right-Thalamus",
    "Left-Caudate", "Right-Caudate",
    "Left-Putamen", "Right-Putamen",
    "Left-Pallidum", "Right-Pallidum",
    "Left-Hippocampus", "Right-Hippocampus",
    "Left-Amygdala", "Right-Amygdala",
    "Left-Accumbens-area", "Right-Accumbens-area",
    "Brain-Stem",
    # White matter tracts (summary)
    "CC_anterior", "CC_posterior",
    "UncF_left", "UncF_right",
    "ILF_left", "ILF_right",
    "CST_left", "CST_right",
]

NUM_REGIONS = len(REGION_NAMES)  # 84


# ---------------------------------------------------------------------------
# Small-world graph construction
# ---------------------------------------------------------------------------

def _watts_strogatz_edges(n: int, k: int, beta: float, rng: np.random.Generator):
    """
    Generate a Watts-Strogatz small-world graph.

    Parameters
    ----------
    n    : number of nodes
    k    : each node is initially connected to k nearest neighbours (must be even)
    beta : rewiring probability in [0, 1]

    Returns (src, dst) arrays of undirected edge indices (each pair appears once).
    """
    edges = set()

    # 1. Ring lattice
    for i in range(n):
        for j in range(1, k // 2 + 1):
            u = i
            v = (i + j) % n
            edges.add((min(u, v), max(u, v)))

    # 2. Random rewiring
    new_edges = set(edges)
    for u, v in edges:
        if rng.random() < beta:
            new_edges.discard((u, v))
            # pick a random target not already connected and not self-loop
            candidates = list(set(range(n)) - {u} - {w for a, b in new_edges for w in (a, b) if (a == u or b == u)})
            if candidates:
                w = rng.choice(candidates)
                new_edges.add((min(u, w), max(u, w)))
            else:
                new_edges.add((u, v))  # keep original if no candidate
    return new_edges


def _edges_to_coo(edges: set, n: int, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert undirected edge set to bidirectional COO format with random weights."""
    src_list, dst_list, w_list = [], [], []
    for u, v in edges:
        w = float(rng.beta(2, 5))  # skewed toward lower values, like real FA
        src_list += [u, v]
        dst_list += [v, u]
        w_list += [w, w]
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_weight = torch.tensor(w_list, dtype=torch.float32)
    return edge_index, edge_weight


# ---------------------------------------------------------------------------
# Node features
# ---------------------------------------------------------------------------

def _build_node_features(n: int, edge_index: torch.Tensor, edge_weight: torch.Tensor,
                          rng: np.random.Generator) -> torch.Tensor:
    """
    Build a 5-dimensional node feature matrix:
      0: normalised degree (weighted)
      1: betweenness-proxy (random — real computation is expensive)
      2: regional volume (proxy, sampled from realistic distribution)
      3: mean resting-state activation (proxy)
      4: cortical thickness (proxy)
    """
    # Weighted degree
    deg = torch.zeros(n)
    deg.scatter_add_(0, edge_index[0], edge_weight)
    deg_norm = deg / (deg.max() + 1e-8)

    # Regional volume proxy (log-normal)
    vol = torch.tensor(rng.lognormal(mean=7.5, sigma=0.5, size=n), dtype=torch.float32)
    vol_norm = (vol - vol.mean()) / (vol.std() + 1e-8)

    # Mean activation (normal, with small subject-specific noise)
    activation = torch.tensor(rng.normal(0, 1, size=n), dtype=torch.float32)

    # Cortical thickness (normal around 2.5mm)
    thickness = torch.tensor(rng.normal(2.5, 0.3, size=n).clip(1.5, 4.0), dtype=torch.float32)
    thickness_norm = (thickness - thickness.mean()) / (thickness.std() + 1e-8)

    # Clustering proxy (local mean of neighbour weights)
    cluster = torch.zeros(n)
    count = torch.zeros(n)
    cluster.scatter_add_(0, edge_index[0], edge_weight)
    count.scatter_add_(0, edge_index[0], torch.ones(edge_index.shape[1]))
    cluster = cluster / (count + 1e-8)

    return torch.stack([deg_norm, cluster, vol_norm, activation, thickness_norm], dim=1)


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------

_TRAIT_NAMES = ["fluid_intelligence", "sustained_attention", "working_memory",
                "processing_speed", "cognitive_flexibility"]


def _generate_label(node_features: torch.Tensor, edge_weight: torch.Tensor,
                    trait_idx: int, rng: np.random.Generator) -> torch.Tensor:
    """
    Generate a binary cognitive trait label.

    The latent score is a noisy linear function of graph statistics,
    mimicking the weak but real brain-behaviour effects in HCP data
    (r ~ 0.2-0.4).
    """
    # Graph-level features: mean degree, mean edge weight, global clustering
    mean_deg = node_features[:, 0].mean().item()
    mean_wt = edge_weight.mean().item()
    cluster = node_features[:, 1].mean().item()

    # Trait-specific weights (fixed seed per trait for reproducibility)
    trait_rng = np.random.default_rng(trait_idx * 1337)
    w = trait_rng.normal(0, 1, 3)

    score = w[0] * mean_deg + w[1] * mean_wt + w[2] * cluster
    score += rng.normal(0, 2.0)  # substantial noise (realistic)

    return torch.tensor(int(score > 0), dtype=torch.long)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_connectome(
    num_regions: int = NUM_REGIONS,
    k: int = 8,
    beta: float = 0.15,
    trait_idx: int = 0,
    subject_id: Optional[str] = None,
    seed: Optional[int] = None,
) -> "ConnectomeGraph":  # noqa: F821 – imported at call site
    """
    Generate a single synthetic connectome graph.

    Parameters
    ----------
    num_regions : number of brain regions (nodes)
    k           : mean degree of initial ring lattice (Watts-Strogatz)
    beta        : rewiring probability; 0 = regular lattice, 1 = random graph
    trait_idx   : which cognitive trait to simulate (0-4)
    subject_id  : optional string identifier
    seed        : random seed for reproducibility

    Returns
    -------
    ConnectomeGraph
    """
    from connectome_gnn.graph import ConnectomeGraph  # late import to avoid circular

    rng = np.random.default_rng(seed)
    if subject_id is None:
        subject_id = f"sub-{rng.integers(10000, 99999)}"

    edges = _watts_strogatz_edges(num_regions, k, beta, rng)
    edge_index, edge_weight = _edges_to_coo(edges, num_regions, rng)
    node_features = _build_node_features(num_regions, edge_index, edge_weight, rng)
    label = _generate_label(node_features, edge_weight, trait_idx, rng)

    return ConnectomeGraph(
        node_features=node_features,
        edge_index=edge_index,
        edge_weight=edge_weight,
        label=label,
        subject_id=subject_id,
    )


def generate_dataset(
    num_subjects: int = 200,
    num_regions: int = NUM_REGIONS,
    k: int = 8,
    beta: float = 0.15,
    trait_idx: int = 0,
    seed: int = 42,
) -> list:
    """
    Generate a dataset of `num_subjects` synthetic connectome graphs.

    Parameters
    ----------
    num_subjects : number of synthetic subjects
    num_regions  : brain regions per subject
    k, beta      : Watts-Strogatz parameters
    trait_idx    : cognitive trait to predict (0-4)
    seed         : master random seed

    Returns
    -------
    List[ConnectomeGraph]
    """
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31, size=num_subjects).tolist()
    return [
        generate_connectome(
            num_regions=num_regions,
            k=k,
            beta=beta,
            trait_idx=trait_idx,
            subject_id=f"sub-{i:04d}",
            seed=int(seeds[i]),
        )
        for i in range(num_subjects)
    ]


def small_world_stats(graphs: list) -> dict:
    """
    Compute mean clustering coefficient and characteristic path length
    for a list of ConnectomeGraph objects.  Useful to verify small-world
    properties (sigma = C/C_random / (L/L_random) >> 1).
    """
    clustering_vals, avg_path_vals = [], []
    for g in graphs:
        A = g.adjacency_matrix().numpy()
        n = A.shape[0]
        # Clustering: fraction of closed triangles per node
        deg = A.sum(axis=1)
        triangles = np.diagonal(A @ A @ A)
        with np.errstate(divide="ignore", invalid="ignore"):
            c = np.where(deg * (deg - 1) > 0, triangles / (deg * (deg - 1)), 0.0)
        clustering_vals.append(float(c.mean()))
        # Average shortest path via BFS (unweighted, sample 20 nodes for speed)
        sample = min(20, n)
        paths = []
        for start in range(sample):
            visited = {start}
            queue = [(start, 0)]
            while queue:
                node, dist = queue.pop(0)
                for nbr in np.where(A[node] > 0)[0]:
                    if nbr not in visited:
                        visited.add(nbr)
                        paths.append(dist + 1)
                        queue.append((nbr, dist + 1))
        avg_path_vals.append(float(np.mean(paths)) if paths else float("nan"))

    return {
        "mean_clustering": float(np.mean(clustering_vals)),
        "mean_avg_path_length": float(np.nanmean(avg_path_vals)),
        "num_graphs": len(graphs),
    }
