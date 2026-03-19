"""Tests for synthetic connectome generation."""

import pytest
import torch
from connectome_gnn.synthetic import (
    generate_connectome,
    generate_dataset,
    small_world_stats,
    NUM_REGIONS,
    REGION_NAMES,
)
from connectome_gnn.graph import ConnectomeGraph


class TestGenerateConnectome:
    def test_returns_connectome_graph(self):
        g = generate_connectome(seed=0)
        assert isinstance(g, ConnectomeGraph)

    def test_node_count(self):
        g = generate_connectome(num_regions=84, seed=0)
        assert g.num_nodes == 84

    def test_feature_dim(self):
        g = generate_connectome(seed=0)
        assert g.num_features == 5

    def test_edge_weight_positive(self):
        g = generate_connectome(seed=0)
        assert (g.edge_weight > 0).all()

    def test_label_binary(self):
        g = generate_connectome(seed=0)
        assert g.label.item() in (0, 1)

    def test_reproducibility(self):
        g1 = generate_connectome(seed=42)
        g2 = generate_connectome(seed=42)
        assert torch.allclose(g1.node_features, g2.node_features)
        assert torch.equal(g1.edge_index, g2.edge_index)

    def test_different_seeds_differ(self):
        g1 = generate_connectome(seed=1)
        g2 = generate_connectome(seed=999)
        # Almost certainly different
        assert not torch.allclose(g1.node_features, g2.node_features)

    def test_edge_index_valid_range(self):
        n = 50
        g = generate_connectome(num_regions=n, seed=0)
        assert g.edge_index.min() >= 0
        assert g.edge_index.max() < n

    def test_bidirectional_edges(self):
        """Both (i,j) and (j,i) should be present for undirected graph."""
        g = generate_connectome(num_regions=20, k=4, seed=0)
        edges = set(zip(g.edge_index[0].tolist(), g.edge_index[1].tolist()))
        for u, v in list(edges)[:5]:
            assert (v, u) in edges


class TestGenerateDataset:
    def test_size(self):
        ds = generate_dataset(num_subjects=30, seed=0)
        assert len(ds) == 30

    def test_all_correct_type(self):
        ds = generate_dataset(num_subjects=10, seed=0)
        assert all(isinstance(g, ConnectomeGraph) for g in ds)

    def test_label_balance(self):
        """Labels should be approximately balanced (not all one class)."""
        ds = generate_dataset(num_subjects=100, seed=0)
        labels = [g.label.item() for g in ds]
        n_pos = sum(labels)
        # Both classes should appear
        assert 5 < n_pos < 95


class TestSmallWorldStats:
    def test_returns_dict(self):
        ds = generate_dataset(num_subjects=5, seed=0)
        stats = small_world_stats(ds)
        assert "mean_clustering" in stats
        assert "mean_avg_path_length" in stats

    def test_clustering_in_range(self):
        ds = generate_dataset(num_subjects=5, seed=0)
        stats = small_world_stats(ds)
        # Real brain graphs: clustering ~ 0.3-0.7 for small-world
        assert 0.0 < stats["mean_clustering"] < 1.0

    def test_region_names_length(self):
        assert len(REGION_NAMES) == NUM_REGIONS
