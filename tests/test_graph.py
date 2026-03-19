"""Tests for ConnectomeGraph and batching utilities."""

import pytest
import torch
from connectome_gnn.graph import ConnectomeGraph, collate_graphs, ConnectomeDataLoader


def make_simple_graph(n=10, e=20, f=5, label=0):
    """Create a simple random ConnectomeGraph for testing."""
    node_features = torch.randn(n, f)
    # Random edges (bidirectional)
    src = torch.randint(0, n, (e,))
    dst = torch.randint(0, n, (e,))
    edge_index = torch.stack([
        torch.cat([src, dst]),
        torch.cat([dst, src]),
    ])
    w = torch.rand(e).abs() + 0.01
    edge_weight = torch.cat([w, w])  # symmetric: same weight both directions
    return ConnectomeGraph(
        node_features=node_features,
        edge_index=edge_index,
        edge_weight=edge_weight,
        label=torch.tensor(label, dtype=torch.long),
        subject_id="test-subject",
    )


class TestConnectomeGraph:
    def test_basic_properties(self):
        g = make_simple_graph(n=84, e=300, f=5)
        assert g.num_nodes == 84
        assert g.num_features == 5
        assert g.num_edges == 600  # bidirectional

    def test_adjacency_matrix_shape(self):
        g = make_simple_graph(n=10, e=5, f=3)
        A = g.adjacency_matrix()
        assert A.shape == (10, 10)

    def test_adjacency_matrix_symmetry(self):
        """For undirected graphs, A should be symmetric."""
        g = make_simple_graph(n=10, e=5, f=3)
        A = g.adjacency_matrix()
        assert torch.allclose(A, A.T, atol=1e-6)

    def test_degree_nonnegative(self):
        g = make_simple_graph(n=10, e=20, f=5)
        deg = g.degree()
        assert deg.shape == (10,)
        assert (deg >= 0).all()

    def test_to_device(self):
        g = make_simple_graph()
        g2 = g.to(torch.device("cpu"))
        assert g2.node_features.device.type == "cpu"


class TestCollate:
    def test_collate_batch_shapes(self):
        graphs = [make_simple_graph(n=84, e=300, f=5, label=i % 2) for i in range(4)]
        batch = collate_graphs(graphs)
        assert batch.node_features.shape == (4 * 84, 5)
        assert batch.batch.shape == (4 * 84,)
        assert batch.labels.shape == (4,)
        assert batch.num_graphs == 4

    def test_batch_index_range(self):
        graphs = [make_simple_graph(n=20, e=50, f=5, label=i % 2) for i in range(3)]
        batch = collate_graphs(graphs)
        assert int(batch.batch.min()) == 0
        assert int(batch.batch.max()) == 2

    def test_ptr_is_cumulative(self):
        graphs = [make_simple_graph(n=10, e=20, f=5) for _ in range(3)]
        batch = collate_graphs(graphs)
        # ptr[0]=0, ptr[1]=10, ptr[2]=20, ptr[3]=30
        assert batch.ptr[0] == 0
        assert batch.ptr[-1] == 30

    def test_edge_index_offset(self):
        """Edge indices must be shifted per graph in the batch."""
        g1 = make_simple_graph(n=5, e=4, f=2, label=0)
        g2 = make_simple_graph(n=5, e=4, f=2, label=1)
        batch = collate_graphs([g1, g2])
        # Second graph's nodes start at index 5
        E1 = g1.edge_index.shape[1]
        assert batch.edge_index[:, E1:].min() >= 5


class TestDataLoader:
    def test_loader_iterations(self):
        graphs = [make_simple_graph(n=10, e=20, f=5, label=i % 2) for i in range(20)]
        loader = ConnectomeDataLoader(graphs, batch_size=8, shuffle=False)
        batches = list(loader)
        # 20 / 8 = ceil 3
        assert len(batches) == 3

    def test_loader_total_graphs(self):
        graphs = [make_simple_graph(n=10, e=20, f=5, label=i % 2) for i in range(15)]
        loader = ConnectomeDataLoader(graphs, batch_size=4, shuffle=False)
        total = sum(b.num_graphs for b in loader)
        assert total == 15
