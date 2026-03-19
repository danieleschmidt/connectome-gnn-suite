"""Tests for GCN and GraphSAGE models."""

import pytest
import torch
from connectome_gnn.graph import collate_graphs
from connectome_gnn.synthetic import generate_dataset
from connectome_gnn.models import GCNConnectome, GraphSAGEConnectome


@pytest.fixture(scope="module")
def small_batch():
    """Reusable small batch of 8 graphs."""
    graphs = generate_dataset(num_subjects=8, num_regions=20, seed=0)
    return collate_graphs(graphs)


class TestGCNConnectome:
    def test_forward_shape(self, small_batch):
        model = GCNConnectome(in_channels=5, hidden_dim=32, num_classes=2)
        model.eval()
        with torch.no_grad():
            out = model(small_batch)
        assert out.shape == (8, 2), f"Expected (8,2), got {out.shape}"

    def test_encode_shape(self, small_batch):
        model = GCNConnectome(in_channels=5, hidden_dim=32, num_classes=2)
        model.eval()
        with torch.no_grad():
            emb = model.encode(small_batch)
        assert emb.shape == (8, 32)

    def test_gradients_flow(self, small_batch):
        model = GCNConnectome(in_channels=5, hidden_dim=32, num_classes=2)
        model.train()
        logits = model(small_batch)
        loss = logits.sum()
        loss.backward()
        # At least one param should have gradient
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad

    def test_custom_num_layers(self, small_batch):
        model = GCNConnectome(in_channels=5, hidden_dim=16, num_classes=2, num_layers=2)
        model.eval()
        with torch.no_grad():
            out = model(small_batch)
        assert out.shape == (8, 2)

    def test_output_is_finite(self, small_batch):
        model = GCNConnectome(in_channels=5, hidden_dim=32, num_classes=2)
        model.eval()
        with torch.no_grad():
            out = model(small_batch)
        assert torch.isfinite(out).all(), "Model output contains NaN/Inf"


class TestGraphSAGEConnectome:
    def test_forward_shape(self, small_batch):
        model = GraphSAGEConnectome(in_channels=5, hidden_dim=32, num_classes=2)
        model.eval()
        with torch.no_grad():
            out = model(small_batch)
        assert out.shape == (8, 2)

    def test_encode_shape(self, small_batch):
        model = GraphSAGEConnectome(in_channels=5, hidden_dim=32, num_classes=2)
        model.eval()
        with torch.no_grad():
            emb = model.encode(small_batch)
        assert emb.shape == (8, 32)

    def test_gradients_flow(self, small_batch):
        model = GraphSAGEConnectome(in_channels=5, hidden_dim=32, num_classes=2)
        model.train()
        logits = model(small_batch)
        loss = logits.sum()
        loss.backward()
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad

    def test_output_is_finite(self, small_batch):
        model = GraphSAGEConnectome(in_channels=5, hidden_dim=32, num_classes=2)
        model.eval()
        with torch.no_grad():
            out = model(small_batch)
        assert torch.isfinite(out).all()


class TestBothModels:
    @pytest.mark.parametrize("ModelClass", [GCNConnectome, GraphSAGEConnectome])
    def test_inference_modes_consistent(self, small_batch, ModelClass):
        """train() and eval() should return same shape."""
        model = ModelClass(in_channels=5, hidden_dim=16, num_classes=2)
        model.train()
        out_train = model(small_batch)
        model.eval()
        with torch.no_grad():
            out_eval = model(small_batch)
        assert out_train.shape == out_eval.shape
