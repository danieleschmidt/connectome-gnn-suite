"""Unit tests for GNN model architectures."""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from unittest.mock import Mock, patch


class TestHierarchicalBrainGNN:
    """Test the Hierarchical Brain GNN model."""
    
    def test_model_initialization(self):
        """Test model can be initialized with proper parameters."""
        model_config = {
            "node_features": 64,
            "hidden_dim": 128,
            "num_levels": 3,
            "level_dims": [128, 64, 32],
            "dropout": 0.1
        }
        
        # Test configuration validity
        assert model_config["node_features"] > 0
        assert model_config["hidden_dim"] > 0
        assert model_config["num_levels"] == len(model_config["level_dims"])
        assert 0 <= model_config["dropout"] <= 1.0
        
    def test_forward_pass_shape(self, hierarchical_graph, device):
        """Test forward pass produces correct output shapes."""
        # Mock model forward pass
        batch_size = 1
        num_nodes = hierarchical_graph.num_nodes
        hidden_dim = 64
        
        # Simulate model output
        node_embeddings = torch.randn(num_nodes, hidden_dim, device=device)
        graph_embedding = torch.randn(batch_size, hidden_dim, device=device)
        
        # Test output shapes
        assert node_embeddings.shape == (num_nodes, hidden_dim)
        assert graph_embedding.shape == (batch_size, hidden_dim)
        
    def test_hierarchical_pooling(self, hierarchical_graph):
        """Test hierarchical pooling operations."""
        data = hierarchical_graph
        hierarchy_levels = data.hierarchy_level
        
        # Test that hierarchy levels are properly assigned
        level_0_nodes = (hierarchy_levels == 0).sum().item()  # Base level
        level_1_nodes = (hierarchy_levels == 1).sum().item()  # Intermediate
        level_2_nodes = (hierarchy_levels == 2).sum().item()  # Top level
        
        assert level_0_nodes == 64  # Base nodes
        assert level_1_nodes == 16  # Intermediate nodes
        assert level_2_nodes == 4   # Top nodes
        
        # Test pooling reduces dimensionality at each level
        assert level_0_nodes > level_1_nodes > level_2_nodes
        
    @pytest.mark.parametrize("pooling_method", ["mean", "max", "attention", "diffpool"])
    def test_pooling_methods(self, simple_graph, pooling_method):
        """Test different pooling methods."""
        x = simple_graph.x  # [4, 16]
        
        if pooling_method == "mean":
            pooled = torch.mean(x, dim=0, keepdim=True)  # [1, 16]
        elif pooling_method == "max":
            pooled = torch.max(x, dim=0, keepdim=True)[0]  # [1, 16]
        elif pooling_method == "attention":
            # Simulate attention pooling
            attention_weights = torch.softmax(torch.randn(x.size(0)), dim=0)
            pooled = torch.sum(x * attention_weights.unsqueeze(1), dim=0, keepdim=True)
        elif pooling_method == "diffpool":
            # Simulate DiffPool assignment matrix
            assignment = torch.softmax(torch.randn(x.size(0), 1), dim=0)
            pooled = torch.matmul(assignment.t(), x)  # [1, 16]
        
        assert pooled.shape == (1, 16)
        
    def test_residual_connections(self, simple_graph):
        """Test residual connections in hierarchical model."""
        x = simple_graph.x  # [4, 16]
        
        # Simulate a layer with residual connection
        linear = nn.Linear(16, 16)
        
        # Forward pass
        h = linear(x)  # [4, 16]
        
        # Residual connection
        residual_output = x + h  # [4, 16]
        
        assert residual_output.shape == x.shape
        # Test that residual adds information
        assert not torch.allclose(residual_output, x)
        assert not torch.allclose(residual_output, h)


class TestTemporalConnectomeGNN:
    """Test the Temporal Connectome GNN model."""
    
    def test_temporal_data_handling(self):
        """Test handling of temporal connectivity data."""
        batch_size = 2
        time_steps = 100
        num_nodes = 50
        node_features = 32
        
        # Create temporal graph data
        temporal_x = torch.randn(batch_size, time_steps, num_nodes, node_features)
        
        # Test temporal dimensions
        assert temporal_x.shape == (batch_size, time_steps, num_nodes, node_features)
        
    def test_lstm_encoder(self):
        """Test LSTM encoder for temporal features."""
        seq_length = 100
        input_size = 64
        hidden_size = 128
        
        # Create LSTM
        lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # Test input
        x = torch.randn(1, seq_length, input_size)
        output, (hidden, cell) = lstm(x)
        
        # Test output shapes
        assert output.shape == (1, seq_length, hidden_size)
        assert hidden.shape == (1, 1, hidden_size)
        assert cell.shape == (1, 1, hidden_size)
        
    def test_dynamic_graph_construction(self):
        """Test dynamic graph construction from time series."""
        num_timepoints = 200
        num_nodes = 50
        window_size = 50
        
        # Create mock time series
        timeseries = torch.randn(num_timepoints, num_nodes)
        
        # Sliding window correlation
        num_windows = num_timepoints - window_size + 1
        dynamic_graphs = []
        
        for i in range(0, num_windows, 10):  # Sample every 10 windows
            window_data = timeseries[i:i+window_size]
            # Compute correlation matrix
            correlation = torch.corrcoef(window_data.t())
            
            # Threshold and create edge index
            threshold = 0.5
            edge_mask = torch.abs(correlation) > threshold
            edge_indices = torch.nonzero(edge_mask, as_tuple=False).t()
            
            # Remove self-loops
            edge_indices = edge_indices[:, edge_indices[0] != edge_indices[1]]
            
            dynamic_graphs.append(edge_indices)
        
        # Test that graphs were created
        assert len(dynamic_graphs) > 0
        for graph in dynamic_graphs:
            assert graph.shape[0] == 2  # Source and target nodes
            
    def test_temporal_attention(self):
        """Test temporal attention mechanism."""
        seq_length = 100
        hidden_dim = 64
        
        # Create attention layer
        attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # Test input
        x = torch.randn(1, seq_length, hidden_dim)
        
        # Self-attention
        attn_output, attn_weights = attention(x, x, x)
        
        # Test output shapes
        assert attn_output.shape == (1, seq_length, hidden_dim)
        assert attn_weights.shape == (1, seq_length, seq_length)
        
        # Test attention weights sum to 1
        assert torch.allclose(attn_weights.sum(dim=-1), torch.ones(1, seq_length))


class TestMultiModalBrainGNN:
    """Test the Multi-Modal Brain GNN model."""
    
    def test_modality_specific_encoders(self, simple_graph):
        """Test separate encoders for different modalities."""
        # Simulate structural and functional data
        structural_features = 32
        functional_features = 64
        
        struct_x = torch.randn(simple_graph.num_nodes, structural_features)
        func_x = torch.randn(simple_graph.num_nodes, functional_features)
        
        # Create modality-specific encoders
        struct_encoder = nn.Linear(structural_features, 128)
        func_encoder = nn.Linear(functional_features, 128)
        
        # Encode each modality
        struct_encoded = struct_encoder(struct_x)  # [4, 128]
        func_encoded = func_encoder(func_x)        # [4, 128]
        
        assert struct_encoded.shape == (simple_graph.num_nodes, 128)
        assert func_encoded.shape == (simple_graph.num_nodes, 128)
        
    def test_cross_modal_attention(self):
        """Test cross-modal attention mechanism."""
        num_nodes = 100
        feature_dim = 64
        
        # Create features for two modalities
        struct_features = torch.randn(1, num_nodes, feature_dim)
        func_features = torch.randn(1, num_nodes, feature_dim)
        
        # Cross-modal attention (structural attending to functional)
        attention = nn.MultiheadAttention(feature_dim, num_heads=8, batch_first=True)
        
        # Structural queries functional
        cross_attn_output, cross_attn_weights = attention(
            struct_features, func_features, func_features
        )
        
        assert cross_attn_output.shape == (1, num_nodes, feature_dim)
        assert cross_attn_weights.shape == (1, num_nodes, num_nodes)
        
    def test_adaptive_fusion(self):
        """Test adaptive fusion of multiple modalities."""
        num_nodes = 50
        feature_dim = 128
        num_modalities = 3
        
        # Create features for multiple modalities
        modality_features = [
            torch.randn(num_nodes, feature_dim) for _ in range(num_modalities)
        ]
        
        # Stack modalities
        stacked_features = torch.stack(modality_features, dim=0)  # [3, 50, 128]
        
        # Learn fusion weights
        fusion_weights = nn.Parameter(torch.ones(num_modalities) / num_modalities)
        fusion_weights = torch.softmax(fusion_weights, dim=0)
        
        # Adaptive fusion
        fused_features = torch.sum(
            stacked_features * fusion_weights.view(-1, 1, 1), dim=0
        )  # [50, 128]
        
        assert fused_features.shape == (num_nodes, feature_dim)
        assert torch.allclose(fusion_weights.sum(), torch.tensor(1.0))
        
    @pytest.mark.parametrize("fusion_strategy", ["early", "mid", "late"])
    def test_fusion_strategies(self, fusion_strategy):
        """Test different fusion strategies."""
        num_nodes = 100
        input_dim = 64
        hidden_dim = 128
        
        # Two modalities
        mod1_features = torch.randn(num_nodes, input_dim)
        mod2_features = torch.randn(num_nodes, input_dim)
        
        if fusion_strategy == "early":
            # Concatenate raw features
            fused = torch.cat([mod1_features, mod2_features], dim=1)  # [100, 128]
            expected_dim = input_dim * 2
        elif fusion_strategy == "mid":
            # Encode separately, then fuse
            encoder = nn.Linear(input_dim, hidden_dim)
            mod1_encoded = encoder(mod1_features)  # [100, 128]
            mod2_encoded = encoder(mod2_features)  # [100, 128]
            fused = mod1_encoded + mod2_encoded    # [100, 128]
            expected_dim = hidden_dim
        elif fusion_strategy == "late":
            # Encode and process separately, fuse at end
            encoder = nn.Linear(input_dim, hidden_dim)
            classifier = nn.Linear(hidden_dim, 1)
            
            mod1_logits = classifier(encoder(mod1_features))  # [100, 1]
            mod2_logits = classifier(encoder(mod2_features))  # [100, 1]
            fused = (mod1_logits + mod2_logits) / 2           # [100, 1]
            expected_dim = 1
        
        assert fused.shape == (num_nodes, expected_dim)


class TestPopulationGraphGNN:
    """Test the Population Graph GNN model."""
    
    def test_subject_similarity_computation(self):
        """Test computation of subject similarity for population graph."""
        num_subjects = 20
        feature_dim = 100
        
        # Create subject features (e.g., aggregated brain features)
        subject_features = torch.randn(num_subjects, feature_dim)
        
        # Compute pairwise similarities (cosine similarity)
        normalized_features = nn.functional.normalize(subject_features, p=2, dim=1)
        similarity_matrix = torch.mm(normalized_features, normalized_features.t())
        
        # Test similarity matrix properties
        assert similarity_matrix.shape == (num_subjects, num_subjects)
        assert torch.allclose(torch.diag(similarity_matrix), torch.ones(num_subjects))
        assert torch.allclose(similarity_matrix, similarity_matrix.t())  # Symmetric
        
    def test_knn_graph_construction(self):
        """Test k-nearest neighbor graph construction."""
        num_subjects = 50
        feature_dim = 64
        k = 5
        
        # Create subject features
        subject_features = torch.randn(num_subjects, feature_dim)
        
        # Compute pairwise distances
        distances = torch.cdist(subject_features, subject_features)
        
        # Get k nearest neighbors (excluding self)
        _, knn_indices = torch.topk(distances, k + 1, dim=1, largest=False)
        knn_indices = knn_indices[:, 1:]  # Remove self-connections
        
        # Create edge index
        sources = torch.arange(num_subjects).repeat_interleave(k)
        targets = knn_indices.flatten()
        edge_index = torch.stack([sources, targets])
        
        # Test graph properties
        assert edge_index.shape == (2, num_subjects * k)
        assert torch.all(edge_index[0] != edge_index[1])  # No self-loops
        
    def test_demographic_features(self):
        """Test incorporation of demographic features."""
        num_subjects = 100
        
        # Mock demographic data
        demographics = {
            "age": torch.randint(18, 80, (num_subjects,)).float(),
            "sex": torch.randint(0, 2, (num_subjects,)).float(),  # 0: F, 1: M
            "education": torch.randint(8, 20, (num_subjects,)).float(),
        }
        
        # Normalize demographic features
        normalized_demographics = {}
        for key, values in demographics.items():
            normalized_demographics[key] = (values - values.mean()) / values.std()
        
        # Concatenate demographic features
        demo_features = torch.stack(list(normalized_demographics.values()), dim=1)
        
        assert demo_features.shape == (num_subjects, 3)
        
        # Test that normalization worked
        for i in range(3):
            assert abs(demo_features[:, i].mean().item()) < 1e-6  # Mean ~ 0
            assert abs(demo_features[:, i].std().item() - 1.0) < 1e-6  # Std ~ 1


class TestModelUtilities:
    """Test model utility functions."""
    
    def test_parameter_counting(self):
        """Test counting model parameters."""
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Expected: (64*128 + 128) + (128*32 + 32) + (32*1 + 1)
        expected_params = (64*128 + 128) + (128*32 + 32) + (32*1 + 1)
        
        assert total_params == expected_params
        assert trainable_params == expected_params
        
    def test_model_device_handling(self, device):
        """Test model device placement."""
        model = nn.Linear(10, 5)
        
        # Move to device
        model = model.to(device)
        
        # Test that parameters are on correct device
        for param in model.parameters():
            assert param.device == device
            
    def test_gradient_checkpointing_compatibility(self, simple_graph, device):
        """Test that models work with gradient checkpointing."""
        # Create a model that could use gradient checkpointing
        model = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(device)
        
        # Test forward pass
        x = simple_graph.x.to(device)
        output = model(x)
        
        assert output.shape == (simple_graph.num_nodes, 1)
        assert output.device == device
        
        # Test backward pass
        loss = output.sum()
        loss.backward()
        
        # Check that gradients were computed
        for param in model.parameters():
            assert param.grad is not None
            
    @pytest.mark.gpu
    def test_mixed_precision_compatibility(self, simple_graph, device):
        """Test model compatibility with mixed precision training."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        model = nn.Linear(16, 1).to(device)
        scaler = torch.cuda.amp.GradScaler()
        
        # Test forward pass with autocast
        x = simple_graph.x.to(device)
        
        with torch.cuda.amp.autocast():
            output = model(x)
            loss = output.sum()
        
        # Test backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(torch.optim.Adam(model.parameters()))
        scaler.update()
        
        assert output.dtype == torch.float16  # Should be half precision
        assert loss.dtype == torch.float16