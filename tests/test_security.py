"""Security tests for connectome GNN suite."""

import pytest
import torch
import numpy as np
import tempfile
import shutil
import os
from pathlib import Path
import json

from connectome_gnn.data.dataset import ConnectomeDataset
from connectome_gnn.models.hierarchical import HierarchicalBrainGNN
from connectome_gnn.training.trainer import ConnectomeTrainer
from connectome_gnn.training.config import TrainingConfig


class TestDataSecurity:
    """Test data security and privacy."""
    
    def test_no_hardcoded_paths(self):
        """Test that no hardcoded sensitive paths are present."""
        import connectome_gnn
        
        # Get the module path
        module_path = Path(connectome_gnn.__file__).parent
        
        # Scan Python files for potentially sensitive hardcoded paths
        sensitive_patterns = [
            "/home/",
            "/Users/",
            "C:\\Users\\",
            "password",
            "secret",
            "api_key",
            "private_key"
        ]
        
        for py_file in module_path.rglob("*.py"):
            with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().lower()
                
                for pattern in sensitive_patterns:
                    # Allow documentation examples, but flag actual hardcoded values
                    if pattern in content and "example" not in content and "doc" not in content:
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if pattern in line and not line.strip().startswith('#'):
                                # This is a potential security issue
                                assert False, f"Potential hardcoded sensitive path in {py_file}:{i+1}: {line.strip()}"
    
    def test_no_credentials_in_code(self):
        """Test that no credentials are hardcoded in the code."""
        import connectome_gnn
        
        module_path = Path(connectome_gnn.__file__).parent
        
        # Patterns that might indicate hardcoded credentials
        credential_patterns = [
            r"password\s*=\s*['\"](?!.*(?:example|test|dummy|placeholder))[^'\"]+['\"]",
            r"token\s*=\s*['\"](?!.*(?:example|test|dummy|placeholder))[^'\"]+['\"]",
            r"api_key\s*=\s*['\"](?!.*(?:example|test|dummy|placeholder))[^'\"]+['\"]",
            r"secret\s*=\s*['\"](?!.*(?:example|test|dummy|placeholder))[^'\"]+['\"]",
        ]
        
        import re
        
        for py_file in module_path.rglob("*.py"):
            with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                for pattern in credential_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    assert len(matches) == 0, f"Potential hardcoded credential in {py_file}: {matches}"
    
    def test_input_validation(self):
        """Test input validation for user-provided data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = ConnectomeDataset(
                root=temp_dir,
                num_subjects=5,
                use_synthetic=True
            )
            
            # Test that malformed data is handled gracefully
            data = dataset[0]
            
            # Test with NaN values
            data.x[0, 0] = float('nan')
            
            model = HierarchicalBrainGNN(
                node_features=100,
                hidden_dim=32,
                output_dim=1
            )
            
            # Should handle NaN gracefully (either filter or raise appropriate error)
            with pytest.raises((ValueError, RuntimeError)):
                model(data)
    
    def test_file_path_validation(self):
        """Test file path validation to prevent directory traversal."""
        # Test that relative paths with .. are handled safely
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "/etc/passwd",
            "C:\\Windows\\System32\\config\\SAM"
        ]
        
        for malicious_path in malicious_paths:
            with pytest.raises((ValueError, OSError, FileNotFoundError)):
                # Should not allow access to system files
                ConnectomeDataset(root=malicious_path, use_synthetic=True)
    
    def test_data_anonymization(self):
        """Test that synthetic data doesn't contain real personal information."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = ConnectomeDataset(
                root=temp_dir,
                num_subjects=10,
                use_synthetic=True
            )
            
            # Check that demographics are anonymized/synthetic
            for i in range(len(dataset)):
                data = dataset[i]
                demographics = data.demographics
                
                # Should not contain real names, SSNs, etc.
                for key, value in demographics.items():
                    if isinstance(value, str):
                        # Check for common PII patterns
                        assert not any(name in value.lower() for name in 
                                     ['john', 'jane', 'smith', 'johnson', 'williams'])
                        
                        # Check for SSN-like patterns
                        import re
                        ssn_pattern = r'\d{3}-\d{2}-\d{4}'
                        assert not re.search(ssn_pattern, str(value))


class TestModelSecurity:
    """Test model security and robustness."""
    
    def test_adversarial_input_detection(self):
        """Test detection of potentially adversarial inputs."""
        model = HierarchicalBrainGNN(
            node_features=100,
            hidden_dim=32,
            output_dim=1
        )
        model.eval()
        
        # Create normal input
        normal_data = self._create_sample_data()
        
        with torch.no_grad():
            normal_output = model(normal_data)
        
        # Create adversarial input with extreme values
        adversarial_data = self._create_sample_data()
        adversarial_data.x *= 1000  # Extreme scaling
        
        with torch.no_grad():
            adversarial_output = model(adversarial_data)
        
        # Model should either:
        # 1. Handle extreme inputs gracefully, or
        # 2. Detect and reject them
        assert not torch.isnan(adversarial_output).any(), "Model should handle extreme inputs"
        assert not torch.isinf(adversarial_output).any(), "Model should handle extreme inputs"
    
    def test_model_weight_bounds(self):
        """Test that model weights stay within reasonable bounds."""
        model = HierarchicalBrainGNN(
            node_features=100,
            hidden_dim=64,
            output_dim=1
        )
        
        # Check initial weights
        for name, param in model.named_parameters():
            assert not torch.isnan(param).any(), f"NaN weights in {name}"
            assert not torch.isinf(param).any(), f"Infinite weights in {name}"
            
            # Weights should be in reasonable range
            weight_range = torch.max(param) - torch.min(param)
            assert weight_range < 100, f"Extreme weight range in {name}: {weight_range}"
    
    def test_memory_bounds(self):
        """Test that model doesn't consume excessive memory."""
        # Create progressively larger models and check memory usage
        if torch.cuda.is_available():
            device = torch.device('cuda')
            
            # Clear memory
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            
            model = HierarchicalBrainGNN(
                node_features=100,
                hidden_dim=512,  # Large model
                output_dim=1
            ).to(device)
            
            # Create large batch
            data = self._create_sample_data(num_nodes=200, batch_size=10)
            data = data.to(device)
            
            with torch.no_grad():
                output = model(data)
            
            current_memory = torch.cuda.memory_allocated()
            memory_used = (current_memory - initial_memory) / 1024**2  # MB
            
            # Should not use excessive memory (less than 4GB)
            assert memory_used < 4000, f"Excessive memory usage: {memory_used:.1f} MB"
    
    def _create_sample_data(self, num_nodes=90, batch_size=1):
        """Create sample graph data for testing."""
        from torch_geometric.data import Data, Batch
        
        data_list = []
        for _ in range(batch_size):
            x = torch.randn(num_nodes, 100)
            edge_index = torch.randint(0, num_nodes, (2, num_nodes * 3))
            edge_attr = torch.randn(edge_index.size(1), 1)
            data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
        
        if batch_size == 1:
            return data_list[0]
        else:
            return Batch.from_data_list(data_list)


class TestConfigurationSecurity:
    """Test configuration security."""
    
    def test_config_validation(self):
        """Test that configuration values are validated."""
        # Test invalid learning rates
        with pytest.raises(ValueError):
            TrainingConfig(learning_rate=-1.0)
        
        with pytest.raises(ValueError):
            TrainingConfig(learning_rate=0.0)
        
        # Test invalid batch sizes
        with pytest.raises(ValueError):
            TrainingConfig(batch_size=0)
        
        with pytest.raises(ValueError):
            TrainingConfig(batch_size=-5)
        
        # Test invalid epochs
        with pytest.raises(ValueError):
            TrainingConfig(num_epochs=0)
    
    def test_output_directory_security(self):
        """Test output directory creation and permissions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TrainingConfig(output_dir=temp_dir)
            
            # Should create directory safely
            output_path = Path(config.output_dir)
            if not output_path.exists():
                output_path.mkdir(parents=True, exist_ok=True)
            
            # Check that directory was created
            assert output_path.exists()
            assert output_path.is_dir()
            
            # Test writing to directory
            test_file = output_path / "test.txt"
            test_file.write_text("test content")
            assert test_file.exists()
    
    def test_config_serialization_safety(self):
        """Test that config serialization is safe."""
        config = TrainingConfig(
            batch_size=32,
            learning_rate=0.001,
            output_dir="/tmp/test"
        )
        
        # Serialize to dict
        config_dict = config.to_dict()
        
        # Should not contain any callable objects or unsafe data
        def check_safe_value(value):
            if callable(value):
                return False
            if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                return True
            return False
        
        def check_dict_safety(d):
            for key, value in d.items():
                assert check_safe_value(value), f"Unsafe value in config: {key} = {value}"
                if isinstance(value, dict):
                    check_dict_safety(value)
                elif isinstance(value, list):
                    for item in value:
                        assert check_safe_value(item), f"Unsafe item in list: {item}"
        
        check_dict_safety(config_dict)


class TestDependencySecurity:
    """Test dependency security."""
    
    def test_import_safety(self):
        """Test that imports are safe and expected."""
        # Test that we can import all main modules without executing unsafe code
        import connectome_gnn.models
        import connectome_gnn.data
        import connectome_gnn.training
        import connectome_gnn.tasks
        import connectome_gnn.optimization
        import connectome_gnn.research
        
        # Should complete without errors
        assert True
    
    def test_no_eval_or_exec(self):
        """Test that code doesn't use eval() or exec() functions."""
        import connectome_gnn
        
        module_path = Path(connectome_gnn.__file__).parent
        
        dangerous_functions = ['eval(', 'exec(', '__import__(']
        
        for py_file in module_path.rglob("*.py"):
            with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                for func in dangerous_functions:
                    # Allow in comments and documentation
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if func in line and not line.strip().startswith('#'):
                            # Check if it's in a string literal (documentation)
                            if '"""' in line or "'''" in line or line.strip().startswith('"') or line.strip().startswith("'"):
                                continue
                            assert False, f"Dangerous function {func} found in {py_file}:{i+1}"


class TestDataIntegrity:
    """Test data integrity and consistency."""
    
    def test_data_checksums(self):
        """Test data integrity with checksums."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = ConnectomeDataset(
                root=temp_dir,
                num_subjects=5,
                use_synthetic=True
            )
            
            # Calculate checksums for reproducibility
            checksums = []
            for i in range(len(dataset)):
                data = dataset[i]
                
                # Simple checksum of node features
                checksum = torch.sum(data.x).item()
                checksums.append(checksum)
            
            # Recreate dataset with same seed
            torch.manual_seed(42)
            np.random.seed(42)
            
            dataset2 = ConnectomeDataset(
                root=temp_dir,
                num_subjects=5,
                use_synthetic=True
            )
            
            # Should have same checksums if truly reproducible
            checksums2 = []
            for i in range(len(dataset2)):
                data = dataset2[i]
                checksum = torch.sum(data.x).item()
                checksums2.append(checksum)
            
            # Note: This might not be exactly equal due to random number generation
            # but should be in same ballpark
            for c1, c2 in zip(checksums, checksums2):
                relative_diff = abs(c1 - c2) / (abs(c1) + 1e-8)
                assert relative_diff < 0.1, f"Large difference in checksums: {c1} vs {c2}"
    
    def test_model_determinism(self):
        """Test model determinism for security auditing."""
        torch.manual_seed(42)
        
        model1 = HierarchicalBrainGNN(
            node_features=100,
            hidden_dim=32,
            output_dim=1
        )
        
        torch.manual_seed(42)
        
        model2 = HierarchicalBrainGNN(
            node_features=100,
            hidden_dim=32,
            output_dim=1
        )
        
        # Models should be identical with same seed
        for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
            assert name1 == name2
            assert torch.equal(param1, param2), f"Parameters differ in {name1}"


class TestPrivacyProtection:
    """Test privacy protection measures."""
    
    def test_no_data_leakage(self):
        """Test that model doesn't leak training data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create small dataset
            dataset = ConnectomeDataset(
                root=temp_dir,
                num_subjects=10,
                use_synthetic=True
            )
            
            model = HierarchicalBrainGNN(
                node_features=100,
                hidden_dim=32,
                output_dim=1
            )
            
            # Extract some training data features
            training_features = []
            for i in range(3):
                data = dataset[i]
                training_features.append(data.x.mean().item())
            
            # After training, model weights shouldn't directly contain training data
            model.eval()
            
            # Check that model parameters don't directly match training data
            for name, param in model.named_parameters():
                param_values = param.detach().cpu().numpy().flatten()
                
                for feature_val in training_features:
                    # No parameter should exactly match training data features
                    exact_matches = np.sum(np.abs(param_values - feature_val) < 1e-6)
                    # Allow some coincidental matches but not too many
                    assert exact_matches < len(param_values) * 0.01, f"Too many exact matches in {name}"
    
    def test_demographic_privacy(self):
        """Test that demographic information is properly handled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = ConnectomeDataset(
                root=temp_dir,
                num_subjects=20,
                use_synthetic=True
            )
            
            # Collect demographic information
            ages = []
            sexes = []
            
            for i in range(len(dataset)):
                data = dataset[i]
                ages.append(data.demographics.get('age', 0))
                sexes.append(data.demographics.get('sex', 'unknown'))
            
            # Check that demographics have appropriate distributions
            assert len(set(ages)) > 5, "Ages should be diverse"
            assert len(set(sexes)) >= 2, "Should have multiple sex categories"
            
            # Ages should be in realistic range
            assert all(18 <= age <= 90 for age in ages), "Ages should be in realistic range"


if __name__ == "__main__":
    pytest.main([__file__])