#!/usr/bin/env python3
"""Basic functionality demo for Connectome-GNN-Suite."""

import torch
import sys
import os

# Add repo to path
sys.path.insert(0, '.')

# Import connectome GNN components
import connectome_gnn
from connectome_gnn.data import ConnectomeDataset, HCPLoader
from connectome_gnn.models import HierarchicalBrainGNN
from connectome_gnn.tasks import CognitiveScorePrediction
from connectome_gnn.training import ConnectomeTrainer


def demo_basic_workflow():
    """Demonstrate basic connectome GNN workflow."""
    print("🧠 Connectome-GNN-Suite Basic Demo")
    print("=" * 50)
    
    # Check available components
    components = connectome_gnn.get_available_components()
    print(f"Available components: {components}")
    
    # Set random seed for reproducibility
    connectome_gnn.set_random_seed(42)
    
    # Create dataset
    print("\n📊 Creating synthetic connectome dataset...")
    dataset = ConnectomeDataset(
        root="demo_data",
        resolution="7mm",
        modality="structural",
        download=True
    )
    print(f"Dataset created with {len(dataset)} samples")
    
    # Examine first sample
    sample = dataset[0]
    print(f"Sample info:")
    print(f"  - Nodes: {sample.num_nodes}")
    print(f"  - Edges: {sample.edge_index.size(1)}")
    print(f"  - Node features: {sample.x.shape}")
    print(f"  - Target: {sample.y.item():.4f}")
    
    # Create model
    print("\n🔬 Creating Hierarchical Brain GNN...")
    model = HierarchicalBrainGNN(
        node_features=sample.x.size(1),
        hidden_dim=64,  # Smaller for demo
        num_levels=3,
        num_classes=1,
        dropout=0.1
    )
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create task
    print("\n🎯 Setting up cognitive score prediction task...")
    task = CognitiveScorePrediction(
        target="y",  # Use the y attribute from synthetic data
        normalize=True
    )
    print(f"Task: {task.task_name} ({task.task_type})")
    
    # Create trainer
    print("\n🏃 Setting up trainer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    trainer = ConnectomeTrainer(
        model=model,
        task=task,
        device=device,
        batch_size=8,  # Small batch for demo
        learning_rate=0.01,
        early_stopping=True,
        patience=5
    )
    
    # Train model
    print("\n🔧 Training model...")
    try:
        history = trainer.fit(
            dataset=dataset,
            epochs=10,  # Short training for demo
            validation_split=0.2,
            verbose=True
        )
        
        print("\n✅ Training completed successfully!")
        print(f"Final train loss: {history['train_loss'][-1]:.4f}")
        if history['val_loss']:
            print(f"Final val loss: {history['val_loss'][-1]:.4f}")
        
        # Test prediction on a sample
        print("\n🔮 Testing prediction...")
        model.eval()
        with torch.no_grad():
            sample_batch = dataset[0:3]  # Get 3 samples
            predictions = model(sample_batch.to(device))
            targets = sample_batch.y.to(device)
            
            print("Predictions vs Targets:")
            for i, (pred, target) in enumerate(zip(predictions, targets)):
                print(f"  Sample {i}: {pred.item():.4f} vs {target.item():.4f}")
        
        print("\n🎉 Demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return False


def demo_data_loading():
    """Demo data loading capabilities."""
    print("\n📚 Data Loading Demo")
    print("-" * 30)
    
    # HCP Loader
    loader = HCPLoader(
        root="demo_data",
        resolution="7mm",
        batch_size=4
    )
    
    stats = loader.get_stats()
    print("Dataset statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Get data splits
    splits = loader.get_train_val_test_loaders()
    print(f"\nData splits:")
    for split_name, dataloader in splits.items():
        print(f"  {split_name}: {len(dataloader.dataset)} samples")


def demo_model_variants():
    """Demo different model architectures."""
    print("\n🏗️  Model Architecture Demo")
    print("-" * 30)
    
    # Create sample data
    dataset = ConnectomeDataset(root="demo_data", download=True)
    sample = dataset[0]
    
    # Test different models
    models = {
        'HierarchicalBrainGNN': HierarchicalBrainGNN(
            node_features=sample.x.size(1),
            hidden_dim=32,
            num_levels=2
        )
    }
    
    for name, model in models.items():
        print(f"\n{name}:")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        try:
            with torch.no_grad():
                output = model(sample)
                print(f"  Output shape: {output.shape}")
                print(f"  ✅ Forward pass successful")
        except Exception as e:
            print(f"  ❌ Forward pass failed: {e}")


if __name__ == "__main__":
    print("🚀 Starting Connectome-GNN-Suite Demo")
    
    # Demo basic workflow
    success = demo_basic_workflow()
    
    if success:
        # Additional demos
        demo_data_loading() 
        demo_model_variants()
        
        print("\n🎊 All demos completed successfully!")
        print("The Connectome-GNN-Suite is working correctly!")
    else:
        print("\n⚠️  Basic demo failed - please check the setup")
        sys.exit(1)