"""Command-line interface for Connectome-GNN-Suite."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeometricDataLoader

from .data import ConnectomeDataset
from .models import (
    HierarchicalBrainGNN,
    TemporalConnectomeGNN,
    MultiModalBrainGNN,
    PopulationGraphGNN
)
from .tasks import (
    CognitiveScorePrediction,
    SubjectClassification,
    AgeRegression,
    ConnectomeTaskSuite
)
from .training import ConnectomeTrainer
from .utils import set_random_seed, get_device


def train():
    """Training command entry point."""
    parser = argparse.ArgumentParser(
        description="Train connectome GNN models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument("--data-root", type=str, required=True,
                       help="Path to dataset root directory")
    parser.add_argument("--dataset", type=str, default="connectome",
                       choices=["connectome", "hcp", "ukb", "custom"],
                       help="Dataset type")
    parser.add_argument("--resolution", type=str, default="7mm",
                       choices=["3mm", "5mm", "7mm"],
                       help="Spatial resolution")
    parser.add_argument("--modality", type=str, default="structural",
                       choices=["structural", "functional", "both"],
                       help="Connectivity modality")
    parser.add_argument("--parcellation", type=str, default="AAL",
                       choices=["AAL", "Schaefer400", "DKT"],
                       help="Brain parcellation atlas")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="hierarchical",
                       choices=["hierarchical", "temporal", "multimodal", "population"],
                       help="Model architecture")
    parser.add_argument("--hidden-dim", type=int, default=256,
                       help="Hidden layer dimension")
    parser.add_argument("--num-layers", type=int, default=3,
                       help="Number of GNN layers")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout probability")
    parser.add_argument("--batch-norm", action="store_true",
                       help="Use batch normalization")
    
    # Task arguments
    parser.add_argument("--task", type=str, default="fluid_intelligence",
                       help="Prediction task")
    parser.add_argument("--task-type", type=str, default="regression",
                       choices=["regression", "classification", "multi_class"],
                       help="Type of prediction task")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                       help="Weight decay")
    parser.add_argument("--patience", type=int, default=20,
                       help="Early stopping patience")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="./outputs",
                       help="Output directory")
    parser.add_argument("--experiment-name", type=str, default="connectome_experiment",
                       help="Experiment name")
    parser.add_argument("--save-best", action="store_true",
                       help="Save best model")
    
    # System arguments
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda"],
                       help="Device to use")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of data loader workers")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--config", type=str,
                       help="JSON config file")
    
    args = parser.parse_args()
    
    # Load config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        # Override args with config values
        for key, value in config.items():
            if hasattr(args, key.replace('-', '_')):
                setattr(args, key.replace('-', '_'), value)
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    try:
        # Load dataset
        print(f"Loading {args.dataset} dataset from {args.data_root}")
        dataset = ConnectomeDataset(
            root=args.data_root,
            resolution=args.resolution,
            modality=args.modality,
            parcellation=args.parcellation,
            download=True
        )
        
        print(f"Dataset loaded: {len(dataset)} subjects")
        
        # Create task
        task_suite = ConnectomeTaskSuite()
        if args.task == "fluid_intelligence":
            task = CognitiveScorePrediction(target="fluid_intelligence")
        elif args.task == "age":
            task = AgeRegression()
        elif args.task == "sex":
            task = SubjectClassification(target="sex", positive_class="M")
        else:
            # Try to create task from suite
            try:
                task = task_suite.get_task("cognitive_prediction", args.task)
            except:
                task = CognitiveScorePrediction(target=args.task)
        
        print(f"Task: {task.get_task_info()}")
        
        # Get sample data to determine input dimensions
        sample_data = dataset[0]
        node_features = sample_data.x.size(1) if sample_data.x is not None else 100
        
        # Create model
        print(f"Creating {args.model} model")
        if args.model == "hierarchical":
            model = HierarchicalBrainGNN(
                node_features=node_features,
                hidden_dim=args.hidden_dim,
                num_classes=task.num_classes,
                num_levels=args.num_layers,
                parcellation=args.parcellation,
                dropout=args.dropout,
                batch_norm=args.batch_norm
            )
        elif args.model == "temporal":
            model = TemporalConnectomeGNN(
                node_features=node_features,
                time_steps=200,  # Default time steps
                hidden_dim=args.hidden_dim,
                num_classes=task.num_classes,
                dropout=args.dropout
            )
        elif args.model == "multimodal":
            model = MultiModalBrainGNN(
                structural_features=node_features,
                functional_features=node_features,
                hidden_dim=args.hidden_dim,
                num_classes=task.num_classes,
                dropout=args.dropout
            )
        elif args.model == "population":
            model = PopulationGraphGNN(
                node_features=node_features,
                hidden_dim=args.hidden_dim,
                num_classes=task.num_classes,
                dropout=args.dropout
            )
        else:
            raise ValueError(f"Unknown model: {args.model}")
        
        model = model.to(device)
        print(f"Model created: {model.count_parameters()} parameters")
        
        # Create trainer
        trainer = ConnectomeTrainer(
            model=model,
            task=task,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            device=device,
            output_dir=output_dir
        )
        
        # Train model
        print("Starting training...")
        trainer.fit(dataset, epochs=args.epochs)
        
        print("Training completed successfully!")
        
        # Save final model if requested
        if args.save_best:
            model_path = output_dir / "best_model.pth"
            trainer.save_best_model(model_path)
            print(f"Best model saved to {model_path}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def evaluate():
    """Evaluation command entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate connectome GNN models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--config-path", type=str,
                       help="Path to model config (auto-detect if not provided)")
    
    # Data arguments  
    parser.add_argument("--data-root", type=str, required=True,
                       help="Path to dataset root directory")
    parser.add_argument("--split", type=str, default="test",
                       choices=["train", "val", "test"],
                       help="Dataset split to evaluate")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="./evaluation",
                       help="Output directory for evaluation results")
    parser.add_argument("--save-predictions", action="store_true",
                       help="Save model predictions")
    
    # System arguments
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size for evaluation")
    
    args = parser.parse_args()
    
    # Get device
    device = get_device(args.device)
    
    try:
        # Load model config
        if args.config_path:
            config_path = args.config_path
        else:
            # Try to find config in same directory as model
            model_dir = Path(args.model_path).parent
            config_path = model_dir / "config.json"
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"Loaded config from {config_path}")
        
        # Load dataset
        dataset = ConnectomeDataset(
            root=args.data_root,
            resolution=config.get('resolution', '7mm'),
            modality=config.get('modality', 'structural'),
            parcellation=config.get('parcellation', 'AAL')
        )
        
        # Create task
        task_name = config.get('task', 'fluid_intelligence')
        if task_name == "fluid_intelligence":
            task = CognitiveScorePrediction(target="fluid_intelligence")
        elif task_name == "age":
            task = AgeRegression()
        else:
            task = CognitiveScorePrediction(target=task_name)
        
        # Load model
        checkpoint = torch.load(args.model_path, map_location=device)
        
        # Create model instance
        model_type = config.get('model', 'hierarchical')
        node_features = dataset[0].x.size(1) if dataset[0].x is not None else 100
        
        if model_type == "hierarchical":
            model = HierarchicalBrainGNN(
                node_features=node_features,
                hidden_dim=config.get('hidden_dim', 256),
                num_classes=task.num_classes,
                parcellation=config.get('parcellation', 'AAL')
            )
        else:
            raise ValueError(f"Model type {model_type} not supported in evaluation")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        print(f"Model loaded from {args.model_path}")
        
        # Create data loader
        data_loader = GeometricDataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # Evaluate
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(device)
                predictions = model(batch)
                targets = task.prepare_targets([batch])
                
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        # Combine results
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        # Compute metrics
        metrics = task.compute_metrics(predictions, targets)
        
        print("\n=== Evaluation Results ===")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'metrics': metrics,
            'config': config,
            'model_path': args.model_path,
            'data_root': args.data_root,
            'split': args.split
        }
        
        results_path = output_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {results_path}")
        
        # Save predictions if requested
        if args.save_predictions:
            predictions_path = output_dir / "predictions.pt"
            torch.save({
                'predictions': predictions,
                'targets': targets
            }, predictions_path)
            print(f"Predictions saved to {predictions_path}")
        
    except Exception as e:
        print(f"Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def preprocess():
    """Preprocessing command entry point."""
    parser = argparse.ArgumentParser(
        description="Preprocess connectome data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--input-dir", type=str, required=True,
                       help="Input data directory")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory")
    parser.add_argument("--parcellation", type=str, default="AAL",
                       help="Brain parcellation")
    parser.add_argument("--edge-threshold", type=float, default=0.01,
                       help="Edge threshold")
    parser.add_argument("--normalization", type=str, default="log_transform",
                       choices=["log_transform", "z_score", "minmax", "none"],
                       help="Normalization method")
    
    args = parser.parse_args()
    
    try:
        from .data.preprocessing import ConnectomeProcessor
        
        processor = ConnectomeProcessor(
            parcellation=args.parcellation,
            edge_threshold=args.edge_threshold,
            normalization=args.normalization if args.normalization != "none" else None
        )
        
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process all connectivity files in input directory
        connectivity_files = list(input_dir.glob("*connectivity*.csv")) + \
                           list(input_dir.glob("*connectivity*.npy"))
        
        print(f"Found {len(connectivity_files)} connectivity files")
        
        for file_path in connectivity_files:
            print(f"Processing {file_path.name}...")
            
            # Load connectivity matrix
            if file_path.suffix == '.csv':
                import pandas as pd
                connectivity = pd.read_csv(file_path, header=None).values
            else:
                connectivity = np.load(file_path)
            
            # Process
            processed = processor.process(connectivity, subject_id=file_path.stem)
            
            # Save processed data
            output_path = output_dir / f"{file_path.stem}_processed.npy"
            np.save(output_path, processed['connectivity'])
        
        print(f"Preprocessing completed. Results saved to {output_dir}")
        
    except Exception as e:
        print(f"Preprocessing failed with error: {e}")
        sys.exit(1)


def visualize():
    """Visualization command entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize connectome data and results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--data-path", type=str, required=True,
                       help="Path to connectome data")
    parser.add_argument("--output-dir", type=str, default="./visualizations",
                       help="Output directory")
    parser.add_argument("--plot-type", type=str, default="connectome",
                       choices=["connectome", "brain", "matrix", "graph"],
                       help="Type of visualization")
    parser.add_argument("--threshold", type=float, default=0.1,
                       help="Edge threshold for visualization")
    
    args = parser.parse_args()
    
    print(f"Visualization functionality will be implemented in Generation 2")
    print(f"Requested: {args.plot_type} plot of {args.data_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Connectome-GNN-Suite CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands',
        required=True
    )
    
    # Add subcommands
    subparsers.add_parser('train', help='Train connectome GNN models')
    subparsers.add_parser('evaluate', help='Evaluate trained models')
    subparsers.add_parser('preprocess', help='Preprocess connectome data')
    subparsers.add_parser('visualize', help='Visualize connectome data')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train()
    elif args.command == 'evaluate':
        evaluate()
    elif args.command == 'preprocess':
        preprocess()
    elif args.command == 'visualize':
        visualize()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()