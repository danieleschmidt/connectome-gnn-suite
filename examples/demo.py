#!/usr/bin/env python3
"""
connectome-gnn-suite demo
=========================
End-to-end example: generate synthetic connectome data, train GCN and
GraphSAGE classifiers, compare accuracy on a held-out test set.

Usage
-----
    ~/anaconda3/bin/python3 examples/demo.py

Expected output: test accuracy ~55-70% for each model (brain-behaviour
correlations are weak, so this is realistic).
"""

import sys
import os

# Allow running from repo root without installing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from connectome_gnn.synthetic import generate_dataset, small_world_stats
from connectome_gnn.graph import ConnectomeDataLoader
from connectome_gnn.models import GCNConnectome, GraphSAGEConnectome
from connectome_gnn.train import Trainer


def print_section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def main():
    torch.manual_seed(42)
    DEVICE = "cpu"
    NUM_SUBJECTS = 300
    NUM_REGIONS = 84  # Desikan-Killiany parcellation
    BATCH_SIZE = 16
    HIDDEN_DIM = 64
    EPOCHS = 30

    # ------------------------------------------------------------------
    # 1. Generate synthetic dataset
    # ------------------------------------------------------------------
    print_section("1. Generating synthetic connectome dataset")
    print(f"  {NUM_SUBJECTS} subjects × {NUM_REGIONS} brain regions")
    print("  Graph type: Watts-Strogatz small-world (k=8, β=0.15)")
    print("  Task: predict fluid intelligence (binary, above/below median)")

    graphs = generate_dataset(
        num_subjects=NUM_SUBJECTS,
        num_regions=NUM_REGIONS,
        k=8,
        beta=0.15,
        trait_idx=0,  # fluid intelligence
        seed=42,
    )

    # Print dataset stats
    g0 = graphs[0]
    print(f"\n  Example subject: {g0.subject_id}")
    print(f"    nodes = {g0.num_nodes}, edges = {g0.num_edges}, features/node = {g0.num_features}")
    print(f"    edge weight range: [{g0.edge_weight.min():.3f}, {g0.edge_weight.max():.3f}]")

    stats = small_world_stats(graphs[:20])
    print(f"\n  Small-world check (sample of 20 subjects):")
    print(f"    mean clustering coefficient = {stats['mean_clustering']:.3f}")
    print(f"    mean avg path length        = {stats['mean_avg_path_length']:.3f}")
    print("  (Real HCP data: clustering ~0.5, path length ~3-4)")

    label_counts = {0: 0, 1: 0}
    for g in graphs:
        label_counts[g.label.item()] += 1
    print(f"\n  Label balance: class 0 = {label_counts[0]}, class 1 = {label_counts[1]}")

    # ------------------------------------------------------------------
    # 2. Train/val/test split
    # ------------------------------------------------------------------
    print_section("2. Data split")
    n_train = int(0.7 * NUM_SUBJECTS)
    n_val   = int(0.15 * NUM_SUBJECTS)
    n_test  = NUM_SUBJECTS - n_train - n_val

    train_graphs = graphs[:n_train]
    val_graphs   = graphs[n_train:n_train + n_val]
    test_graphs  = graphs[n_train + n_val:]

    print(f"  train: {n_train}  |  val: {n_val}  |  test: {n_test}")

    train_loader = ConnectomeDataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = ConnectomeDataLoader(val_graphs,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = ConnectomeDataLoader(test_graphs,  batch_size=BATCH_SIZE, shuffle=False)

    # ------------------------------------------------------------------
    # 3. Train GCN
    # ------------------------------------------------------------------
    print_section("3. Training GCNConnectome")
    gcn = GCNConnectome(
        in_channels=g0.num_features,
        hidden_dim=HIDDEN_DIM,
        num_classes=2,
        num_layers=3,
        dropout=0.3,
    )
    n_params = sum(p.numel() for p in gcn.parameters())
    print(f"  Parameters: {n_params:,}")

    gcn_trainer = Trainer(gcn, torch.optim.Adam(gcn.parameters(), lr=1e-3, weight_decay=1e-4), device=DEVICE)
    gcn_history = gcn_trainer.fit(train_loader, val_loader, num_epochs=EPOCHS, patience=8, verbose=True)

    gcn_test = gcn_trainer.evaluate(test_loader)
    print(f"\n  GCN Test accuracy: {gcn_test['accuracy']:.3f}  ({gcn_test['correct']}/{gcn_test['total']})")

    # ------------------------------------------------------------------
    # 4. Train GraphSAGE
    # ------------------------------------------------------------------
    print_section("4. Training GraphSAGEConnectome")
    sage = GraphSAGEConnectome(
        in_channels=g0.num_features,
        hidden_dim=HIDDEN_DIM,
        num_classes=2,
        num_layers=3,
        dropout=0.3,
    )
    n_params = sum(p.numel() for p in sage.parameters())
    print(f"  Parameters: {n_params:,}")

    sage_trainer = Trainer(sage, torch.optim.Adam(sage.parameters(), lr=1e-3, weight_decay=1e-4), device=DEVICE)
    sage_history = sage_trainer.fit(train_loader, val_loader, num_epochs=EPOCHS, patience=8, verbose=True)

    sage_test = sage_trainer.evaluate(test_loader)
    print(f"\n  SAGE Test accuracy: {sage_test['accuracy']:.3f}  ({sage_test['correct']}/{sage_test['total']})")

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print_section("5. Results Summary")
    print(f"  {'Model':<20}  {'Test Acc':>10}  {'Val Loss':>10}")
    print(f"  {'-'*45}")
    best_gcn_val = min(gcn_history['val_loss'])
    best_sage_val = min(sage_history['val_loss'])
    print(f"  {'GCN':<20}  {gcn_test['accuracy']:>10.3f}  {best_gcn_val:>10.4f}")
    print(f"  {'GraphSAGE':<20}  {sage_test['accuracy']:>10.3f}  {best_sage_val:>10.4f}")
    print()
    print("  Note: ~55-70% accuracy is realistic for weak brain-behaviour")
    print("  correlations (r~0.2-0.3) typical in neuroimaging studies.")
    print("  Real HCP datasets with richer features achieve ~65-75%.")


if __name__ == "__main__":
    main()
