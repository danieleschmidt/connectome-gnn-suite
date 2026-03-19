"""
Training utilities for connectome GNN classifiers.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from connectome_gnn.graph import ConnectomeBatch, ConnectomeDataLoader


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Simple training loop for ConnectomeBatch-based GNN classifiers.

    Parameters
    ----------
    model     : GCNConnectome or GraphSAGEConnectome
    optimizer : any torch.optim.Optimizer
    device    : 'cpu' or 'cuda'
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()

    def train_epoch(self, loader: ConnectomeDataLoader) -> float:
        """Run one training epoch. Returns mean loss."""
        self.model.train()
        total_loss, total_samples = 0.0, 0
        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(batch)
            loss = self.loss_fn(logits, batch.labels)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss.detach()) * batch.num_graphs
            total_samples += batch.num_graphs
        return total_loss / max(total_samples, 1)

    @torch.no_grad()
    def evaluate(self, loader: ConnectomeDataLoader) -> dict:
        """Evaluate on loader. Returns accuracy and mean loss."""
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for batch in loader:
            batch = batch.to(self.device)
            logits = self.model(batch)
            loss = self.loss_fn(logits, batch.labels)
            preds = logits.argmax(dim=1)
            correct += int((preds == batch.labels).sum())
            total += batch.num_graphs
            total_loss += float(loss) * batch.num_graphs
        return {
            "accuracy": correct / max(total, 1),
            "loss": total_loss / max(total, 1),
            "correct": correct,
            "total": total,
        }

    def fit(
        self,
        train_loader: ConnectomeDataLoader,
        val_loader: ConnectomeDataLoader,
        num_epochs: int = 50,
        patience: int = 10,
        verbose: bool = True,
    ) -> dict:
        """
        Train with early stopping on validation loss.

        Returns
        -------
        history : dict with lists 'train_loss', 'val_loss', 'val_acc'
        """
        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        best_val_loss = float("inf")
        best_epoch = 0
        best_state = None

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])

            if verbose:
                print(
                    f"Epoch {epoch:3d} | "
                    f"train_loss={train_loss:.4f} | "
                    f"val_loss={val_metrics['loss']:.4f} | "
                    f"val_acc={val_metrics['accuracy']:.3f}"
                )

            # Early stopping
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            if epoch - best_epoch >= patience:
                if verbose:
                    print(f"Early stop at epoch {epoch} (best={best_epoch})")
                break

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return history
