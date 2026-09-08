"""
Model training pipeline for Aqua-Predict-NB.
Implements the training loop with validation, early stopping,
AdamW optimizer, gradient clipping, and checkpoint persistence.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.config import (
    CHECKPOINT_DIR,
    model_config
)
from src.models.lstm_forecaster import AquaLSTMForecaster


class ModelTrainer:
    """Manages model training, evaluation, and checkpoint saving."""

    def __init__(
        self,
        model: AquaLSTMForecaster,
        device: str = "cpu",
        checkpoint_dir: Path = CHECKPOINT_DIR
    ):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(
        self,
        dataloader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ) -> float:
        """Trains model for a single epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            x_seq = batch["x_seq"].to(self.device)
            x_static = batch["x_static"].to(self.device)
            y_target = batch["y_target"].to(self.device)

            optimizer.zero_grad()
            preds = self.model(x_seq, x_static)
            loss = criterion(preds, y_target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def validate(
        self,
        dataloader,
        criterion: nn.Module
    ) -> float:
        """Evaluates model loss on validation dataset."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                x_seq = batch["x_seq"].to(self.device)
                x_static = batch["x_static"].to(self.device)
                y_target = batch["y_target"].to(self.device)

                preds = self.model(x_seq, x_static)
                loss = criterion(preds, y_target)
                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int = 35,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 8
    ) -> Dict:
        """
        Executes full training process with early stopping.
        """
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        criterion = nn.MSELoss()

        history = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
            "best_epoch": 0,
            "best_val_loss": float("inf")
        }

        best_val_loss = float("inf")
        patience_counter = 0
        best_checkpoint_path = self.checkpoint_dir / "best_model.pt"

        print(f">>> Starting LSTM training ({epochs} epochs, lr={lr}, patience={patience})...")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            val_loss = self.validate(val_loader, criterion)
            current_lr = optimizer.param_groups[0]["lr"]

            scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["lr"].append(current_lr)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                history["best_val_loss"] = best_val_loss
                history["best_epoch"] = epoch
                patience_counter = 0

                # Save checkpoint
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "model_config": {
                        "seq_feature_dim": self.model.seq_feature_dim,
                        "static_feature_dim": self.model.static_feature_dim,
                        "hidden_dim": self.model.hidden_dim,
                        "num_layers": self.model.num_layers,
                        "forecast_horizon": self.model.forecast_horizon,
                        "dropout": self.model.dropout_rate
                    }
                }, best_checkpoint_path)
            else:
                patience_counter += 1

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")

            if patience_counter >= patience:
                print(f"  [EARLY STOPPING] Triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.4f} at epoch {history['best_epoch']}.")
                break

        # Save training history JSON
        with open(self.checkpoint_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        print(f"[SUCCESS] Best model checkpoint saved to: {best_checkpoint_path}")
        return history
