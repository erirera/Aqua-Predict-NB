"""
PyTorch LSTM Architecture with Static Covariate Fusion and Monte Carlo Dropout.
Designed for multi-step (1-6 month) groundwater level forecasting in New Brunswick.
"""

from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import model_config


class AquaLSTMForecaster(nn.Module):
    """
    Multivariate Sequence-to-Sequence LSTM forecaster:
    - Encodes 24-month temporal dynamics (precipitation, temperature, cyclical lags).
    - Fuses temporal latent vectors with static geological & well construction covariates.
    - Projects to multi-step future groundwater levels (1 to 6 months).
    """

    def __init__(
        self,
        seq_feature_dim: int,
        static_feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        forecast_horizon: int = 6,
        dropout: float = 0.2
    ):
        super().__init__()
        self.seq_feature_dim = seq_feature_dim
        self.static_feature_dim = static_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.dropout_rate = dropout

        # Temporal Sequence Encoder
        self.lstm = nn.LSTM(
            input_size=seq_feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        # Static Covariate Embedding MLP
        self.static_embed_dim = 32
        if static_feature_dim > 0:
            self.static_mlp = nn.Sequential(
                nn.Linear(static_feature_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, self.static_embed_dim),
                nn.ReLU()
            )
            fusion_dim = hidden_dim + self.static_embed_dim
        else:
            self.static_mlp = None
            fusion_dim = hidden_dim

        # Forecast Projection Head
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 96),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(48, forecast_horizon)
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        x_static: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass:
        - x_seq: (batch_size, seq_len, seq_feature_dim)
        - x_static: (batch_size, static_feature_dim)
        Returns:
        - out: (batch_size, forecast_horizon)
        """
        lstm_out, (hn, cn) = self.lstm(x_seq)
        # Take final hidden state from top layer
        last_hidden = hn[-1]  # (batch_size, hidden_dim)

        if self.static_mlp is not None and x_static is not None and x_static.shape[1] > 0:
            static_rep = self.static_mlp(x_static)
            fused = torch.cat([last_hidden, static_rep], dim=-1)
        else:
            fused = last_hidden

        forecast = self.head(fused)
        return forecast


def enable_monte_carlo_dropout(model: nn.Module) -> None:
    """
    Enables dropout layers during inference while keeping BatchNorm / LayerNorm in eval mode.
    Key to Bayesian Monte Carlo Dropout approximation.
    """
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


class MonteCarloDropoutPredictor:
    """
    Wraps AquaLSTMForecaster to perform stochastic forward passes (N=100).
    Produces mean prediction, standard deviation, and P10/P50/P90 percentile bounds.
    """

    def __init__(self, model: AquaLSTMForecaster, n_samples: int = 100, device: str = "cpu"):
        self.model = model
        self.n_samples = n_samples
        self.device = device
        self.model.to(self.device)

    def predict_with_uncertainty(
        self,
        x_seq: torch.Tensor,
        x_static: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """
        Executes n_samples forward passes with dropout active.
        """
        enable_monte_carlo_dropout(self.model)
        x_seq = x_seq.to(self.device)
        x_static = x_static.to(self.device)

        mc_predictions = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                pred = self.model(x_seq, x_static)
                mc_predictions.append(pred.cpu().numpy())

        # Shape: (n_samples, batch_size, forecast_horizon)
        mc_array = np.array(mc_predictions)

        p10 = np.percentile(mc_array, 10, axis=0)
        p50 = np.percentile(mc_array, 50, axis=0)
        p90 = np.percentile(mc_array, 90, axis=0)
        mean = np.mean(mc_array, axis=0)
        std = np.std(mc_array, axis=0)

        return {
            "p10": p10,      # High drought risk scenario
            "p50": p50,      # Median forecast
            "p90": p90,      # Low risk / recharge scenario
            "mean": mean,
            "std": std,
            "all_samples": mc_array
        }


class QuantileLoss(nn.Module):
    """
    Pinball loss for quantile regression.
    loss = max(q * (y - y_hat), (q - 1) * (y - y_hat))
    """

    def __init__(self, quantiles: Tuple[float, ...] = (0.10, 0.50, 0.90)):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        target: (batch_size, horizon)
        preds: (batch_size, horizon, num_quantiles)
        """
        losses = []
        for i, q in enumerate(self.quantiles):
            error = target - preds[..., i]
            q_loss = torch.max((q - 1) * error, q * error)
            losses.append(torch.mean(q_loss))
        return torch.sum(torch.stack(losses))
