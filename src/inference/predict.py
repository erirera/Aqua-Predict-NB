"""
Inference module for Aqua-Predict-NB.
Loads model checkpoints and executes Bayesian Monte Carlo Dropout (N=100)
to generate multi-step groundwater forecasts with uncertainty envelopes.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import json
import numpy as np
import pandas as pd
import torch

from src.config import (
    CHECKPOINT_DIR,
    PROCESSED_DATA_DIR,
    model_config
)
from src.models.lstm_forecaster import AquaLSTMForecaster, MonteCarloDropoutPredictor


class WellForecastingEngine:
    """Performs batch Monte Carlo inference across New Brunswick wells."""

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        device: str = "cpu"
    ):
        self.device = device
        self.checkpoint_path = checkpoint_path or (CHECKPOINT_DIR / "best_model.pt")
        self.metadata_path = metadata_path or (PROCESSED_DATA_DIR / "features_metadata.json")

        # Load metadata
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)

        # Load scalers
        self.feature_scaler = joblib.load(PROCESSED_DATA_DIR / "feature_scaler.joblib")
        self.target_scaler = joblib.load(PROCESSED_DATA_DIR / "target_scaler.joblib")

        # Load model
        checkpoint = torch.load(self.checkpoint_path, map_location=device)
        cfg = checkpoint["model_config"]

        self.model = AquaLSTMForecaster(
            seq_feature_dim=cfg["seq_feature_dim"],
            static_feature_dim=cfg["static_feature_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            forecast_horizon=cfg["forecast_horizon"],
            dropout=cfg["dropout"]
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.predictor = MonteCarloDropoutPredictor(
            self.model,
            n_samples=model_config.mc_dropout_samples,
            device=device
        )

    def inverse_transform_targets(self, scaled_targets: np.ndarray) -> np.ndarray:
        """Converts scaled predictions back to actual metres below surface."""
        # target_scaler was fit on shape (N, 1)
        orig_shape = scaled_targets.shape
        flat = scaled_targets.reshape(-1, 1)
        inversed = self.target_scaler.inverse_transform(flat)
        return inversed.reshape(orig_shape)

    def forecast_all_wells_latest(
        self,
        processed_df: pd.DataFrame
    ) -> List[Dict]:
        """
        Extracts the most recent 24-month window for each well and computes 6-month forecast.
        """
        results = []
        scaled_feats = self.metadata["scaled_feature_columns"]
        scaled_static = [f"sc_{c}" for c in self.metadata["static_columns"]]
        lookback = model_config.lookback_window
        horizon = model_config.forecast_horizon

        for well_id, group in processed_df.groupby("well_id"):
            group = group.sort_values("date").reset_index(drop=True)
            if len(group) < lookback:
                continue

            # Latest lookback window
            latest_window = group.iloc[-lookback:]
            x_seq = torch.tensor(latest_window[scaled_feats].values, dtype=torch.float32).unsqueeze(0)
            x_static = torch.tensor(group[scaled_static].iloc[-1].values, dtype=torch.float32).unsqueeze(0)

            # Historical last 6 months in physical metres
            hist_6mo = group["groundwater_level_m"].iloc[-6:].values.tolist()
            hist_dates = group["date"].astype(str).iloc[-6:].values.tolist()

            # Monte Carlo Dropout Inference
            preds_scaled = self.predictor.predict_with_uncertainty(x_seq, x_static)

            # Invert to actual metres
            p10_actual = self.inverse_transform_targets(preds_scaled["p10"])[0]
            p50_actual = self.inverse_transform_targets(preds_scaled["p50"])[0]
            p90_actual = self.inverse_transform_targets(preds_scaled["p90"])[0]
            std_actual = preds_scaled["std"][0] * self.target_scaler.scale_[0]

            last_date = pd.to_datetime(group["date"].iloc[-1])
            forecast_dates = [
                (last_date + pd.DateOffset(months=m)).strftime("%Y-%m")
                for m in range(1, horizon + 1)
            ]

            results.append({
                "well_id": well_id,
                "county": group["county"].iloc[0],
                "latitude": float(group["latitude"].iloc[0]),
                "longitude": float(group["longitude"].iloc[0]),
                "total_depth_m": float(group["total_depth_m"].iloc[0]),
                "casing_depth_m": float(group["casing_depth_m"].iloc[0]),
                "aquifer_type": group["aquifer_type"].iloc[0],
                "geo_threat": group["geo_threat"].iloc[0],
                "baseline_gwl_m": float(group["baseline_gwl_m"].iloc[0]),
                "pump_intake_depth_m": float(group["pump_intake_depth_m"].iloc[0]),
                "historical_recent_gwl": [round(float(v), 3) for v in hist_6mo],
                "historical_dates": hist_dates,
                "forecast_p10": [round(float(v), 3) for v in p10_actual],
                "forecast_p50": [round(float(v), 3) for v in p50_actual],
                "forecast_p90": [round(float(v), 3) for v in p90_actual],
                "uncertainty_std": [round(float(v), 3) for v in std_actual],
                "forecast_dates": forecast_dates,
                "all_historical_gwl": group["groundwater_level_m"].values.tolist()
            })

        return results
