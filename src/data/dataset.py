"""
PyTorch Dataset and DataLoader module for Aqua-Predict-NB.
Slices multi-scale temporal sequences and prepares tensors for the LSTM model.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.config import model_config


class GroundwaterSequenceDataset(Dataset):
    """
    Slices historical groundwater and meteorological time series into:
    - Input sequence: (batch_size, lookback_window, feature_dim)
    - Static covariates: (batch_size, static_dim)
    - Future targets: (batch_size, forecast_horizon)
    """

    def __init__(
        self,
        X_seq: np.ndarray,
        X_static: np.ndarray,
        y_seq: np.ndarray,
        well_ids: List[str],
        forecast_dates: List[str]
    ):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.X_static = torch.tensor(X_static, dtype=torch.float32)
        self.y_seq = torch.tensor(y_seq, dtype=torch.float32)
        self.well_ids = well_ids
        self.forecast_dates = forecast_dates

    def __len__(self) -> int:
        return len(self.X_seq)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x_seq": self.X_seq[idx],
            "x_static": self.X_static[idx],
            "y_target": self.y_seq[idx],
            "well_id": self.well_ids[idx],
            "forecast_date": self.forecast_dates[idx]
        }


def build_sequences_from_dataframe(
    df: pd.DataFrame,
    scaled_feature_cols: List[str],
    scaled_static_cols: List[str],
    scaled_target_col: str,
    lookback: int = 24,
    horizon: int = 6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Slices a continuous well-grouped dataframe into overlapping (lookback -> horizon) pairs.
    """
    X_seq_list = []
    X_static_list = []
    y_target_list = []
    well_ids_list = []
    forecast_dates_list = []

    for well_id, well_group in df.groupby("well_id"):
        well_group = well_group.sort_values("date").reset_index(drop=True)
        total_len = len(well_group)

        if total_len < (lookback + horizon):
            continue

        feature_matrix = well_group[scaled_feature_cols].values
        static_vec = well_group[scaled_static_cols].iloc[0].values
        target_series = well_group[scaled_target_col].values
        dates = well_group["date"].astype(str).values

        # Sliding window
        for t in range(total_len - lookback - horizon + 1):
            x_window = feature_matrix[t : t + lookback]
            y_window = target_series[t + lookback : t + lookback + horizon]
            fc_date = dates[t + lookback]

            X_seq_list.append(x_window)
            X_static_list.append(static_vec)
            y_target_list.append(y_window)
            well_ids_list.append(well_id)
            forecast_dates_list.append(fc_date)

    return (
        np.array(X_seq_list, dtype=np.float32),
        np.array(X_static_list, dtype=np.float32),
        np.array(y_target_list, dtype=np.float32),
        well_ids_list,
        forecast_dates_list
    )


def create_train_val_test_dataloaders(
    processed_df: pd.DataFrame,
    metadata: Dict,
    val_split_date: str = "2021-01-01",
    test_split_date: str = "2023-01-01",
    batch_size: int = 64
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Splits data temporally to emulate realistic forecasting into future years.
    Train: dates < val_split_date
    Val: val_split_date <= dates < test_split_date
    Test: dates >= test_split_date
    """
    scaled_feats = metadata["scaled_feature_columns"]
    scaled_static = [f"sc_{c}" for c in metadata["static_columns"]]
    scaled_target = metadata["scaled_target_column"]
    lookback = model_config.lookback_window
    horizon = model_config.forecast_horizon

    # Build all sequences
    X_seq, X_static, y_target, w_ids, dates = build_sequences_from_dataframe(
        processed_df,
        scaled_feature_cols=scaled_feats,
        scaled_static_cols=scaled_static,
        scaled_target_col=scaled_target,
        lookback=lookback,
        horizon=horizon
    )

    dates_arr = np.array(dates)
    train_mask = dates_arr < val_split_date
    val_mask = (dates_arr >= val_split_date) & (dates_arr < test_split_date)
    test_mask = dates_arr >= test_split_date

    # Datasets
    train_ds = GroundwaterSequenceDataset(
        X_seq[train_mask], X_static[train_mask], y_target[train_mask],
        [w for i, w in enumerate(w_ids) if train_mask[i]],
        [d for i, d in enumerate(dates) if train_mask[i]]
    )
    val_ds = GroundwaterSequenceDataset(
        X_seq[val_mask], X_static[val_mask], y_target[val_mask],
        [w for i, w in enumerate(w_ids) if val_mask[i]],
        [d for i, d in enumerate(dates) if val_mask[i]]
    )
    test_ds = GroundwaterSequenceDataset(
        X_seq[test_mask], X_static[test_mask], y_target[test_mask],
        [w for i, w in enumerate(w_ids) if test_mask[i]],
        [d for i, d in enumerate(dates) if test_mask[i]]
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    dims = {
        "sequence_feature_dim": X_seq.shape[2] if len(X_seq) > 0 else 0,
        "static_feature_dim": X_static.shape[1] if len(X_static) > 0 else 0,
        "lookback_window": lookback,
        "forecast_horizon": horizon,
        "num_train_samples": len(train_ds),
        "num_val_samples": len(val_ds),
        "num_test_samples": len(test_ds)
    }

    return train_loader, val_loader, test_loader, dims
