"""
Preprocessing and Feature Engineering pipeline for Aqua-Predict-NB.
Constructs rolling climate lag features, cyclical seasonal encodings,
and normalizes multivariate inputs for the LSTM forecasting model.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    spatial_config,
    model_config
)


class HydroFeatureEngineer:
    """
    Computes hydrogeological features for groundwater level time series:
    - Multi-scale rolling precipitation sums (1m, 3m, 6m, 12m)
    - Cyclical month encoding (sin/cos)
    - Temperature lag and rolling averages
    - Static well construction & aquifer features
    """

    def __init__(self):
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.feature_columns: List[str] = []
        self.static_columns: List[str] = []
        self.is_fitted = False

    def engineer_time_series_features(self, ts_df: pd.DataFrame) -> pd.DataFrame:
        """Adds rolling meteorological lags and cyclical seasonal indicators."""
        df = ts_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["well_id", "date"]).reset_index(drop=True)

        # Cyclical month encodings
        df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0)
        df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0)

        # Rolling meteorological features per well
        grouped = df.groupby("well_id")

        df["precip_lag1"] = grouped["precip_mm"].shift(1).bfill()
        df["precip_roll3"] = grouped["precip_mm"].transform(lambda x: x.rolling(3, min_periods=1).mean())
        df["precip_roll6"] = grouped["precip_mm"].transform(lambda x: x.rolling(6, min_periods=1).mean())
        df["precip_roll12"] = grouped["precip_mm"].transform(lambda x: x.rolling(12, min_periods=1).mean())

        df["temp_lag1"] = grouped["temp_mean_c"].shift(1).bfill()
        df["temp_roll3"] = grouped["temp_mean_c"].transform(lambda x: x.rolling(3, min_periods=1).mean())

        # Rolling Standardized Precipitation proxy (SPI anomaly proxy)
        p_mean = df["precip_roll3"].mean()
        p_std = df["precip_roll3"].std() + 1e-6
        df["precip_spi_proxy"] = (df["precip_roll3"] - p_mean) / p_std

        return df

    def merge_static_attributes(
        self,
        ts_df: pd.DataFrame,
        wells_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merges well construction and geological categorical attributes."""
        wells = wells_df.copy()
        wells["casing_ratio"] = wells["casing_depth_m"] / (wells["total_depth_m"] + 1e-5)
        
        # One-hot encode aquifer types
        aquifer_dummies = pd.get_dummies(wells["aquifer_type"], prefix="aq", dtype=float)
        # Ensure all standard NB aquifer types are present
        for aq in spatial_config.AQUIFER_TYPES:
            col = f"aq_{aq}"
            if col not in aquifer_dummies.columns:
                aquifer_dummies[col] = 0.0

        wells_merged = pd.concat([wells, aquifer_dummies], axis=1)

        merged = pd.merge(ts_df, wells_merged, on="well_id", how="left")
        return merged

    def fit_transform(
        self,
        df: pd.DataFrame,
        train_split_date: str = "2022-01-01"
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Fits scalers on data before train_split_date and transforms full dataset.
        Prevents forward-looking information leakage into the test set.
        """
        dynamic_cols = [
            "precip_mm", "temp_mean_c", "sin_month", "cos_month",
            "precip_lag1", "precip_roll3", "precip_roll6", "precip_roll12",
            "temp_lag1", "temp_roll3", "precip_spi_proxy"
        ]
        
        aq_cols = [c for c in df.columns if c.startswith("aq_")]
        static_cols = [
            "latitude", "longitude", "total_depth_m", "casing_depth_m",
            "overburden_depth_m", "bedrock_depth_m", "casing_ratio",
            "baseline_gwl_m"
        ] + aq_cols

        self.feature_columns = dynamic_cols + static_cols
        self.static_columns = static_cols

        train_mask = pd.to_datetime(df["date"]) < pd.to_datetime(train_split_date)
        train_features = df.loc[train_mask, self.feature_columns]
        train_target = df.loc[train_mask, ["groundwater_level_m"]]

        self.feature_scaler.fit(train_features)
        self.target_scaler.fit(train_target)
        self.is_fitted = True

        # Transform all rows
        scaled_features = self.feature_scaler.transform(df[self.feature_columns])
        scaled_features_df = pd.DataFrame(
            scaled_features,
            columns=[f"sc_{c}" for c in self.feature_columns],
            index=df.index
        )
        scaled_target = self.target_scaler.transform(df[["groundwater_level_m"]])

        processed_df = df.copy()
        processed_df = pd.concat([processed_df, scaled_features_df], axis=1)
        processed_df["sc_groundwater_level_m"] = scaled_target.ravel()

        # Save metadata and scalers
        metadata = {
            "feature_columns": self.feature_columns,
            "scaled_feature_columns": [f"sc_{c}" for c in self.feature_columns],
            "static_columns": self.static_columns,
            "target_column": "groundwater_level_m",
            "scaled_target_column": "sc_groundwater_level_m",
            "train_split_date": train_split_date
        }

        joblib.dump(self.feature_scaler, PROCESSED_DATA_DIR / "feature_scaler.joblib")
        joblib.dump(self.target_scaler, PROCESSED_DATA_DIR / "target_scaler.joblib")
        with open(PROCESSED_DATA_DIR / "features_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        processed_df.to_csv(PROCESSED_DATA_DIR / "preprocessed_dataset.csv", index=False)
        return processed_df, metadata


def run_preprocessing_pipeline() -> Tuple[pd.DataFrame, Dict]:
    """Runs end-to-end preprocessing."""
    print("=" * 60)
    print(">>> Running Feature Engineering & Preprocessing Pipeline")
    
    wells_path = RAW_DATA_DIR / "nb_owls_wells.csv"
    ts_path = RAW_DATA_DIR / "nb_wells_timeseries.csv"

    if not wells_path.exists() or not ts_path.exists():
        from src.data.ingestion import run_ingestion_pipeline
        run_ingestion_pipeline()

    wells_df = pd.read_csv(wells_path)
    ts_df = pd.read_csv(ts_path)

    fe = HydroFeatureEngineer()
    enriched_ts = fe.engineer_time_series_features(ts_df)
    full_df = fe.merge_static_attributes(enriched_ts, wells_df)
    processed_df, meta = fe.fit_transform(full_df)

    print(f"[SUCCESS] Engineered {len(meta['feature_columns'])} hydrogeological features.")
    print(f"[SUCCESS] Preprocessed dataset written to: {PROCESSED_DATA_DIR / 'preprocessed_dataset.csv'}")
    print("=" * 60)
    return processed_df, meta


if __name__ == "__main__":
    run_preprocessing_pipeline()
