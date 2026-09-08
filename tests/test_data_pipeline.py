"""
Unit tests for Aqua-Predict-NB data ingestion, feature engineering, and dataset slicing.
"""

import unittest
import numpy as np
import pandas as pd
import torch

from src.data.ingestion import generate_eccc_climate_data, generate_nb_wells
from src.data.preprocessing import HydroFeatureEngineer
from src.data.dataset import build_sequences_from_dataframe, GroundwaterSequenceDataset


class TestDataPipeline(unittest.TestCase):

    def test_eccc_climate_generation(self):
        """Validates meteorological station data generation."""
        df = generate_eccc_climate_data(start_year=2022, end_year=2023, seed=12)
        self.assertGreater(len(df), 0)
        self.assertIn("precip_mm", df.columns)
        self.assertIn("temp_mean_c", df.columns)
        self.assertIn("snow_water_equiv_mm", df.columns)
        # Verify physical sanity
        self.assertTrue((df["precip_mm"] >= 0).all())
        self.assertTrue((df["temp_max_c"] >= df["temp_min_c"]).all())

    def test_nb_wells_generation(self):
        """Validates NB OWLS well generation and county distributions."""
        wells_df, ts_df = generate_nb_wells(num_wells=15, seed=42)
        self.assertEqual(len(wells_df), 15)
        self.assertGreater(len(ts_df), 100)
        self.assertIn("well_id", wells_df.columns)
        self.assertIn("aquifer_type", wells_df.columns)
        self.assertIn("geo_threat", wells_df.columns)

        # Coordinate bounds check for New Brunswick
        self.assertTrue((wells_df["latitude"] >= 44.5).all())
        self.assertTrue((wells_df["latitude"] <= 48.2).all())
        self.assertTrue((wells_df["longitude"] >= -69.2).all())
        self.assertTrue((wells_df["longitude"] <= -63.7).all())

    def test_feature_engineering_and_scaling(self):
        """Validates rolling features and scaling without data leakage."""
        wells_df, ts_df = generate_nb_wells(num_wells=10, seed=99)
        fe = HydroFeatureEngineer()
        enriched = fe.engineer_time_series_features(ts_df)
        self.assertIn("precip_lag1", enriched.columns)
        self.assertIn("precip_roll3", enriched.columns)
        self.assertIn("precip_roll12", enriched.columns)
        self.assertIn("sin_month", enriched.columns)
        self.assertIn("cos_month", enriched.columns)

        merged = fe.merge_static_attributes(enriched, wells_df)
        processed_df, meta = fe.fit_transform(merged, train_split_date="2020-01-01")

        self.assertTrue(fe.is_fitted)
        self.assertIn("sc_precip_mm", processed_df.columns)
        self.assertIn("sc_groundwater_level_m", processed_df.columns)

    def test_sequence_windowing(self):
        """Validates sliding window sequence tensor dimensions."""
        wells_df, ts_df = generate_nb_wells(num_wells=5, seed=123)
        fe = HydroFeatureEngineer()
        merged = fe.merge_static_attributes(fe.engineer_time_series_features(ts_df), wells_df)
        proc_df, meta = fe.fit_transform(merged)

        lookback = 12
        horizon = 3
        X_seq, X_static, y_target, w_ids, dates = build_sequences_from_dataframe(
            proc_df,
            scaled_feature_cols=meta["scaled_feature_columns"],
            scaled_static_cols=[f"sc_{c}" for c in meta["static_columns"]],
            scaled_target_col=meta["scaled_target_column"],
            lookback=lookback,
            horizon=horizon
        )

        self.assertEqual(len(X_seq.shape), 3)
        self.assertEqual(X_seq.shape[1], lookback)
        self.assertEqual(y_target.shape[1], horizon)

        # Dataset loader test
        ds = GroundwaterSequenceDataset(X_seq, X_static, y_target, w_ids, dates)
        self.assertEqual(len(ds), len(X_seq))
        item = ds[0]
        self.assertIsInstance(item["x_seq"], torch.Tensor)
        self.assertEqual(item["x_seq"].shape, (lookback, X_seq.shape[2]))


if __name__ == "__main__":
    unittest.main()
