"""
Unit tests for AquaLSTMForecaster model, Monte Carlo Dropout, and evaluation metrics.
"""

import unittest
import numpy as np
import torch

from src.models.lstm_forecaster import AquaLSTMForecaster, MonteCarloDropoutPredictor, QuantileLoss
from src.models.evaluate import compute_nse, compute_kge, compute_picp, compute_mpiw, evaluate_forecasts


class TestModel(unittest.TestCase):

    def test_lstm_forward_pass(self):
        """Tests that forward pass produces expected forecast horizon shape."""
        batch_size = 8
        seq_len = 24
        feature_dim = 10
        static_dim = 5
        horizon = 6

        model = AquaLSTMForecaster(
            seq_feature_dim=feature_dim,
            static_feature_dim=static_dim,
            hidden_dim=32,
            num_layers=2,
            forecast_horizon=horizon,
            dropout=0.2
        )

        x_seq = torch.randn(batch_size, seq_len, feature_dim)
        x_static = torch.randn(batch_size, static_dim)

        out = model(x_seq, x_static)
        self.assertEqual(out.shape, (batch_size, horizon))

    def test_monte_carlo_dropout_uncertainty(self):
        """Tests that MCDO produces P10 <= P50 <= P90 monotonically."""
        model = AquaLSTMForecaster(
            seq_feature_dim=6,
            static_feature_dim=4,
            hidden_dim=32,
            num_layers=1,
            forecast_horizon=4,
            dropout=0.3
        )
        predictor = MonteCarloDropoutPredictor(model, n_samples=25, device="cpu")

        x_seq = torch.randn(2, 12, 6)
        x_static = torch.randn(2, 4)

        preds = predictor.predict_with_uncertainty(x_seq, x_static)
        self.assertIn("p10", preds)
        self.assertIn("p50", preds)
        self.assertIn("p90", preds)
        self.assertIn("std", preds)

        # Statistical sanity check: P10 <= P50 <= P90
        self.assertTrue(np.all(preds["p10"] <= preds["p50"] + 1e-5))
        self.assertTrue(np.all(preds["p50"] <= preds["p90"] + 1e-5))
        self.assertTrue(np.all(preds["std"] >= 0))

    def test_evaluation_metrics(self):
        """Tests NSE, KGE, and PICP calculations."""
        y_true = np.array([[10.0, 11.0], [12.0, 13.0]])
        y_pred = np.array([[10.1, 10.9], [12.2, 12.9]])
        p10 = y_pred - 0.5
        p90 = y_pred + 0.5

        metrics = evaluate_forecasts(y_true, y_pred, p10, p90)
        self.assertIn("rmse_m", metrics)
        self.assertIn("nse", metrics)
        self.assertIn("kge", metrics)
        self.assertIn("picp", metrics)
        self.assertGreaterEqual(metrics["picp"], 0.0)
        self.assertLessEqual(metrics["picp"], 1.0)
        self.assertGreater(metrics["nse"], 0.8)


if __name__ == "__main__":
    unittest.main()
