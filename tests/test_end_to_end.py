"""
Integration test running a fast end-to-end pipeline pass and validating outputs.
"""

import json
import unittest
from pathlib import Path

from run_pipeline import execute_pipeline
from src.config import DATA_DIR, FIGURES_DIR, CHECKPOINT_DIR


class TestEndToEndPipeline(unittest.TestCase):

    def test_pipeline_execution(self):
        """Runs a fast multi-step pipeline on a small subset of wells."""
        execute_pipeline(
            stage="all",
            num_wells=12,
            epochs=2,
            mc_samples=10,
            device_str="cpu"
        )

        # 1. Verify checkpoint
        self.assertTrue((CHECKPOINT_DIR / "best_model.pt").exists())
        self.assertTrue((CHECKPOINT_DIR / "training_history.json").exists())

        # 2. Verify dashboard JSON exports
        wells_json = DATA_DIR / "processed_wells.json"
        summary_json = DATA_DIR / "pipeline_summary.json"
        geojson_file = DATA_DIR / "nb_wells_risk.geojson"

        self.assertTrue(wells_json.exists())
        self.assertTrue(summary_json.exists())
        self.assertTrue(geojson_file.exists())

        with open(wells_json, "r") as f:
            data = json.load(f)
            self.assertEqual(len(data), 12)
            w0 = data[0]
            self.assertIn("forecastP50", w0)
            self.assertIn("forecastP10", w0)
            self.assertIn("forecastP90", w0)
            self.assertEqual(len(w0["forecastP50"]), 6)

        with open(summary_json, "r") as f:
            summary = json.load(f)
            self.assertEqual(summary["total_wells"], 12)
            self.assertIn("high_drought_risk_count", summary)

        # 3. Verify figure exports
        expected_figs = [
            "hydrographs_with_uncertainty.png",
            "provincial_drought_risk_map.png",
            "training_validation_curves.png",
            "meteorological_lag_correlations.png",
            "risk_tier_summary.png"
        ]
        for fig_name in expected_figs:
            fig_p = FIGURES_DIR / fig_name
            self.assertTrue(fig_p.exists(), f"Expected figure not found: {fig_p}")
            self.assertGreater(fig_p.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
