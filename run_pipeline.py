"""
Aqua-Predict-NB: Master End-to-End Pipeline Orchestrator.
Command-line interface to execute individual pipeline stages or complete end-to-end flow:
Ingest -> Preprocess -> Train -> Evaluate -> Infer (MCDO) -> Risk Tiering -> Export (Data & Figures).
"""

import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

from src.config import (
    CHECKPOINT_DIR,
    DATA_DIR,
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    model_config
)
from src.data.ingestion import run_ingestion_pipeline
from src.data.preprocessing import run_preprocessing_pipeline
from src.data.dataset import create_train_val_test_dataloaders
from src.models.lstm_forecaster import AquaLSTMForecaster
from src.models.train import ModelTrainer
from src.models.evaluate import evaluate_forecasts
from src.inference.predict import WellForecastingEngine
from src.inference.risk_tiering import compute_provincial_risk_summary
from src.export.export_dashboard import export_dashboard_json
from src.export.export_figures import run_figures_export_pipeline


def execute_pipeline(
    stage: str = "all",
    num_wells: int = 250,
    epochs: int = 30,
    mc_samples: int = 100,
    device_str: str = "auto"
):
    """Orchestrates pipeline execution."""
    start_time = time.time()
    device = "cuda" if (device_str == "auto" and torch.cuda.is_available()) else ("cpu" if device_str == "auto" else device_str)

    print("\n" + "=" * 70)
    print("      Aqua-Predict-NB: AI Groundwater Level Forecasting Pipeline")
    print("      New Brunswick Private Wells | PyTorch LSTM + Monte Carlo Dropout")
    print(f"      Execution Mode: Stage='{stage}' | Compute Device: '{device}'")
    print("=" * 70 + "\n")

    # 1. INGESTION
    if stage in ["ingest", "all"]:
        run_ingestion_pipeline(num_wells=num_wells)
        if stage == "ingest":
            return

    # 2. PREPROCESSING
    if stage in ["preprocess", "all"]:
        processed_df, meta = run_preprocessing_pipeline()
        if stage == "preprocess":
            return
    else:
        import pandas as pd
        proc_path = PROCESSED_DATA_DIR / "preprocessed_dataset.csv"
        meta_path = PROCESSED_DATA_DIR / "features_metadata.json"
        if not proc_path.exists() or not meta_path.exists():
            processed_df, meta = run_preprocessing_pipeline()
        else:
            processed_df = pd.read_csv(proc_path)
            with open(meta_path, "r") as f:
                meta = json.load(f)

    # 3. TRAIN
    train_loader, val_loader, test_loader, dims = create_train_val_test_dataloaders(
        processed_df=processed_df,
        metadata=meta,
        batch_size=model_config.batch_size
    )

    if stage in ["train", "all"]:
        print("\n" + "-" * 60)
        print(">>> Stage 3: Training AquaLSTMForecaster...")
        model = AquaLSTMForecaster(
            seq_feature_dim=dims["sequence_feature_dim"],
            static_feature_dim=dims["static_feature_dim"],
            hidden_dim=model_config.hidden_dim,
            num_layers=model_config.num_layers,
            forecast_horizon=model_config.forecast_horizon,
            dropout=model_config.dropout
        )
        trainer = ModelTrainer(model=model, device=device)
        training_history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=model_config.learning_rate,
            patience=model_config.early_stopping_patience
        )
        if stage == "train":
            return
    else:
        with open(CHECKPOINT_DIR / "training_history.json", "r") as f:
            training_history = json.load(f)

    # 4. EVALUATION
    eval_results = {}
    if stage in ["evaluate", "infer", "export", "all"]:
        print("\n" + "-" * 60)
        print(">>> Stage 4: Benchmarking Test Set Performance (NSE, RMSE, MAE)...")
        engine = WellForecastingEngine(device=device)
        engine.predictor.n_samples = mc_samples

        # Evaluate on test set
        all_y_true = []
        all_p10 = []
        all_p50 = []
        all_p90 = []

        with torch.no_grad():
            for batch in test_loader:
                preds_dict = engine.predictor.predict_with_uncertainty(batch["x_seq"], batch["x_static"])
                y_true_scaled = batch["y_target"].numpy()

                all_y_true.append(engine.inverse_transform_targets(y_true_scaled))
                all_p10.append(engine.inverse_transform_targets(preds_dict["p10"]))
                all_p50.append(engine.inverse_transform_targets(preds_dict["p50"]))
                all_p90.append(engine.inverse_transform_targets(preds_dict["p90"]))

        if len(all_y_true) > 0:
            y_true_mat = np.vstack(all_y_true)
            p10_mat = np.vstack(all_p10)
            p50_mat = np.vstack(all_p50)
            p90_mat = np.vstack(all_p90)

            eval_results = evaluate_forecasts(y_true_mat, p50_mat, p10_mat, p90_mat)
            print(f"  [METRIC] Overall Test RMSE: {eval_results['rmse_m']} m")
            print(f"  [METRIC] Overall Test MAE:  {eval_results['mae_m']} m")
            print(f"  [METRIC] Nash-Sutcliffe Efficiency (NSE): {eval_results['nse']}")
            print(f"  [METRIC] Kling-Gupta Efficiency (KGE):    {eval_results['kge']}")
            print(f"  [METRIC] P10-P90 Coverage (PICP):         {round(eval_results['picp']*100, 1)}%")

            with open(CHECKPOINT_DIR / "evaluation_metrics.json", "w") as f:
                json.dump(eval_results, f, indent=2)

        if stage == "evaluate":
            return

    # 5. INFERENCE & RISK TIERING
    print("\n" + "-" * 60)
    print(f">>> Stage 5: Monte Carlo Inference (N={mc_samples} passes) & Risk Tiering...")
    engine = WellForecastingEngine(device=device)
    engine.predictor.n_samples = mc_samples
    raw_forecasts = engine.forecast_all_wells_latest(processed_df)

    enriched_wells, risk_summary = compute_provincial_risk_summary(
        well_forecasts=raw_forecasts,
        horizon_months=3
    )

    print(f"  [SUMMARY] Total Monitored Wells:     {risk_summary['total_wells']}")
    print(f"  [SUMMARY] High Drought Risk Wells:   {risk_summary['high_drought_risk_count']} ({risk_summary['high_risk_pct']}%)")
    print(f"  [SUMMARY] Medium Risk Wells:         {risk_summary['medium_drought_risk_count']}")
    print(f"  [SUMMARY] Combined Geo-Risk Threat: {risk_summary['combined_georisk_count']}")

    if stage == "infer":
        return

    # 6. EXPORT (DASHBOARD JSON & FIGURES)
    print("\n" + "-" * 60)
    print(">>> Stage 6: Exporting Artifacts (Dashboard JSON & Publication Figures)...")
    export_dashboard_json(
        wells_data=enriched_wells,
        summary_stats=risk_summary,
        evaluation_metrics=eval_results,
        output_dir=DATA_DIR
    )

    fig_paths = run_figures_export_pipeline(
        wells_data=enriched_wells,
        summary_stats=risk_summary,
        training_history=training_history,
        processed_df=processed_df
    )

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"[PIPELINE COMPLETED SUCCESSFULLY] in {elapsed:.1f} seconds.")
    print("Generated Artifacts:")
    print(f"  |-- Data:    {DATA_DIR / 'processed_wells.json'}")
    print(f"  |-- Summary: {DATA_DIR / 'pipeline_summary.json'}")
    print(f"  |-- GeoJSON: {DATA_DIR / 'nb_wells_risk.geojson'}")
    print(f"  +-- Figures ({len(fig_paths)}):")
    for p in fig_paths:
        print(f"        * {p.name}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Aqua-Predict-NB Pipeline Orchestrator")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["ingest", "preprocess", "train", "evaluate", "infer", "export", "all"],
        help="Pipeline stage to execute"
    )
    parser.add_argument("--wells", type=int, default=250, help="Number of NB wells to generate/ingest")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--mc-samples", type=int, default=100, help="Monte Carlo Dropout samples for inference")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cpu, cuda)")

    args = parser.parse_args()
    execute_pipeline(
        stage=args.stage,
        num_wells=args.wells,
        epochs=args.epochs,
        mc_samples=args.mc_samples,
        device_str=args.device
    )


if __name__ == "__main__":
    main()
