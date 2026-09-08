"""
Model evaluation module for Aqua-Predict-NB.
Computes hydrogeological forecasting metrics:
- Nash-Sutcliffe Efficiency (NSE)
- Kling-Gupta Efficiency (KGE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Prediction Interval Coverage Probability (PICP)
- Mean Prediction Interval Width (MPIW)
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def compute_nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Nash-Sutcliffe Efficiency (standard hydrological benchmark)."""
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    if denominator < 1e-8:
        return 1.0
    numerator = np.sum((y_true - y_pred) ** 2)
    return float(1.0 - (numerator / denominator))


def compute_kge(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Kling-Gupta Efficiency."""
    std_true = np.std(y_true)
    std_pred = np.std(y_pred)
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    if std_true < 1e-8 or std_pred < 1e-8:
        return 0.0

    # Pearson correlation
    r = np.corrcoef(y_true.ravel(), y_pred.ravel())[0, 1]
    alpha = std_pred / std_true
    beta = mean_pred / (mean_true + 1e-8)

    kge = 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    return float(kge)


def compute_picp(y_true: np.ndarray, lower_bound: np.ndarray, upper_bound: np.ndarray) -> float:
    """Computes Prediction Interval Coverage Probability (P10 to P90 interval)."""
    covered = (y_true >= lower_bound) & (y_true <= upper_bound)
    return float(np.mean(covered))


def compute_mpiw(lower_bound: np.ndarray, upper_bound: np.ndarray) -> float:
    """Computes Mean Prediction Interval Width."""
    return float(np.mean(upper_bound - lower_bound))


def evaluate_forecasts(
    y_true_actual_m: np.ndarray,
    p50_preds_m: np.ndarray,
    p10_preds_m: np.ndarray,
    p90_preds_m: np.ndarray
) -> Dict[str, float]:
    """
    Computes overall and horizon-specific metrics in physical metres.
    """
    rmse_overall = float(np.sqrt(np.mean((y_true_actual_m - p50_preds_m) ** 2)))
    mae_overall = float(np.mean(np.abs(y_true_actual_m - p50_preds_m)))
    nse_overall = compute_nse(y_true_actual_m, p50_preds_m)
    kge_overall = compute_kge(y_true_actual_m, p50_preds_m)
    picp_overall = compute_picp(y_true_actual_m, p10_preds_m, p90_preds_m)
    mpiw_overall = compute_mpiw(p10_preds_m, p90_preds_m)

    results = {
        "rmse_m": round(rmse_overall, 4),
        "mae_m": round(mae_overall, 4),
        "nse": round(nse_overall, 4),
        "kge": round(kge_overall, 4),
        "picp": round(picp_overall, 4),
        "mpiw_m": round(mpiw_overall, 4)
    }

    # By-horizon breakdown (Month 1 through 6)
    horizon = y_true_actual_m.shape[1]
    for h in range(horizon):
        yt_h = y_true_actual_m[:, h]
        yp_h = p50_preds_m[:, h]
        p10_h = p10_preds_m[:, h]
        p90_h = p90_preds_m[:, h]

        results[f"horizon_m{h+1}_rmse_m"] = round(float(np.sqrt(np.mean((yt_h - yp_h) ** 2))), 4)
        results[f"horizon_m{h+1}_mae_m"] = round(float(np.mean(np.abs(yt_h - yp_h))), 4)
        results[f"horizon_m{h+1}_nse"] = round(compute_nse(yt_h, yp_h), 4)
        results[f"horizon_m{h+1}_picp"] = round(compute_picp(yt_h, p10_h, p90_h), 4)

    return results
