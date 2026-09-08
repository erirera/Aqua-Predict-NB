"""
Scientific figure and image export pipeline for Aqua-Predict-NB.
Generates publication-ready 300-DPI visualisations:
1. hydrographs_with_uncertainty.png
2. provincial_drought_risk_map.png
3. training_validation_curves.png
4. meteorological_lag_correlations.png
5. risk_tier_summary.png
"""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    FIGURES_DIR,
    spatial_config
)

# Publication styling settings
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--"
})


def export_hydrographs_with_uncertainty(
    wells_data: List[Dict],
    output_path: Optional[Path] = None
) -> Path:
    """
    Plots representative hydrographs for 3 distinct NB aquifer types showing
    historical records, LSTM median forecast (P50), and P10-P90 Monte Carlo Dropout bounds.
    """
    output_path = output_path or (FIGURES_DIR / "hydrographs_with_uncertainty.png")

    # Select 3 representative wells across different aquifers
    target_aquifers = ["Fractured Bedrock", "Carboniferous Sandstone", "Glaciofluvial Sand/Gravel"]
    selected_wells = []
    for aq in target_aquifers:
        found = next((w for w in wells_data if w["aquifer"] == aq), None)
        if found:
            selected_wells.append(found)

    if len(selected_wells) < 3:
        selected_wells = wells_data[:3]

    fig, axes = plt.subplots(len(selected_wells), 1, figsize=(11, 4 * len(selected_wells)), sharex=False)
    if len(selected_wells) == 1:
        axes = [axes]

    for ax, w in zip(axes, selected_wells):
        hist_gwl = w["historical"]
        hist_dates = w["historicalDates"]
        fc_p50 = w["forecastP50"]
        fc_p10 = w["forecastP10"]
        fc_p90 = w["forecastP90"]
        fc_dates = w["forecastDates"]

        # Timeline setup
        all_dates = hist_dates + fc_dates
        x_hist = np.arange(len(hist_dates))
        x_fc = np.arange(len(hist_dates) - 1, len(all_dates))

        # Align first forecast point with last historical point for continuity
        connected_p50 = [hist_gwl[-1]] + fc_p50
        connected_p10 = [hist_gwl[-1]] + fc_p10
        connected_p90 = [hist_gwl[-1]] + fc_p90

        # Plot historical
        ax.plot(x_hist, hist_gwl, color="#334155", linewidth=2.0, marker="o", markersize=4.5, label="Observed GWL")

        # Uncertainty envelope
        ax.fill_between(
            x_fc,
            connected_p10,
            connected_p90,
            color="#3b82f6",
            alpha=0.22,
            label="LSTM Uncertainty (P10–P90 MCDO)"
        )

        # Median forecast
        ax.plot(x_fc, connected_p50, color="#2563eb", linewidth=2.2, linestyle="--", marker="s", markersize=4, label="Median Forecast (P50)")
        # High drought risk line
        ax.plot(x_fc, connected_p10, color="#dc2626", linewidth=1.4, linestyle=":", label="Drought Stress (P10)")

        # Critical pump intake threshold
        pump_y = w.get("pumpIntake", -30.0)
        # Only show if in reasonable viewing range
        if pump_y > min(min(hist_gwl), min(fc_p10)) - 10:
            ax.axhline(pump_y, color="#991b1b", linestyle="-.", linewidth=1.2, label="Pump Intake Depth")

        # Vertical transition line
        ax.axvline(x_hist[-1], color="#94a3b8", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.text(x_hist[-1] + 0.1, max(hist_gwl) + 0.2, "Forecast Origin", color="#475569", fontsize=9, style="italic")

        ax.set_xticks(np.arange(len(all_dates)))
        ax.set_xticklabels(all_dates, rotation=35, ha="right")
        ax.set_ylabel("GWL (m below surface)")
        
        threat_str = f" | Threat: {w['quality']}" if w['quality'] != 'None' else ""
        risk_badge = f"[{w['risk'].upper()} RISK]"
        ax.set_title(
            f"{w['id']} ({w['county']} Co.) — Aquifer: {w['aquifer']} — Depth: {w['depth']}m {threat_str} {risk_badge}",
            fontweight="bold",
            color="#1e293b"
        )
        ax.legend(loc="lower left", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[FIGURE EXPORTED] {output_path}")
    return output_path


def export_provincial_drought_risk_map(
    wells_data: List[Dict],
    output_path: Optional[Path] = None
) -> Path:
    """
    Exports a spatial distribution map of New Brunswick private wells categorized by risk tier,
    with distinct markers for geochemical threats (Arsenic, Uranium, Methane).
    """
    output_path = output_path or (FIGURES_DIR / "provincial_drought_risk_map.png")

    df = pd.DataFrame(wells_data)
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color palette
    color_map = {"high": "#ef4444", "medium": "#f59e0b", "low": "#10b981"}
    
    # Plot low, medium, high in order
    for risk_lvl in ["low", "medium", "high"]:
        sub = df[df["risk"] == risk_lvl]
        ax.scatter(
            sub["lng"],
            sub["lat"],
            c=color_map[risk_lvl],
            s=50 if risk_lvl == "high" else 35,
            alpha=0.85,
            edgecolors="#1e293b",
            linewidth=0.5,
            label=f"{risk_lvl.capitalize()} Drought Risk (n={len(sub)})"
        )

    # Highlight geochemical threat zones with symbol overlays
    geo_threats = df[df["quality"].isin(["Arsenic", "Uranium", "Methane"])]
    markers = {"Arsenic": "x", "Uranium": "^", "Methane": "s"}
    for threat, marker in markers.items():
        sub_t = geo_threats[geo_threats["quality"] == threat]
        if len(sub_t) > 0:
            if marker == "x":
                ax.scatter(
                    sub_t["lng"],
                    sub_t["lat"],
                    c="#7c3aed",
                    marker=marker,
                    s=110,
                    linewidth=1.8,
                    label=f"Threat: {threat} (n={len(sub_t)})"
                )
            else:
                ax.scatter(
                    sub_t["lng"],
                    sub_t["lat"],
                    facecolors="none",
                    edgecolors="#7c3aed",
                    marker=marker,
                    s=110,
                    linewidth=1.4,
                    label=f"Threat: {threat} (n={len(sub_t)})"
                )

    ax.set_title("AquaPredict NB: Provincial Private Well Groundwater Drought Risk & Geochemical Overlays", fontweight="bold")
    ax.set_xlabel("Longitude (°W)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_xlim(spatial_config.LON_MIN, spatial_config.LON_MAX)
    ax.set_ylim(spatial_config.LAT_MIN, spatial_config.LAT_MAX)

    # Add NB reference coordinate grid & styling
    ax.text(
        0.02, 0.04,
        f"Total Monitored Wells: {len(df)}\nHigh Drought Risk: {len(df[df['risk']=='high'])}\nModel: PyTorch 2-Layer Seq2Seq LSTM (MCDO N=100)",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.95)
    )

    ax.legend(loc="upper left", framealpha=0.95, fontsize=8.5)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[FIGURE EXPORTED] {output_path}")
    return output_path


def export_training_validation_curves(
    training_history: Dict,
    output_path: Optional[Path] = None
) -> Path:
    """
    Plots training loss, validation loss, and learning rate schedule.
    """
    output_path = output_path or (FIGURES_DIR / "training_validation_curves.png")

    train_loss = training_history.get("train_loss", [])
    val_loss = training_history.get("val_loss", [])
    lrs = training_history.get("lr", [])
    best_epoch = training_history.get("best_epoch", len(val_loss))

    epochs = np.arange(1, len(train_loss) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss curve
    ax1.plot(epochs, train_loss, label="Training Loss (MSE)", color="#2563eb", linewidth=2.0)
    ax1.plot(epochs, val_loss, label="Validation Loss (MSE)", color="#dc2626", linewidth=2.0)
    ax1.axvline(best_epoch, color="#059669", linestyle="--", linewidth=1.5, label=f"Best Checkpoint (Ep {best_epoch})")
    ax1.scatter([best_epoch], [val_loss[best_epoch - 1]], color="#059669", s=70, zorder=5)

    ax1.set_title("LSTM Convergence & Early Stopping", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (Scaled MSE)")
    ax1.legend(framealpha=0.9)

    # Learning rate schedule
    ax2.plot(epochs, lrs, color="#7c3aed", linewidth=2.0, marker="o", markersize=3)
    ax2.set_title("Learning Rate Schedule (ReduceLROnPlateau)", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate")
    ax2.set_yscale("log")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[FIGURE EXPORTED] {output_path}")
    return output_path


def export_meteorological_lag_correlations(
    processed_df: pd.DataFrame,
    output_path: Optional[Path] = None
) -> Path:
    """
    Cross-correlation analysis showing how different aquifer types respond to
    1-month, 3-month, 6-month, and 12-month precipitation accumulation.
    """
    output_path = output_path or (FIGURES_DIR / "meteorological_lag_correlations.png")

    lag_cols = ["precip_lag1", "precip_roll3", "precip_roll6", "precip_roll12"]
    aquifers = processed_df["aquifer_type"].unique()

    corrs = []
    for aq in aquifers:
        sub = processed_df[processed_df["aquifer_type"] == aq]
        row = []
        for col in lag_cols:
            r = np.corrcoef(sub[col].values, sub["groundwater_level_m"].values)[0, 1]
            row.append(r)
        corrs.append(row)

    corr_df = pd.DataFrame(
        corrs,
        index=aquifers,
        columns=["1-Mo Lag", "3-Mo Rolling", "6-Mo Rolling", "12-Mo Rolling"]
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(
        corr_df,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar_kws={"label": "Pearson Correlation with GWL"},
        ax=ax,
        linewidths=1.0,
        linecolor="#f1f5f9"
    )

    ax.set_title("Aquifer Recharge Dynamics: Precipitation Accumulation vs. Water Table Response", fontweight="bold")
    ax.set_ylabel("NB Aquifer Type")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[FIGURE EXPORTED] {output_path}")
    return output_path


def export_risk_tier_summary(
    wells_data: List[Dict],
    summary_stats: Dict,
    output_path: Optional[Path] = None
) -> Path:
    """
    Provincial risk tier breakdown infographic (donut chart + hazard breakdown).
    """
    output_path = output_path or (FIGURES_DIR / "risk_tier_summary.png")

    df = pd.DataFrame(wells_data)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Donut chart of drought risk
    counts = [
        summary_stats["high_drought_risk_count"],
        summary_stats["medium_drought_risk_count"],
        summary_stats["low_drought_risk_count"]
    ]
    labels = ["High Drought Risk", "Medium Risk", "Low Risk"]
    colors = ["#ef4444", "#f59e0b", "#10b981"]

    wedges, texts, autotexts = ax1.pie(
        counts,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        wedgeprops=dict(width=0.4, edgecolor="w", linewidth=2)
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")

    ax1.set_title(f"NB Wells Drought Risk Breakdown (N={summary_stats['total_wells']})", fontweight="bold")

    # 2. Bar chart of geochemical threats within drought risk tiers
    threat_counts = df.groupby(["risk", "quality"]).size().unstack(fill_value=0)
    # Ensure column ordering
    for q in ["None", "Arsenic", "Uranium", "Methane"]:
        if q not in threat_counts.columns:
            threat_counts[q] = 0

    threat_counts = threat_counts[["None", "Arsenic", "Uranium", "Methane"]]
    threat_counts.plot(
        kind="bar",
        stacked=True,
        color=["#94a3b8", "#dc2626", "#d97706", "#2563eb"],
        ax=ax2,
        edgecolor="#1e293b",
        linewidth=0.5
    )

    ax2.set_title("Geochemical Threat Distribution by Drought Tier", fontweight="bold")
    ax2.set_xlabel("Drought Risk Tier")
    ax2.set_ylabel("Number of Wells")
    ax2.legend(title="Geochemical Hazard", framealpha=0.9)
    ax2.set_xticklabels([t.get_text().capitalize() for t in ax2.get_xticklabels()], rotation=0)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[FIGURE EXPORTED] {output_path}")
    return output_path


def run_figures_export_pipeline(
    wells_data: List[Dict],
    summary_stats: Dict,
    training_history: Dict,
    processed_df: pd.DataFrame
) -> List[Path]:
    """Generates all 5 publication-ready visualisations."""
    print("=" * 60)
    print(">>> Exporting Publication-Ready Figures and Visualizations...")
    
    paths = [
        export_hydrographs_with_uncertainty(wells_data),
        export_provincial_drought_risk_map(wells_data),
        export_training_validation_curves(training_history),
        export_meteorological_lag_correlations(processed_df),
        export_risk_tier_summary(wells_data, summary_stats)
    ]
    
    print(f"[SUCCESS] Exported {len(paths)} figures to {FIGURES_DIR}")
    print("=" * 60)
    return paths
