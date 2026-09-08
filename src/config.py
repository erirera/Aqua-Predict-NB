"""
Configuration module for Aqua-Predict-NB.
Centralizes paths, hydrogeological constants, model hyperparameters, and risk thresholds.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"

for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, CHECKPOINT_DIR, OUTPUT_DIR, FIGURES_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


@dataclass
class SpatialConfig:
    """New Brunswick spatial and geological configuration."""
    # Rough geographic bounds for New Brunswick (lat min/max, lon min/max)
    LAT_MIN: float = 44.60
    LAT_MAX: float = 48.05
    LON_MIN: float = -69.05
    LON_MAX: float = -63.75

    # Center coordinates for Leaflet / GIS mapping
    CENTER_LAT: float = 46.5653
    CENTER_LON: float = -66.4619

    # Aquifer classifications dominant in New Brunswick
    AQUIFER_TYPES: List[str] = field(default_factory=lambda: [
        "Fractured Bedrock",
        "Carboniferous Sandstone",
        "Glaciofluvial Sand/Gravel",
        "Windsor Group Karst",
        "Mafic Volcanic"
    ])

    # Geochemical threat types
    GEOCHEMICAL_HAZARDS: List[str] = field(default_factory=lambda: [
        "Arsenic",
        "Uranium",
        "Methane"
    ])


@dataclass
class ClimateConfig:
    """ECCC reference climate stations in New Brunswick."""
    REFERENCE_STATIONS: List[dict] = field(default_factory=lambda: [
        {"id": "ECCC_FREDERICTON", "name": "Fredericton CDA", "lat": 45.92, "lon": -66.62, "elevation": 45.0},
        {"id": "ECCC_MONCTON", "name": "Moncton Airport", "lat": 46.11, "lon": -64.68, "elevation": 71.0},
        {"id": "ECCC_SAINT_JOHN", "name": "Saint John A", "lat": 45.32, "lon": -65.89, "elevation": 103.0},
        {"id": "ECCC_MIRAMICHI", "name": "Miramichi Airport", "lat": 47.01, "lon": -65.45, "elevation": 33.0},
        {"id": "ECCC_BATHURST", "name": "Bathurst A", "lat": 47.63, "lon": -65.74, "elevation": 89.0},
        {"id": "ECCC_EDMUNDSTON", "name": "Edmundston", "lat": 47.37, "lon": -68.33, "elevation": 149.0},
        {"id": "ECCC_SUSSEX", "name": "Sussex Four Corners", "lat": 45.72, "lon": -65.51, "elevation": 21.0},
        {"id": "ECCC_WOODSTOCK", "name": "Woodstock", "lat": 46.15, "lon": -67.58, "elevation": 140.0}
    ])


@dataclass
class ModelConfig:
    """LSTM and Monte Carlo Dropout hyperparameters."""
    lookback_window: int = 24       # 24 months lookback
    forecast_horizon: int = 6       # 6 months forecasting horizon
    hidden_dim: int = 128           # LSTM hidden units
    num_layers: int = 2             # Stacked LSTM layers
    dropout: float = 0.2            # Dropout probability for regularisation & MC Dropout
    learning_rate: float = 1e-3     # AdamW learning rate
    weight_decay: float = 1e-4      # Weight decay
    batch_size: int = 64
    epochs: int = 35
    early_stopping_patience: int = 8
    mc_dropout_samples: int = 100   # Number of stochastic forward passes for Bayesian bounds
    quantiles: Tuple[float, ...] = (0.10, 0.50, 0.90)  # P10 (drought), P50 (median), P90 (recharge)


@dataclass
class RiskThresholds:
    """Hydrogeological drought and water quality risk classification."""
    HIGH_DROUGHT_PERCENTILE: float = 10.0   # Forecast GWL < 10th percentile of historical levels
    MEDIUM_DROUGHT_PERCENTILE: float = 25.0 # Forecast GWL between 10th and 25th percentile
    
    # Static level drop buffer in metres below baseline considered severe
    CRITICAL_DROP_METRES: float = 2.5


# Instantiated singletons
spatial_config = SpatialConfig()
climate_config = ClimateConfig()
model_config = ModelConfig()
risk_thresholds = RiskThresholds()
