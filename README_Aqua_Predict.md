# Aqua-Predict-NB: AI Groundwater Level Forecasting
### New Brunswick Private Wells

> **Status: Research Design & Prototype Visualisation**
> This repository contains the full research design, proposed methodology, and interactive proof-of-concept dashboard for an LSTM-based groundwater level forecasting system for New Brunswick private wells. The PyTorch model implementation is under active development — see the [Roadmap](#roadmap) for current progress.

---

## Overview

Approximately 30% of New Brunswick residents rely on private wells for drinking water. Groundwater levels in these wells are sensitive to drought, seasonal variability, and long-term climate trends — yet no publicly available forecasting tool exists for NB private well owners or regulators.

This project proposes and designs an **LSTM (Long Short-Term Memory) neural network** to forecast groundwater levels 1–6 months ahead using meteorological drivers, geological context, and historical well records from the NB Online Well Log System (OWLS).

**The practical outcome:** A risk-tiered early warning system identifying private wells at elevated drought risk, enabling proactive water management decisions by well owners and municipal planners.

---

## What's in This Repository

| File / Folder | Description |
|---|---|
| `index.html` + `script.js` + `styles.css` | Interactive proof-of-concept dashboard demonstrating the proposed system: spatial risk map of NB private wells, simulated LSTM forecast charts with uncertainty bounds, and water quality risk overlays |
| `README.md` | Full research design, model architecture, data pipeline specification, and implementation roadmap |

The **dashboard visualises the proposed system using simulated data**, not trained model outputs. Well locations are randomly generated; forecast charts simulate LSTM output with 10th/90th percentile uncertainty bounds to illustrate the intended user experience. Live at: [erirera.github.io/Aqua-Predict-NB](https://erirera.github.io/Aqua-Predict-NB/)

---

## Scientific Background

Groundwater level in shallow bedrock and overburden wells is controlled by:

- **Precipitation and snowmelt recharge** — primary driver of seasonal fluctuations
- **Geological substrate** — hydraulic conductivity of bedrock aquifer type (fractured granite vs. sandstone vs. limestone)
- **Well depth and casing** — determines which aquifer horizon is being measured
- **Regional groundwater flow** — topographic position relative to recharge and discharge zones

LSTMs are well-suited to this problem because groundwater response to precipitation is non-linear and lagged — shallow wells respond quickly, deep bedrock wells may show 6–12 month lags. The LSTM's memory cells can learn these variable lag structures from training data.

---

## Proposed System Architecture

```
Data Pipeline:
  ├── NB OWLS (Online Well Log System) — historical water level records + well metadata
  ├── ECCC meteorological stations — daily precipitation, temperature, snow depth
  └── NRCan surficial geology — aquifer type classification per well

Feature Engineering:
  ├── Meteorological: 30-day, 90-day, 365-day rolling precipitation totals
  ├── Seasonal: month-of-year encoding (sin/cos)
  ├── Geological: aquifer type (categorical), overburden depth, bedrock depth
  └── Well-specific: casing depth, static water level baseline

Model: LSTM Sequence-to-Sequence
  ├── Input: 24-month lookback window, multivariate features
  ├── Architecture: 2-layer LSTM (128 hidden units) + dropout (0.2) + dense output
  ├── Output: 6-month forecast horizon, mean prediction + uncertainty
  └── Uncertainty: Monte Carlo Dropout (N=100 forward passes at inference)

Risk Tiering:
  ├── High risk: forecast level < 10th percentile of historical distribution
  ├── Medium risk: forecast level between 10th and 25th percentile
  └── Low risk: forecast level > 25th percentile of historical distribution

Water Quality Overlay (secondary):
  └── Geological risk layers: Arsenic (mafic bedrock zones), Uranium (granite), Methane (organic sediments)
```

---

## Dashboard Features (Prototype)

| Feature | Description |
|---|---|
| Spatial Risk Map | NB private wells colour-coded by simulated drought risk (High / Medium / Low) |
| LSTM Forecast Charts | Per-well time-series with 1–6 month forecast and 10th/90th percentile uncertainty bounds |
| Water Quality Overlays | Toggle-able geological risk layers (As, U, CH4) |
| Risk Statistics | Province-wide summary: wells at high / medium / low risk |

---

## Data Sources

| Dataset | Source | Status |
|---|---|---|
| Well records & static levels | [NB OWLS (Online Well Log System)](https://www.gnb.ca/0062/OWLS/index-e.asp) | Identified |
| Daily meteorological data | [ECCC Climate Data](https://climate.weather.gc.ca/) | Identified |
| Surficial geology | [NB Geological Survey](https://www2.gnb.ca/content/gnb/en/departments/erd/energy/content/minerals/content/geology_data.html) | Identified |
| Bedrock geology (aquifer type proxy) | NB GSB Open Data | Identified |

---

## Python Stack (Implementation)

```python
torch          # LSTM model implementation and training
pandas         # Time-series data management
numpy          # Array operations
geopandas      # Spatial well data
matplotlib     # Forecast visualisation
scikit-learn   # Preprocessing, metrics
```

---

## Roadmap

- [x] Research design and model architecture specified
- [x] Interactive dashboard (proof-of-concept with simulated data)
- [x] Data sources identified and access confirmed
- [ ] Data ingestion pipeline (OWLS + ECCC)
- [ ] Feature engineering and time-series alignment
- [ ] LSTM implementation (PyTorch)
- [ ] Model training and cross-validation
- [ ] Monte Carlo Dropout uncertainty quantification
- [ ] Risk tiering and alert logic
- [ ] Deployment as static web app with real model inference

---

## Why Groundwater Matters for Geoscience

Groundwater modelling sits at the intersection of hydrogeology, climate science, and AI — and the skills involved (subsurface characterisation, uncertainty quantification, time-series prediction, spatial analysis) transfer directly to mineral exploration and environmental geoscience. This project is designed to build and demonstrate those skills on a real, socially relevant problem using publicly available NB data.

---

## Author

**Dele Falebita, PhD** — GIT APEGNB & Data Scientist  
[github.com/erirera](https://github.com/erirera) | Moncton, New Brunswick, Canada

---

*Research design completed April 2026. PyTorch implementation in progress.*  
*License: CC0-1.0*
