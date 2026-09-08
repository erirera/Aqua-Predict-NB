"""
Dashboard data exporter for Aqua-Predict-NB.
Serializes well forecasts, risk tiers, and provincial summary metrics into
JSON and GeoJSON formats consumed by the Leaflet + Chart.js web dashboard.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

from src.config import DATA_DIR, REPORTS_DIR


def export_dashboard_json(
    wells_data: List[Dict],
    summary_stats: Dict,
    evaluation_metrics: Dict,
    output_dir: Path = DATA_DIR
) -> Tuple[Path, Path, Path]:
    """
    Serializes:
    1. processed_wells.json: well array for map and forecast charts.
    2. pipeline_summary.json: metrics for dashboard cards.
    3. nb_wells_risk.geojson: spatial standard GeoJSON for GIS integration.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    wells_json_path = output_dir / "processed_wells.json"
    summary_json_path = output_dir / "pipeline_summary.json"
    geojson_path = output_dir / "nb_wells_risk.geojson"

    # Merge evaluation metrics into summary stats
    full_summary = {**summary_stats, **evaluation_metrics}

    # 1. Write processed_wells.json
    with open(wells_json_path, "w") as f:
        json.dump(wells_data, f, indent=2)

    # 2. Write pipeline_summary.json
    with open(summary_json_path, "w") as f:
        json.dump(full_summary, f, indent=2)

    # 3. Construct GeoJSON FeatureCollection
    features = []
    for w in wells_data:
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [w["lng"], w["lat"]]
            },
            "properties": {
                "well_id": w["id"],
                "county": w["county"],
                "aquifer": w["aquifer"],
                "depth_m": w["depth"],
                "risk": w["risk"],
                "quality_threat": w["quality"],
                "baseline_gwl_m": w["baseLevel"],
                "min_forecast_p50_m": min(w["forecastP50"]),
                "min_forecast_p10_m": min(w["forecastP10"])
            }
        }
        features.append(feature)

    geojson_doc = {
        "type": "FeatureCollection",
        "features": features
    }

    with open(geojson_path, "w") as f:
        json.dump(geojson_doc, f, indent=2)

    print(f"[SUCCESS] Exported dashboard wells JSON: {wells_json_path}")
    print(f"[SUCCESS] Exported pipeline summary JSON: {summary_json_path}")
    print(f"[SUCCESS] Exported GeoJSON dataset: {geojson_path}")

    return wells_json_path, summary_json_path, geojson_path
