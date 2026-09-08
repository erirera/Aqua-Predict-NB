"""
Risk tiering and hazard classification module for Aqua-Predict-NB.
Implements hydrogeological drought risk ranking based on historical percentiles,
critical pump intake thresholds, and geochemical threat overlays.
"""

from typing import Dict, List, Tuple
import numpy as np

from src.config import risk_thresholds


def classify_well_drought_risk(
    historical_gwl: List[float],
    forecast_p50: List[float],
    forecast_p10: List[float],
    pump_intake_depth_m: float,
    horizon_months: int = 3
) -> Dict[str, any]:
    """
    Classifies well drought risk:
    - High: P10 forecast or P50 forecast breaches 10th percentile of historical record,
      or water level drops within 2.0m of pump intake.
    - Medium: Forecast level is between 10th and 25th percentile of historical record.
    - Low: Forecast level remains above 25th percentile.
    """
    hist = np.array(historical_gwl)
    q10 = np.percentile(hist, risk_thresholds.HIGH_DROUGHT_PERCENTILE)
    q25 = np.percentile(hist, risk_thresholds.MEDIUM_DROUGHT_PERCENTILE)

    # We evaluate minimum forecasted water level (most negative) over the designated horizon
    fc_eval_p50 = np.min(forecast_p50[:horizon_months])
    fc_eval_p10 = np.min(forecast_p10[:horizon_months])

    pump_intake_buffer = fc_eval_p50 - pump_intake_depth_m

    # Note: GWL is negative metres (depth below ground surface, e.g. -12m is deeper than -5m)
    # So "lower water table" means a smaller / more negative number: fc_eval < q10
    if fc_eval_p10 <= q10 or pump_intake_buffer <= risk_thresholds.CRITICAL_DROP_METRES:
        risk = "high"
    elif fc_eval_p50 <= q25:
        risk = "medium"
    else:
        risk = "low"

    return {
        "risk": risk,
        "historical_q10_m": round(float(q10), 3),
        "historical_q25_m": round(float(q25), 3),
        "min_forecast_p50_m": round(float(fc_eval_p50), 3),
        "min_forecast_p10_m": round(float(fc_eval_p10), 3),
        "pump_intake_buffer_m": round(float(pump_intake_buffer), 2)
    }


def compute_provincial_risk_summary(
    well_forecasts: List[Dict],
    horizon_months: int = 3
) -> Tuple[List[Dict], Dict[str, any]]:
    """
    Iterates over all wells, classifies risk, determines combined geo-risk,
    and returns enriched well list + province-wide summary statistics.
    """
    enriched_wells = []
    total_wells = len(well_forecasts)
    high_count = 0
    medium_count = 0
    low_count = 0
    combined_georisk_count = 0

    for w in well_forecasts:
        classification = classify_well_drought_risk(
            historical_gwl=w["all_historical_gwl"],
            forecast_p50=w["forecast_p50"],
            forecast_p10=w["forecast_p10"],
            pump_intake_depth_m=w["pump_intake_depth_m"],
            horizon_months=horizon_months
        )

        risk = classification["risk"]
        geo_threat = w["geo_threat"]

        # Combined geo-risk: well is at risk of drying AND has geological contamination threat
        is_georisk = (risk in ["high", "medium"]) and (geo_threat != "None")
        if is_georisk:
            combined_georisk_count += 1

        if risk == "high":
            high_count += 1
        elif risk == "medium":
            medium_count += 1
        else:
            low_count += 1

        enriched = {
            "id": w["well_id"],
            "county": w["county"],
            "lat": w["latitude"],
            "lng": w["longitude"],
            "depth": int(w["total_depth_m"]),
            "casingDepth": int(w["casing_depth_m"]),
            "aquifer": w["aquifer_type"],
            "quality": geo_threat,
            "risk": risk,
            "baseLevel": w["baseline_gwl_m"],
            "pumpIntake": w["pump_intake_depth_m"],
            "historical": w["historical_recent_gwl"],
            "historicalDates": w["historical_dates"],
            "forecastP10": w["forecast_p10"],
            "forecastP50": w["forecast_p50"],
            "forecastP90": w["forecast_p90"],
            "forecastDates": w["forecast_dates"],
            "uncertaintyStd": w["uncertainty_std"],
            "droughtBufferM": classification["pump_intake_buffer_m"]
        }
        enriched_wells.append(enriched)

    summary = {
        "total_wells": total_wells,
        "high_drought_risk_count": high_count,
        "medium_drought_risk_count": medium_count,
        "low_drought_risk_count": low_count,
        "combined_georisk_count": combined_georisk_count,
        "high_risk_pct": round((high_count / max(total_wells, 1)) * 100, 1),
        "horizon_evaluated_months": horizon_months
    }

    return enriched_wells, summary
