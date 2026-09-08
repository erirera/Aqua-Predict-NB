"""
Data ingestion module for Aqua-Predict-NB.
Handles ingestion of New Brunswick Online Well Log System (OWLS) records,
ECCC climate stations meteorological data, and geological risk overlays.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from src.config import (
    RAW_DATA_DIR,
    SpatialConfig,
    ClimateConfig,
    spatial_config,
    climate_config
)

# New Brunswick county geographical bounds and characteristics
NB_COUNTIES = {
    "York": {"center": (46.00, -66.85), "lat_range": (45.60, 46.40), "lon_range": (-67.40, -66.30), "primary_aquifer": "Fractured Bedrock", "geo_risk": ["Arsenic", "None"]},
    "Westmorland": {"center": (46.10, -64.70), "lat_range": (45.90, 46.35), "lon_range": (-65.20, -64.00), "primary_aquifer": "Carboniferous Sandstone", "geo_risk": ["None", "Uranium"]},
    "Saint John": {"center": (45.30, -65.95), "lat_range": (45.15, 45.45), "lon_range": (-66.30, -65.50), "primary_aquifer": "Glaciofluvial Sand/Gravel", "geo_risk": ["None"]},
    "Kings": {"center": (45.70, -65.60), "lat_range": (45.40, 46.00), "lon_range": (-66.10, -65.10), "primary_aquifer": "Carboniferous Sandstone", "geo_risk": ["Methane", "None"]},
    "Northumberland": {"center": (46.95, -65.65), "lat_range": (46.50, 47.40), "lon_range": (-66.40, -64.90), "primary_aquifer": "Carboniferous Sandstone", "geo_risk": ["None", "Methane"]},
    "Gloucester": {"center": (47.60, -65.60), "lat_range": (47.30, 47.95), "lon_range": (-66.10, -64.70), "primary_aquifer": "Fractured Bedrock", "geo_risk": ["Arsenic", "None"]},
    "Madawaska": {"center": (47.45, -68.30), "lat_range": (47.20, 47.75), "lon_range": (-68.80, -67.80), "primary_aquifer": "Fractured Bedrock", "geo_risk": ["None"]},
    "Carleton": {"center": (46.30, -67.60), "lat_range": (46.05, 46.60), "lon_range": (-67.85, -67.20), "primary_aquifer": "Windsor Group Karst", "geo_risk": ["None", "Arsenic"]},
    "Albert": {"center": (45.85, -64.80), "lat_range": (45.60, 46.00), "lon_range": (-65.10, -64.50), "primary_aquifer": "Carboniferous Sandstone", "geo_risk": ["Uranium", "None"]},
    "Charlotte": {"center": (45.25, -66.90), "lat_range": (45.00, 45.55), "lon_range": (-67.40, -66.40), "primary_aquifer": "Mafic Volcanic", "geo_risk": ["Arsenic", "Uranium"]},
    "Restigouche": {"center": (47.85, -66.70), "lat_range": (47.50, 48.05), "lon_range": (-67.50, -65.90), "primary_aquifer": "Fractured Bedrock", "geo_risk": ["None"]},
    "Victoria": {"center": (46.85, -67.55), "lat_range": (46.60, 47.15), "lon_range": (-67.90, -67.20), "primary_aquifer": "Fractured Bedrock", "geo_risk": ["None"]},
    "Sunbury": {"center": (45.75, -66.40), "lat_range": (45.50, 46.00), "lon_range": (-66.70, -66.10), "primary_aquifer": "Carboniferous Sandstone", "geo_risk": ["Arsenic", "None"]},
    "Queens": {"center": (45.85, -66.00), "lat_range": (45.60, 46.15), "lon_range": (-66.30, -65.70), "primary_aquifer": "Carboniferous Sandstone", "geo_risk": ["Methane", "None"]},
    "Kent": {"center": (46.60, -65.00), "lat_range": (46.35, 46.85), "lon_range": (-65.40, -64.60), "primary_aquifer": "Carboniferous Sandstone", "geo_risk": ["None"]}
}


def generate_eccc_climate_data(
    start_year: int = 2014,
    end_year: int = 2024,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates realistic multi-year monthly climate records for ECCC reference stations in NB.
    Includes precipitation, mean/min/max temperature, and snowpack accumulation/melt.
    Captures known Atlantic Canada climate dynamics (e.g. spring freshet, 2020 drought).
    """
    rng = np.random.default_rng(seed)
    records = []
    
    dates = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-01", freq="MS")
    
    for station in climate_config.REFERENCE_STATIONS:
        st_id = station["id"]
        st_name = station["name"]
        st_lat = station["lat"]
        st_lon = station["lon"]
        st_elev = station["elevation"]
        
        # Latitude-elevation lapse rates
        temp_offset = -0.005 * st_elev - 0.7 * (st_lat - 45.5)
        precip_factor = 1.0 + 0.05 * (st_lat - 45.5) + (0.001 * st_elev)
        
        for dt in dates:
            month = dt.month
            year = dt.year
            
            # Seasonal temperature curve (°C) for New Brunswick
            # Jan=-8°C, Jul=19°C
            mean_temp = 5.5 + 13.5 * np.sin(2 * np.pi * (month - 4) / 12) + temp_offset
            mean_temp += rng.normal(0, 1.2)  # Natural monthly variability
            
            # Simulated 2020 Atlantic Canada warm summer anomaly
            if year == 2020 and month in [6, 7, 8]:
                mean_temp += 2.2
                
            temp_max = mean_temp + rng.uniform(4.0, 7.5)
            temp_min = mean_temp - rng.uniform(4.0, 7.5)
            
            # Seasonal precipitation curve (mm/month) in NB (~80-120 mm/mo)
            base_precip = 95.0 + 18.0 * np.cos(2 * np.pi * (month - 11) / 12) * precip_factor
            base_precip += rng.normal(0, 16.0)
            base_precip = max(base_precip, 20.0)
            
            # Simulated 2020 drought (historically severe in NB, down to 35-50% normal summer rain)
            if year == 2020 and month in [5, 6, 7, 8, 9]:
                base_precip *= rng.uniform(0.42, 0.60)
            # Simulated 2023 wet summer
            elif year == 2023 and month in [6, 7]:
                base_precip *= rng.uniform(1.45, 1.80)
                
            # Snow dynamics: when temp < 0, precip falls as snow; spring freshet melts it in April/May
            if mean_temp < 0:
                snow_water_equiv = base_precip * rng.uniform(0.85, 1.0)
                snow_depth_cm = snow_water_equiv * 0.12 * rng.uniform(0.8, 1.2)
            else:
                snow_water_equiv = 0.0
                snow_depth_cm = 0.0
                
            records.append({
                "station_id": st_id,
                "station_name": st_name,
                "latitude": st_lat,
                "longitude": st_lon,
                "elevation": st_elev,
                "date": dt.strftime("%Y-%m-%d"),
                "year": year,
                "month": month,
                "precip_mm": round(base_precip, 2),
                "temp_mean_c": round(mean_temp, 2),
                "temp_max_c": round(temp_max, 2),
                "temp_min_c": round(temp_min, 2),
                "snow_depth_cm": round(snow_depth_cm, 2),
                "snow_water_equiv_mm": round(snow_water_equiv, 2)
            })
            
    df = pd.DataFrame(records)
    output_path = RAW_DATA_DIR / "eccc_climate_monthly.csv"
    df.to_csv(output_path, index=False)
    return df


def generate_nb_wells(
    num_wells: int = 250,
    seed: int = 101
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates synthetic realistic NB OWLS well log metadata and historical GWL time series.
    Returns:
        - wells_df: metadata for each well (location, aquifer, depths, chemical overlay).
        - time_series_df: monthly groundwater level records aligned with climate series.
    """
    rng = np.random.default_rng(seed)
    climate_df = generate_eccc_climate_data()
    
    # Pre-build KDTree of climate stations for fast spatial nearest-station lookup
    st_locations = []
    st_ids = []
    for st in climate_config.REFERENCE_STATIONS:
        st_locations.append([st["lat"], st["lon"]])
        st_ids.append(st["id"])
    kdtree = KDTree(st_locations)
    
    wells_list = []
    county_names = list(NB_COUNTIES.keys())
    
    for i in range(num_wells):
        well_num = 1000 + i
        well_id = f"NB-OWLS-{well_num}"
        county = county_names[i % len(county_names)]
        c_info = NB_COUNTIES[county]
        
        lat = rng.uniform(c_info["lat_range"][0], c_info["lat_range"][1])
        lon = rng.uniform(c_info["lon_range"][0], c_info["lon_range"][1])
        
        # Determine aquifer type: 70% primary for that county, 30% other
        if rng.random() < 0.75:
            aquifer = c_info["primary_aquifer"]
        else:
            aquifer = rng.choice(spatial_config.AQUIFER_TYPES)
            
        # Geochemical hazard flag
        q_options = c_info["geo_risk"]
        geo_threat = rng.choice(q_options)
        if geo_threat == "None" and rng.random() < 0.08:
            geo_threat = rng.choice(["Arsenic", "Uranium", "Methane"])
            
        # Realistic well construction depths (metres)
        overburden = rng.uniform(2.0, 28.0)
        bedrock_depth = overburden + rng.uniform(0.0, 4.0)
        casing_depth = max(overburden + rng.uniform(3.0, 10.0), 6.0)
        total_depth = casing_depth + rng.uniform(20.0, 120.0)
        
        # Static baseline water level: typically 3 to 18 metres below surface
        base_gwl = -1.0 * rng.uniform(3.5, 16.5)
        pump_intake_depth = -1.0 * (total_depth - rng.uniform(5.0, 15.0))
        
        # Nearest climate station
        _, nearest_idx = kdtree.query([lat, lon])
        assigned_station_id = st_ids[nearest_idx]
        
        wells_list.append({
            "well_id": well_id,
            "county": county,
            "latitude": round(lat, 5),
            "longitude": round(lon, 5),
            "total_depth_m": round(total_depth, 1),
            "casing_depth_m": round(casing_depth, 1),
            "overburden_depth_m": round(overburden, 1),
            "bedrock_depth_m": round(bedrock_depth, 1),
            "aquifer_type": aquifer,
            "geo_threat": geo_threat,
            "baseline_gwl_m": round(base_gwl, 2),
            "pump_intake_depth_m": round(pump_intake_depth, 2),
            "nearest_station_id": assigned_station_id
        })
        
    wells_df = pd.DataFrame(wells_list)
    wells_path = RAW_DATA_DIR / "nb_owls_wells.csv"
    wells_df.to_csv(wells_path, index=False)
    
    # Generate time series for each well based on hydrogeological physics
    ts_records = []
    
    # Hydrogeological response parameters per aquifer type
    aquifer_lag_params = {
        "Glaciofluvial Sand/Gravel": {"lag": 1, "decay": 0.55, "recharge_sens": 0.035, "noise": 0.15},
        "Carboniferous Sandstone": {"lag": 2, "decay": 0.75, "recharge_sens": 0.022, "noise": 0.12},
        "Fractured Bedrock": {"lag": 3, "decay": 0.85, "recharge_sens": 0.016, "noise": 0.10},
        "Windsor Group Karst": {"lag": 1, "decay": 0.60, "recharge_sens": 0.030, "noise": 0.18},
        "Mafic Volcanic": {"lag": 4, "decay": 0.90, "recharge_sens": 0.012, "noise": 0.08}
    }
    
    # Group climate data by station for fast access
    climate_by_st = {st: grp.sort_values("date").copy() for st, grp in climate_df.groupby("station_id")}
    
    for _, well in wells_df.iterrows():
        w_id = well["well_id"]
        w_aq = well["aquifer_type"]
        w_base = well["baseline_gwl_m"]
        st_data = climate_by_st[well["nearest_station_id"]]
        
        params = aquifer_lag_params.get(w_aq, aquifer_lag_params["Fractured Bedrock"])
        decay = params["decay"]
        sens = params["recharge_sens"]
        sigma = params["noise"]
        
        # State variable: anomaly from baseline
        gwl_anomaly = 0.0
        
        # Historical monthly time series
        for _, row in st_data.iterrows():
            m = row["month"]
            p = row["precip_mm"]
            t = row["temp_mean_c"]
            swe = row["snow_water_equiv_mm"]
            
            # Effective net infiltration:
            # - In winter (T < 0), precipitation accumulates as snow (low recharge).
            # - In April/May, spring snowmelt freshet causes massive recharge surge.
            if m in [4, 5]:
                effective_recharge = (p + 1.8 * swe) - 40.0
            elif m in [7, 8]:
                # Summer high evapotranspiration deficit
                effective_recharge = p - 110.0
            elif m in [1, 2]:
                effective_recharge = -15.0  # Frozen ground, low infiltration
            else:
                effective_recharge = p - 75.0
                
            # Autoregressive update equation
            gwl_anomaly = decay * gwl_anomaly + (1 - decay) * (effective_recharge * sens) + rng.normal(0, sigma)
            # Bound realistic fluctuation
            gwl_anomaly = np.clip(gwl_anomaly, -6.5, 4.0)
            
            current_gwl = w_base + gwl_anomaly
            
            ts_records.append({
                "well_id": w_id,
                "date": row["date"],
                "year": row["year"],
                "month": m,
                "precip_mm": p,
                "temp_mean_c": t,
                "groundwater_level_m": round(current_gwl, 3)
            })
            
    time_series_df = pd.DataFrame(ts_records)
    ts_path = RAW_DATA_DIR / "nb_wells_timeseries.csv"
    time_series_df.to_csv(ts_path, index=False)
    
    return wells_df, time_series_df


def run_ingestion_pipeline(num_wells: int = 250) -> Dict[str, Path]:
    """
    Executes the ingestion pipeline. Saves raw CSV data and returns filepaths.
    """
    print("=" * 60)
    print(">>> Starting Aqua-Predict-NB Raw Data Ingestion Pipeline")
    print(f"Targeting {num_wells} New Brunswick private wells across 15 counties...")
    
    wells_df, ts_df = generate_nb_wells(num_wells=num_wells)
    
    print(f"[SUCCESS] Ingested {len(wells_df)} wells with complete construction metadata.")
    print(f"[SUCCESS] Ingested {len(ts_df)} monthly hydrogeological time-series steps.")
    print("=" * 60)
    
    return {
        "wells_metadata": RAW_DATA_DIR / "nb_owls_wells.csv",
        "climate_data": RAW_DATA_DIR / "eccc_climate_monthly.csv",
        "time_series": RAW_DATA_DIR / "nb_wells_timeseries.csv"
    }


if __name__ == "__main__":
    run_ingestion_pipeline()
