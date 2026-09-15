"""
Cascading hazard simulator — models how a glacial lake outburst (GLOF) at one
village propagates downstream over time and escalates risk at other villages,
instead of treating each village's risk as an independent, static number.

This is a simplified physical model (distance/elevation-based wave
propagation), not a full hydrodynamic simulation (e.g. HEC-RAS) — it is
intended to demonstrate the cascading-hazard concept end-to-end in a
hackathon prototype. Swap `WAVE_SPEED_KMPH` and the elevation-drop rule for
a real routed hydrograph model in production.
"""

import numpy as np
import pandas as pd

# Typical GLOF debris/flood-wave travel speed in steep Himalayan terrain
# (order-of-magnitude estimate used in several published GLOF studies for
# the initial surge in steep upper reaches; slows further downstream).
WAVE_SPEED_KMPH = 18.0

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def breach_severity(source_row) -> float:
    """0-1 severity score for a hypothetical breach at this lake, based on
    the same GLOF precursor signals used in the main risk model."""
    growth = source_row["glacial_lake_growth_rate_pct_5yr"] / 65.0  # normalize vs sample max
    melt = source_row["glacier_melt_index"]
    dam_risk = {"low": 0.3, "medium": 0.6, "high": 1.0, "not_applicable": 0.0}.get(
        source_row["moraine_dam_stability"], 0.5
    )
    area_factor = min(source_row["glacial_lake_area_km2"] / 2.5, 1.0)
    severity = 0.30 * growth + 0.25 * melt + 0.30 * dam_risk + 0.15 * area_factor
    return float(np.clip(severity, 0.05, 1.0))


def simulate_cascade(df: pd.DataFrame, source_village_id: str, max_hours: int = 6) -> pd.DataFrame:
    """
    Simulates a GLOF breach at `source_village_id` and computes, for every
    other village: distance from source, arrival time of the flood wave, and
    an impact score that decays with distance/time and scales with the
    source lake's breach severity. Only villages at or below the source's
    elevation are considered "downstream" (a simple proxy for real flow
    routing, since real river-network data isn't in this sample dataset).
    """
    source = df[df["village_id"] == source_village_id].iloc[0]
    severity = breach_severity(source)

    rows = []
    for _, v in df.iterrows():
        dist_km = haversine_km(source["lat"], source["lon"], v["lat"], v["lon"])
        is_downstream = (v["elevation_m"] <= source["elevation_m"]) and (v["village_id"] != source["village_id"])
        if v["village_id"] == source["village_id"]:
            arrival_hr = 0.0
            impact = severity
        elif is_downstream and dist_km <= max_hours * WAVE_SPEED_KMPH:
            arrival_hr = dist_km / WAVE_SPEED_KMPH
            # impact decays with distance (inverse-ish) and scales with source severity
            impact = severity * max(0.0, 1 - dist_km / (max_hours * WAVE_SPEED_KMPH))
        else:
            arrival_hr = None
            impact = 0.0

        rows.append(dict(
            village_id=v["village_id"], village_name=v["village_name"],
            district=v["district"], lat=v["lat"], lon=v["lon"],
            distance_km=round(dist_km, 1),
            is_source=(v["village_id"] == source["village_id"]),
            downstream=is_downstream,
            arrival_hr=None if arrival_hr is None else round(arrival_hr, 2),
            impact_score=round(impact, 3),
        ))

    result = pd.DataFrame(rows).sort_values("distance_km")
    result.attrs["source_village"] = source["village_name"]
    result.attrs["severity"] = severity
    return result


def impact_level(impact_score: float) -> str:
    if impact_score <= 0:
        return "Not affected"
    if impact_score < 0.2:
        return "Minor"
    if impact_score < 0.45:
        return "Moderate"
    if impact_score < 0.7:
        return "Severe"
    return "Extreme"
