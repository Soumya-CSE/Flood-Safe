"""
Generates a sample village-level dataset for the Flash Flood / Landslide /
GLOF (Glacial Lake Outburst Flood) Prediction System pilot (modelled loosely
on real glacier-belt hill districts: Chamoli & Rudraprayag (2013 Kedarnath /
Chorabari Lake event, 2021 Chamoli event), North Sikkim (2023 South Lhonak
Lake GLOF), Kinnaur, and Kullu).

This is SYNTHETIC / SAMPLE data meant as a starting point for the prototype.
Replace this file's output (data/villages_sample.csv) with real
IMD / ISRO Bhoonidhi / Bhuvan DEM / GSI Bhukosh / ICIMOD glacial lake
inventory values once available.
Every row is tagged in `data_provenance` so real vs simulated stays traceable.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# (name, district, lat, lon, elevation_m, shelter, glacial_lake_present)
villages = [
    ("Joshimath", "Chamoli", 30.5553, 79.5644, 1875, "Community Hall Joshimath", True),
    ("Lambagad", "Chamoli", 30.5901, 79.6390, 1650, "Govt School Lambagad", True),
    ("Pipalkoti", "Chamoli", 30.4667, 79.4667, 1265, "Relief Camp Pipalkoti", False),
    ("Reni", "Chamoli", 30.5892, 79.7213, 1850, "Reni Panchayat Bhawan", True),
    ("Tapovan", "Chamoli", 30.5300, 79.5850, 1920, "Tapovan Community Centre", True),
    ("Kedarnath", "Rudraprayag", 30.7346, 79.0669, 3583, "Kedarnath Base Camp Shelter", True),
    ("Sonprayag", "Rudraprayag", 30.6167, 78.9500, 1829, "Sonprayag Transit Shelter", False),
    ("Guptkashi", "Rudraprayag", 30.5333, 79.0833, 1319, "Guptkashi Govt School", False),
    ("Chamba", "Tehri Garhwal", 30.3667, 78.4833, 1676, "Chamba Relief Centre", False),
    ("Ghansali", "Tehri Garhwal", 30.3833, 78.6333, 1150, "Ghansali Community Hall", False),
    ("Uttarkashi Town", "Uttarkashi", 30.7300, 78.4500, 1352, "Uttarkashi District Shelter", False),
    ("Bhatwari", "Uttarkashi", 30.7833, 78.6333, 1500, "Bhatwari Panchayat Ghar", False),
    ("Kalpa", "Kinnaur (HP)", 31.5375, 78.2588, 2960, "Kalpa Community Shelter", True),
    ("Sangla", "Kinnaur (HP)", 31.4257, 78.2649, 2680, "Sangla Relief Camp", True),
    ("Manali", "Kullu (HP)", 32.2432, 77.1892, 2050, "Manali Bus Stand Shelter", True),
    ("Kasol", "Kullu (HP)", 32.0100, 77.3145, 1640, "Kasol Community Hall", False),
    ("Aizawl Outskirts", "Aizawl (Mizoram)", 23.7271, 92.7176, 1132, "Aizawl Relief Centre", False),
    ("Gangtok Rural", "East Sikkim", 27.3389, 88.6065, 1650, "Gangtok Relief Shelter", True),
    ("Lachung", "North Sikkim", 27.6891, 88.7452, 2750, "Lachung Community Hall", True),
    ("Munsiyari", "Pithoragarh", 30.0668, 80.2394, 2298, "Munsiyari Govt Inter College", True),
]

rows = []
for i, (name, district, lat, lon, elev, shelter, has_lake) in enumerate(villages, start=1):
    slope_deg = rng.uniform(15, 55)
    dist_river_km = rng.uniform(0.1, 5.0)
    lulc_forest_pct = rng.uniform(20, 85)
    lithology_risk = rng.choice(["low", "medium", "high"], p=[0.3, 0.4, 0.3])
    hist_landslide_count = rng.poisson(1.5 if lithology_risk != "low" else 0.4)

    rainfall_today_mm = rng.gamma(2.0, 12.0)
    rainfall_3day_mm = rainfall_today_mm + rng.gamma(2.0, 20.0)
    soil_moisture_pct = np.clip(rng.normal(35 + slope_deg * 0.3, 8), 5, 95)

    # --- GLOF (Glacial Lake Outburst Flood) fields ---
    if has_lake:
        glacial_lake_area_km2 = round(float(rng.uniform(0.08, 2.6)), 3)
        glacial_lake_growth_rate_pct_5yr = round(float(rng.uniform(4, 65)), 1)  # lake expansion = glacier melt/moraine stress signal
        distance_to_glacial_lake_km = round(float(rng.uniform(1.5, 22)), 2)
        glacier_melt_index = round(float(np.clip(rng.normal(0.5 + (elev - 2000) / 4000, 0.15), 0.05, 0.98)), 2)
        moraine_dam_stability = rng.choice(["low", "medium", "high"], p=[0.35, 0.4, 0.25])  # risk level, not literal stability
    else:
        glacial_lake_area_km2 = 0.0
        glacial_lake_growth_rate_pct_5yr = 0.0
        distance_to_glacial_lake_km = 999.0  # effectively "not applicable" / far away
        glacier_melt_index = 0.0
        moraine_dam_stability = "not_applicable"

    moraine_risk_term = {"low": 0.3, "medium": 0.9, "high": 1.6, "not_applicable": 0.0}[moraine_dam_stability]
    glof_component = (
        0.02 * glacial_lake_growth_rate_pct_5yr
        + 1.1 * glacier_melt_index
        + moraine_risk_term
        - 0.04 * min(distance_to_glacial_lake_km, 30)
    ) if has_lake else 0.0

    # --- Traditional Ecological Knowledge (TEK) signal ---
    # Community/elder-observed precursor signs (animal behavior, spring water
    # changes, informal warnings) — probability of a flagged sign is higher
    # in already-unstable locations (communities notice real risk), so this
    # is correlated with but NOT redundant with the physical features above.
    tek_flag_prob = 0.12 + (0.15 if lithology_risk == "high" else 0.0) + (0.10 if has_lake else 0.0)
    traditional_knowledge_signal = int(rng.random() < min(tek_flag_prob, 0.5))
    tek_notes_pool = [
        "Elders reported unusual livestock restlessness before dawn",
        "Spring water near the village turned cloudy/muddy",
        "Community noted birds/wildlife leaving the upper slope area",
        "Local elder flagged a recurring pre-monsoon warning sign",
        "Unusual stream sound/flow change reported by residents",
    ]
    traditional_knowledge_notes = (
        str(rng.choice(tek_notes_pool)) if traditional_knowledge_signal else ""
    )

    # synthetic ground-truth label: higher slope + rainfall + soil moisture +
    # past landslides + GLOF susceptibility + community/traditional warning
    # signs -> higher chance of a historically recorded event
    risk_score = (
        0.02 * slope_deg
        + 0.01 * rainfall_3day_mm
        + 0.015 * soil_moisture_pct
        + 0.35 * hist_landslide_count
        + (1.2 if lithology_risk == "high" else 0.4 if lithology_risk == "medium" else 0)
        - 0.4 * dist_river_km
        + glof_component
        + 0.8 * traditional_knowledge_signal
        + rng.normal(0, 0.6)
    )

    rows.append(dict(
        village_id=f"V{i:03d}",
        village_name=name,
        district=district,
        state="",
        lat=lat,
        lon=lon,
        elevation_m=elev,
        slope_deg=round(slope_deg, 1),
        distance_to_river_km=round(dist_river_km, 2),
        lulc_forest_pct=round(lulc_forest_pct, 1),
        lithology_risk=lithology_risk,
        rainfall_today_mm=round(rainfall_today_mm, 1),
        rainfall_3day_mm=round(rainfall_3day_mm, 1),
        soil_moisture_pct=round(soil_moisture_pct, 1),
        historical_landslide_count=hist_landslide_count,
        glacial_lake_present=has_lake,
        glacial_lake_area_km2=glacial_lake_area_km2,
        glacial_lake_growth_rate_pct_5yr=glacial_lake_growth_rate_pct_5yr,
        distance_to_glacial_lake_km=distance_to_glacial_lake_km,
        glacier_melt_index=glacier_melt_index,
        moraine_dam_stability=moraine_dam_stability,
        traditional_knowledge_signal=traditional_knowledge_signal,
        traditional_knowledge_notes=traditional_knowledge_notes,
        raw_risk_score=risk_score,
        nearest_shelter=shelter,
        # shelter placed at a small random offset from the village as a stand-in
        shelter_lat=round(lat + rng.uniform(-0.01, 0.01), 5),
        shelter_lon=round(lon + rng.uniform(-0.01, 0.01), 5),
        data_provenance="synthetic_sample",
    ))

df = pd.DataFrame(rows)

# derive a binary historical-event label from the top ~35% of raw_risk_score
threshold = df["raw_risk_score"].quantile(0.65)
df["label"] = (df["raw_risk_score"] > threshold).astype(int)
df.drop(columns=["raw_risk_score"], inplace=True)

df.to_csv("data/villages_sample.csv", index=False)
print(df[["village_name", "glacial_lake_present", "glacier_melt_index",
          "moraine_dam_stability", "traditional_knowledge_signal",
          "slope_deg", "rainfall_3day_mm", "label"]])
print(f"\nSaved {len(df)} rows to data/villages_sample.csv")
