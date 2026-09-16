import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

FEATURES = [
    "elevation_m",
    "slope_deg",
    "distance_to_river_km",
    "lulc_forest_pct",
    "rainfall_today_mm",
    "rainfall_3day_mm",
    "soil_moisture_pct",
    "historical_landslide_count",
    # GLOF (Glacial Lake Outburst Flood) features
    "glacial_lake_area_km2",
    "glacial_lake_growth_rate_pct_5yr",
    "distance_to_glacial_lake_km",
    "glacier_melt_index",
    # Traditional Ecological Knowledge (TEK) — community/elder-observed
    # precursor signs (animal behavior, spring water changes, informal
    # warnings), collected via the app's community feedback form and fed
    # back into the model as a genuine, weighted feature — not a gimmick
    # tacked on for show. See: Simeulue Island "smong" oral-tradition
    # tsunami warning (2004) as the real-world precedent for why this matters.
    "traditional_knowledge_signal",
]

LITHOLOGY_MAP = {"low": 0, "medium": 1, "high": 2}
MORAINE_MAP = {"low": 0, "medium": 1, "high": 2, "not_applicable": 0}

ENCODED_FEATURES = FEATURES + ["lithology_risk_code", "moraine_dam_stability_code"]


def prep_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["lithology_risk_code"] = out["lithology_risk"].map(LITHOLOGY_MAP).fillna(1)
    out["moraine_dam_stability_code"] = out["moraine_dam_stability"].map(MORAINE_MAP).fillna(0)
    return out


def train_model(df: pd.DataFrame):
    data = prep_features(df)
    X = data[ENCODED_FEATURES]
    y = data["label"]
    model = RandomForestClassifier(
        n_estimators=200, max_depth=6, random_state=42, class_weight="balanced"
    )
    model.fit(X, y)
    return model


def predict_risk(model, df: pd.DataFrame) -> pd.DataFrame:
    data = prep_features(df)
    X = data[ENCODED_FEATURES]
    proba = model.predict_proba(X)[:, 1]
    out = df.copy()
    out["risk_probability"] = proba
    out["risk_level"] = pd.cut(
        proba,
        bins=[-0.01, 0.25, 0.5, 0.75, 1.01],
        labels=["Low", "Medium", "High", "Critical"],
    )
    return out


RISK_COLORS = {
    "Low": [34, 139, 34, 200],
    "Medium": [255, 193, 7, 200],
    "High": [255, 111, 0, 200],
    "Critical": [211, 47, 47, 220],
}


def risk_color(level: str):
    return RISK_COLORS.get(level, [128, 128, 128, 180])


def simulate_next_reading(row: pd.Series, rng: np.random.Generator) -> pd.Series:
    """Random-walk perturbation of the dynamic fields, standing in for a live
    IoT / satellite feed. Static terrain fields are left untouched."""
    r = row.copy()
    r["rainfall_today_mm"] = max(0.0, r["rainfall_today_mm"] + rng.normal(0, 6))
    r["rainfall_3day_mm"] = max(0.0, r["rainfall_3day_mm"] + rng.normal(0, 8))
    r["soil_moisture_pct"] = float(np.clip(r["soil_moisture_pct"] + rng.normal(0, 3), 2, 98))
    if bool(r.get("glacial_lake_present", False)):
        # daily temperature-driven melt variation + slow lake-area creep
        r["glacier_melt_index"] = float(np.clip(r["glacier_melt_index"] + rng.normal(0, 0.03), 0.02, 0.99))
        r["glacial_lake_area_km2"] = float(max(0.0, r["glacial_lake_area_km2"] + rng.normal(0, 0.01)))
    r["data_provenance"] = "simulated_live"
    return r
