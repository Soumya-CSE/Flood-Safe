"""
Automated glacial lake change-detection pipeline.

Demonstrates the actual algorithm used in real satellite-based GLOF
monitoring (water-body extraction via a spectral water index + pixel
counting to get area, repeated across time to get a growth trend) —
running here on synthetic raster snapshots so the pipeline works fully
offline in this prototype.

To point this at REAL data in production, replace `generate_synthetic_snapshots()`
with a Google Earth Engine / Sentinel Hub pull, e.g.:

    import ee
    ee.Initialize()
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(lake_point)
        .filterDate(start_date, end_date)
        .map(lambda img: img.normalizedDifference(["B3", "B8"]).rename("NDWI"))
    )
    # threshold NDWI > 0.2 to get a water mask, then use ee.Image.pixelArea()
    # summed over the water mask to get lake area in m^2 for each date.

The detection logic below (threshold -> water mask -> pixel-area sum) is
IDENTICAL in structure to that real pipeline; only the image source differs.
"""

import numpy as np
import pandas as pd

GRID_SIZE = 300          # synthetic raster is GRID_SIZE x GRID_SIZE pixels (~3km x 3km scene at 10m)
PIXEL_RESOLUTION_M = 10  # matches Sentinel-2 10m bands, for realistic area math
PIXEL_AREA_KM2 = (PIXEL_RESOLUTION_M / 1000) ** 2
WATER_INDEX_THRESHOLD = 0.2  # analogous to an NDWI water threshold


def generate_synthetic_snapshots(village_row: pd.Series, n_snapshots: int = 6, seed: int = None):
    """
    Builds a time series of synthetic 'satellite' rasters showing a glacial
    lake growing, calibrated so the FINAL snapshot's detected area matches
    the village's current glacial_lake_area_km2, and the total growth over
    the series matches its glacial_lake_growth_rate_pct_5yr. Each raster is
    a water-index-like grid (values 0-1) exactly like an NDWI image would be.
    """
    rng = np.random.default_rng(seed or hash(village_row["village_id"]) % (2**32))
    final_area_km2 = max(float(village_row["glacial_lake_area_km2"]), 0.02)
    growth_pct = float(village_row["glacial_lake_growth_rate_pct_5yr"])
    start_area_km2 = final_area_km2 / (1 + growth_pct / 100)

    center = GRID_SIZE // 2 + rng.integers(-3, 3)
    years = np.linspace(0, 5, n_snapshots)
    areas_target = np.linspace(start_area_km2, final_area_km2, n_snapshots)

    snapshots = []
    for target_area in areas_target:
        target_pixels = target_area / PIXEL_AREA_KM2
        radius = max(1.5, np.sqrt(target_pixels / np.pi))

        yy, xx = np.mgrid[0:GRID_SIZE, 0:GRID_SIZE]
        dist = np.sqrt((yy - center) ** 2 + (xx - center) ** 2)
        # water index: high (near 1) inside the lake radius, low outside, with
        # a noisy realistic-looking edge like a real spectral index would have
        noise = rng.normal(0, 0.05, size=dist.shape)
        raster = np.clip(1.2 - (dist / radius) + noise, 0, 1)
        snapshots.append(raster)

    return years, snapshots


def detect_water_area_km2(raster: np.ndarray) -> float:
    """The actual change-detection step: threshold the water index and sum
    pixel area — this line is unchanged whether the raster is synthetic or
    a real Sentinel band-math result."""
    water_mask = raster > WATER_INDEX_THRESHOLD
    return float(water_mask.sum() * PIXEL_AREA_KM2)


def run_change_detection(village_row: pd.Series, n_snapshots: int = 6) -> pd.DataFrame:
    years, snapshots = generate_synthetic_snapshots(village_row, n_snapshots)
    records = []
    for yr, raster in zip(years, snapshots):
        records.append(dict(
            year_offset=round(float(yr), 1),
            detected_area_km2=round(detect_water_area_km2(raster), 4),
        ))
    df = pd.DataFrame(records)
    df["pct_change_from_start"] = (
        (df["detected_area_km2"] - df["detected_area_km2"].iloc[0])
        / df["detected_area_km2"].iloc[0] * 100
    ).round(1)
    return df, snapshots
