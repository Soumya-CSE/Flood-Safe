# Flash Flood Prediction System — Prototype

Prototype for MHA Problem Statement 192 (Flash Flood Prediction System for
Hilly Regions using Multi-Source Data), built with Streamlit.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

## What's inside

| Tab | What it does |
|---|---|
| 🗺️ Live Risk Map | Village-level risk map (Low/Medium/High/Critical), color + size coded, blue ring = glacial lake nearby |
| 📍 Village Detail | Risk gauge, live rainfall/soil-moisture trend, **glacial lake / GLOF status panel**, **explainable AI** (SHAP) breakdown |
| 🚨 Alerts | Lists villages at High/Critical risk, simulated Email/SMS/Siren dispatch + log |
| 🧭 Safe Route & Shelters | Nearest shelter per village + indicative evacuation path on map |
| 🗣️ Community Feedback | Citizens/volunteers submit ground reports → feeds back into model retraining |
| ℹ️ About | Explains what's real data vs simulated in this demo |

Sidebar controls:
- **Simulate next sensor reading** — advances a random-walk "live" feed (stand-in for real IoT/satellite ingestion) and recomputes risk for every village
- **Reset to baseline data** — clears the live feed and alert/history log

## Editing the dataset

Everything the model trains on lives in `data/villages_sample.csv`. To plug in
real data:

1. Keep the same column names (`village_id, village_name, district, lat, lon,
   elevation_m, slope_deg, distance_to_river_km, lulc_forest_pct,
   lithology_risk, rainfall_today_mm, rainfall_3day_mm, soil_moisture_pct,
   historical_landslide_count, glacial_lake_present, glacial_lake_area_km2,
   glacial_lake_growth_rate_pct_5yr, distance_to_glacial_lake_km,
   glacier_melt_index, moraine_dam_stability, label, nearest_shelter,
   shelter_lat, shelter_lon, data_provenance`)
2. Replace the synthetic values with real ones:
   - Rainfall → IMD / IMDLIB
   - Soil moisture → ISRO Bhoonidhi (EOS-04 SAR) or NASA SMAP
   - Elevation / slope → Bhuvan or SRTM DEM (derive slope via QGIS/`richdem`)
   - Lithology / historical landslide label → GSI Bhukosh inventory
   - Glacial lake area / growth rate / melt index → ICIMOD glacial lake
     inventory or ISRO glacier monitoring products (for villages with no
     nearby lake, set `glacial_lake_present=False` and leave the other GLOF
     columns at `0` / `999` as in the sample)
3. Restart the app — it retrains on whatever is in the CSV at startup.

`generate_data.py` shows exactly how the sample file was built, in case you
want to regenerate it for a different pilot region.

## Known simplifications (be upfront about these in your demo)

- The "live sensor feed" is a simulated random walk, not a real IoT connection
- Alert dispatch (Email/SMS/Siren) is logged in-app, not actually sent
- The evacuation path is a straight line between village and shelter, not
  routed over the real road network
- The historical-event label used to train the demo model is synthetic

