import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import shap
import streamlit as st

from model_utils import ENCODED_FEATURES, FEATURES, predict_risk, risk_color, simulate_next_reading, train_model
from cascade_utils import simulate_cascade, impact_level, WAVE_SPEED_KMPH
from satellite_monitor import run_change_detection, GRID_SIZE, PIXEL_RESOLUTION_M

# Free, no-API-key satellite imagery basemap (Esri World Imagery) — gives the
# map a real green/brown terrain look instead of a blank canvas. Used as the
# bottom layer under every pydeck map in this app.
#

def satellite_basemap_layer():
    return pdk.Layer(
        "TileLayer",
        data="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        min_zoom=0,
        max_zoom=19,
        tile_size=256,
    )

# FALLBACK (verified, zero-setup) — if the satellite tiles above don't show:
#   1. Delete `satellite_basemap_layer()` from each `layers=[...]` list below
#   2. Change `map_provider=None` to `map_provider="carto", map_style="light"`
#      in each st.pydeck_chart(pdk.Deck(...)) call

st.set_page_config(page_title="Flash Flood Prediction System", layout="wide", page_icon="🌧️")

# ----------------------------------------------------------------------------
# Custom styling — replaces default Streamlit look with a cleaner,
# dashboard-style theme (gradient header, card-style metrics/containers,
# styled tabs). Pure CSS injection, no extra dependency.
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    .block-container { padding-top: 1.2rem; max-width: 1300px; }

    .app-header {
        background: linear-gradient(135deg, #0f766e 0%, #134e6f 55%, #1e3a5f 100%);
        padding: 1.6rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.4rem;
        box-shadow: 0 4px 18px rgba(15, 118, 110, 0.25);
    }
    .app-header h1 {
        color: #ffffff; font-size: 1.9rem; font-weight: 700; margin: 0 0 0.35rem 0;
    }
    .app-header p {
        color: #d7f0ec; font-size: 0.95rem; margin: 0;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.9rem 1rem 0.6rem 1rem;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.06);
    }
    div[data-testid="stMetricLabel"] { font-weight: 600; color: #475569; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 2px solid #e2e8f0; }
    .stTabs [data-baseweb="tab"] {
        height: 44px; padding: 0 18px; border-radius: 10px 10px 0 0;
        background-color: #f1f5f9; font-weight: 600; color: #475569;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0f766e !important; color: #ffffff !important;
    }

    /* Buttons */
    .stButton > button, .stFormSubmitButton > button {
        border-radius: 8px; font-weight: 600; border: none;
    }
    .stButton > button[kind="primary"] { background-color: #0f766e; }

    /* Containers used for alert cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #f8fafc; }

    h2, h3 { color: #134e6f; }
</style>
""", unsafe_allow_html=True)

DATA_PATH = "data/villages_sample.csv"


# ----------------------------------------------------------------------------
# Data / model loading (cached so edits to the CSV require a rerun to pick up)
# ----------------------------------------------------------------------------
@st.cache_data
def load_baseline():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def get_model(df: pd.DataFrame):
    return train_model(df)


def init_state():
    if "live_data" not in st.session_state:
        st.session_state.live_data = load_baseline().copy()
    if "model" not in st.session_state:
        st.session_state.model = get_model(load_baseline())
    if "history" not in st.session_state:
        # rolling per-village history of (timestamp, rainfall, soil_moisture, risk_probability)
        st.session_state.history = {
            vid: [] for vid in st.session_state.live_data["village_id"]
        }
    if "alert_log" not in st.session_state:
        st.session_state.alert_log = []
    if "feedback_log" not in st.session_state:
        st.session_state.feedback_log = []
    if "tick" not in st.session_state:
        st.session_state.tick = 0
    if "rng" not in st.session_state:
        st.session_state.rng = np.random.default_rng(7)


init_state()

ALERT_THRESHOLD = "High"  # alerts fire for High or Critical
LEVEL_ORDER = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}


def record_history_and_alerts(scored: pd.DataFrame):
    now = dt.datetime.now().strftime("%H:%M:%S")
    for _, r in scored.iterrows():
        st.session_state.history[r["village_id"]].append(
            dict(t=now, rainfall_3day_mm=r["rainfall_3day_mm"],
                 soil_moisture_pct=r["soil_moisture_pct"],
                 risk_probability=r["risk_probability"])
        )
        st.session_state.history[r["village_id"]] = st.session_state.history[r["village_id"]][-30:]
        if LEVEL_ORDER[r["risk_level"]] >= LEVEL_ORDER[ALERT_THRESHOLD]:
            st.session_state.alert_log.append(
                dict(time=now, village=r["village_name"], level=r["risk_level"],
                     probability=round(r["risk_probability"], 2))
            )
    st.session_state.alert_log = st.session_state.alert_log[-50:]


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("🌧️ Control Panel")
    st.caption("Flash Flood Prediction System — Prototype")

    if st.button("🔄 Simulate next sensor reading", use_container_width=True):
        st.session_state.tick += 1
        rng = st.session_state.rng
        st.session_state.live_data = st.session_state.live_data.apply(
            lambda row: simulate_next_reading(row, rng), axis=1
        )

    if st.button("↩️ Reset to baseline data", use_container_width=True):
        st.session_state.live_data = load_baseline().copy()
        st.session_state.history = {vid: [] for vid in st.session_state.live_data["village_id"]}
        st.session_state.alert_log = []
        st.session_state.tick = 0

    st.divider()
    village_names = st.session_state.live_data["village_name"].tolist()
    selected_village = st.selectbox("Select village / ward", village_names)

    st.divider()
    st.caption(
        "📌 **Data provenance in this demo**\n\n"
        "- Terrain, rainfall & soil-moisture baselines: *synthetic sample* "
        "(replace `data/villages_sample.csv` with real IMD / ISRO Bhoonidhi / "
        "Bhuvan DEM / GSI Bhukosh values)\n"
        "- Live sensor feed: *simulated* random-walk over the baseline\n"
        "- Historical event label: *synthetic*, for demo model training only"
    )

# score current live data
scored = predict_risk(st.session_state.model, st.session_state.live_data)
if st.session_state.tick > 0:
    record_history_and_alerts(scored)

row = scored[scored["village_name"] == selected_village].iloc[0]

st.markdown("""
<div class="app-header">
    <h1>🌧️ Flash Flood Prediction System — Live Prototype</h1>
    <p>Hyper-local risk monitoring for hilly regions · Ministry of Home Affairs — Problem Statement 192</p>
</div>
""", unsafe_allow_html=True)

tab_map, tab_village, tab_cascade, tab_satellite, tab_alerts, tab_route, tab_feedback, tab_about = st.tabs(
    ["🗺️ Live Risk Map", "📍 Village Detail", "🌊 Cascade Simulator", "🛰️ Satellite Lake Monitor",
     "🚨 Alerts", "🧭 Safe Route & Shelters", "🗣️ Community Feedback", "ℹ️ About"]
)

# ----------------------------------------------------------------------------
# TAB 1 — Live risk map
# ----------------------------------------------------------------------------
with tab_map:
    c1, c2, c3, c4, c5 = st.columns(5)
    counts = scored["risk_level"].value_counts()
    c1.metric("🟢 Low", int(counts.get("Low", 0)))
    c2.metric("🟡 Medium", int(counts.get("Medium", 0)))
    c3.metric("🟠 High", int(counts.get("High", 0)))
    c4.metric("🔴 Critical", int(counts.get("Critical", 0)))
    c5.metric("🧊 Glacial lake villages", int(scored["glacial_lake_present"].sum()))

    map_df = scored.copy()

# Convert pandas Categorical to normal Python strings
    map_df["risk_level"] = map_df["risk_level"].astype(object)

# Assign map colors without pandas apply()
    color_map = {
    "Low": [46, 125, 50, 200],
    "Medium": [249, 199, 79, 200],
    "High": [255, 111, 0, 200],
    "Critical": [211, 47, 47, 200],
}

    map_df["color"] = [
    color_map.get(str(level), [128, 128, 128, 200])
    for level in map_df["risk_level"]
]

    map_df["radius"] = 800 + map_df["risk_probability"] * 2500

    map_df["position"] = map_df[["lon", "lat"]].values.tolist()

    layer = pdk.Layer(
      "ScatterplotLayer",
       data=map_df,
       get_position="position",
       get_fill_color="color",
       get_radius="radius",
       pickable=True,
       opacity=0.8,
)
    glof_df = map_df[map_df["glacial_lake_present"]].copy()
    glof_df["position"] = glof_df[["lon", "lat"]].values.tolist()
    glof_df["ring_radius"] = glof_df["radius"] + 500

    glof_ring_layer = pdk.Layer(
        "ScatterplotLayer",
        data=glof_df,
        get_position="position",
        get_radius="ring_radius",
        stroked=True,
        filled=False,
        get_line_color=[33, 150, 243, 220],
        line_width_min_pixels=2,
)
    view_state = pdk.ViewState(
        latitude=map_df["lat"].mean(), longitude=map_df["lon"].mean(), zoom=6.3
    )
    st.pydeck_chart(pdk.Deck(
        layers=[layer, glof_ring_layer], initial_view_state=view_state,
        map_provider="carto", map_style="light",
        tooltip={"text": "{village_name} ({district})\nRisk: {risk_level}\nProbability: {risk_probability}\nGlacial lake nearby: {glacial_lake_present}"},
    ))
    st.caption("🟢 Low · 🟡 Medium · 🟠 High · 🔴 Critical — marker size scales with risk probability. "
               "🔵 Blue ring = village has a glacial lake nearby (GLOF-relevant). "
               "Click **Simulate next sensor reading** in the sidebar to advance the live feed.")

    with st.expander("View underlying village data table"):
        st.dataframe(
            scored[["village_id", "village_name", "district", "rainfall_today_mm",
                    "rainfall_3day_mm", "soil_moisture_pct", "glacial_lake_present",
                    "glacier_melt_index", "moraine_dam_stability", "traditional_knowledge_signal",
                    "risk_probability", "risk_level", "data_provenance"]],
            use_container_width=True,
        )

# ----------------------------------------------------------------------------
# TAB 2 — Village detail (trend + explainability)
# ----------------------------------------------------------------------------
with tab_village:
    left, right = st.columns([1, 1])

    with left:
        st.subheader(f"{row['village_name']}, {row['district']}")
        st.metric("Current risk level", row["risk_level"],
                   help="Low < 25% · Medium 25–50% · High 50–75% · Critical > 75%")
        st.progress(min(float(row["risk_probability"]), 1.0),
                    text=f"Risk probability: {row['risk_probability']:.0%}")

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=row["risk_probability"] * 100,
            number={"suffix": "%"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": "#1f2937"},
                   "steps": [
                       {"range": [0, 25], "color": "#2e7d32"},
                       {"range": [25, 50], "color": "#f9c74f"},
                       {"range": [50, 75], "color": "#ff6f00"},
                       {"range": [75, 100], "color": "#d32f2f"},
                   ]},
        ))
        gauge.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(gauge, use_container_width=True)

        hist = st.session_state.history.get(row["village_id"], [])
        if hist:
            hdf = pd.DataFrame(hist)
            trend = go.Figure()
            trend.add_trace(go.Scatter(x=hdf["t"], y=hdf["rainfall_3day_mm"],
                                        name="3-day rainfall (mm)", mode="lines+markers"))
            trend.add_trace(go.Scatter(x=hdf["t"], y=hdf["soil_moisture_pct"],
                                        name="Soil moisture (%)", mode="lines+markers", yaxis="y2"))
            trend.update_layout(
                height=280, margin=dict(l=20, r=20, t=30, b=20),
                yaxis=dict(title="Rainfall (mm)"),
                yaxis2=dict(title="Soil moisture (%)", overlaying="y", side="right"),
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(trend, use_container_width=True)
        else:
            st.info("Click **Simulate next sensor reading** a few times in the sidebar to build a trend.")

        st.divider()
        st.subheader("🧊 Glacial Lake / GLOF status")
        if bool(row["glacial_lake_present"]):
            g1, g2 = st.columns(2)
            g1.metric("Lake area", f"{row['glacial_lake_area_km2']:.2f} km²")
            g2.metric("5-yr growth rate", f"{row['glacial_lake_growth_rate_pct_5yr']:.0f}%")
            g3, g4 = st.columns(2)
            g3.metric("Glacier melt index", f"{row['glacier_melt_index']:.2f}")
            g4.metric("Distance to lake", f"{row['distance_to_glacial_lake_km']:.1f} km")
            stability = row["moraine_dam_stability"]
            badge = {"low": "🟢 Low risk", "medium": "🟡 Medium risk", "high": "🔴 High risk"}.get(stability, stability)
            st.markdown(f"**Moraine dam stability risk:** {badge}")
            st.caption(
                "A growing lake area + a weak/high-risk moraine dam + a high melt index is the "
                "classic GLOF precursor pattern (as seen before the 2013 Kedarnath/Chorabari Lake "
                "event and the 2023 South Lhonak Lake GLOF in Sikkim)."
            )
        else:
            st.caption("No glacial lake mapped near this village — GLOF is not a contributing hazard here.")

        st.divider()
        st.subheader("🗣️ Traditional / Community Knowledge")
        if int(row.get("traditional_knowledge_signal", 0)) == 1:
            note = row.get("traditional_knowledge_notes", "") or "Community-flagged warning sign"
            st.warning(f"⚠️ **Community/elder-flagged sign active:** {note}")
            st.caption(
                "This is treated as a real, weighted input to the risk model — not just a "
                "comment. Communities have historically caught precursor signs that instruments "
                "miss (e.g. Simeulue Island's oral 'smong' tradition saved tens of thousands of "
                "lives in the 2004 Indian Ocean tsunami)."
            )
        else:
            st.caption(
                "No traditional/community warning signs currently flagged for this village. "
                "Residents can report one via the **Community Feedback** tab."
            )

    with right:
        st.subheader("Explainable AI result — why this risk level?")
        model = st.session_state.model
        from model_utils import prep_features
        X_row = prep_features(pd.DataFrame([row]))[ENCODED_FEATURES]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_row)
        # shap_values shape handling for binary RF: take class-1 contributions
        sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0, :, 1]

        contrib = pd.DataFrame({
            "feature": X_row.columns,
            "contribution": sv,
            "value": X_row.iloc[0].values,
        }).sort_values("contribution", key=abs, ascending=True)

        fig = go.Figure(go.Bar(
            x=contrib["contribution"], y=contrib["feature"], orientation="h",
            marker_color=["#d32f2f" if v > 0 else "#2e7d32" for v in contrib["contribution"]],
        ))
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                           xaxis_title="Impact on risk (SHAP value)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔴 Pushes risk higher · 🟢 Pushes risk lower. Values shown are the village's current "
                   "feature readings (rainfall, soil moisture, slope, etc.).")

# ----------------------------------------------------------------------------
# TAB — Cascade Simulator (cascading GLOF hazard propagation)
# ----------------------------------------------------------------------------
with tab_cascade:
    st.subheader("🌊 GLOF Cascade Simulator")
    st.caption(
        "Most flood predictors score each village independently. This simulates what "
        "actually happens physically: a glacial lake breach at one point sends a flood "
        "wave downstream that reaches — and escalates risk at — other villages over the "
        "next several hours. Pick a glacial-lake village and simulate a hypothetical breach."
    )

    glof_villages = scored[scored["glacial_lake_present"]]["village_name"].tolist()
    if not glof_villages:
        st.info("No glacial-lake villages in the current dataset.")
    else:
        c1, c2 = st.columns([2, 1])
        source_name = c1.selectbox("Source glacial lake (breach location)", glof_villages)
        max_hours = c2.slider("Simulation window (hours)", 2, 12, 6)

        if st.button("💥 Simulate GLOF breach", type="primary"):
            source_id = scored[scored["village_name"] == source_name].iloc[0]["village_id"]
            cascade_df = simulate_cascade(scored, source_id, max_hours=max_hours)
            st.session_state.cascade_result = cascade_df

        if "cascade_result" in st.session_state:
            cascade_df = st.session_state.cascade_result
            severity = cascade_df.attrs.get("severity", None)
            source_village = cascade_df.attrs.get("source_village", source_name)

            affected = cascade_df[cascade_df["impact_score"] > 0].copy()
            affected["impact_level"] = affected["impact_score"].apply(impact_level)

            m1, m2, m3 = st.columns(3)
            m1.metric("Breach severity", f"{severity:.0%}" if severity else "—")
            m2.metric("Villages affected", len(affected) - 1)  # exclude source itself
            m3.metric("Wave speed (assumed)", f"{WAVE_SPEED_KMPH:.0f} km/h")

            # Map: color by impact level, source highlighted
            cascade_map = cascade_df.copy()
            cascade_map["impact_level"] = cascade_map["impact_score"].apply(impact_level)
            cascade_map.loc[cascade_map["is_source"], "impact_level"] = "Source (breach point)"

            # Convert to plain Python strings before building the color list,
            # same fix used on the Live Risk Map tab (avoids Categorical/dtype
            # serialization issues when pydeck JSON-encodes the dataframe)
            cascade_map["impact_level"] = cascade_map["impact_level"].astype(object)
            level_colors = {
                "Source (breach point)": [80, 0, 160, 240],
                "Extreme": [211, 47, 47, 230], "Severe": [255, 111, 0, 220],
                "Moderate": [255, 193, 7, 200], "Minor": [255, 235, 59, 160],
                "Not affected": [200, 200, 200, 90],
            }
            cascade_map["color"] = [
                level_colors.get(str(level), [128, 128, 128, 180])
                for level in cascade_map["impact_level"]
            ]
            cascade_map["radius"] = np.where(cascade_map["is_source"], 1400,
                                              600 + cascade_map["impact_score"] * 2500)
            cascade_map["position"] = cascade_map[["lon", "lat"]].values.tolist()

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=cascade_map,
                get_position="position",
                get_fill_color="color",
                get_radius="radius",
                pickable=True,
                opacity=0.85,
            )
            view_state = pdk.ViewState(
                latitude=cascade_map["lat"].mean(), longitude=cascade_map["lon"].mean(), zoom=6.5
            )
            st.pydeck_chart(pdk.Deck(
                layers=[layer], initial_view_state=view_state, map_provider="carto", map_style="light",
                tooltip={"text": "{village_name}\nImpact: {impact_level}\nArrival: {arrival_hr}h\nDistance: {distance_km}km"},
            ))
            st.caption("🟣 Breach point · 🔴 Extreme · 🟠 Severe · 🟡 Moderate · 🟨 Minor · ⚪ Not reached in this window")

            st.markdown(f"**Downstream impact timeline from {source_village}:**")
            display_cols = ["village_name", "district", "distance_km", "arrival_hr", "impact_level"]
            st.dataframe(
                affected[affected["village_id"] != cascade_df[cascade_df["is_source"]]["village_id"].iloc[0]]
                [display_cols].rename(columns={
                    "village_name": "Village", "district": "District",
                    "distance_km": "Distance (km)", "arrival_hr": "Arrival (hr)",
                    "impact_level": "Impact"
                }),
                use_container_width=True, hide_index=True,
            )
            st.info(
                "⚠️ This is a simplified distance/elevation-based propagation model for "
                "demonstration — not a routed hydrodynamic simulation. A production version "
                "would route the flood wave along the actual river network and channel "
                "geometry (e.g. using HEC-RAS or a similar dam-break/outburst-flow model)."
            )
        else:
            st.caption("Click **Simulate GLOF breach** to run the cascade.")

# ----------------------------------------------------------------------------
# TAB — Satellite Lake Monitor (automated change detection)
# ----------------------------------------------------------------------------
with tab_satellite:
    st.subheader("🛰️ Automated Glacial Lake Change Detection")
    st.caption(
        "Instead of manually entering a lake's area, this pipeline detects it automatically "
        "from satellite imagery: threshold a water index (like NDWI) into a water mask, "
        "then sum pixel area — repeated over time to get a growth trend. Running here on "
        "synthetic rasters so it works fully offline; the detection logic itself is the "
        "same as a real Sentinel-1/2 pipeline (see code note below)."
    )

    sat_villages = scored[scored["glacial_lake_present"]]["village_name"].tolist()
    if not sat_villages:
        st.info("No glacial-lake villages in the current dataset.")
    else:
        sel_village = st.selectbox("Glacial lake village", sat_villages, key="sat_village")
        row_sat = scored[scored["village_name"] == sel_village].iloc[0]

        n_snap = st.slider("Number of satellite snapshots (over 5 years)", 3, 10, 6)
        result_df, snapshots = run_change_detection(row_sat, n_snapshots=n_snap)

        left, right = st.columns([1, 1])
        with left:
            st.markdown("**First snapshot (Year 0)**")
            fig0 = go.Figure(go.Heatmap(z=snapshots[0], colorscale="Blues", showscale=False))
            fig0.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                                xaxis_visible=False, yaxis_visible=False)
            st.plotly_chart(fig0, use_container_width=True)
        with right:
            st.markdown(f"**Latest snapshot (Year {result_df['year_offset'].iloc[-1]:.0f})**")
            figN = go.Figure(go.Heatmap(z=snapshots[-1], colorscale="Blues", showscale=False))
            figN.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                                xaxis_visible=False, yaxis_visible=False)
            st.plotly_chart(figN, use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Detected area (latest)", f"{result_df['detected_area_km2'].iloc[-1]:.3f} km²")
        m2.metric("Change since Year 0", f"{result_df['pct_change_from_start'].iloc[-1]:+.1f}%")
        m3.metric("Scene resolution", f"{PIXEL_RESOLUTION_M} m/pixel ({GRID_SIZE}×{GRID_SIZE})")

        area_fig = go.Figure(go.Scatter(
            x=result_df["year_offset"], y=result_df["detected_area_km2"],
            mode="lines+markers", line=dict(color="#0f766e", width=3),
        ))
        area_fig.update_layout(
            height=260, margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Years", yaxis_title="Detected lake area (km²)",
        )
        st.plotly_chart(area_fig, use_container_width=True)

        with st.expander("How this would run on real satellite data"):
            st.code(
                'import ee\n'
                'ee.Initialize()\n'
                'collection = (\n'
                '    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\n'
                '    .filterBounds(lake_point)\n'
                '    .filterDate(start_date, end_date)\n'
                '    .map(lambda img: img.normalizedDifference(["B3", "B8"]).rename("NDWI"))\n'
                ')\n'
                '# threshold NDWI > 0.2 -> water mask -> sum ee.Image.pixelArea()\n'
                '# over the mask to get lake area in m^2 for each date',
                language="python",
            )
            st.caption(
                "The threshold-and-sum logic is identical to the synthetic version above — "
                "only the image source (Sentinel-2 via Google Earth Engine) changes."
            )

# ----------------------------------------------------------------------------
# TAB 3 — Alerts
# ----------------------------------------------------------------------------
with tab_alerts:
    st.subheader("Active alerts (High / Critical risk)")
    active = scored[scored["risk_level"].isin(["High", "Critical"])].sort_values(
        "risk_probability", ascending=False)

    if active.empty:
        st.success("No villages currently at High or Critical risk.")
    else:
        for _, r in active.iterrows():
            with st.container(border=True):
                c1, c2, c3 = st.columns([2, 1, 2])
                c1.markdown(f"**{r['village_name']}** ({r['district']}) — "
                            f"{r['risk_level']} · {r['risk_probability']:.0%}")
                if c2.button("📣 Dispatch alert", key=f"dispatch_{r['village_id']}"):
                    ts = dt.datetime.now().strftime("%H:%M:%S")
                    st.session_state.alert_log.append(
                        dict(time=ts, village=r["village_name"], level=r["risk_level"],
                             probability=round(r["risk_probability"], 2),
                             channel="Email + SMS + Siren (simulated)")
                    )
                    st.toast(f"Alert dispatched for {r['village_name']}", icon="🚨")
                c3.markdown("📧 Email &nbsp;·&nbsp; 📱 SMS &nbsp;·&nbsp; 📢 Siren")

    st.divider()
    st.subheader("Alert log")
    if st.session_state.alert_log:
        st.dataframe(pd.DataFrame(st.session_state.alert_log[::-1]), use_container_width=True)
    else:
        st.caption("No alerts dispatched yet.")

    st.info(
        "⚠️ In this prototype, alert dispatch is **simulated** — no real email/SMS/siren is sent. "
        "Wire this panel to an SMTP service, an SMS gateway (e.g. an approved DLT-registered "
        "provider), and MQTT-triggered local sirens for a production deployment."
    )

# ----------------------------------------------------------------------------
# TAB 4 — Safe route & shelters
# ----------------------------------------------------------------------------
with tab_route:
    st.subheader(f"Nearest safe shelter — {row['village_name']}")
    st.markdown(f"🏠 **Suggested shelter:** {row['nearest_shelter']}")

    route_df = pd.DataFrame([
        dict(name=row["village_name"], lat=row["lat"], lon=row["lon"], type="Village"),
        dict(name=row["nearest_shelter"], lat=row["shelter_lat"], lon=row["shelter_lon"], type="Shelter"),
    ])
    path = [[row["lon"], row["lat"]], [row["shelter_lon"], row["shelter_lat"]]]

    route_df["position"] = route_df[["lon", "lat"]].values.tolist()

    route_df["color"] = [
       [33, 150, 243, 220] if t == "Shelter"
       else [211, 47, 47, 220]
       for t in route_df["type"]
]

    path_layer = pdk.Layer(
        "PathLayer",
        data=[{"path": path}],
        get_path="path",
        get_color=[76, 175, 80, 220],
        get_width=6,
    )

    point_layer = pdk.Layer(
       "ScatterplotLayer",
        data=route_df,
        get_position="position",
        get_fill_color="color",
        get_radius=600,
        pickable=True,
)
    
    view_state = pdk.ViewState(latitude=row["lat"], longitude=row["lon"], zoom=11)
    st.pydeck_chart(pdk.Deck(layers=[path_layer, point_layer], initial_view_state=view_state,
                              map_provider="carto", map_style="light", tooltip={"text": "{name}"}))
    st.caption(
        "🔴 Village · 🔵 Shelter — the line shown is an **indicative straight-line path**, not a "
        "road-routed evacuation path. Wire this to a routing engine (e.g. OSRM/GraphHopper) with "
        "the local road network for turn-by-turn evacuation directions in production."
    )

    st.divider()
    st.subheader("All shelters in the pilot region")
    st.dataframe(
        scored[["village_name", "district", "nearest_shelter", "shelter_lat", "shelter_lon"]],
        use_container_width=True,
    )

# ----------------------------------------------------------------------------
# TAB 5 — Community feedback (closes the loop back into the model)
# ----------------------------------------------------------------------------
with tab_feedback:
    st.subheader("Submit a ground report")
    st.caption("Citizen / field-volunteer reports help verify predictions and improve future model training.")

    with st.form("feedback_form", clear_on_submit=True):
        fb_village = st.selectbox("Village / ward", village_names, key="fb_village")
        fb_actual = st.radio(
            "What actually happened here?",
            ["No event — conditions look normal", "Minor warning signs (cracks, seepage, small slides)",
             "Confirmed landslide / flash flood event", "False alarm — system over-predicted"],
        )
        st.markdown("**🗣️ Traditional / community knowledge**")
        st.caption(
            "Communities often notice real precursor signs before instruments do — animal "
            "behavior, spring/stream water changes, elder-recognized patterns. This is treated "
            "as a genuine model input, not just a comment box."
        )
        fb_tek = st.checkbox("Community/elders flagged an unusual traditional warning sign")
        fb_tek_notes = st.text_input(
            "Briefly describe the sign (optional)",
            placeholder="e.g. livestock restless at night, spring water turned cloudy...",
            disabled=not fb_tek,
        )
        fb_notes = st.text_area("Additional notes (optional)")
        fb_reporter = st.text_input("Reporter name / role (optional)")
        submitted = st.form_submit_button("Submit report")

        if submitted:
            ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.feedback_log.append(
                dict(time=ts, village=fb_village, report=fb_actual,
                     notes=fb_notes, reporter=fb_reporter or "Anonymous",
                     traditional_signal=fb_tek,
                     traditional_notes=fb_tek_notes if fb_tek else "")
            )
            if fb_tek:
                vmask = st.session_state.live_data["village_name"] == fb_village
                st.session_state.live_data.loc[vmask, "traditional_knowledge_signal"] = 1
                st.session_state.live_data.loc[vmask, "traditional_knowledge_notes"] = (
                    fb_tek_notes or "Community-flagged warning sign"
                )
            st.success("Report submitted — thank you. It has been added to the feedback queue below.")

    st.divider()
    st.subheader("Feedback queue")
    if st.session_state.feedback_log:
        fb_df = pd.DataFrame(st.session_state.feedback_log)
        st.dataframe(fb_df[::-1], use_container_width=True)

        if st.button("🔁 Retrain model with this feedback"):
            live = st.session_state.live_data.copy()
            new_rows = []
            for entry in st.session_state.feedback_log:
                vid_row = live[live["village_name"] == entry["village"]].iloc[0].to_dict()
                vid_row["label"] = 1 if "Confirmed" in entry["report"] else (
                    0 if "False alarm" in entry["report"] or "No event" in entry["report"] else vid_row.get("label", 0))
                if entry.get("traditional_signal"):
                    vid_row["traditional_knowledge_signal"] = 1
                vid_row["data_provenance"] = "community_feedback"
                new_rows.append(vid_row)
            augmented = pd.concat([load_baseline(), pd.DataFrame(new_rows)], ignore_index=True)
            st.session_state.model = train_model(augmented)
            st.success(f"Model retrained using baseline data + {len(new_rows)} community report(s), "
                       f"including any traditional-knowledge signals.")
    else:
        st.caption("No feedback submitted yet.")

# ----------------------------------------------------------------------------
# TAB 6 — About
# ----------------------------------------------------------------------------
with tab_about:
    st.subheader("About this prototype")
    st.markdown(f"""
This is a working **prototype** for MHA Problem Statement 192 — *Flash Flood Prediction
System for Hilly Regions using Multi-Source Data*.

**Pipeline shown here:** data sources (rainfall, soil moisture, terrain, historical
records, **glacial lake / GLOF data**, **traditional/community knowledge**) →
data integration & feature engineering → ML model (Random Forest) → risk
classification (Low/Medium/High/Critical) → explainable AI (SHAP) → alerts
(email/SMS/siren, simulated) → safe route & shelter suggestion → community
feedback loop back into retraining → GLOF cascade simulation → automated
satellite lake monitoring.

**Model:** RandomForestClassifier, trained live on `data/{'villages_sample.csv'}`
at app startup — **{len(load_baseline())} villages**, features: {', '.join(FEATURES)}
+ lithology risk + moraine dam stability risk.

**GLOF (Glacial Lake Outburst Flood) coverage:** {int(load_baseline()['glacial_lake_present'].sum())}
of {len(load_baseline())} villages in this sample have a glacial lake mapped nearby
(modelled loosely on real glacier-belt locations — Chamoli/Rudraprayag, Kinnaur,
Kullu, North Sikkim). For these villages the model also uses lake area, 5-year lake
growth rate, distance to the lake, glacier melt index, and moraine dam stability —
the standard precursor signals used in real GLOF susceptibility studies.

**🗣️ Traditional Ecological Knowledge (TEK):** {int(load_baseline()['traditional_knowledge_signal'].sum())}
of {len(load_baseline())} villages currently have a community/elder-flagged
precursor sign, submitted via the Community Feedback tab. This is a genuine,
weighted model feature (visible in the SHAP panel) — not a cosmetic add-on.
The rationale: Simeulue Island's oral "smong" tsunami-warning tradition
prevented mass casualties in the 2004 Indian Ocean tsunami even without
instruments, because generations of local knowledge recognized the precursor
signs before the wave hit. This system treats that kind of knowledge as data,
not folklore.

**What's real vs simulated in this demo:**
- Village coordinates and district names: real place names in Uttarakhand/HP/Sikkim/Mizoram pilot belt
- Rainfall, soil moisture, slope, historical-event labels: **synthetic sample data** —
  replace `data/villages_sample.csv` with real IMD / ISRO Bhoonidhi / Bhuvan DEM / GSI Bhukosh values
- Glacial lake area, growth rate, melt index, moraine dam stability: **synthetic sample data** —
  replace with real ICIMOD / ISRO glacial lake inventory values
- Traditional knowledge signals: **synthetic seed data** for the demo; in real use these come
  entirely from actual community reports submitted through the Community Feedback tab
- Live sensor feed: **simulated** random-walk, standing in for real IoT ingestion
- Alert dispatch: **simulated**, no real email/SMS/siren is sent
- Evacuation path: **straight-line indicative**, not road-routed
- Cascade simulation: simplified distance/elevation-based propagation, not a routed hydrodynamic model
- Satellite change detection: runs on synthetic rasters; identical threshold-and-pixel-count logic
  to a real Sentinel-1/2 pipeline (see code sample in the Satellite Lake Monitor tab)

**To use your own data:** edit `data/villages_sample.csv` directly (keep the same column names),
or replace it entirely with your GEE/QGIS-exported village-level table. The app will retrain
the model on whatever is in that file the next time it starts.
""")
    