# 🌧️ Flash Flood Prediction System — Prototype

A **Streamlit-based prototype** for **Flash Flood Prediction System for Hilly Regions using Multi-Source Data**.

The system combines rainfall, soil moisture, terrain, historical landslide information, and glacial-lake indicators to estimate village-level disaster risk and demonstrate an end-to-end early-warning workflow.

> ⚠️ **Prototype:** Some components use simulated/sample data for demonstration. The **About** tab clearly identifies real vs simulated components.

---

## 🚀 Run It

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

## 🧩 Features

| Tab                            | Purpose                                                                        |
| ------------------------------ | ------------------------------------------------------------------------------ |
| 🗺️ **Live Risk Map**          | Village-level Low/Medium/High/Critical risk map with glacial-lake indicators   |
| 📍 **Village Detail**          | Risk gauge, rainfall/soil-moisture trends, GLOF status and SHAP explainability |
| 🌊 **Cascade Simulator**       | Simulates a GLOF breach and downstream flood-wave impact                       |
| 🛰️ **Satellite Lake Monitor** | Demonstrates glacial-lake change detection using time-series raster data       |
| 🚨 **Alerts**                  | Identifies High/Critical villages and simulates Email/SMS/Siren dispatch       |
| 🧭 **Safe Route & Shelters**   | Shows nearest shelters and indicative evacuation paths                         |
| 🗣️ **Community Feedback**     | Collects local observations and Traditional Ecological Knowledge               |
| ℹ️ **About**                   | Explains architecture, data sources and prototype limitations                  |

### Sidebar

* **Simulate next sensor reading** — generates a simulated live reading and recalculates risk.
* **Reset to baseline** — restores the original dataset and clears alert history.

---

## 🌍 Multi-Source Data

| Data                     | Intended Source                     |
| ------------------------ | ----------------------------------- |
| 🌧️ Rainfall             | IMD / IMDLIB                        |
| 💧 Soil Moisture         | ISRO Bhoonidhi / EOS-04 / NASA SMAP |
| ⛰️ Elevation & Slope     | Bhuvan / SRTM DEM                   |
| 🪨 Landslide & Lithology | GSI Bhukosh                         |
| 🏔️ Glacial Lakes        | ICIMOD / ISRO products              |
| 🛰️ Satellite Monitoring | Sentinel-1 / Sentinel-2             |

The current prototype uses sample data in:

```text
data/villages_sample.csv
```

To integrate real data, retain the existing column structure and replace the synthetic values.

---

## 🌊 Cascade Simulator

Demonstrates the cascading hazard chain:

```text
Glacial Lake
     ↓
GLOF Breach
     ↓
Flood Wave
     ↓
Downstream Villages
     ↓
Impact & Early Warning
```

The prototype estimates downstream impact using distance, elevation and simplified flood-wave propagation.

> This is an illustrative simulation, not a physically validated hydraulic model.

---

## 🛰️ Satellite Lake Monitor

Demonstrates automated lake-change detection by:

1. Processing time-series raster data
2. Applying a water-index threshold
3. Estimating water area
4. Measuring lake growth over time

The same concept can later be connected to real Sentinel-1/2 processing pipelines.

---

## 🤖 Explainable AI

The system uses **SHAP** to explain individual risk predictions.

Important factors can include:

* Rainfall
* Soil moisture
* Slope
* Elevation
* Historical landslides
* Glacial-lake growth
* Distance to glacial lake
* Moraine-dam stability

Instead of only showing:

```text
Risk = HIGH
```

the system also answers:

> **Why is this village considered high risk?**

---

## 🗣️ Community Feedback & Traditional Ecological Knowledge

The system includes a **human-in-the-loop** layer where citizens, volunteers and local communities can report:

* Heavy or unusual rainfall
* Rapid water-level changes
* Landslide signs
* Blocked roads
* Changes in rivers/streams
* Historically vulnerable locations
* Local safe/unsafe routes

### 🌿 Traditional Ecological Knowledge

Local communities may also provide **Traditional Ecological Knowledge (TEK)** developed through generations of observing their environment.

Examples include:

* Changes in river behaviour
* Changes in springs or water sources
* Local flood/landslide warning signs
* Unusual environmental patterns
* Traditional knowledge of safe areas

A well-known example is **Simeulue's "Smong" tradition**, where generations of local knowledge helped communities recognize tsunami warning signs and move to higher ground.

The project therefore combines:

```text
Sensor Data
     +
Satellite Data
     +
Historical Data
     +
AI Prediction
     +
Community Reports
     +
Traditional Ecological Knowledge
     ↓
Context-Aware Early Warning
```

> TEK complements scientific measurements; it does not replace official warning systems.

---

## 🚨 Alerts

The system identifies:

```text
HIGH
CRITICAL
```

risk villages and demonstrates:

```text
Risk Detection
      ↓
Alert Generation
      ↓
Email / SMS / Siren
      ↓
Alert Log
```

Currently, dispatch is **simulated and logged inside the application**.

---

## 🧭 Safe Routes & Shelters

The system identifies the nearest shelter and displays an indicative evacuation path.

> Current routes are simplified straight-line paths. Future versions can use real road networks, blocked-road information and dynamic evacuation routing.

---

## ⚠️ Known Limitations

This is a **prototype**, not a production disaster-management system.

* Live sensor feed → simulated random walk
* Alert dispatch → simulated
* Evacuation route → simplified
* Training labels → synthetic
* GLOF propagation → simplified
* Dataset → sample/demo data

These limitations are intentionally disclosed in the **About** section.

---

## 🔮 Future Scope

* 📡 Real-time IoT sensors
* 🌧️ Live rainfall feeds
* 🛰️ Automated Sentinel-1/2 processing
* 🧠 LSTM / GNN-based prediction
* 🗄️ Real-time database
* ⚡ FastAPI prediction service
* 📧 Real email alerts
* 📱 SMS/mobile alerts
* 🗺️ Real road-network evacuation routing
* 🌊 Physically validated GLOF/flood modelling
* 👥 Community + TEK data integration

---

## 📁 Project Structure

```text
Flood-Safe/
│
├── app.py
├── generate_data.py
├── model_utils.py
├── requirements.txt
├── README.md
│
└── data/
    └── villages_sample.csv
```
## 📸 Screenshots

<img width="1895" height="915" alt="Screenshot 2026-09-16 103949" src="https://github.com/user-attachments/assets/aae02ad6-7ebd-4e14-9477-64a72b315d7d" />


---

## 🎯 Goal

The long-term goal is to build a **multi-source, explainable and community-aware early-warning system** that can:

```text
Detect → Predict → Explain → Alert → Evacuate → Learn
```

to provide earlier and more actionable warnings for vulnerable communities in hilly regions.

---

## 📜 Disclaimer

This project is an **academic/hackathon prototype**. It should not be used for real-world evacuation or emergency decisions without validation using authoritative data, domain expertise, and validated disaster-management models.
