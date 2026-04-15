# AeroFaultML ✈️
### Catching Hidden Flight Anomalies Before They Become Problems

---

## The Story Behind This Project

Every time an aircraft takes flight, it generates a continuous stream of thousands of sensor readings — engine temperatures, pressures, vibration levels, airspeed, and more. These readings flow silently in the background, logged but rarely examined in real time.

Now imagine a scenario: no single sensor reading looks alarming on its own. The engine temperature is within normal range. The pressure readings are fine. The vibration levels seem acceptable. But together — in a very specific combination — these readings are quietly whispering that something is *off*.

Traditional monitoring systems, built around fixed thresholds for individual sensors, would miss this entirely. A human analyst reviewing the data manually would likely miss it too — there are simply too many data points to inspect.

**This is the problem AeroFaultML was built to solve.**

---

## What This Tool Does

AeroFaultML is an intelligent monitoring system that learns what "normal" flight looks like by analysing historical sensor data. Once it understands normal, it continuously watches for moments where the *combination* of sensor readings deviates from that pattern — even when no individual sensor is technically out of bounds.

Think of it like a highly experienced aircraft engineer who has reviewed thousands of flights and developed an intuition for when something "just doesn't feel right." AeroFaultML provides that same intuition, automatically, at scale, and without fatigue.

**The end result:** flagged time windows during a flight where the sensor data collectively suggests something unusual is happening — giving maintenance teams a precise starting point for investigation, rather than a haystack to search through.

---

## Why It Matters

| Without AeroFaultML | With AeroFaultML |
|---|---|
| Analysts manually review thousands of data points | The system automatically highlights suspicious moments |
| Single-sensor alerts generate noise and miss complex faults | Multi-sensor patterns are evaluated together for higher accuracy |
| Issues may go unnoticed until they become critical | Early warnings surface subtle degradation before it escalates |
| Dataset-specific tooling requires code changes per flight | Works with any aircraft sensor CSV, no reconfiguration needed |

---

## The Technology 

Under the hood, AeroFaultML uses a technique called an **Isolation Forest** — a machine learning algorithm that is particularly well-suited to finding rare, unusual events in large datasets.

Here's the intuition: imagine you have a forest of decision trees, each randomly splitting data points. Normal data points, which cluster together, are hard to isolate — it takes many splits to separate them from the crowd. Anomalous data points, which sit far from the norm, are easy to isolate — they get separated quickly. The algorithm scores each moment in the flight data based on how easy it is to isolate; easy to isolate = likely anomalous.

**The pipeline, step by step:**

1. **Ingest** — Load flight sensor data from a CSV file (column names are automatically detected; no manual configuration needed)
2. **Clean & Normalise** — Prepare the data so that all sensors are compared on equal footing, regardless of their individual units or scales
3. **Train** — Build an Isolation Forest model that learns the signature of normal flight behaviour
4. **Score** — Assign an anomaly score to every moment in the flight timeline
5. **Visualise** — Generate clear plots showing when anomalies occurred and which sensors contributed

---

## What the Output Looks Like

The tool produces two visual outputs:

**Anomaly Score Timeline** — A single chart showing the anomaly score across the full flight. Spikes indicate moments where the system detected unusual behaviour.

![Anomaly timeline](https://github.com/RocketManGit/Flight-Anomaly-Detection-Isolation-Forest-Model-/raw/main/assets/anomaly_score_timeline.png)

**Per-Sensor Breakdown** — A set of charts showing each individual sensor's readings over time, with anomalous periods highlighted — helping engineers pinpoint which sensors were behaving unusually during a flagged event.

![Sensor subplots](https://github.com/RocketManGit/Flight-Anomaly-Detection-Isolation-Forest-Model-/raw/main/assets/sensor_subplots.png)

---

## Data Source

This project was developed and validated using real-world flight data from **NASA's DASHLINK Flight Operations Quality Assurance (FOQA)** programme — a publicly available dataset of recorded aircraft sensor telemetry.

🔗 [NASA DASHlink Dataset](https://c3.nasa.gov/dashlink/resources/132/)

---

## Project Structure (For the Technical Team)

```
AeroFaultML/
├── main.py              # Entry point — run this to execute the full pipeline
├── preprocess.py        # Data loading, cleaning, and automatic feature detection
├── model.py             # Isolation Forest training and anomaly scoring
├── visualize.py         # Plot generation
├── requirements.txt     # Python library dependencies
├── assets/              # Output plots
│   ├── anomaly_score_timeline.png
│   └── sensor_subplots.png
└── data/
    └── flight_data.csv  # Place your sensor CSV here
```

---

## Getting Started (For Developers)

**Step 1 — Install dependencies:**
```bash
pip install -r requirements.txt
```

**Step 2 — Add your data:**
Place your flight sensor CSV in the `data/` folder and update the filename in `main.py`:
```python
DATA_FILE = "data/your_flight_data.csv"
```

**Step 3 — Run the pipeline:**
```bash
python main.py
```

Plots are saved to the `plots/` directory. If no CSV is provided, the system automatically generates synthetic demo data so you can explore the output immediately.

---

## A Note on Versatility

AeroFaultML is not tied to any specific aircraft type, airline, or sensor configuration. It automatically detects all numeric sensor columns in whatever CSV you provide, filters out non-sensor fields, and adapts accordingly. Switching to a different dataset requires no code changes.

---

## Requirements

```
numpy
pandas
scikit-learn
matplotlib
```

---

*Built to bring machine learning–powered safety awareness to aviation maintenance workflows.*
