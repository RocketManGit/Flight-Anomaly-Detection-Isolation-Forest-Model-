# AeroFaultML 🛩️
Multivariate anomaly detection on aircraft flight sensor data using Isolation Forest.

## Problem
Aircraft generate thousands of sensor readings per flight.
Maintenance teams struggle to spot early signs of sensor failure or system
degradation before it becomes critical. This tool automatically flags timesteps
where the *combination* of sensor readings deviates from normal flight behaviour —
catching faults that single-sensor thresholds would miss.

## How it works
1. Load and clean any aircraft sensor CSV — column names are auto-detected
2. Standardize features to zero mean and unit variance
3. Train Isolation Forest (100 trees, 2% contamination)
4. Score every timestep — short isolation path = anomaly, scored 0 to 1
5. Plot flagged anomalies over time and per sensor

## Project structure
```
AeroFaultML/
├── main.py              # entry point — run this
├── preprocess.py        # load, clean, auto-detect features
├── model.py             # train Isolation Forest, compute scores
├── visualize.py         # generate anomaly plots
├── requirements.txt     # dependencies
├── assets/              # output plots for display
│   ├── anomaly_score_timeline.png
│   └── sensor_subplots.png
└── data/
    └── flight_data.csv  # place your CSV here
```

## Quickstart

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Add your CSV** to the `data/` folder and update the filename in `main.py`:
```python
DATA_FILE = "data/your_flight_data.csv"
```

**3. Run the full pipeline:**
```bash
python main.py
```

Plots are saved to `plots/`. If no CSV is provided, synthetic demo data is
generated automatically so the pipeline runs out of the box.

## Versatility
The pipeline accepts any aircraft sensor CSV regardless of column names.
It automatically detects all numeric sensor columns, filters out index and
timestamp columns, and adapts the model and plots accordingly. No code
changes are needed when switching datasets.

## Results
![Anomaly timeline](assets/anomaly_score_timeline.png)
![Sensor subplots](assets/sensor_subplots.png)

## Dataset
NASA DASHLINK Flight Operations Quality Assurance (FOQA)  
https://c3.nasa.gov/dashlink/resources/132/

## Requirements
```
numpy
pandas
scikit-learn
matplotlib
```
