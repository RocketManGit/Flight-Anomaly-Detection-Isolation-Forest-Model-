import os
import numpy as np
import pandas as pd
import pickle
from preprocess import load_and_clean, scale_features
from model import train_isolation_forest, get_anomaly_scores
from visualize import plot_anomalies

# ── Config ──────────────────────
DATA_FILE = "data/flight_data.csv"
MODEL_PATH = "data/iso_forest.pkl"
RESULTS_PATH = "data/results.csv"
# ────────────────────────────────

if __name__ == "__main__":

    # Step 1: Generate synthetic data if no CSV exists
    if not os.path.exists(DATA_FILE):
        print("No CSV found. Generating synthetic demo data...")
        np.random.seed(42)
        n = 2000
        df_synth = pd.DataFrame({
            "time":           np.arange(n),
            "airspeed":       np.random.normal(250, 10, n),
            "altitude":       np.random.normal(30000, 500, n),
            "vertical_speed": np.random.normal(0, 200, n),
            "engine_rpm":     np.random.normal(85, 3, n),
            "engine_temp":    np.random.normal(650, 15, n),
            "fuel_flow":      np.random.normal(2500, 100, n),
        })
        anomaly_idx = np.random.choice(n, 40, replace=False)
        df_synth.loc[anomaly_idx, "engine_temp"] += np.random.uniform(100, 200, 40)
        df_synth.loc[anomaly_idx, "engine_rpm"]  -= np.random.uniform(20, 40, 40)
        df_synth.loc[anomaly_idx, "airspeed"]    += np.random.uniform(80, 150, 40)
        os.makedirs("data", exist_ok=True)
        df_synth.to_csv(DATA_FILE, index=False)
        print(f"Synthetic data saved to {DATA_FILE}")

    # Step 2: Load and clean
    print("\nStep 1: Preprocessing...")
    df, feature_cols = load_and_clean(DATA_FILE)
    X, scaler = scale_features(df, feature_cols)

    # Step 3: Train model and score
    print("\nStep 2: Training model...")
    model = train_isolation_forest(X, contamination=0.02)
    labels, scores = get_anomaly_scores(model, X)

    df["anomaly_label"] = labels
    df["anomaly_score"] = scores
    df.to_csv(RESULTS_PATH, index=False)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler,
                     "feature_cols": feature_cols}, f)

    n_anomalies = (labels == -1).sum()
    print(f"Total samples  : {len(labels)}")
    print(f"Anomalies found: {n_anomalies} ({100*n_anomalies/len(labels):.1f}%)")

    # Step 4: Visualize
    print("\nStep 3: Generating plots...")
    plot_anomalies(results_path=RESULTS_PATH, model_path=MODEL_PATH)

    print("\nDone! Check data/ and plots/ for outputs.")