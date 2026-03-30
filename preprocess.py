import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    Load NASA DASHLINK flight sensor CSV and return a cleaned DataFrame.
    Expected columns: time, airspeed, altitude, vertical_speed,
                      engine_rpm, engine_temp, fuel_flow
    """
    df = pd.read_csv(filepath)

    # Drop rows where more than 50% of values are missing
    df.dropna(thresh=int(0.5 * len(df.columns)), inplace=True)

    # Forward-fill then back-fill remaining NaNs (sensor dropout recovery)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Remove physically impossible readings (domain-specific bounds)
    bounds = {
        "airspeed":       (0,    600),   # knots
        "altitude":       (-500, 45000), # feet
        "vertical_speed": (-8000, 8000), # fpm
        "engine_rpm":     (0,    110),   # % N1
        "engine_temp":    (0,    1000),  # Celsius (EGT)
        "fuel_flow":      (0,    10000), # kg/hr
    }
    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            df = df[(df[col] >= lo) & (df[col] <= hi)]

    df.reset_index(drop=True, inplace=True)
    return df


def scale_features(df: pd.DataFrame, feature_cols: list) -> tuple:
    """
    Standardize features to zero mean, unit variance.
    Returns scaled array and the fitted scaler (needed for inverse transform).
    Standardization: x' = (x - μ) / σ
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols].values)
    return X, scaler


if __name__ == "__main__":
    DATA_PATH = "data/flight_data.csv"

    if not os.path.exists(DATA_PATH):
        print("Generating synthetic data for demo...")
        # Synthetic stand-in: 2000 normal timesteps + 40 injected anomalies
        np.random.seed(42)
        n = 2000
        df_synth = pd.DataFrame({
            "time":          np.arange(n),
            "airspeed":      np.random.normal(250, 10, n),
            "altitude":      np.random.normal(30000, 500, n),
            "vertical_speed":np.random.normal(0, 200, n),
            "engine_rpm":    np.random.normal(85, 3, n),
            "engine_temp":   np.random.normal(650, 15, n),
            "fuel_flow":     np.random.normal(2500, 100, n),
        })
        # Inject anomalies at random indices
        anomaly_idx = np.random.choice(n, 40, replace=False)
        df_synth.loc[anomaly_idx, "engine_temp"]   += np.random.uniform(100, 200, 40)
        df_synth.loc[anomaly_idx, "engine_rpm"]    -= np.random.uniform(20, 40, 40)
        df_synth.loc[anomaly_idx, "airspeed"]      += np.random.uniform(80, 150, 40)
        os.makedirs("data", exist_ok=True)
        df_synth.to_csv(DATA_PATH, index=False)
        print(f"Synthetic data saved to {DATA_PATH}")

    df = load_and_clean(DATA_PATH)
    print(f"Loaded {len(df)} samples with {df.shape[1]} features.")
    print(df.describe())