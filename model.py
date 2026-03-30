import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from preprocess import load_and_clean, scale_features
import pickle, os

FEATURE_COLS = [
    "airspeed", "altitude", "vertical_speed",
    "engine_rpm", "engine_temp", "fuel_flow"
]

def train_isolation_forest(X: np.ndarray,
                           n_estimators: int = 100,
                           contamination: float = 0.02) -> IsolationForest:
    """
    Train Isolation Forest.

    n_estimators : number of trees in the forest
    contamination: expected proportion of anomalies (sets score threshold)

    The model assigns each sample:
      label  +1 → normal
      label  -1 → anomaly
      score  s(x,n) ∈ (0,1), closer to 1 = more anomalous
    """
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42,
        n_jobs=-1          # use all CPU cores
    )
    model.fit(X)
    return model


def get_anomaly_scores(model: IsolationForest,
                       X: np.ndarray) -> tuple:
    """
    Returns:
      labels : np.ndarray of +1 (normal) or -1 (anomaly)
      scores : np.ndarray in (0,1) — higher = more anomalous

    Scikit-learn returns raw decision scores (negative = anomaly).
    We flip and normalize to the (0,1) intuitive range.
    """
    labels       = model.predict(X)
    raw_scores   = model.decision_function(X)  # more negative = more anomalous
    scores       = -raw_scores                  # flip sign
    # Min-max normalize to (0,1) for interpretability
    scores = (scores - scores.min()) / (scores.max() - scores.min())
    return labels, scores


if __name__ == "__main__":
    DATA_PATH  = "data/flight_data.csv"
    MODEL_PATH = "data/iso_forest.pkl"

    df        = load_and_clean(DATA_PATH)
    X, scaler = scale_features(df, FEATURE_COLS)

    model            = train_isolation_forest(X, contamination=0.02)
    labels, scores   = get_anomaly_scores(model, X)

    n_anomalies = (labels == -1).sum()
    print(f"Total samples : {len(labels)}")
    print(f"Anomalies found: {n_anomalies} ({100*n_anomalies/len(labels):.1f}%)")

    # Save model and results alongside the data
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)

    df["anomaly_label"] = labels
    df["anomaly_score"] = scores
    df.to_csv("data/results.csv", index=False)
    print("Model saved → data/iso_forest.pkl")
    print("Results saved → data/results.csv")