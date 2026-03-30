import numpy as np
from sklearn.ensemble import IsolationForest

def train_isolation_forest(X: np.ndarray,
                           n_estimators: int = 100,
                           contamination: float = 0.02) -> IsolationForest:
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X)
    return model


def get_anomaly_scores(model: IsolationForest,
                       X: np.ndarray) -> tuple:
    labels     = model.predict(X)
    raw_scores = model.decision_function(X)
    scores     = -raw_scores
    scores     = (scores - scores.min()) / (scores.max() - scores.min())
    return labels, scores