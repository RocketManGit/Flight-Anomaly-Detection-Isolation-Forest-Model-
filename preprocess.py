import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def load_and_clean(filepath: str) -> tuple:
    df = pd.read_csv(filepath)

    print(f"Columns found: {list(df.columns)}")

    df.dropna(thresh=int(0.5 * len(df.columns)), inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    feature_cols = []
    for col in numeric_cols:
        unique_ratio = df[col].nunique() / len(df)
        is_sequential = (df[col].diff().dropna() == 1).mean() > 0.95
        if not is_sequential and unique_ratio > 0.01:
            feature_cols.append(col)

    if len(feature_cols) == 0:
        raise ValueError("No usable numeric feature columns found in the CSV.")

    print(f"Features selected: {feature_cols}")

    df = df[feature_cols].copy()
    df.reset_index(drop=True, inplace=True)
    df.insert(0, "time", range(len(df)))

    return df, feature_cols


def scale_features(df: pd.DataFrame, feature_cols: list) -> tuple:
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols].values)
    return X, scaler