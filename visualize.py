import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle, os, math

def plot_anomalies(results_path: str = "data/results.csv",
                  model_path:   str = "data/iso_forest.pkl",
                  output_dir:   str = "plots"):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(results_path)

    with open(model_path, "rb") as f:
        saved = pickle.load(f)
    feature_cols = saved["feature_cols"]

    normal    = df[df["anomaly_label"] ==  1]
    anomalies = df[df["anomaly_label"] == -1]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df["time"], df["anomaly_score"],
            color="#3B8BD4", linewidth=0.7, label="Anomaly score")
    ax.scatter(anomalies["time"], anomalies["anomaly_score"],
               color="#E8593C", s=25, zorder=5, label="Flagged anomaly")
    ax.axhline(df[df["anomaly_label"]==-1]["anomaly_score"].min(),
               color="#E8593C", linestyle="--", linewidth=0.8,
               alpha=0.5, label="Detection threshold")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Anomaly score  (0 = normal, 1 = anomalous)")
    ax.set_title("Isolation Forest — anomaly score over flight")
    ax.legend(framealpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{output_dir}/anomaly_score_timeline.png", dpi=150)
    plt.close()
    print("Saved → plots/anomaly_score_timeline.png")

    n_features = len(feature_cols)
    n_cols     = 2
    n_rows     = math.ceil(n_features / n_cols)

    fig = plt.figure(figsize=(16, 5 * n_rows))
    gs  = gridspec.GridSpec(n_rows, n_cols, hspace=0.45, wspace=0.3)

    for i, feat in enumerate(feature_cols):
        ax = fig.add_subplot(gs[i // n_cols, i % n_cols])
        ax.plot(normal["time"], normal[feat],
                color="#3B8BD4", linewidth=0.6, alpha=0.7, label="Normal")
        ax.scatter(anomalies["time"], anomalies[feat],
                   color="#E8593C", s=18, zorder=5, label="Anomaly")
        ax.set_title(feat.replace("_", " ").title(), fontsize=10)
        ax.set_xlabel("Time step", fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7, framealpha=0.3)

    if n_features % 2 != 0:
        fig.delaxes(fig.axes[-1])

    fig.suptitle("Sensor readings — normal vs detected anomalies",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(f"{output_dir}/sensor_subplots.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → plots/sensor_subplots.png")