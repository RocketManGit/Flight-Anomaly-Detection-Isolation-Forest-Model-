import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

def plot_anomalies(results_path: str = "data/results.csv",
                  output_dir:   str = "plots"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(results_path)

    normal   = df[df["anomaly_label"] ==  1]
    anomalies= df[df["anomaly_label"] == -1]

    # ── Figure 1: Anomaly score over time ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df["time"], df["anomaly_score"],
            color="#3B8BD4", linewidth=0.7, label="Anomaly score")
    ax.scatter(anomalies["time"], anomalies["anomaly_score"],
               color="#E8593C", s=25, zorder=5, label="Flagged anomaly")
    ax.axhline(df[df["anomaly_label"]==-1]["anomaly_score"].min(),
               color="#E8593C", linestyle="--", linewidth=0.8, alpha=0.5,
               label="Detection threshold")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Anomaly score  (0 = normal, 1 = anomalous)")
    ax.set_title("Isolation Forest — anomaly score over flight")
    ax.legend(framealpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{output_dir}/anomaly_score_timeline.png", dpi=150)
    plt.close()
    print("Saved → plots/anomaly_score_timeline.png")

    # ── Figure 2: Feature subplots with flagged anomalies overlaid ─────────
    features = ["airspeed", "altitude", "engine_rpm",
                "engine_temp", "vertical_speed", "fuel_flow"]
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.3)

    for i, feat in enumerate(features):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        ax.plot(normal["time"],    normal[feat],
                color="#3B8BD4", linewidth=0.6, alpha=0.7, label="Normal")
        ax.scatter(anomalies["time"], anomalies[feat],
                   color="#E8593C", s=18, zorder=5, label="Anomaly")
        ax.set_title(feat.replace("_", " ").title(), fontsize=10)
        ax.set_xlabel("Time step", fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7, framealpha=0.3)

    fig.suptitle("Sensor readings — normal vs detected anomalies",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(f"{output_dir}/sensor_subplots.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → plots/sensor_subplots.png")


if __name__ == "__main__":
    plot_anomalies()