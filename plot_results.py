import pandas as pd
import matplotlib.pyplot as plt
import sys
import os


def plot_metrics(edm_log_file, af3_log_file, output_image):
    """
    Reads two CSV log files and generates a comparison plot.
    """
    if not os.path.exists(edm_log_file):
        print(f"Error: Log file not found: {edm_log_file}")
        return
    if not os.path.exists(af3_log_file):
        print(f"Error: Log file not found: {af3_log_file}")
        return

    try:
        df_edm = pd.read_csv(edm_log_file)
        df_af3 = pd.read_csv(af3_log_file)
    except pd.errors.EmptyDataError:
        print("Error: One of the log files is empty. Did training run?")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Global Accuracy (MSE)
    ax1.plot(
        df_edm["epoch"],
        df_edm["global_mse"],
        label="EDM-style (MSE only)",
        color="blue",
        alpha=0.8,
    )
    ax1.plot(
        df_af3["epoch"],
        df_af3["global_mse"],
        label="AF3-style (Composite)",
        color="red",
        alpha=0.8,
    )
    ax1.set_title("Global Accuracy (Vertex MSE)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Mean Squared Error (Lower is better)")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Plot 2: Local Accuracy (Angle MAE)
    ax2.plot(
        df_edm["epoch"],
        df_edm["local_angle_mae"],
        label="EDM-style (MSE only)",
        color="blue",
        alpha=0.8,
    )
    ax2.plot(
        df_af3["epoch"],
        df_af3["local_angle_mae"],
        label="AF3-style (Composite)",
        color="red",
        alpha=0.8,
    )
    ax2.set_title("Local Accuracy (Angle MAE)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Mean Absolute Error (deg) (Lower is better)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(
        "GeoDiffusion Experiment Results: EDM-style vs. AF3-style Loss", fontsize=16
    )
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))

    plt.savefig(output_image)
    print(f"Results plot saved to {output_image}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python plot_results.py <edm_log_file.csv> <af3_log_file.csv>")
        sys.exit(1)

    edm_file = sys.argv[1]
    af3_file = sys.argv[2]
    plot_metrics(edm_file, af3_file, "experiment_results.png")
