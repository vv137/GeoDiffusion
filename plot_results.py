import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse


def plot_metrics(log_files, labels, output_image):
    """
    Reads multiple CSV log files and generates a comparison plot.

    Args:
        log_files (list): List of paths to the CSV log files.
        labels (list): List of labels for each log file/condition.
        output_image (str): Path to save the output plot image.
    """
    if len(log_files) != len(labels):
        print("Error: Number of log files must match number of labels.")
        return

    dfs = []
    for log_file in log_files:
        if not os.path.exists(log_file):
            print(f"Warning: Log file not found: {log_file}. Skipping.")
            dfs.append(None)  # Keep placeholder to maintain label correspondence
            continue
        try:
            df = pd.read_csv(log_file)
            if df.empty:
                print(f"Warning: Log file is empty: {log_file}. Skipping.")
                dfs.append(None)
            else:
                dfs.append(df)
        except Exception as e:
            print(f"Error reading log file {log_file}: {e}. Skipping.")
            dfs.append(None)

    if all(df is None for df in dfs):
        print("Error: No valid log files found to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))  # Increased width
    colors = plt.cm.viridis(np.linspace(0, 1, len(dfs)))  # Use colormap for distinction
    linestyles = ["-", "--", ":", "-."]

    # Plot 1: Global Accuracy (MSE)
    for i, (df, label) in enumerate(zip(dfs, labels)):
        if df is not None:
            ax1.plot(
                df["epoch"],
                df["global_mse"],
                label=label,
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)],
                alpha=0.9,
            )
    ax1.set_title("Global Accuracy (Vertex MSE)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Mean Squared Error (Lower is better)")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)
    # ax1.set_ylim(bottom=-1)  # Start y-axis slightly below 0

    # Plot 2: Local Accuracy (Angle MAE)
    for i, (df, label) in enumerate(zip(dfs, labels)):
        if df is not None:
            ax2.plot(
                df["epoch"],
                df["local_angle_mae"],
                label=label,
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)],
                alpha=0.9,
            )
    ax2.set_title("Local Accuracy (Internal Angle MAE)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Mean Absolute Error (deg) (Lower is better)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)
    # ax2.set_ylim(bottom=-5)  # Start y-axis slightly below 0

    fig.suptitle(
        "GeoDiffusion Experiment Results: Loss Function Comparison", fontsize=18
    )
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.94))  # Adjust rect

    plt.savefig(output_image)
    print(f"Results comparison plot saved to {output_image}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot GeoDiffusion Experiment Results")
    parser.add_argument(
        "log_files",
        nargs=4,
        help="Paths to the four CSV log files (e.g., edm_no.csv edm_with.csv af3_no.csv af3_with.csv)",
    )
    parser.add_argument(
        "--labels",
        nargs=4,
        default=[
            "EDM (MSE)",
            "EDM + SmoothLDDT",
            "AF3 (MSE Weight)",
            "AF3 + SmoothLDDT",
        ],
        help="Labels for the four experiments in the plot legend.",
    )
    parser.add_argument(
        "--output", default="experiment_comparison.png", help="Output image file name."
    )

    args = parser.parse_args()
    plot_metrics(args.log_files, args.labels, args.output)
