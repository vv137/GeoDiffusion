import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse


def plot_metrics(log_files, labels, output_image, window_size):
    """
    Reads multiple CSV log files and generates a comparison plot.
    Includes rolling average for smoothing and std deviation band for fluctuation.

    Args:
        log_files (list): List of paths to the CSV log files.
        labels (list): List of labels for each log file/condition.
        output_image (str): Path to save the output plot image.
        window_size (int): Window size for rolling average/std.
    """
    if len(log_files) != len(labels):
        print("Error: Number of log files must match number of labels.")
        return

    dfs = []
    for log_file in log_files:
        if not os.path.exists(log_file):
            print(f"Warning: Log file not found: {log_file}. Skipping.")
            dfs.append(None)
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))  # Increased width
    colors = plt.cm.viridis(np.linspace(0, 1, len(dfs)))
    linestyles = ["-", "--", ":", "-."]

    # Plot 1: Global Accuracy (MSE)
    for i, (df, label) in enumerate(zip(dfs, labels)):
        if df is not None:
            # Calculate rolling mean and std
            # center=True: Causal이 아닌, 해당 지점 중심의 평균을 계산
            # min_periods=1: 데이터가 1개만 있어도 계산 (처음 부분)
            global_mean = (
                df["global_mse"]
                .rolling(window=window_size, center=True, min_periods=1)
                .mean()
            )
            global_std = (
                df["global_mse"]
                .rolling(window=window_size, center=True, min_periods=1)
                .std()
                .fillna(0)
            )  # std가 NaN인 경우 0으로

            color = colors[i % len(colors)]
            style = linestyles[i % len(linestyles)]

            # Plot the smoothed rolling average line
            ax1.plot(
                df["epoch"],
                global_mean,
                label=f"{label} (Smooth Trend)",
                color=color,
                linestyle=style,
                alpha=0.9,
                linewidth=2,
            )
            # Plot the fluctuation band (mean +/- 1 std)
            ax1.fill_between(
                df["epoch"],
                global_mean - global_std,
                global_mean + global_std,
                color=color,
                alpha=0.15,
                label=f"{label} (Fluctuation)",
            )

    ax1.set_title("Global Accuracy (Vertex MSE)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Mean Squared Error (Lower is better)")
    ax1.legend(fontsize="small")
    ax1.grid(True, linestyle="--", alpha=0.5)
    # ax1.set_ylim(bottom=0)
    ax1.set_yscale("log")

    # Plot 2: Local Accuracy (Angle MAE)
    for i, (df, label) in enumerate(zip(dfs, labels)):
        if df is not None:
            # Calculate rolling mean and std
            local_mean = (
                df["local_angle_mae"]
                .rolling(window=window_size, center=True, min_periods=1)
                .mean()
            )
            local_std = (
                df["local_angle_mae"]
                .rolling(window=window_size, center=True, min_periods=1)
                .std()
                .fillna(0)
            )

            color = colors[i % len(colors)]
            style = linestyles[i % len(linestyles)]

            # Plot the smoothed rolling average line
            ax2.plot(
                df["epoch"],
                local_mean,
                label=f"{label} (Smooth Trend)",
                color=color,
                linestyle=style,
                alpha=0.9,
                linewidth=2,
            )
            # Plot the fluctuation band (mean +/- 1 std)
            ax2.fill_between(
                df["epoch"],
                local_mean - local_std,
                local_mean + local_std,
                color=color,
                alpha=0.15,
                label=f"{label} (Fluctuation)",
            )

    ax2.set_title("Local Accuracy (Internal Angle MAE)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Mean Absolute Error (deg) (Lower is better)")
    ax2.legend(fontsize="small")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.set_ylim(bottom=0)

    fig.suptitle(
        f"GeoDiffusion Experiment Results (Smoothed, Window={window_size})", fontsize=18
    )
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.94))

    plt.savefig(output_image)
    print(f"Results comparison plot saved to {output_image}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot GeoDiffusion Experiment Results")
    parser.add_argument(
        "log_files",
        nargs="+",
        help="Paths to the CSV log files (e.g., log1.csv log2.csv ...)",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Labels for the experiments in the plot legend. Must match log_files.",
    )
    parser.add_argument(
        "--output", default="experiment_comparison.png", help="Output image file name."
    )
    # Add window size argument
    parser.add_argument(
        "--window_size",
        type=int,
        default=10,
        help="Window size for rolling average smoothing.",
    )

    args = parser.parse_args()

    labels = args.labels
    if not labels:
        labels = [os.path.basename(f) for f in args.log_files]

    if len(labels) != len(args.log_files):
        print("Error: The number of --labels must match the number of log_files.")
    else:
        # Pass window_size to the plotting function
        plot_metrics(args.log_files, labels, args.output, args.window_size)
