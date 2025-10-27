import numpy as np
import matplotlib.pyplot as plt
import os

# Constants from the dataset generator
DATA_FILE = "zigzag_dataset.npz"
N_POINTS = 50
NUM_TO_PLOT = 4


def plot_samples(data_file):
    """
    Loads the generated dataset and plots a few samples.
    """
    if not os.path.exists(data_file):
        print(f"Error: Dataset file not found at {data_file}")
        print("Please run generate_dataset.py first.")
        return

    try:
        data = np.load(data_file)
        structures = data["structures"]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Plotting {NUM_TO_PLOT} samples from {data_file}...")

    # Create a 2x2 grid for plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()  # Flatten the 2x2 grid to a 1D array

    num_samples = min(structures.shape[0], NUM_TO_PLOT)

    for i in range(num_samples):
        # (N_POINTS, 3, 2)
        sample = structures[i]

        # Reshape to (N_Atoms, 2)
        atoms = sample.reshape(-1, 2)  # (150, 2)
        vertices = sample[:, 1, :]  # (50, 2)

        ax = axes[i]
        # Plot all atoms
        ax.scatter(atoms[:, 0], atoms[:, 1], s=10, c="blue", alpha=0.6, label="Atoms")
        # Highlight vertices
        ax.plot(
            vertices[:, 0],
            vertices[:, 1],
            "o-",
            c="red",
            alpha=0.8,
            markersize=5,
            label="Vertices (Global)",
        )

        ax.set_title(f"Sample {i + 1}")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.legend()
        ax.axis("equal")

    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Ground Truth Dataset Samples", fontsize=16)
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))

    output_image = "dataset_samples.png"
    plt.savefig(output_image)
    print(f"Sample plot saved to {output_image}")


if __name__ == "__main__":
    plot_samples(DATA_FILE)
