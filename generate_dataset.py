import numpy as np


# --- Constants for the dataset ---
# N_POINTS will now be a range
MIN_POINTS = 30
MAX_POINTS = 70
# INTERNAL_ANGLE_DEG will also be a range
MIN_ANGLE_DEG = 30.0
MAX_ANGLE_DEG = 120.0

AMPLITUDE = 3.0
FREQUENCY = 0.5
BOND_LENGTH = 1.0
N_SAMPLES = 2000  # Increased sample size for more variability
OUTPUT_FILE = "zigzag_variable_dataset.npz"


def generate_sample(n_points, internal_angle_deg, amplitude, frequency, bond_length):
    """
    Generates a single polymer sample with variable length and angle.

    Returns:
        np.ndarray: shape (n_points, 3, 2)
                    The 3D array of atom coordinates for the polymer.
    """
    # 1. Calculate the global structure (the sine wave vertices)
    x_centers = amplitude * np.linspace(0, (2 * np.pi) / frequency, n_points)
    x_centers -= x_centers.mean()
    y_centers = amplitude * np.sin(frequency * x_centers)
    y_centers -= y_centers.mean()
    centers = np.stack([x_centers, y_centers], axis=1)

    # 2. Define the local structure (the rigid V-shape)
    half_angle_rad = np.deg2rad(internal_angle_deg) / 2.0
    rel_atom0 = np.array(
        [-bond_length * np.sin(half_angle_rad), bond_length * np.cos(half_angle_rad)]
    )
    rel_atom2 = np.array(
        [bond_length * np.sin(half_angle_rad), bond_length * np.cos(half_angle_rad)]
    )

    # 3. Combine global and local structures
    monomers = np.zeros((n_points, 3, 2))
    for i in range(n_points):
        vertex_coord = centers[i]
        monomers[i, 0] = vertex_coord + rel_atom0
        monomers[i, 1] = vertex_coord
        monomers[i, 2] = vertex_coord + rel_atom2

    return monomers


def main():
    """
    Generates the full dataset with variable properties and saves it.
    """
    print(f"Generating {N_SAMPLES} samples with variable lengths and angles...")

    dataset_structures = []
    dataset_angles = []
    dataset_n_points = []

    for _ in range(N_SAMPLES):
        # Sample random properties for each structure
        n_points = np.random.randint(MIN_POINTS, MAX_POINTS + 1)
        internal_angle = np.random.uniform(MIN_ANGLE_DEG, MAX_ANGLE_DEG)

        sample = generate_sample(
            n_points, internal_angle, AMPLITUDE, FREQUENCY, BOND_LENGTH
        )

        # Reshape to (n_atoms, 2) for easier handling of variable lengths
        n_atoms = n_points * 3
        dataset_structures.append(sample.reshape(n_atoms, 2))
        dataset_angles.append(internal_angle)
        dataset_n_points.append(n_points)

    # Save as a list of arrays in the .npz file
    np.savez_compressed(
        OUTPUT_FILE,
        structures=np.array(dataset_structures, dtype=object),
        true_angle_degs=np.array(dataset_angles),
        n_points=np.array(dataset_n_points),
    )

    print(f"Variable dataset saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
