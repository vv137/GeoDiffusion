import numpy as np

# --- Constants for the dataset ---
N_POINTS = 50  # Number of "monomers" (V-shapes) in each polymer
AMPLITUDE = 2.0  # Amplitude of the sine wave
FREQUENCY = 0.5  # Frequency of the sine wave
INTERNAL_ANGLE_DEG = 45.0  # Sharp local angle (ground truth)
BOND_LENGTH = 0.5  # Length of the V-shape arms
N_SAMPLES = 1000  # Total number of polymers to generate
OUTPUT_FILE = "zigzag_dataset.npz"


def generate_sample(n_points, amplitude, frequency, internal_angle_deg, bond_length):
    """
    Generates a single polymer sample.

    Each sample consists of 'n_points' monomers. Each monomer is a 3-atom
    V-shape, with the vertex (atom 1) lying on a sine wave.

    Returns:
        np.ndarray: shape (n_points, 3, 2)
                    The 3D array of atom coordinates for the polymer.
                    [monomer_index, atom_index, (x, y)]
                    - atom_index 0: Left arm atom
                    - atom_index 1: Vertex atom (on the sine wave)
                    - atom_index 2: Right arm atom
    """

    # 1. Calculate the global structure (the sine wave vertices)
    x_centers = np.linspace(0, (2 * np.pi) / frequency, n_points)
    y_centers = amplitude * np.sin(frequency * x_centers)

    # (n_points, 2) array of vertex coordinates
    centers = np.stack([x_centers, y_centers], axis=1)

    # 2. Define the local structure (the rigid V-shape)
    half_angle_rad = np.deg2rad(internal_angle_deg) / 2.0

    # Relative coordinates of the arm atoms from the vertex
    # We orient the V-shape to point "up" (positive y)
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
        monomers[i, 0] = vertex_coord + rel_atom0  # Left arm
        monomers[i, 1] = vertex_coord  # Vertex
        monomers[i, 2] = vertex_coord + rel_atom2  # Right arm

    return monomers


def main():
    """
    Generates the full dataset and saves it to a .npz file.
    """
    print(f"Generating {N_SAMPLES} samples...")

    dataset = []
    for _ in range(N_SAMPLES):
        sample = generate_sample(
            N_POINTS, AMPLITUDE, FREQUENCY, INTERNAL_ANGLE_DEG, BOND_LENGTH
        )
        dataset.append(sample)

    # Stack all samples into a single numpy array
    # Shape: (N_SAMPLES, N_POINTS, 3, 2)
    dataset_array = np.array(dataset)

    # Save to a compressed .npz file
    np.savez_compressed(
        OUTPUT_FILE,
        structures=dataset_array,
        true_angle_deg=np.array([INTERNAL_ANGLE_DEG]),  # Save metadata
        n_points=np.array([N_POINTS]),
    )

    print(f"Dataset saved to {OUTPUT_FILE}")
    print(f"Dataset shape: {dataset_array.shape}")


if __name__ == "__main__":
    main()
