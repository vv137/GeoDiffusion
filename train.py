import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import csv
from model import SimpleDenoiser
import matplotlib.pyplot as plt

# --- PyTorch-based Loss Functions ---


def l2_loss(pred_clean, true_clean):
    """Standard L2 (MSE) loss."""
    return (pred_clean - true_clean).pow(2).mean()


def local_angle_loss(pred_clean_flat, n_points, true_angle_deg):
    """
    Torch-based version of the local angle metric.
    In this version, it is used for *evaluation* (logging) only.

    Args:
        pred_clean_flat (torch.Tensor): (B, N_Atoms, 2)
        n_points (int): Number of V-shapes (monomers)
        true_angle_deg (float): Ground truth angle
    """
    # Reshape (B, N_Atoms, 2) -> (B, N_Points, 3, 2)
    B = pred_clean_flat.shape[0]
    pred_structures = pred_clean_flat.view(B, n_points, 3, 2)

    # Get atom coordinates for all monomers in the batch
    # Shape: (B, n_points, 2)
    atom0 = pred_structures[:, :, 0, :]  # Left arm
    atom1 = pred_structures[:, :, 1, :]  # Vertex
    atom2 = pred_structures[:, :, 2, :]  # Right arm

    # Calculate the two vectors from the vertex
    # Shape: (B, n_points, 2)
    vec_a = atom0 - atom1
    vec_b = atom2 - atom1

    # Calculate the dot product per monomer
    dot_product = torch.einsum("...i,...i->...", vec_a, vec_b)

    # Calculate the magnitudes per vector
    norm_a = torch.norm(vec_a, dim=-1)
    norm_b = torch.norm(vec_b, dim=-1)

    # Calculate cosine of the angle
    cos_theta = dot_product / ((norm_a * norm_b) + 1e-9)
    cos_theta = torch.clip(cos_theta, -1.0, 1.0)

    # Get the angle in degrees
    pred_angles_rad = torch.acos(cos_theta)
    pred_angles_deg = torch.rad2deg(pred_angles_rad)

    # Calculate the mean absolute error (this is our loss)
    angle_error = torch.abs(pred_angles_deg - true_angle_deg)
    return angle_error.mean()


def smooth_lddt_loss(
    pred_clean,
    true_clean,
    cutoff_radius=2.0,
    cutoffs=[0.1, 0.25, 0.5, 1.0],
    device=torch.device("cpu"),
):
    """
    A general-purpose local distance preservation loss, inspired by AF3's
    smooth LDDT (Algorithm 27).

    Args:
        pred_clean (torch.Tensor): Predicted clean structures (B, N_Atoms, 2)
        true_clean (torch.Tensor): Ground truth structures (B, N_Atoms, 2)
        cutoff_radius (float): Local neighborhood radius.
        cutoffs (list): Thresholds for the sigmoids (data-scale dependent).
        device (str): PyTorch device.
    """
    B, N_Atoms, _ = pred_clean.shape

    # Calculate pairwise distance matrices
    # Shape: (B, N_Atoms, N_Atoms)
    dist_pred = torch.cdist(pred_clean, pred_clean)
    dist_true = torch.cdist(true_clean, true_clean)

    # Calculate distance difference
    # Shape: (B, N_Atoms, N_Atoms)
    delta = torch.abs(dist_true - dist_pred)

    # Create a local neighborhood mask
    # We only care about pairs where the true distance is within the radius
    mask = (dist_true < cutoff_radius).float()

    # Exclude self-comparisons (diagonal)
    mask = mask * (1.0 - torch.eye(N_Atoms, device=device)).unsqueeze(0)

    if mask.sum() == 0:
        # Avoid division by zero if batch is small or cutoff is too low
        return torch.tensor(0.0, device=device)

    # AF3-style smooth sigmoids
    # We generalize the cutoffs for our toy data scale
    s1 = torch.sigmoid(cutoffs[0] - delta)
    s2 = torch.sigmoid(cutoffs[1] - delta)
    s3 = torch.sigmoid(cutoffs[2] - delta)
    s4 = torch.sigmoid(cutoffs[3] - delta)

    # Average score
    score = 0.25 * (s1 + s2 + s3 + s4)

    # Calculate masked mean lDDT score
    lddt = (score * mask).sum() / mask.sum()

    # Loss is 1 - lDDT
    return 1.0 - lddt


# --- PyTorch-based Metrics for Logging ---


def global_mse_metric(pred_clean_flat, true_clean_flat, n_points):
    """Torch-based Global MSE metric for validation logging."""
    B = pred_clean_flat.shape[0]
    pred_centers = pred_clean_flat.view(B, n_points, 3, 2)[:, :, 1, :]
    true_centers = true_clean_flat.view(B, n_points, 3, 2)[:, :, 1, :]
    return (pred_centers - true_centers).pow(2).mean()


# --- Weighting Functions ---


def get_edm_weight(sigma, sigma_data):
    """
    EDM-style loss weighting (from Eq. 6 in the supp. info).
    This is Lambda(t) which multiplies the MSE loss.
    """
    sigma_data = torch.tensor(sigma_data, device=sigma.device)
    num = sigma.pow(2) + sigma_data.pow(2)
    den = (sigma * sigma_data).pow(2)
    return num / den


def get_af3_weight(sigma, sigma_data):
    """
    AlphaFold 3-style loss weighting (from Eq. 6 in the supp. info).
    This is Lambda(t) which multiplies the MSE loss.
    """
    sigma_data = torch.tensor(sigma_data, device=sigma.device)
    num = sigma.pow(2) + sigma_data.pow(2)
    den = (sigma + sigma_data).pow(2)
    return num / den


# --- Plotting Function ---


def plot_validation_samples(
    pred_structures, true_structures, epoch, loss_type, save_dir, num_to_plot=4
):
    """
    Plots a comparison of predicted vs. ground truth validation samples.

    Args:
        pred_structures (np.ndarray): (B, N_Points, 3, 2)
        true_structures (np.ndarray): (B, N_Points, 3, 2)
        epoch (int): Current epoch number
        loss_type (str): 'edm' or 'af3'
        save_dir (str): Directory to save plots
        num_to_plot (int): Number of samples to plot
    """

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    num_samples = min(pred_structures.shape[0], num_to_plot)

    for i in range(num_samples):
        # (N_Points, 3, 2)
        pred_sample = pred_structures[i]
        true_sample = true_structures[i]

        ax = axes[i]

        # Plot all true atoms as dots
        true_atoms = true_sample.reshape(-1, 2)
        ax.scatter(
            true_atoms[:, 0],
            true_atoms[:, 1],
            s=10,
            c="blue",
            alpha=0.4,
            label="Ground Truth Atoms",
        )

        # Plot all pred atoms as 'x'
        pred_atoms = pred_sample.reshape(-1, 2)
        ax.scatter(
            pred_atoms[:, 0],
            pred_atoms[:, 1],
            s=10,
            c="red",
            alpha=0.4,
            marker="x",
            label="Predicted Atoms",
        )

        # Plot the arms (lines)
        for j in range(true_sample.shape[0]):  # Iterate over N_Points
            true_p0, true_p1, true_p2 = true_sample[j]
            pred_p0, pred_p1, pred_p2 = pred_sample[j]

            # Plot true arms
            ax.plot(
                [true_p0[0], true_p1[0]],
                [true_p0[1], true_p1[1]],
                c="blue",
                alpha=0.7,
                linewidth=2,
                label="GT Arms" if j == 0 else None,
            )
            ax.plot(
                [true_p1[0], true_p2[0]],
                [true_p1[1], true_p2[1]],
                c="blue",
                alpha=0.7,
                linewidth=2,
            )

            # Plot pred arms
            ax.plot(
                [pred_p0[0], pred_p1[0]],
                [pred_p0[1], pred_p1[1]],
                c="red",
                alpha=0.7,
                linestyle="-",
                linewidth=2,
                label="Pred Arms" if j == 0 else None,
            )
            ax.plot(
                [pred_p1[0], pred_p2[0]],
                [pred_p1[1], pred_p2[1]],
                c="red",
                alpha=0.7,
                linestyle="-",
                linewidth=2,
            )

        ax.set_title(f"Sample {i + 1}")
        ax.legend()
        ax.axis("equal")

    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis("off")

    plt.suptitle(
        f"Validation Samples - Epoch {epoch} - {loss_type.upper()} Loss", fontsize=16
    )
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))

    output_image = os.path.join(
        save_dir, f"validation_plot_epoch_{epoch:04d}_{loss_type}.png"
    )
    plt.savefig(output_image)
    plt.close(fig)  # Close figure to free memory


# --- Data Loader ---


class ZigZagDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.structures = torch.from_numpy(data["structures"]).float()
        self.true_angle_deg = data["true_angle_deg"][0]
        self.n_points = data["n_points"][0]

        # Reshape for the model: (B, N_Points, 3, 2) -> (B, N_Atoms, 2)
        n_samples, n_points, n_atoms_per_point, n_dims = self.structures.shape
        self.n_atoms = n_points * n_atoms_per_point
        self.structures = self.structures.view(n_samples, self.n_atoms, n_dims)

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        return self.structures[idx]


# --- Main Training Script ---


def main(args):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Load data
    dataset = ZigZagDataset(args.data_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    n_atoms = dataset.n_atoms
    n_points = dataset.n_points
    true_angle_deg = dataset.true_angle_deg

    # Create a fixed validation batch
    val_batch = next(iter(DataLoader(dataset, batch_size=32, shuffle=False))).to(device)

    # Initialize model
    model = SimpleDenoiser(n_atoms_in=n_atoms, n_atoms_out=n_atoms).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Open log file
    with open(args.log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "loss_type", "total_loss", "global_mse", "local_angle_mae"]
        )

        print(f"--- Training model with loss: {args.loss_type} ---")

        for epoch in range(args.epochs):
            model.train()
            for true_clean in dataloader:
                true_clean = true_clean.to(device)
                B = true_clean.shape[0]

                # 1. Sample noise level (sigma)
                # Log-normal distribution, a common choice
                sigma = torch.exp(torch.randn(B, 1, 1, device=device) * 1.2 - 0.5)

                # 2. Create noisy input
                noise = torch.randn_like(true_clean)
                x_noisy = true_clean + noise * sigma

                # 3. Get prediction
                # (B, N_Atoms, 2), (B, 1, 1) -> (B, N_Atoms, 2)
                pred_clean = model(x_noisy, sigma.squeeze(-1))

                # 4. Calculate losses
                loss_mse = l2_loss(pred_clean, true_clean)

                if args.loss_type == "af3":
                    # Use the general-purpose Smooth LDDT loss
                    loss_local = smooth_lddt_loss(pred_clean, true_clean, device=device)
                    weight = get_af3_weight(sigma.squeeze(), args.sigma_data)
                    # AF3-style loss: L_total = Lambda(t) * L_mse + L_smooth_lddt
                    total_loss = (weight * loss_mse).mean() + loss_local
                else:  # 'edm'
                    # EDM-style control
                    weight = get_edm_weight(sigma.squeeze(), args.sigma_data)
                    total_loss = (weight * loss_mse).mean()

                # 5. Optimization step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            # --- Validation ---
            model.eval()
            with torch.no_grad():
                # For validation, we test at a fixed, low noise level
                val_sigma = torch.tensor([[0.05]], device=device).expand(
                    val_batch.shape[0], 1
                )
                val_noise = torch.randn_like(val_batch)
                val_noisy = val_batch + val_noise * val_sigma.unsqueeze(-1)

                val_pred_clean = model(val_noisy, val_sigma)

                # Calculate metrics
                val_global_mse = global_mse_metric(
                    val_pred_clean, val_batch, n_points
                ).item()
                # We still *track* the local angle MAE to see if our
                # smooth_lddt loss was successful at solving it.
                val_local_mae = local_angle_loss(
                    val_pred_clean, n_points, true_angle_deg
                ).item()

            # Log to console and file
            print(
                f"Epoch {epoch + 1}/{args.epochs} | Loss: {total_loss.item():.4f} | "
                f"Global MSE: {val_global_mse:.4f} | Local Angle MAE: {val_local_mae:.4f}"
            )
            writer.writerow(
                [
                    epoch + 1,
                    args.loss_type,
                    total_loss.item(),
                    val_global_mse,
                    val_local_mae,
                ]
            )

            # --- Plot Validation Samples ---
            if (epoch + 1) % args.plot_every == 0:
                print(f"Plotting validation samples for epoch {epoch + 1}...")
                # Reshape for plotting
                B_val = val_batch.shape[0]
                val_pred_plot = (
                    val_pred_clean.view(B_val, n_points, 3, 2).detach().cpu().numpy()
                )
                val_true_plot = (
                    val_batch.view(B_val, n_points, 3, 2).detach().cpu().numpy()
                )

                plot_validation_samples(
                    val_pred_plot,
                    val_true_plot,
                    epoch + 1,
                    args.loss_type,
                    args.checkpoint_dir,
                )

    # Save final checkpoint
    checkpoint_path = os.path.join(
        args.checkpoint_dir, f"final_model_{args.loss_type}.pth"
    )
    torch.save(model.state_dict(), checkpoint_path)
    print(f"--- Training finished. Model saved to {checkpoint_path} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GeoDiffusion Toy Experiment")
    parser.add_argument(
        "--loss_type",
        type=str,
        choices=["edm", "af3"],
        required=True,
        help="Type of loss function to use ('edm' or 'af3')",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="zigzag_dataset.npz",
        help="Path to the .npz dataset file",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        required=True,
        help="Path to save the output CSV log",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--sigma_data",
        type=float,
        default=1.0,
        help="Data variance parameter (sigma_data in loss)",
    )
    parser.add_argument(
        "--plot_every",
        type=int,
        default=10,
        help="Save a plot of validation samples every N epochs",
    )

    args = parser.parse_args()
    main(args)
