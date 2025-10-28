import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import csv

# Import the new EDM parameterized model
from model import EDMPrecondSimpleDenoiser
import matplotlib.pyplot as plt


# --- Loss Functions ---
# (l2_loss, smooth_lddt_loss remain the same as previous version)
def l2_loss(pred_clean, true_clean):
    """Standard L2 (MSE) loss."""
    # Flatten N_Atoms and coordinates for mean calculation
    return (pred_clean - true_clean).pow(2).mean()


def smooth_lddt_loss(
    pred_clean,
    true_clean,
    cutoff_radius=2.0,
    cutoffs=[0.1, 0.25, 0.5, 1.0],
    eps=1e-9,
    device=torch.device("cpu"),
):
    """
    A general-purpose local distance preservation loss, inspired by AF3's
    smooth LDDT (Algorithm 27). Args/Returns same as before.
    """
    B, N_Atoms, _ = pred_clean.shape
    dist_pred = torch.cdist(pred_clean, pred_clean)
    dist_true = torch.cdist(true_clean, true_clean)
    delta = torch.abs(dist_true - dist_pred)
    mask = (dist_true < cutoff_radius).float()
    mask = mask * (1.0 - torch.eye(N_Atoms, device=device)).unsqueeze(0)
    mask_sum = mask.sum()
    if mask_sum < eps:
        return torch.tensor(0.0, device=device)
    # AF3-style smooth sigmoids (inverted logic compared to paper for loss)
    s1 = torch.sigmoid(delta - cutoffs[0])  # Score increases as delta increases
    s2 = torch.sigmoid(delta - cutoffs[1])
    s3 = torch.sigmoid(delta - cutoffs[2])
    s4 = torch.sigmoid(delta - cutoffs[3])
    score = 0.25 * (s1 + s2 + s3 + s4)
    lddt_error_score = (score * mask).sum() / mask_sum
    return lddt_error_score  # Return 1-lddt if you want the score itself


# --- Metrics ---
# (global_mse_metric, local_angle_metric remain the same)
def global_mse_metric(pred_clean_flat, true_clean_flat, n_points):
    """Torch-based Global MSE metric (on vertices) for validation logging."""
    B = pred_clean_flat.shape[0]
    pred_centers = pred_clean_flat.view(B, n_points, 3, 2)[:, :, 1, :]
    true_centers = true_clean_flat.view(B, n_points, 3, 2)[:, :, 1, :]
    return (pred_centers - true_centers).pow(2).mean()


def local_angle_metric(pred_clean_flat, n_points, true_angle_deg):
    """
    Torch-based version of the local angle metric (MAE).
    Used for *evaluation* (logging) only. Args/Returns same as before.
    """
    B = pred_clean_flat.shape[0]
    pred_structures = pred_clean_flat.view(B, n_points, 3, 2)
    atom0 = pred_structures[:, :, 0, :]
    atom1 = pred_structures[:, :, 1, :]
    atom2 = pred_structures[:, :, 2, :]
    vec_a = atom0 - atom1
    vec_b = atom2 - atom1
    dot_product = torch.einsum("...i,...i->...", vec_a, vec_b)
    norm_a = torch.norm(vec_a, dim=-1)
    norm_b = torch.norm(vec_b, dim=-1)
    cos_theta = dot_product / ((norm_a * norm_b) + 1e-9)
    cos_theta = torch.clip(cos_theta, -1.0, 1.0)
    pred_angles_rad = torch.acos(cos_theta)
    pred_angles_deg = torch.rad2deg(pred_angles_rad)
    angle_error = torch.abs(pred_angles_deg - true_angle_deg)
    return angle_error.mean()


# --- Weighting Functions ---
# (get_edm_weight, get_af3_weight remain the same)
def get_edm_weight(sigma, sigma_data):
    """
    EDM-style loss weighting (from Eq. 22 in EDM paper, 1/c_out^2).
    Multiplies the raw MSE loss.
    lambda(sigma) = (sigma^2 + sigma_data^2) / (sigma^2 * sigma_data^2)
    """
    sigma = sigma.squeeze()  # Ensure sigma is shape [B]
    sigma_data = torch.tensor(sigma_data, device=sigma.device)
    num = sigma.pow(2) + sigma_data.pow(2)
    den = (sigma * sigma_data).pow(
        2
    ) + 1e-9  # Add epsilon for numerical stability at sigma=0
    weight = num / den
    # Reshape weight to (B, 1, 1) for broadcasting with loss (B, N, C)
    return weight.view(-1, 1, 1)


def get_af3_weight(sigma, sigma_data):
    """
    AlphaFold 3-style loss weighting (from Eq. 6 in the supp. info).
    Multiplies the raw MSE loss.
    Lambda(t) = (t^2 + sigma_data^2) / (t + sigma_data)^2
    """
    sigma = sigma.squeeze()  # Ensure sigma is shape [B]
    sigma_data = torch.tensor(sigma_data, device=sigma.device)
    num = sigma.pow(2) + sigma_data.pow(2)
    den = (sigma + sigma_data).pow(2) + 1e-9  # Add epsilon for numerical stability
    weight = num / den
    # Reshape weight to (B, 1, 1) for broadcasting with loss (B, N, C)
    return weight.view(-1, 1, 1)


# --- Plotting Function ---
# (plot_validation_samples remains the same)
def plot_validation_samples(
    pred_structures,
    true_structures,
    epoch,
    loss_type,
    smooth_lddt_weight,
    save_dir,
    num_to_plot=4,
):
    """Plots validation samples. Args/Body same as before."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    num_samples = min(pred_structures.shape[0], num_to_plot)
    for i in range(num_samples):
        pred_sample = pred_structures[i]
        true_sample = true_structures[i]
        ax = axes[i]
        true_atoms = true_sample.reshape(-1, 2)
        ax.scatter(
            true_atoms[:, 0],
            true_atoms[:, 1],
            s=10,
            c="blue",
            alpha=0.4,
            label="Ground Truth Atoms",
        )
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
        for j in range(true_sample.shape[0]):
            true_p0, true_p1, true_p2 = true_sample[j]
            pred_p0, pred_p1, pred_p2 = pred_sample[j]
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
    for i in range(num_samples, len(axes)):
        axes[i].axis("off")
    lddt_status = "ON" if smooth_lddt_weight > 0 else "OFF"
    plt.suptitle(
        f"Validation Samples - Epoch {epoch}\n"
        f"Loss Weight: {loss_type.upper()}, SmoothLDDT: {lddt_status} (w={smooth_lddt_weight})",
        fontsize=16,
    )
    plt.tight_layout(rect=(0, 0.03, 1, 0.93))
    output_image = os.path.join(save_dir, f"validation_plot_epoch_{epoch:04d}.png")
    plt.savefig(output_image)
    plt.close(fig)


# --- Data Loader ---
# (ZigZagDataset remains the same)
class ZigZagDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.structures = torch.from_numpy(data["structures"]).float()
        self.true_angle_deg = data["true_angle_deg"][0]
        self.n_points = data["n_points"][0]
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
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    n_atoms = dataset.n_atoms
    n_points = dataset.n_points
    true_angle_deg = dataset.true_angle_deg

    # Create a fixed validation batch
    val_batch = next(iter(DataLoader(dataset, batch_size=32, shuffle=False))).to(device)

    # Initialize model - USE THE NEW EDM PARAMETERIZED MODEL
    model = EDMPrecondSimpleDenoiser(
        n_atoms_in=n_atoms,
        n_atoms_out=n_atoms,
        sigma_data=args.sigma_data,  # Pass sigma_data to the model
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Log file setup
    log_file_path = os.path.join(args.log_dir, args.log_file_name)
    os.makedirs(args.log_dir, exist_ok=True)
    with open(log_file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "loss_type",
                "smooth_lddt_weight",
                "total_loss",
                "global_mse",
                "local_angle_mae",
            ]
        )
        # ... [Logging header remains the same] ...
        print("--- Training model (EDM Parameterized) ---")  # Indicate parameterization
        print(f"Loss Weighting: {args.loss_type}")
        print(f"Smooth LDDT Weight: {args.smooth_lddt_weight}")
        print(f"Sigma Data: {args.sigma_data}")
        print(f"Checkpoints: {args.checkpoint_dir}")
        print(f"Log File: {log_file_path}")
        print("-" * 20)

        for epoch in range(args.epochs):
            model.train()
            epoch_loss = 0.0
            num_batches = 0
            for true_clean in dataloader:
                true_clean = true_clean.to(device)
                B = true_clean.shape[0]

                # 1. Sample noise level (sigma) - Log-normal distribution
                log_sigma = torch.randn(B, 1, device=device) * 1.2 - 1.0  # (B, 1)
                sigma = torch.exp(log_sigma)  # (B, 1)

                # 2. Create noisy input
                noise = torch.randn_like(true_clean)
                # Expand sigma for broadcasting: (B, 1) -> (B, 1, 1)
                x_noisy = true_clean + noise * sigma.unsqueeze(-1)

                # 3. Get prediction D_theta(x_noisy, sigma) from EDM parameterized model
                # Pass sigma as (B, 1)
                pred_denoised = model(x_noisy, sigma)  # Model now returns D_theta

                # 4. Calculate weighted MSE loss using the output D_theta
                # Note: raw_loss_mse uses pred_denoised (D_theta), not pred_clean
                raw_loss_mse = (pred_denoised - true_clean).pow(2)  # (B, N_Atoms, 2)
                if args.loss_type == "af3":
                    weight = get_af3_weight(sigma, args.sigma_data)
                else:  # 'edm'
                    weight = get_edm_weight(sigma, args.sigma_data)

                # Apply weight and take mean
                weighted_mse_loss = (weight * raw_loss_mse).mean()

                # 5. Calculate Smooth LDDT loss (optional) - use D_theta as prediction
                loss_local = 0.0
                if args.smooth_lddt_weight > 0:
                    # Use the denoised prediction for LDDT loss
                    loss_local = smooth_lddt_loss(
                        pred_denoised,
                        true_clean,
                        device=device,
                        cutoff_radius=args.lddt_cutoff,
                    )

                # 6. Combine losses
                total_loss = weighted_mse_loss + args.smooth_lddt_weight * loss_local

                # 7. Optimization step
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += total_loss.item()
                num_batches += 1

            avg_epoch_loss = epoch_loss / num_batches

            # --- Validation ---
            model.eval()
            with torch.no_grad():
                # For validation, test at a fixed, low noise level
                val_sigma_val = 0.01  # Small sigma for near-clean evaluation
                val_sigma = torch.full(
                    (val_batch.shape[0], 1), val_sigma_val, device=device
                )

                val_noise = torch.randn_like(val_batch) * val_sigma_val
                val_noisy = val_batch + val_noise

                # Get denoised prediction D_theta for validation
                val_pred_denoised = model(val_noisy, val_sigma)

                # Calculate metrics using the denoised prediction
                val_global_mse = global_mse_metric(
                    val_pred_denoised, val_batch, n_points
                ).item()
                val_local_mae = local_angle_metric(
                    val_pred_denoised, n_points, true_angle_deg
                ).item()

            # Log to console and file
            print(
                f"Epoch {epoch + 1}/{args.epochs} | Avg Loss: {avg_epoch_loss:.4f} | "
                f"Val Global MSE: {val_global_mse:.4f} | Val Local Angle MAE: {val_local_mae:.4f}"
            )
            # ... [Writing to log file remains the same] ...
            writer.writerow(
                [
                    epoch + 1,
                    args.loss_type,
                    args.smooth_lddt_weight,
                    avg_epoch_loss,
                    val_global_mse,
                    val_local_mae,
                ]
            )
            f.flush()

            # --- Plot Validation Samples ---
            if (epoch + 1) % args.plot_every == 0:
                print(f"Plotting validation samples for epoch {epoch + 1}...")
                B_val = val_batch.shape[0]
                # Use the denoised prediction for plotting
                val_pred_plot = (
                    val_pred_denoised.view(B_val, n_points, 3, 2).detach().cpu().numpy()
                )
                val_true_plot = (
                    val_batch.view(B_val, n_points, 3, 2).detach().cpu().numpy()
                )
                plot_validation_samples(
                    val_pred_plot,
                    val_true_plot,
                    epoch + 1,
                    args.loss_type,
                    args.smooth_lddt_weight,
                    args.checkpoint_dir,
                )

            # --- Save Checkpoint ---
            if (epoch + 1) % args.save_every == 0:
                # ... [Saving checkpoint remains the same] ...
                ckpt_path = os.path.join(
                    args.checkpoint_dir, f"model_epoch_{epoch + 1:04d}.pth"
                )
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

    # Save final model
    # ... [Saving final model remains the same] ...
    final_model_path = os.path.join(args.checkpoint_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"--- Training finished. Final model saved to {final_model_path} ---")


if __name__ == "__main__":
    # --- Argument Parser ---
    # Arguments remain the same as the previous version
    parser = argparse.ArgumentParser(
        description="GeoDiffusion Toy Experiment (EDM Parameterized)"
    )  # Updated description
    parser.add_argument(
        "--loss_type",
        type=str,
        choices=["edm", "af3"],
        required=True,
        help="Type of MSE loss weighting ('edm' or 'af3')",
    )
    parser.add_argument(
        "--smooth_lddt_weight",
        type=float,
        default=0.0,
        help="Weight for the smooth LDDT loss component (0.0 to disable)",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="zigzag_dataset.npz",
        help="Path to the .npz dataset file",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory to save the output CSV log files",
    )
    parser.add_argument(
        "--log_file_name",
        type=str,
        required=True,
        help="Base name for the output CSV log file (e.g., edm_no_lddt_log.csv)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory to save model checkpoints and validation plots",
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Training batch size"
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument(
        "--sigma_data",
        type=float,
        default=1.0,
        help="Data variance parameter (sigma_data for model and loss)",
    )
    parser.add_argument(
        "--lddt_cutoff",
        type=float,
        default=2.0,
        help="Cutoff radius (Angstroms) for smooth LDDT loss neighborhood.",
    )
    parser.add_argument(
        "--plot_every",
        type=int,
        default=10,
        help="Save a plot of validation samples every N epochs",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=50,
        help="Save a model checkpoint every N epochs",
    )
    args = parser.parse_args()
    main(args)
