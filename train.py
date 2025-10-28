import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import argparse
import os
import csv
import matplotlib.pyplot as plt

# Import the UPDATED Transformer model
from model import GeoDiffusionTransformer


# --- Loss Functions & Metrics ---
def l2_loss(pred_clean, true_clean, mask):
    """L2 loss, masked for padded sequences."""
    loss = (pred_clean - true_clean).pow(2)
    # mask is (B, N_max), we need (B, N_max, 2)
    masked_loss = loss * mask.unsqueeze(-1)
    # Sum up all valid losses and divide by the number of valid atoms
    return masked_loss.sum() / mask.sum()


def smooth_lddt_loss(pred_clean, true_clean, mask, cutoff_radius=2.0, eps=1e-9):
    """
    Masked Smooth LDDT loss for variable-length batch.
    Args:
        mask (torch.Tensor): Boolean mask, True for valid atoms (B, N_max)
    """
    B, N_Atoms_Max, _ = pred_clean.shape
    device = pred_clean.device

    dist_pred = torch.cdist(pred_clean, pred_clean)
    dist_true = torch.cdist(true_clean, true_clean)
    delta = torch.abs(dist_true - dist_pred)

    valid_mask = mask.unsqueeze(2) * mask.unsqueeze(1)
    valid_mask = valid_mask * (1.0 - torch.eye(N_Atoms_Max, device=device)).unsqueeze(0)
    cutoff_mask = (dist_true < cutoff_radius).float() * valid_mask

    mask_sum = cutoff_mask.sum()
    if mask_sum < eps:
        return torch.tensor(0.0, device=device)

    # AF3-style smooth sigmoids
    s1 = torch.sigmoid(delta - 0.5)
    s2 = torch.sigmoid(delta - 1.0)
    s3 = torch.sigmoid(delta - 2.0)
    s4 = torch.sigmoid(delta - 4.0)
    score = 0.25 * (s1 + s2 + s3 + s4)

    lddt_error_score = (score * cutoff_mask).sum() / mask_sum
    return lddt_error_score


# --- Weighting Functions ---
def get_edm_weight(sigma, sigma_data):
    """EDM-style loss weighting."""
    sigma = sigma.squeeze()
    sigma_data = torch.tensor(sigma_data, device=sigma.device)
    num = sigma.pow(2) + sigma_data.pow(2)
    den = (sigma * sigma_data).pow(2) + 1e-9
    return (num / den).view(-1, 1, 1)  # (B, 1, 1)


def get_af3_weight(sigma, sigma_data):
    """AlphaFold 3-style loss weighting."""
    sigma = sigma.squeeze()
    sigma_data = torch.tensor(sigma_data, device=sigma.device)
    num = sigma.pow(2) + sigma_data.pow(2)
    den = (sigma + sigma_data).pow(2) + 1e-9
    return (num / den).view(-1, 1, 1)  # (B, 1, 1)


# --- Updated Metrics to handle batches ---
def global_mse_metric(pred_clean, true_clean, n_points_batch, mask):
    """Global MSE on vertices for a batch."""
    total_mse = 0
    num_samples = pred_clean.shape[0]
    for i in range(num_samples):
        n_p = n_points_batch[i].item()
        n_a = n_p * 3
        # Get vertices (atom 1, 4, 7, ...)
        pred_vertices = pred_clean[i, 1:n_a:3, :]
        true_vertices = true_clean[i, 1:n_a:3, :]
        total_mse += (pred_vertices - true_vertices).pow(2).mean()
    return total_mse / num_samples


def local_angle_metric(pred_clean, n_points_batch, true_angle_batch):
    """Local angle MAE for a batch."""
    total_mae = 0
    num_samples = pred_clean.shape[0]
    for i in range(num_samples):
        n_p = n_points_batch[i].item()
        n_a = n_p * 3
        true_angle = true_angle_batch[i]

        pred_struct = pred_clean[i, :n_a, :].view(n_p, 3, 2)
        atom0 = pred_struct[:, 0, :]
        atom1 = pred_struct[:, 1, :]
        atom2 = pred_struct[:, 2, :]

        vec_a, vec_b = atom0 - atom1, atom2 - atom1
        dot = torch.einsum("...i,...i->...", vec_a, vec_b)
        norm_a = torch.norm(vec_a, dim=-1)
        norm_b = torch.norm(vec_b, dim=-1)

        cos_theta = dot / ((norm_a * norm_b) + 1e-9)
        pred_angles = torch.rad2deg(torch.acos(torch.clip(cos_theta, -1.0, 1.0)))
        total_mae += torch.abs(pred_angles - true_angle).mean()

    return total_mae / num_samples


# --- Plotting Functions (Inlined) ---
def calculate_mean_angle(structure_tensor):
    """Calculates the mean internal angle for a single (N_p, 3, 2) structure tensor."""
    with torch.no_grad():
        atom0 = structure_tensor[:, 0, :]
        atom1 = structure_tensor[:, 1, :]
        atom2 = structure_tensor[:, 2, :]
        vec_a = atom0 - atom1
        vec_b = atom2 - atom1
        dot_product = torch.einsum("...i,...i->...", vec_a, vec_b)
        norm_a = torch.norm(vec_a, dim=-1)
        norm_b = torch.norm(vec_b, dim=-1)
        cos_theta = dot_product / ((norm_a * norm_b) + 1e-9)
        cos_theta = torch.clip(cos_theta, -1.0, 1.0)
        pred_angles_rad = torch.acos(cos_theta)
        pred_angles_deg = torch.rad2deg(pred_angles_rad)
        return pred_angles_deg.mean().item()


def plot_validation_samples(
    pred_structures_batch,  # This is the *final* generated structure
    true_structures_batch,  # This is for comparison
    n_points_batch,
    true_angle_batch,
    epoch,
    loss_type,
    smooth_lddt_weight,
    save_dir,
    num_to_plot=4,
):
    """
    Plots validation samples from a padded batch, handling variable lengths.
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    num_samples = min(pred_structures_batch.shape[0], num_to_plot)

    for i in range(num_samples):
        ax = axes[i]

        n_p = n_points_batch[i].item()
        n_atoms = n_p * 3
        true_angle = true_angle_batch[i].item()

        # --- Slice the unpadded data ---
        true_sample_flat = true_structures_batch[i, :n_atoms, :].cpu()
        pred_sample_flat = pred_structures_batch[i, :n_atoms, :].cpu()
        true_sample = true_sample_flat.view(n_p, 3, 2)
        pred_sample = pred_sample_flat.view(n_p, 3, 2)

        pred_angle = calculate_mean_angle(pred_sample)

        # --- Plotting logic (using numpy for plotting) ---
        true_sample_np = true_sample.numpy()
        pred_sample_np = pred_sample.numpy()

        true_atoms_all = true_sample_np.reshape(-1, 2)
        true_vertices = true_sample_np[:, 1, :]
        pred_atoms_all = pred_sample_np.reshape(-1, 2)
        pred_vertices = pred_sample_np[:, 1, :]

        # Plot Ground Truth
        ax.scatter(
            true_atoms_all[:, 0],
            true_atoms_all[:, 1],
            s=10,
            c="blue",
            alpha=0.3,
            label="GT Atoms",
        )
        ax.plot(
            true_vertices[:, 0],
            true_vertices[:, 1],
            "o-",
            c="blue",
            alpha=0.6,
            markersize=4,
            label="GT Vertices",
        )

        # Plot Prediction
        ax.scatter(
            pred_atoms_all[:, 0],
            pred_atoms_all[:, 1],
            s=10,
            c="red",
            alpha=0.3,
            marker="x",
            label="Pred Atoms",
        )
        ax.plot(
            pred_vertices[:, 0],
            pred_vertices[:, 1],
            "x-",
            c="red",
            alpha=0.6,
            markersize=4,
            label="Pred Vertices",
        )

        # Plot connecting arms for detail
        for j in range(n_p):  # Plot all arms
            true_p0, true_p1, true_p2 = true_sample_np[j]
            pred_p0, pred_p1, pred_p2 = pred_sample_np[j]

            label_gt = "GT Arms" if j == 0 else None
            ax.plot(
                [true_p0[0], true_p1[0]],
                [true_p0[1], true_p1[1]],
                c="blue",
                alpha=0.4,
                label=label_gt,
            )
            ax.plot(
                [true_p1[0], true_p2[0]], [true_p1[1], true_p2[1]], c="blue", alpha=0.4
            )

            label_pred = "Pred Arms" if j == 0 else None
            ax.plot(
                [pred_p0[0], pred_p1[0]],
                [pred_p0[1], pred_p1[1]],
                c="red",
                alpha=0.4,
                linestyle="--",
                label=label_pred,
            )
            ax.plot(
                [pred_p1[0], pred_p2[0]],
                [pred_p1[1], pred_p2[1]],
                c="red",
                alpha=0.4,
                linestyle="--",
            )

        ax.set_title(
            f"Sample {i + 1} (N={n_p}) | Angle (True/Pred): {true_angle:.1f}° / {pred_angle:.1f}°"
        )
        ax.legend()
        ax.axis("equal")

    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis("off")

    lddt_status = "ON" if smooth_lddt_weight > 0 else "OFF"
    plt.suptitle(
        f"Validation Samples (Generated from Noise) - Epoch {epoch}\n"  # Updated title
        f"Loss Weight: {loss_type.upper()}, SmoothLDDT: {lddt_status} (w={smooth_lddt_weight})",
        fontsize=16,
    )
    plt.tight_layout(rect=(0, 0.03, 1, 0.93))
    output_image = os.path.join(save_dir, f"validation_plot_epoch_{epoch:04d}.png")
    plt.savefig(output_image)
    plt.close(fig)


# --- Data Handling ---
class VariableZigZagDataset(Dataset):
    """Dataset for variable-length zigzag polymers."""

    def __init__(self, npz_file):
        try:
            data = np.load(npz_file, allow_pickle=True)
        except FileNotFoundError:
            print(f"Error: Dataset file not found at {npz_file}")
            print("Please run generate_dataset.py first.")
            exit(1)

        self.structures = data["structures"]  # List of (N_atoms, 2) arrays
        self.true_angles = torch.from_numpy(data["true_angle_degs"]).float()
        self.n_points = data["n_points"]  # Array of ints

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        structure = torch.from_numpy(self.structures[idx]).float()
        return {
            "structure": structure,
            "n_points": self.n_points[idx],
            "true_angle": self.true_angles[idx],
        }


def collate_fn(batch):
    """Pads sequences for batching."""
    structures = [item["structure"] for item in batch]
    n_points = torch.tensor([item["n_points"] for item in batch], dtype=torch.int)
    true_angles = torch.tensor(
        [item["true_angle"] for item in batch], dtype=torch.float
    )

    # (B, N_max, 2)
    padded_structures = pad_sequence(structures, batch_first=True, padding_value=0.0)

    # Create padding mask (True for padded values, False for real atoms)
    max_len = padded_structures.shape[1]
    # (B, N_max)
    mask = torch.arange(max_len)[None, :] >= (n_points * 3)[:, None]

    return {
        "structures": padded_structures,
        "n_points": n_points,
        "true_angles": true_angles,
        "pad_mask": mask,  # True for padding
    }


# --- Main Training Script ---
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    dataset = VariableZigZagDataset(args.data_file)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )

    # Create a fixed validation batch
    val_dataloader = DataLoader(
        dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
    )
    val_batch = next(iter(val_dataloader))
    for key in val_batch:
        val_batch[key] = val_batch[key].to(device)

    model = GeoDiffusionTransformer(
        embed_dim=args.embed_dim,
        nhead=args.nhead,
        num_layers=args.num_layers,
        sigma_data=args.sigma_data,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --- Define EDM Sampling Schedule (for validation) ---
    num_steps = args.sample_steps
    sigma_max = 160.0
    sigma_min = 4e-4
    rho = 7.0
    sigmas = (
        sigma_max ** (1 / rho)
        + (torch.arange(num_steps, device=device) / (num_steps - 1))
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    # --- End Sampler Def ---

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

        print("--- Training Conditional Transformer Model ---")
        print(f"Loss Weighting: {args.loss_type}")
        print(f"Smooth LDDT Weight: {args.smooth_lddt_weight}")
        print(f"Sigma Data: {args.sigma_data}")
        print(f"Dataset: {args.data_file}")
        print(f"Checkpoints: {args.checkpoint_dir}")
        print(f"Log File: {log_file_path}")
        print("-" * 20)

        for epoch in range(args.epochs):
            model.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                true_clean = batch["structures"].to(device)
                pad_mask = batch["pad_mask"].to(device)  # True for padding
                loss_mask = ~pad_mask  # True for valid atoms
                angle_cond = batch["true_angles"].to(device).unsqueeze(-1)  # (B, 1)
                B = true_clean.shape[0]

                # 1. Sample noise level (sigma)
                log_sigma = torch.randn(B, 1, device=device) * 1.2 - 1.0
                sigma = torch.exp(log_sigma)

                # 2. Create noisy input (noise applied everywhere)
                noise = torch.randn_like(true_clean)
                x_noisy = true_clean + noise * sigma.unsqueeze(-1)
                # Mask noisy input to 0 where padding is
                x_noisy = x_noisy * loss_mask.unsqueeze(-1)

                # 3. Get prediction (Pass angle condition)
                pred_denoised = model(x_noisy, sigma, angle_cond, pad_mask)

                # 4. Calculate weighted MSE loss
                raw_loss_mse = (pred_denoised - true_clean).pow(2)

                if args.loss_type == "af3":
                    weight = get_af3_weight(sigma, args.sigma_data)
                else:  # 'edm'
                    weight = get_edm_weight(sigma, args.sigma_data)

                weighted_mse_loss = (
                    weight * raw_loss_mse * loss_mask.unsqueeze(-1)
                ).sum() / loss_mask.sum()

                # 5. Calculate Smooth LDDT loss (optional)
                loss_local = 0.0
                if args.smooth_lddt_weight > 0:
                    loss_local = smooth_lddt_loss(
                        pred_denoised,
                        true_clean,
                        loss_mask,
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
                # Get validation batch conditions
                val_true_clean = val_batch["structures"]
                val_n_points = val_batch["n_points"]
                val_true_angles = val_batch["true_angles"]
                val_pad_mask = val_batch["pad_mask"]
                val_loss_mask = ~val_pad_mask
                val_angle_cond = val_true_angles.unsqueeze(-1)
                B_val = val_true_clean.shape[0]

                # Start from pure noise x_T
                x_t = torch.randn_like(val_true_clean) * sigmas[0]

                for i in range(num_steps - 1):
                    sigma_t = sigmas[i]
                    sigma_next = sigmas[i + 1]

                    sigma_t_model = sigma_t.view(1, 1).expand(B_val, 1)  # (B, 1)

                    # Get Denoised prediction D(x_t, sigma_t, cond_angle)
                    D_t = model(x_t, sigma_t_model, val_angle_cond, val_pad_mask)

                    # Euler step
                    d_t = (x_t - D_t) / sigma_t
                    x_t = x_t + d_t * (sigma_next - sigma_t)

                    # Mask out padding area after each step
                    x_t = x_t * val_loss_mask.unsqueeze(-1)

                val_pred = x_t  # This is our final x_0 prediction
                # --- End Generation ---

                # Calculate validation metrics using the *generated* sample
                val_global_mse = global_mse_metric(
                    val_pred, val_true_clean, val_n_points, val_loss_mask
                ).item()
                val_local_mae = local_angle_metric(
                    val_pred, val_n_points, val_true_angles
                ).item()

            # --- Logging ---
            print(
                f"Epoch {epoch + 1}/{args.epochs} | Avg Loss: {avg_epoch_loss:.4f} | "
                f"Val Global MSE: {val_global_mse:.4f} | Val Local Angle MAE: {val_local_mae:.4f}"
            )
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

            # --- Plotting Call (uses the generated val_pred) ---
            if (epoch + 1) % args.plot_every == 0:
                print(f"Plotting validation samples for epoch {epoch + 1}...")
                plot_validation_samples(
                    pred_structures_batch=val_pred,
                    true_structures_batch=val_true_clean,
                    n_points_batch=val_n_points,
                    true_angle_batch=val_true_angles,
                    epoch=epoch + 1,
                    loss_type=args.loss_type,
                    smooth_lddt_weight=args.smooth_lddt_weight,
                    save_dir=args.checkpoint_dir,
                )

            # --- Save Checkpoint ---
            if (epoch + 1) % args.save_every == 0:
                ckpt_path = os.path.join(
                    args.checkpoint_dir, f"model_epoch_{epoch + 1:04d}.pth"
                )
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

    final_model_path = os.path.join(args.checkpoint_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"--- Training finished. Final model saved to {final_model_path} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GeoDiffusion Conditional Transformer Experiment"
    )

    # --- Experiment ---
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
        "--lddt_cutoff",
        type=float,
        default=15.0,
        help="Cutoff radius (Angstroms) for smooth LDDT loss neighborhood.",
    )

    # --- Data and Logging ---
    parser.add_argument(
        "--data_file",
        type=str,
        default="zigzag_variable_dataset.npz",
        help="Path to the .npz dataset file",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs_transformer",
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

    # --- Training ---
    parser.add_argument(
        "--epochs", type=int, default=300, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Training batch size"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--sigma_data",
        type=float,
        default=16.0,
        help="Data variance parameter (sigma_data for model and loss)",
    )

    # --- Model ---
    parser.add_argument(
        "--embed_dim", type=int, default=128, help="Transformer embedding dimension"
    )
    parser.add_argument(
        "--num_layers", type=int, default=6, help="Number of transformer layers"
    )
    parser.add_argument(
        "--nhead", type=int, default=8, help="Number of attention heads"
    )

    # --- Misc ---
    parser.add_argument(
        "--plot_every",
        type=int,
        default=25,
        help="Save a plot of validation samples every N epochs",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=50,
        help="Save a model checkpoint every N epochs",
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=200,
        help="Number of steps for Euler sampler in validation",
    )

    args = parser.parse_args()
    main(args)
