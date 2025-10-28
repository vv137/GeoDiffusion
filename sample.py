import numpy as np
import torch
import argparse
import os

# Import the new EDM parameterized model
from model import EDMPrecondSimpleDenoiser
from train import plot_validation_samples  # Reuse plotting function
from train import ZigZagDataset  # To get metadata


# (get_noise_schedule remains the same)
def get_noise_schedule(sigma_min, sigma_max, n_steps, rho=7.0, device="cpu"):
    """Generate Karras noise schedule."""
    step_indices = torch.arange(n_steps, device=device)
    sigma_max_pow_1_rho = sigma_max ** (1 / rho)
    sigma_min_pow_1_rho = sigma_min ** (1 / rho)
    sigmas = (
        sigma_max_pow_1_rho
        + step_indices / (n_steps - 1) * (sigma_min_pow_1_rho - sigma_max_pow_1_rho)
    ).pow(rho)
    sigmas = torch.cat([sigmas, torch.zeros_like(sigmas[:1])])  # Append sigma=0
    return sigmas


# (edm_sampler_step remains the same, as it expects model to predict x0=D_theta)
def edm_sampler_step(
    x_curr,
    sigma_curr,
    sigma_next,
    model,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    """
    Perform one step of the EDM sampler (Algorithm 2 from Karras et al. 2022).
    Assumes the model predicts x0 (clean data) = D_theta(x, sigma).
    """
    gamma = (
        min(s_churn / sigma_curr.numel(), np.sqrt(2) - 1)
        if s_tmin <= sigma_curr <= s_tmax
        else 0.0
    )
    sigma_hat = sigma_curr * (gamma + 1)
    if gamma > 0:
        eps = torch.randn_like(x_curr) * s_noise
        x_hat = x_curr + eps * torch.sqrt(sigma_hat**2 - sigma_curr**2)
    else:
        x_hat = x_curr

    # Euler step: x_next = x_hat + d_x * (sigma_next - sigma_hat)
    # where d_x = (x_hat - pred_x0) / sigma_hat
    with torch.no_grad():
        # Model takes noisy input and current sigma_hat, returns D_theta = pred_x0
        # Ensure sigma_hat is passed with correct shape (B, 1)
        pred_x0 = model(x_hat, sigma_hat.view(-1, 1))

    d_x = (x_hat - pred_x0) / sigma_hat  # Estimated direction (noise)
    x_next = x_hat + d_x * (sigma_next - sigma_hat)  # First order update

    # Apply 2nd order correction
    if sigma_next != 0:
        with torch.no_grad():
            # Model takes first-order prediction and next sigma, returns D_theta_next = pred_x0_next
            # Ensure sigma_next is passed with correct shape (B, 1)
            pred_x0_next = model(x_next, sigma_next.view(-1, 1))
        d_x_prime = (x_next - pred_x0_next) / sigma_next  # Corrected direction
        x_next = x_hat + (d_x + d_x_prime) / 2 * (
            sigma_next - sigma_hat
        )  # Second order update

    return x_next


def sample(args):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset metadata
    try:
        dataset_meta = ZigZagDataset(args.data_file)
        n_atoms = dataset_meta.n_atoms
        n_points = dataset_meta.n_points
        print(f"Loaded metadata: n_atoms={n_atoms}, n_points={n_points}")
    except FileNotFoundError:
        print(f"Error: Data file {args.data_file} not found.")
        return
    except Exception as e:
        print(f"Error loading data file {args.data_file}: {e}")
        return

    # Load model - USE THE NEW EDM PARAMETERIZED MODEL
    model = EDMPrecondSimpleDenoiser(
        n_atoms_in=n_atoms,
        n_atoms_out=n_atoms,
        sigma_data=args.sigma_data,  # Pass sigma_data used during training
    ).to(device)
    try:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.eval()
        print(f"Loaded model checkpoint from: {args.checkpoint}")
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # Generate noise schedule
    sigmas = get_noise_schedule(
        args.sigma_min, args.sigma_max, args.num_steps, rho=args.rho, device=device
    )

    # Initial noise (scaled by largest sigma)
    x_curr = torch.randn(args.num_samples, n_atoms, 2, device=device) * sigmas[0]

    # Sampling loop
    print(f"Starting sampling with {args.num_steps} steps...")
    for i in range(args.num_steps):
        sigma_curr = sigmas[i]
        sigma_next = sigmas[i + 1]
        # Sampler step expects scalar sigmas for internal calcs, model expects (B, 1)
        x_curr = edm_sampler_step(
            x_curr,
            sigma_curr.view(-1),  # Pass scalar sigma_curr
            sigma_next.view(-1),  # Pass scalar sigma_next
            model,  # The model itself handles the (B, 1) sigma input shape
        )
        if (i + 1) % (args.num_steps // 10) == 0:  # Print progress roughly 10 times
            print(
                f"  Step {i + 1}/{args.num_steps} (sigma={sigma_curr.item():.4f}) complete."
            )

    print("Sampling finished.")

    # Save samples
    # ... [Saving logic remains the same] ...
    generated_samples_np = x_curr.detach().cpu().numpy()
    output_npz = os.path.join(
        args.output_dir, f"generated_samples_{args.num_steps}steps.npz"
    )
    np.savez_compressed(output_npz, structures=generated_samples_np)
    print(f"Generated samples saved to {output_npz}")

    # Plot samples
    if args.plot:
        # ... [Plotting logic remains the same] ...
        samples_reshaped = generated_samples_np.reshape(
            args.num_samples, n_points, 3, 2
        )
        dummy_gt = (
            dataset_meta.structures[: args.num_samples]
            .view(args.num_samples, n_points, 3, 2)
            .numpy()
        )
        plot_validation_samples(
            samples_reshaped,
            dummy_gt,
            epoch=args.num_steps,
            loss_type="generated",
            smooth_lddt_weight=-1,
            save_dir=args.output_dir,
            num_to_plot=min(args.num_samples, 4),
        )
        plot_path = os.path.join(
            args.output_dir, f"validation_plot_epoch_{args.num_steps}.png"
        )
        final_plot_path = os.path.join(
            args.output_dir, f"generated_samples_plot_{args.num_steps}steps.png"
        )
        # Ensure the old plot exists before renaming, handle potential errors
        if os.path.exists(plot_path):
            os.rename(plot_path, final_plot_path)
            print(f"Plot of generated samples saved to {final_plot_path}")
        else:
            print(f"Warning: Expected plot file not found at {plot_path}")


if __name__ == "__main__":
    # Add sigma_data argument needed for model initialization
    parser = argparse.ArgumentParser(
        description="Generate samples using a trained GeoDiffusion model (EDM Parameterized)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="zigzag_dataset.npz",
        help="Path to the original .npz dataset file (for metadata)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="generated_samples",
        help="Directory to save generated samples (.npz) and plot",
    )
    parser.add_argument(
        "--num_samples", type=int, default=4, help="Number of samples to generate"
    )
    parser.add_argument(
        "--num_steps", type=int, default=100, help="Number of diffusion sampling steps"
    )
    parser.add_argument(
        "--sigma_min",
        type=float,
        default=0.002,
        help="Minimum noise level (sigma) in the schedule",
    )
    parser.add_argument(
        "--sigma_max",
        type=float,
        default=80.0,
        help="Maximum noise level (sigma) in the schedule",
    )
    parser.add_argument(
        "--rho", type=float, default=7.0, help="Exponent for the Karras noise schedule"
    )
    parser.add_argument(
        "--sigma_data",
        type=float,
        default=1.0,
        help="Data variance parameter used during training",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate a plot of the first few generated samples",
    )
    args = parser.parse_args()
    sample(args)
