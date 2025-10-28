import torch
import torch.nn as nn


# Helper functions for EDM preconditioning scalings
def edm_c_in(sigma, sigma_data=1.0):
    return 1.0 / (sigma**2 + sigma_data**2).sqrt()


def edm_c_skip(sigma, sigma_data=1.0):
    return sigma_data**2 / (sigma**2 + sigma_data**2)


def edm_c_out(sigma, sigma_data=1.0):
    return sigma * sigma_data / (sigma**2 + sigma_data**2).sqrt()


class EDMPrecondSimpleDenoiser(nn.Module):
    """
    A simple point-wise MLP denoiser implementing the EDM preconditioning.
    The internal network F_theta predicts the scaled residual component.
    The forward pass computes D_theta(x, sigma).
    """

    def __init__(self, n_atoms_in, n_atoms_out, embed_dim=128, sigma_data=1.0):
        super().__init__()

        self.n_atoms_in = n_atoms_in
        self.n_atoms_out = n_atoms_out
        self.sigma_data = sigma_data

        # A simple network to embed the noise level (sigma)
        # Input to F_theta is sigma directly (not log-sigma or other embeddings initially)
        self.sigma_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # The main MLP (F_theta) that processes each atom independently
        self.mlp = nn.Sequential(
            # Input is (scaled x, scaled y) + sigma_embed_dim
            nn.Linear(2 + embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            # Output is the F_theta component (scaled residual), (x, y)
            nn.Linear(embed_dim, 2),
        )

    def forward(self, x_noisy, sigma):
        """
        Implements the EDM forward pass: D_theta(x, sigma).

        Args:
            x_noisy (torch.Tensor): Noisy point cloud, shape (B, N_Atoms, 2)
            sigma (torch.Tensor): Noise levels, shape (B, 1) or broadcastable.

        Returns:
            torch.Tensor: Predicted clean point cloud D_theta, shape (B, N_Atoms, 2)
        """
        # Ensure sigma has the right shape (B, 1, 1) for broadcasting with (B, N, 2)
        if sigma.ndim == 1:
            sigma = sigma.unsqueeze(-1)  # (B, 1)
        sigma = sigma.unsqueeze(-1)  # (B, 1, 1)

        # Calculate preconditioning constants
        c_in_val = edm_c_in(sigma, self.sigma_data)
        c_skip_val = edm_c_skip(sigma, self.sigma_data)
        c_out_val = edm_c_out(sigma, self.sigma_data)

        # Prepare input for F_theta
        model_input = c_in_val * x_noisy

        # Embed sigma - unsqueeze sigma to (B, 1) for linear layer
        # Note: EDM paper conditions F_theta also on sigma directly.
        sigma_emb = self.sigma_embed(
            sigma.squeeze(-1)
        )  # Input (B, 1) -> Output (B, embed_dim)

        # Expand sigma embedding to match atom dimension: (B, N_Atoms, embed_dim)
        sigma_emb = sigma_emb.unsqueeze(1).expand(-1, model_input.shape[1], -1)

        # Concatenate scaled input with sigma embedding
        # (B, N_Atoms, 2) + (B, N_Atoms, embed_dim) -> (B, N_Atoms, 2 + embed_dim)
        f_theta_input = torch.cat([model_input, sigma_emb], dim=-1)

        # Pass through the MLP (F_theta)
        # (B, N_Atoms, 2 + embed_dim) -> (B, N_Atoms, 2)
        f_theta_output = self.mlp(f_theta_input)

        # Compute the final denoised output D_theta
        # D_theta = c_skip * x_noisy + c_out * F_theta(c_in * x_noisy, sigma)
        denoised_output = c_skip_val * x_noisy + c_out_val * f_theta_output

        return denoised_output
