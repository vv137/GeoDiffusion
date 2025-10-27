import torch
import torch.nn as nn


class SimpleDenoiser(nn.Module):
    """
    A simple point-wise MLP denoiser.

    It takes a noisy point cloud and the noise level (sigma) and
    predicts the *clean* point cloud.
    """

    def __init__(self, n_atoms_in, n_atoms_out, embed_dim=128):
        super().__init__()

        self.n_atoms_in = n_atoms_in
        self.n_atoms_out = n_atoms_out

        # A simple network to embed the noise level (sigma)
        self.sigma_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # The main MLP that processes each atom independently
        self.mlp = nn.Sequential(
            # Input is (x, y) + embed_dim
            nn.Linear(2 + embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            # Output is (x, y)
            nn.Linear(embed_dim, 2),
        )

    def forward(self, x_noisy, sigma):
        """
        Args:
            x_noisy (torch.Tensor): Noisy point cloud, shape (B, N_Atoms, 2)
            sigma (torch.Tensor): Noise levels, shape (B, 1)

        Returns:
            torch.Tensor: Predicted clean point cloud, shape (B, N_Atoms, 2)
        """
        # Embed sigma and broadcast it to match the number of atoms
        # sigma: (B, 1) -> (B, embed_dim)
        sigma_emb = self.sigma_embed(sigma)

        # sigma_emb: (B, embed_dim) -> (B, N_Atoms, embed_dim)
        sigma_emb = sigma_emb.unsqueeze(1).expand(-1, x_noisy.shape[1], -1)

        # Concatenate noisy coordinates with the sigma embedding
        # (B, N_Atoms, 2) + (B, N_Atoms, embed_dim) -> (B, N_Atoms, 2 + embed_dim)
        x_in = torch.cat([x_noisy, sigma_emb], dim=-1)

        # Pass each atom's features through the MLP
        # (B, N_Atoms, 2 + embed_dim) -> (B, N_Atoms, 2)
        x_pred_clean = self.mlp(x_in)

        return x_pred_clean
