import torch
import torch.nn as nn
import math


# --- EDM Preconditioning Scalings ---
def edm_c_in(sigma, sigma_data=1.0):
    return 1.0 / (sigma**2 + sigma_data**2).sqrt()


def edm_c_skip(sigma, sigma_data=1.0):
    return sigma_data**2 / (sigma**2 + sigma_data**2)


def edm_c_out(sigma, sigma_data=1.0):
    return sigma * sigma_data / (sigma**2 + sigma_data**2).sqrt()


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim] or [B, L, D] (if batch_first)
        """
        # (B, L, D) -> (L, B, D) for PE
        x = x.permute(1, 0, 2)
        x = x + self.pe[: x.size(0)]
        # (L, B, D) -> (B, L, D) back
        x = self.dropout(x.permute(1, 0, 2))
        return x


class GeoDiffusionTransformer(nn.Module):
    """
    Transformer denoiser implementing EDM preconditioning AND
    conditional input for angles.
    """

    def __init__(self, embed_dim=128, nhead=8, num_layers=6, sigma_data=1.0):
        super().__init__()
        self.sigma_data = sigma_data
        self.embed_dim = embed_dim

        # 1. Atom Coordinate Embedding
        self.coord_embed = nn.Linear(2, embed_dim)

        # 2. Sigma (Noise Level) Embedding
        self.sigma_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # 3. Angle (Condition) Embedding
        self.angle_embed = nn.Sequential(
            nn.Linear(1, embed_dim),  # Input: 1D normalized angle
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # 4. Positional Encoding
        self.pos_encoder = PositionalEncoding(embed_dim)

        # 5. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # 6. Output projection
        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, 2),  # Predicts the (x,y) residual
        )

        print(
            f"Initialized Conditional GeoDiffusionTransformer (dim={embed_dim}, layers={num_layers})"
        )

    def forward(self, x_noisy, sigma, angle_cond, src_key_padding_mask):
        """
        Args:
            x_noisy (torch.Tensor): Padded noisy point clouds (B, Max_Atoms, 2)
            sigma (torch.Tensor): Noise levels (B, 1)
            angle_cond (torch.Tensor): Angle conditions (B, 1)
            src_key_padding_mask (torch.Tensor): Mask for padded atoms (B, Max_Atoms)

        Returns:
            torch.Tensor: Predicted clean point cloud D_theta (B, Max_Atoms, 2)
        """
        if sigma.ndim == 1:
            sigma = sigma.unsqueeze(-1)

        # 1. Preconditioning
        c_in = edm_c_in(sigma, self.sigma_data).unsqueeze(-1)
        c_skip = edm_c_skip(sigma, self.sigma_data).unsqueeze(-1)
        c_out = edm_c_out(sigma, self.sigma_data).unsqueeze(-1)

        # 2. Embeddings
        coord_emb = self.coord_embed(c_in * x_noisy)
        sigma_emb = (
            self.sigma_embed(sigma).unsqueeze(1).expand(-1, x_noisy.shape[1], -1)
        )

        # --- Process Angle Condition ---
        # Normalize angle (e.g., 0-180 degrees -> 0-1)
        angle_norm = angle_cond / 180.0
        angle_emb = self.angle_embed(angle_norm)
        angle_emb = angle_emb.unsqueeze(1).expand(-1, x_noisy.shape[1], -1)
        # --- End ---

        # 3. Prepare input for transformer
        # (B, L, D)
        transformer_input = coord_emb + sigma_emb + angle_emb
        transformer_input = self.pos_encoder(transformer_input)

        # 4. Transformer forward pass
        f_theta_output = self.transformer_encoder(
            transformer_input, src_key_padding_mask=src_key_padding_mask
        )

        # 5. Output projection (Predicts F_theta)
        residual = self.output_layer(f_theta_output)

        # 6. Final Denoised Output (D_theta)
        denoised_output = c_skip * x_noisy + c_out * residual

        return denoised_output
