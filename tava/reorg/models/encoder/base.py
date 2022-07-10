import math
from typing import Any, Dict, Tuple

import torch
from tava.reorg.models.encoder.abstract import AbstractEncoder


class SinusoidalEncoder(AbstractEncoder):
    """ Sinusoidal Positional Encoder used in NeRF. """
    def __init__(
        self, x_dim, min_deg, max_deg, use_identity: bool = True
    ):
        super().__init__()
        self.x_dim = x_dim
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.use_identity = use_identity
        self.register_buffer(
            "scales", 
            torch.tensor([2**i for i in range(min_deg, max_deg)])
        )

    @property
    def latent_dim(self) -> int:
        return (
            int(self.use_identity) + self.scales.shape[-1] * 2
        ) * self.x_dim

    def forward(self, x: torch.Tensor, meta: Dict = None) -> Dict:
        """
        Args:
            x: [..., x_dim]
        Returns:
            latent: [..., latent_dim]
            warp: same as x
        """
        xb = torch.reshape(
            (x[Ellipsis, None, :] * self.scales[:, None]),
            list(x.shape[:-1]) + [self.scales.shape[-1] * self.x_dim],
        )
        latent = torch.sin(torch.cat([xb, xb + 0.5 * math.pi], dim=-1))
        if self.use_identity:
            latent = torch.cat([x] + [latent], dim=-1)
        return {"latent": latent}


class IntegratedSinusoidalEncoder(AbstractEncoder):
    """ Integrated Sinusoidal Positional Encoder used in Mip-NeRF. """
    
    def __init__(
        self, x_dim, min_deg, max_deg, diag: bool = True
    ):
        super().__init__()
        self.x_dim = x_dim
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.diag = diag
        if diag:
            scales = torch.tensor(
                [2**i for i in range(min_deg, max_deg)]
            )
        else:
            scales = torch.cat(
                [
                    2**i * torch.eye(x_dim) 
                    for i in range(min_deg, max_deg)
                ], 
                dim=-1
            )
        self.register_buffer("scales", scales)

    @property
    def latent_dim(self) -> int:
        return self.scales.shape[-1] * 2 * self.x_dim

    def forward(
        self, x: Tuple[torch.Tensor, torch.Tensor], meta: Dict = None
    ) -> Dict:
        """
        Args:
            x: if `self.diag` is True: ([..., x_dim], [..., x_dim])
                else ([..., x_dim], [..., x_dim, x_dim])
        Returns:
            latent: [..., latent_dim]
            warp: same as x
        """
        x, x_cov = x
        if self.diag:
            shape = list(x.shape[:-1]) + [self.scales.shape[-1] * self.x_dim]
            y = torch.reshape(x[..., None, :] * self.scales[:, None], shape)
            y_var = torch.reshape(
                x_cov[..., None, :] * self.scales[:, None] ** 2, shape
            )
        else:
            y = torch.matmul(x, self.scales)
            # Get the diagonal of a covariance matrix (ie, variance).
            # This is equivalent to jax.vmap(torch.diag)((basis.T @ covs) @ basis).
            y_var = torch.sum((torch.matmul(x_cov, self.scales)) * self.scales, -2)
        latent = self._expected_sin(
            torch.cat([y, y + 0.5 * math.pi], dim=-1),
            torch.cat([y_var] * 2, dim=-1),
        )[0]
        return {"latent": latent}

    def _expected_sin(self, x, x_var):
        """Estimates mean and variance of sin(z), z ~ N(x, var)."""
        # When the variance is wide, shrink sin towards zero.
        y = torch.exp(-0.5 * x_var) * torch.sin(x)
        y_var = torch.clip(
            0.5 * (1 - torch.exp(-2 * x_var) * torch.cos(2 * x)) - y**2, min=0
        )
        return y, y_var
