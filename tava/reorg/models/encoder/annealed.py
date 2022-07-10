""" Annealed Sinusoidal Encoder` in Nefies.

Reference:
    https://github.com/google/nerfies/blob/main/nerfies/modules.py#L231
"""
import math
from typing import Dict, Tuple

import torch
from tava.reorg.models.encoder.base import (
    IntegratedSinusoidalEncoder,
    SinusoidalEncoder,
)
from tava.reorg.utils.schedules import Schedule


def cosine_easing_window(min_deg, max_deg, alpha):
    assert alpha >= min_deg and alpha <= max_deg
    bands = torch.linspace(min_deg, max_deg - 1, max_deg - min_deg)
    x = torch.clamp(alpha - bands, 0.0, 1.0)
    return 0.5 * (1 + torch.cos(math.pi * x + math.pi))
    

class AnnealedSinusoidalEncoder(SinusoidalEncoder):

    def __init__(
        self, 
        x_dim: int, 
        min_deg: int, 
        max_deg: int, 
        use_identity: bool = True,
        alpha_sched: Schedule = None,
    ):
        super().__init__(x_dim, min_deg, max_deg, use_identity)
        self.alpha_sched = alpha_sched

    def forward(self, x: torch.Tensor, meta: Dict) -> Dict:
        alpha = meta.get("alpha", None)
        if alpha is None:
            step = meta.get("step", self.max_deg)
            alpha = self.alpha_sched(step)
        latent = super().forward(x)["latent"]
        identity, sinusoidal = torch.split(
            latent, 
            (
                int(self.use_identity) * self.x_dim, 
                self.scales.shape[-1] * self.x_dim * 2,
            ), 
            dim=-1
        )
        sinusoidal = sinusoidal.reshape(
            list(sinusoidal.shape[:-1]) + 
            [self.scales.shape[-1], self.x_dim, 2]
        )
        window = cosine_easing_window(
            self.min_deg, self.max_deg, alpha
        ).reshape((self.scales.shape[-1], 1, 1)).to(sinusoidal)
        sinusoidal = window * sinusoidal
        latent = torch.cat([
            identity, sinusoidal.reshape(list(x.shape[:-1]) + [-1])
        ], dim=-1)
        return {"latent": latent}


class AnnealedIntegratedSinusoidalEncoder(IntegratedSinusoidalEncoder):
    
    def __init__(
        self, 
        x_dim: int, 
        min_deg: int, 
        max_deg: int, 
        diag: bool = True,
        alpha_sched: Schedule = None,
    ):
        super().__init__(x_dim, min_deg, max_deg, diag)
        self.alpha_sched = alpha_sched

    def forward(
        self, x: Tuple[torch.Tensor, torch.Tensor], meta: Dict
    ) -> Dict:
        alpha = meta.get("alpha", None)
        if alpha is None:
            step = meta.get("step", self.max_deg)
            alpha = self.alpha_sched(step)
        sinusoidal = super().forward(x)["latent"]
        sinusoidal = sinusoidal.reshape(
            list(sinusoidal.shape[:-1]) + 
            [self.scales.shape[-1], self.x_dim, 2]
        )
        window = cosine_easing_window(
            self.min_deg, self.max_deg, alpha
        ).reshape((self.scales.shape[-1], 1, 1)).to(sinusoidal)
        latent = torch.reshape(
            window * sinusoidal, 
            list(sinusoidal.shape[:-3]) + [self.latent_dim]
        )
        return {"latent": latent}
