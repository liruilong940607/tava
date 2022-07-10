from typing import Any, Dict, Tuple

# pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
import tinycudann as tcnn
import torch
from tava.reorg.models.encoder.abstract import AbstractEncoder


class TCNNHashPositionalEncoder(AbstractEncoder):
    """ Hash Positinal Encoder from Instant-NGP.
    
    https://github.com/NVlabs/instant-ngp
    """
    def __init__(
        self,
        bounding_box: Tuple[float, float, float, float, float, float], 
        x_dim: int = 3,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        per_level_scale: float = 2.0,
    ):
        super().__init__()
        # [min_x, min_y, min_z, max_x, max_y, max_z]
        self.register_buffer("bounding_box", torch.tensor(bounding_box))
        self.x_dim = x_dim
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        # The input to the tcnn.Encoding should be normalized
        # to (0, 1) using `self.bounding_box`
        self.encoder = tcnn.Encoding(
            n_input_dims=x_dim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": n_features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
            },
        )

    @property
    def latent_dim(self):
        return self.n_levels * self.n_features_per_level

    def forward(self, x: torch.Tensor, meta: Dict) -> Dict:
        """
        :params x: [..., 3],
        :return x_enc: [..., self.latent_dim]
        """
        bb_min, bb_max = torch.split(
            self.bounding_box, [self.x_dim, self.x_dim], dim=0
        )
        x = (x - bb_min) / (bb_max - bb_min)
        x = self.encoder(
            x.reshape(-1, x.shape[-1]).half()
        ).to(x).reshape(list(x.shape[:-1]) + [self.latent_dim])
        return {"latent": x}

    def contains(self, x: torch.Tensor):
        bb_min, bb_max = torch.split(
            self.bounding_box, [self.x_dim, self.x_dim], dim=0
        )
        x = (x - bb_min) / (bb_max - bb_min)
        return ((x > 0) & (x < 1)).all(dim=-1)


class TCNNSHViewEncoder(AbstractEncoder):
    """ SH Viewdir Encoder from Instant-NGP.
    
    https://github.com/NVlabs/instant-ngp
    """
    def __init__(self, x_dim: int = 3, degree: int = 4):
        super().__init__()
        self.x_dim = x_dim
        self.degree = degree
        # SH encoding requires inputs to be in [0, 1]
        self.encoder = tcnn.Encoding(
            n_input_dims=x_dim,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": degree,
            },
        )

    @property
    def latent_dim(self):
        return self.degree ** 2

    def forward(self, x: torch.Tensor, meta: Dict) -> Dict:
        """
        :params x: dirs [..., 3],
        :return latent: [..., self.latent_dim]
        """
        # SH encoding requires inputs to be in [0, 1]
        x = (x + 1) / 2
        x = self.encoder(x)
        return {"latent": x}

