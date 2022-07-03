# Copyright (c) Meta Platforms, Inc. and affiliates.
""" Positional Encoding. """
import math
from typing import List

# pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
import tinycudann as tcnn
import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int = 3,
        min_deg: int = 0,
        max_deg: int = 10,
        append_identity: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.append_identity = append_identity

    @property
    def out_dim(self):
        return (
            self.in_dim if self.append_identity else 0
        ) + self.in_dim * 2 * (self.max_deg - self.min_deg)

    def forward(self, x: torch.Tensor):
        """
        :params x: [..., 3]
        :return x_enc: [..., self.out_dim]
        """
        scales = torch.tensor(
            [2**i for i in range(self.min_deg, self.max_deg)],
            dtype=x.dtype,
            device=x.device,
        )
        xb = torch.reshape(
            (x[Ellipsis, None, :] * scales[:, None]),
            list(x.shape[:-1]) + [scales.shape[0] * x.shape[-1]],
        )
        four_feat = torch.sin(torch.cat([xb, xb + 0.5 * math.pi], dim=-1))
        if self.append_identity:
            return torch.cat([x] + [four_feat], dim=-1)
        else:
            return four_feat


class IntegratedPositionalEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int = 3,
        min_deg: int = 0,
        max_deg: int = 10,
        diag: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.diag = diag

    @property
    def out_dim(self):
        return self.in_dim * 2 * (self.max_deg - self.min_deg)

    def forward(self, x_coord: torch.Tensor):
        """
        :params x_coord: ([..., 3], [..., 3] or [..., 3, 3])
        :return x_enc: [..., self.out_dim]
        """
        if self.diag:
            x, x_cov_diag = x_coord
            scales = torch.tensor(
                [2**i for i in range(self.min_deg, self.max_deg)],
                device=x.device,
            )
            shape = list(x.shape[:-1]) + [x.shape[-1] * scales.shape[0]]
            y = torch.reshape(x[..., None, :] * scales[:, None], shape)
            y_var = torch.reshape(
                x_cov_diag[..., None, :] * scales[:, None] ** 2, shape
            )
        else:
            x, x_cov = x_coord
            num_dims = x.shape[-1]
            basis = torch.cat(
                [
                    2**i * torch.eye(num_dims, device=x.device)
                    for i in range(self.min_deg, self.max_deg)
                ],
                1,
            )
            y = torch.matmul(x, basis)
            # Get the diagonal of a covariance matrix (ie, variance).
            # This is equivalent to jax.vmap(torch.diag)((basis.T @ covs) @ basis).
            y_var = torch.sum((torch.matmul(x_cov, basis)) * basis, -2)
        return self._expected_sin(
            torch.cat([y, y + 0.5 * math.pi], dim=-1),
            torch.cat([y_var] * 2, dim=-1),
        )[0]

    def _expected_sin(self, x, x_var):
        """Estimates mean and variance of sin(z), z ~ N(x, var)."""
        # When the variance is wide, shrink sin towards zero.
        y = torch.exp(-0.5 * x_var) * torch.sin(x)
        y_var = torch.clip(
            0.5 * (1 - torch.exp(-2 * x_var) * torch.cos(2 * x)) - y**2, min=0
        )
        return y, y_var


class WindowedPositionalEncoder(PositionalEncoder):
    """ `AnnealedSinusoidalEncoder` in Nefies:
    https://github.com/google/nerfies/blob/main/nerfies/modules.py#L231
    """
    
    def forward(self, x: torch.Tensor, alpha: float):
        """
        :params x: [..., 3]
        :params alpha: float
        :return x_enc: [..., self.out_dim]
        """
        features = super().forward(x)
        if self.append_identity:
            identity, features = torch.split(
                features, 
                (self.in_dim, self.in_dim * 2 * (self.max_deg - self.min_deg)), 
                dim=-1
            )
        features = features.reshape(
            list(x.shape[:-1]) + [self.max_deg - self.min_deg, self.in_dim, 2]
        )
        window = self.cosine_easing_window(alpha).reshape(
            (self.max_deg - self.min_deg, 1, 1)
        ).to(features)
        features = window * features
        if self.append_identity:
            return torch.cat([
                identity, features.reshape(list(x.shape[:-1]) + [-1])
            ], dim=-1)
        else:
            return features
        
    def cosine_easing_window(self, alpha):
        bands = torch.linspace(0, self.max_deg - 1, self.max_deg)
        x = torch.clamp(alpha - bands, 0.0, 1.0)
        return 0.5 * (1 + torch.cos(math.pi * x + math.pi))


class TCNNHashPositionalEncoder(nn.Module):
    """ Hash Positinal Encoder from Instant-NGP.
    
    https://github.com/NVlabs/instant-ngp
    """
    def __init__(
        self,
        bounding_box: List[float], 
        in_dim: int = 3,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        per_level_scale: float = 2.0,
    ):
        super().__init__()
        # [min_x, min_y, min_z, max_x, max_y, max_z]
        self.bounding_box = torch.tensor(bounding_box)
        self.in_dim = in_dim
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        # The input to the tcnn.Encoding should be normalized
        # to (0, 1) using `self.bounding_box`
        self.encoder = tcnn.Encoding(
            n_input_dims=in_dim,
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
    def out_dim(self):
        return self.n_levels * self.n_features_per_level

    def forward(self, x: torch.Tensor):
        """
        :params x: [..., 3],
        :return x_enc: [..., self.out_dim]
        """
        bb_min, bb_max = torch.split(
            self.bounding_box.to(x), [3, 3], dim=0
        )
        x = (x - bb_min) / (bb_max - bb_min)
        mask = ((x > 0) & (x < 1)).all(dim=-1)
        x = self.encoder(
            x.reshape(-1, x.shape[-1]).half()
        ).to(x).reshape(list(x.shape[:-1]) + [self.out_dim])
        return x, mask
        

class TCNNSHViewEncoder(nn.Module):
    """ SH Viewdir Encoder from Instant-NGP.
    
    https://github.com/NVlabs/instant-ngp
    """
    def __init__(
        self,
        in_dim: int = 3,
        degree: int = 4,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.degree = degree
        # SH encoding requires inputs to be in [0, 1]
        self.encoder = tcnn.Encoding(
            n_input_dims=in_dim,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": degree,
            },
        )

    @property
    def out_dim(self):
        return self.degree ** 2

    def forward(self, dir: torch.Tensor):
        """
        :params dir: [..., 3],
        :return dir_enc: [..., self.out_dim]
        """
        # SH encoding requires inputs to be in [0, 1]
        dir = (dir + 1) / 2
        dir = self.encoder(dir)
        return dir


if __name__ == "__main__":
    import time
    encoder = TCNNHashPositionalEncoder([0, 0, 0, 1, 1, 1]).to("cuda")
    
    # with torch.no_grad():
    for _ in range(100):
        x = torch.rand([100000, 3]).to("cuda")
        x.requires_grad_ = True
        out = encoder(x)
        out[0].sum().backward()
        time.sleep(0.1)
        print (out[0].shape)
