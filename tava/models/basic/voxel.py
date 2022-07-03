""" The Voxel for NeRF / Mip-NeRF """
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class Voxel(nn.Module):
    """A Simple Voxel."""

    def __init__(
        self, 
        bbox: Tuple = [0, 0, 0, 1, 1, 1],
        res: Union[int, Tuple] = 32,  # The resolution of the voxel.
        input_dim: int = 3,  # The dim of the input coordinates. 
        output_dim: int = 4,  # The dim of the output features
        interp_mode: str = "bilinear",  # "bilinear" | "bicubic"
        padding_mode: str = "zeros",  # "zeros" | "border"
        align_corners: bool = True,
        init_value: float = 0.1,
    ):
        super().__init__()
        if isinstance(res, int):
            res = [res] * input_dim
        assert len(res) == input_dim
        assert len(bbox) == input_dim * 2
        self.register_buffer("bbox", torch.tensor(bbox))
        self.res = res
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.interp_mode = interp_mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.data = nn.Parameter(
            init_value * torch.ones([1, output_dim] + self.res)
        )

    def normalize(self, x):
        # As we can't garentee the input x is within the bbox 
        # during the entire optimization, we further apply sigmoid
        # after bounding-box normalization to make sure the 
        # output coordinate is definitely within (-1, 1).
        bbox_min, bbox_max = torch.split(self.bbox, [3, 3])
        x = (x - bbox_min) / (bbox_max - bbox_min)  # to soft (0, 1)
        # x = F.sigmoid((x - 0.5) * 8)  # to hard (0, 1)
        x = (x - 0.5) * 2  # to hard (-1, 1)
        return x
    
    def forward(self, x):
        assert x.shape[-1] == self.input_dim
        x = self.normalize(x)  # to (-1, 1)
        # print ((x.abs() < 1).all(dim=-1).float().mean())
        x = x.flip(dims=(-1,))  # the convention is k, j, i
        out = F.grid_sample(
            self.data,
            x.view([1] * self.input_dim + [-1, self.input_dim]),
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
            mode=self.interp_mode,
        ).transpose(1, -1)
        out = out.view(list(x.shape[:-1]) + [self.output_dim])
        return out 


class NeRFVoxel(nn.Module):

    def __init__(
        self,
        bbox: Tuple = [0, 0, 0, 1, 1, 1],
        res: Union[int, Tuple] = 32,  # The resolution of the voxel.
        input_dim: int = 3,  # The dim of the input coordinates. 
        interp_mode: str = "bilinear",  # "bilinear" | "bicubic"
        padding_mode: str = "zeros",  # "zeros" | "border"
        align_corners: bool = True,
    ) -> None:
        super().__init__()
        self.voxel_rgb = Voxel(
            bbox=bbox,
            res=res,
            input_dim=input_dim,
            output_dim=3,
            interp_mode=interp_mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
            init_value=0.1,
        )
        self.voxel_sigma = Voxel(
            bbox=bbox,
            res=res,
            input_dim=input_dim,
            output_dim=1,
            interp_mode=interp_mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
            init_value=10,
        )

    def forward(self, x, masks=None):
        """
        :params x: [..., 2 or 3].
        :return
            raw_rgb [..., 3], raw_sigma [..., 1]
        """
        raw_rgb = self.voxel_rgb(x)
        raw_sigma = self.voxel_sigma(x)
        return raw_rgb, raw_sigma

    def query_sigma(self, x, masks=None):
        return self.forward(x)[1]

    def query_rgb(self, x, masks=None):
        return self.forward(x)[0]
