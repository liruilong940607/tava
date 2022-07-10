from abc import abstractmethod
from typing import Dict

import torch
import torch.nn as nn


class AbstractEncoder(nn.Module):
    """ Point-based encoder that maps (x, meta) to latent. 
    
    The desired inputs are:
    - x: points. Normally it is a tensor with shape [..., x_dim]. 
        It can also be in other forms such as a tuple for Mip-NeRF.
    - meta: (Optional) meta data attached to the points. 
        For example timestamp in dynamic NeRFs. Should be a dict that
        contains whatever is needed.

    The desired output is a Dict contains:
    - latent: a latent code for this point, usually with shape 
        [..., latent_dim], but can also be in other forms such as tuple.
    - warp: (Optional) a warped x after this encoder. Usually this
        would have the same shape and form with the input x.
    - ...

    Note: the output latent code is designed to be used as the input to
    the point decoder, to read out color, density etc using MLPs / Voxels. 
    While the output warp is designed for building correspondence, which
    in some case might be the same as the output latent, in some case might
    be different.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__()

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        # the output dim of the latent code
        raise NotImplementedError

    @property
    @abstractmethod
    def warp_dim(self) -> int:
        # the output dim of the warp
        raise NotImplementedError
    
    @abstractmethod
    def forward(self, x: torch.Tensor, meta: Dict = None) -> Dict:
        """ See doc above. """
        raise NotImplementedError
