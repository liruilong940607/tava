from typing import Dict

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from tava.reorg.models.encoder.abstract import AbstractEncoder
from tava.reorg.utils.transforms import se3_exp_map
from tava.utils.point import transform_points


class SE3Encoder(AbstractEncoder):
    """ SE3 Deformable Encoder in Nerfies. 
    
    Nerfies config: tava/reorg/configs/encoder/warp_nerfies.yaml
    D-NeRF config: tava/reorg/configs/encoder/warp_dnerf.yaml
    """
    
    def __init__(
        self, 
        warp_x_encoder: DictConfig,
        warp_meta_encoder: DictConfig,
        x_encoder: DictConfig,
        trunk: DictConfig,
        branch_r: DictConfig = None,
        branch_t: DictConfig = None,
        **kwargs,
    ):
        super().__init__()
        self.warp_x_encoder = instantiate(warp_x_encoder)
        self.warp_meta_encoder = instantiate(warp_meta_encoder)
        self.x_encoder = instantiate(x_encoder)

        input_dim = (
            self.warp_x_encoder.latent_dim + 
            self.warp_meta_encoder.latent_dim
        )
        self.trunk = instantiate(trunk, input_dim=input_dim)
        
        if branch_r and branch_t:
            self.branch_r = instantiate(
                branch_r, input_dim=self.trunk.output_dim)
            self.branch_t = instantiate(
                branch_t, input_dim=self.trunk.output_dim)
        else:
            self.branch_r = branch_t = None

    @property
    def latent_dim(self) -> int:
        return self.x_encoder.latent_dim

    @property
    def warp_dim(self) -> int:
        return 3

    def forward(self, x: torch.Tensor, meta: Dict) -> Dict:
        timestamp = meta["timestamp"]
        step = meta.get("step", None)

        if timestamp.shape[:-1] != x.shape[:-1]:
            timestamp = torch.broadcast_to(meta, x[..., :1])
            
        x_embed = self.warp_x_encoder(x, meta={"step": step})["latent"]
        meta_embed = self.warp_meta_encoder(
            timestamp, meta={"step": step}
        )["latent"]

        inputs = torch.cat([x_embed, meta_embed], dim=-1)
        trunk_outputs = self.trunk(inputs)

        if self.branch_r and self.branch_t:
            # nerfies
            log_translations = self.branch_t(trunk_outputs)
            log_rotations = self.branch_r(trunk_outputs)
            transforms = se3_exp_map(
                torch.cat([log_translations, log_rotations], dim=-1)
            )
            x_warp = transform_points(x, transforms)
        else:
            # d-nerf
            x_warp = x + trunk_outputs

        x_enc = self.x_encoder(x_warp, meta={"step": step})["latent"]
        return {"latent": x_enc, "warp": x_warp}
