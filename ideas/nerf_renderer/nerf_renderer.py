import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import raymarching
import raymarching2

# class NeRFRenderer(nn.Module):
#     def __init__(
#         self,
#         bbox: Tuple[float] = None,  # e.g. [0, 0, 0, 1, 1, 1]
#         near: float = None,  # near plane.
#         far: float = None,  # far plane.
#         density_thresh: float = 1e-2,  # density threshold
#         density_grid_size: int = 128,  # density grid size
#     ):


# class NeRFRenderer(nn.Module):
#     def __init__(
#         self,
#         bbox: Tuple[float] = None,  # e.g. [0, 0, 0, 1, 1, 1]
#         near: float = None,  # near plane.
#         far: float = None,  # far plane.
#         density_thresh: float = 1e-2,  # density threshold
#         density_grid_size: int = 128,  # density grid size
#     ):
#         super().__init__()
#         if bbox is not None:
#             bbox = torch.tensor(bbox, dtype=torch.float32)
#         self.register_buffer('bbox', bbox)
#         self.near = near
#         self.far = far
#         self.density_thresh = density_thresh
#         self.density_grid_size = density_grid_size

#         # TODO(ruilongli): origin impl. uses uint8 for bitfield. not sure 
#         # if that is faster.
#         density_grid = torch.ones(density_grid_size ** 3)
#         density_bitfield = torch.ones(density_grid_size ** 3 // 8, dtype=torch.uint8)
#         self.register_buffer('density_grid', density_grid)
#         self.register_buffer('density_bitfield', density_bitfield)

#         # step counter: 16 is hardcoded for averaging...
#         step_counter = torch.zeros(16, 2, dtype=torch.int32)
#         self.register_buffer('step_counter', step_counter)
#         self.mean_count = 0
#         self.local_step = 0

#     def forward(self, x, d):
#         raise NotImplementedError()

#     def run(
#         self, 
#         rays_o: torch.Tensor,  # [..., 3] 
#         rays_d: torch.Tensor,  # [..., 3]
#         max_steps: int = 1024,
#     ):
#         # return rgb [..., 3], depth [...,]
#         input_shape = rays_o.shape[:-1]
#         rays_o = rays_o.contiguous().view(-1, 3)  # [n_rays, 3]
#         rays_d = rays_d.contiguous().view(-1, 3)  # [n_rays, 3]

#         # nears, fars: both [n_rays, 1]
#         if self.bbox is not None:
#             nears, fars = raymarching.near_far_from_aabb(
#                 rays_o, rays_d, self.bbox, 0.
#             )
#             nears.unsqueeze_(-1)
#             fars.unsqueeze_(-1)
#         else:
#             nears = torch.ones_like(rays_o[:, :1]) * self.near
#             fars = torch.ones_like(rays_o[:, :1]) * self.far
        
#         if self.training:
#             # setup counter
#             counter = self.step_counter[self.local_step % 16]
#             counter.zero_() # set to 0
#             self.local_step += 1

#             cascade = 1
#             perturb = False
#             force_all_rays = True
#             dt_gamma = 0.
#             bound = 1
#             xyzs, dirs, deltas, rays = raymarching.march_rays_train(
#                 rays_o, rays_d, bound, self.density_bitfield, 
#                 cascade, self.density_grid_size, nears, fars, counter, 
#                 self.mean_count, perturb, 128, force_all_rays, 
#                 dt_gamma, max_steps
#             )
#             print ("xyzs", xyzs.shape)  # [n_pts, 3]
#             print ("dirs", dirs.shape)  # [n_pts, 3]
#             print ("deltas", deltas.shape)  # [n_pts, 2]
#             print ("rays", rays.shape)  # [n_rays, 3]
            


if __name__ == "__main__":
    import tqdm

    # renderer = NeRFRenderer(
    #     bbox=[0, 0, 0, 1, 1, 1],
    #     # near=0.1,
    #     # far=1.0,
    # ).to("cuda")
    rays_o = (torch.zeros((100, 100, 3)) + 0.1).to("cuda")
    rays_d = torch.randn((100, 100, 3)).to("cuda")
    rays_d = F.normalize(rays_d, dim=-1)
    # for _ in tqdm.tqdm(range(1)):
    #     renderer.run(rays_o, rays_d)

    density_bitfield = (torch.ones(
        (5, 128 ** 3 // 8), dtype=torch.uint8
    ) * 255).to("cuda")
    
    aabb = torch.tensor([0., 0., 0., 1., 1., 1.]).to("cuda")

    for _ in tqdm.tqdm(range(1)):
        positions, dirs, deltas, nears, fars = raymarching2.generate_training_samples(
            rays_o, rays_d, aabb, density_bitfield
        )
        torch.cuda.synchronize()
    # print (nears.shape, nears[0, :10])
    # print (positions.shape, positions)

    # for _ in tqdm.tqdm(range(1000)):
    #     nears, fars = raymarching.near_far_from_aabb(
    #         rays_o, rays_d, aabb, 0
    #     )
    #     torch.cuda.synchronize()
    # # print (nears.shape, nears[:10])
