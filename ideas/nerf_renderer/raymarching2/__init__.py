import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

try:
    import _raymarching2 as _backend
except ImportError:
    from .backend import _backend




class _generate_training_samples(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(
        ctx, 
        rays_o: torch.Tensor, 
        rays_d: torch.Tensor, 
        aabb: torch.Tensor, 
        density_bitfield: torch.Tensor, 
        max_samples: int = 10_000,
    ):
        input_shape = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)
        indices, positions, dirs, deltas, ts, nears, fars = _backend.generate_training_samples(
            rays_o, rays_d, aabb, density_bitfield, max_samples
        )
        # # check
        # indices = indices.long()
        # ray_ids, sample_ids, sample_cnts = torch.split(indices, 1, dim=-1)
        # for ray_id, sample_id, sample_cnt in zip(
        #     ray_ids, sample_ids, sample_cnts
        # ):
        #     if sample_cnt == 0:
        #         continue
        #     errs = dirs[sample_id: sample_id + sample_cnt] - rays_d[ray_id]
        #     max_err = torch.linalg.norm(errs).max()
        #     print ("max_err", max_err)
        
        # nears = nears.view(input_shape)
        # fars = fars.view(input_shape)
        # indices = indices.view(input_shape + torch.Size([3]))

        return indices, positions, dirs, deltas, ts, nears, fars
generate_training_samples = _generate_training_samples.apply


class _volumetric_rendering(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(
        ctx, 
        indices: torch.Tensor, 
        positions: torch.Tensor, 
        deltas: torch.Tensor, 
        ts: torch.Tensor, 
        sigmas: torch.Tensor, 
        rgbs: torch.Tensor, 
        bkgd_rgb: torch.Tensor,
    ):
        (
            accumulated_weight, 
            accumulated_depth, 
            accumulated_color, 
            accumulated_position
        ) = _backend.volumetric_rendering(
            indices, positions, deltas, ts, 
            sigmas, rgbs, 
            bkgd_rgb
        )
        ctx.save_for_backward(
            accumulated_color,
            indices, positions, deltas, ts,
            sigmas, rgbs, 
            bkgd_rgb,
        )
        return (
            accumulated_weight, 
            accumulated_depth, 
            accumulated_color, 
            accumulated_position
        )

    @staticmethod
    @custom_bwd
    def backward(
        ctx, 
        grad_weight: torch.Tensor, 
        grad_depth: torch.Tensor, 
        grad_color: torch.Tensor, 
        grad_position: torch.Tensor, 
    ):
        grad_color = grad_color.contiguous()

        (
            accumulated_color,
            indices, positions, deltas, ts,
            sigmas, rgbs, 
            bkgd_rgb,
        ) = ctx.saved_tensors

        (
            grad_sigmas, grad_rgbs
        ) = _backend.volumetric_rendering_backward(
            accumulated_color, grad_color,
            indices, positions, deltas, ts, 
            sigmas, rgbs, 
            bkgd_rgb
        )
        return (
            None, None, None, None,
            grad_sigmas, grad_rgbs, 
            None,
        )

volumetric_rendering = _volumetric_rendering.apply

