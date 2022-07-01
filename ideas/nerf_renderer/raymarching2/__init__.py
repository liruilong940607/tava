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
        positions, dirs, deltas, nears, fars = _backend.generate_training_samples(
            rays_o, rays_d, aabb, density_bitfield, max_samples
        )
        nears = nears.view(input_shape)
        fars = fars.view(input_shape)
        return positions, dirs, deltas, nears, fars
generate_training_samples = _generate_training_samples.apply

