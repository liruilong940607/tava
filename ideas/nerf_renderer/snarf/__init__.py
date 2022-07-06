import torch
from torch.cuda.amp import custom_bwd, custom_fwd

try:
    import _snarf as _C
except ImportError:
    from .backend import _backend as _C

class _grid_sample(torch.autograd.Function):
    
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(
        ctx, 
        positions: torch.Tensor,
        grid: torch.Tensor,
        grid_type: _C.GridType = _C.GridType.Hash,
        n_levels: int = 16, 
        # n_features_per_level: int = 2,
        base_resolution: int = 16, 
        per_level_scale: float = 2.0,
        log2_hashmap_size: int = 19,
        prepare_input_gradients: bool = True,
    ):
        # assert grid.shape(-1) == n_levels * n_features_per_level

        outputs = _C.grid_sample(
            positions, grid, grid_type,
            n_levels, base_resolution, per_level_scale, log2_hashmap_size,
            prepare_input_gradients,
        )
        if prepare_input_gradients:
            encoded_positions, dy_dx = outputs
        else:
            encoded_positions = outputs
        return encoded_positions

grid_sample = _grid_sample.apply

class _root_finding(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x_init: torch.Tensor, x_jac: torch.Tensor = None):
        if x_jac is None:
            x_jac = torch.eye(3).to(x_init)[None].expand(x_init.shape[0], -1, -1)
        x_root, success = _C.root_finding(x_init, x_jac)
        return x_root, success 

root_finding = _root_finding.apply
