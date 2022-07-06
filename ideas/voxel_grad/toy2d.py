import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.backends.backend_agg import FigureCanvasAgg
from tava.models.basic.posi_enc import PositionalEncoder
from tqdm import tqdm


class MLP(nn.Module):
    def __init__(
        self, 
        input_dim=2, 
        net_depth=8, 
        net_width=128, 
        skip_layer=4, 
        output_dim=1,
        input_activation=nn.Sigmoid(),  # to [0, 1]
    ):
        super().__init__()
        # self.encoder = PositionalEncoder(input_dim)
        self.input_dim = input_dim
        self.net_depth = net_depth
        self.skip_layer = skip_layer
        self.input_layers = nn.ModuleList()
        in_features = self.input_dim
        for i in range(net_depth):
            self.input_layers.append(nn.Linear(in_features, net_width))
            if i % skip_layer == 0 and i > 0:
                in_features = net_width + self.input_dim
            else:
                in_features = net_width
        hidden_features = in_features
        self.output_layer = nn.Linear(hidden_features, output_dim)
        self.net_activation = torch.nn.ReLU()
        self.input_activation = input_activation

    def forward(self, x, enable_act = True):
        if enable_act:
            x = self.input_activation(x)  # to [0, 1]
        # x = self.encoder(x)
        inputs = x
        for i in range(self.net_depth):
            x = self.input_layers[i](x)
            x = self.net_activation(x)
            if i % self.skip_layer == 0 and i > 0:
                x = torch.cat([x, inputs], dim=-1)
        return self.output_layer(x)


class Voxel(nn.Module):
    def __init__(
        self, 
        res, 
        input_dim=2, 
        output_dim=1, 
        input_activation=nn.Sigmoid(),  # to [0, 1]
        interp_mode="bicubic",  # "bilinear" | "bicubic"
    ):
        super().__init__()
        self.res = res
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_activation = input_activation
        self.interp_mode = interp_mode
        self.data_shape = [1] + [output_dim] + [res] * self.input_dim
        self.data = nn.Parameter(torch.rand(self.data_shape))
    
    def forward(self, x, enable_act = True):
        assert x.shape[-1] == self.input_dim
        if enable_act:
            x = self.input_activation(x)  # to [0, 1]
        x = x * 2. - 1.  # to [-1, 1]
        x = x.flip(dims=(-1,))  # the convention is k, j, i
        out = F.grid_sample(
            self.data,
            x.view([1] * self.input_dim + [-1, self.input_dim]),
            padding_mode='border',
            align_corners=True,
            mode=self.interp_mode,
        ).transpose(1, -1)
        out = out.view(list(x.shape[:-1]) + [self.output_dim])
        return out 


class InterpMultiResVoxel(nn.Module):
    def __init__(
        self, 
        multi_res, 
        input_dim=2, 
        output_dim=1, 
        input_activation=nn.Sigmoid(),  # to [0, 1]
        interp_mode="bicubic",  # "bilinear" | "bicubic"
    ):
        super().__init__()
        self.multi_res = multi_res
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_activation = input_activation
        self.interp_mode = interp_mode
        self.data_shape = [1] + [output_dim] + [multi_res[-1]] * self.input_dim
        self.data = nn.Parameter(torch.rand(self.data_shape))

    def forward(self, x, enable_act = True):
        assert x.shape[-1] == self.input_dim
        if enable_act:
            x = self.input_activation(x)  # to [0, 1]
        x = x * 2. - 1.  # to [-1, 1]
        x = x.flip(dims=(-1,))  # the convention is k, j, i
        
        outputs = []
        for res in self.multi_res:
            if res == self.multi_res[-1]:
                data = self.data
            else:
                data = F.interpolate(
                    self.data, 
                    size=[res] * self.input_dim,
                    align_corners=False,
                    mode=self.interp_mode,
                )
            out = F.grid_sample(
                data,
                x.view([1] * self.input_dim + [-1, self.input_dim]),
                padding_mode='border',
                align_corners=False,
                mode=self.interp_mode,
            ).transpose(1, -1)
            out = out.view(list(x.shape[:-1]) + [self.output_dim])
            outputs.append(out)
        outputs = torch.cat(outputs, dim=-1)
        return outputs 


class MultiResVoxel(nn.Module):
    def __init__(
        self, 
        res, 
        input_dim=2, 
        output_dim=[1], 
        input_activation=nn.Sigmoid(),  # to [0, 1]
        interp_mode="bicubic",  # "bilinear" | "bicubic"
    ):
        super().__init__()
        self.input_activation = input_activation
        self.voxels = nn.ModuleList([
            Voxel(
                res=_res, 
                input_dim=input_dim, 
                output_dim=_output_dim, 
                input_activation=input_activation, 
                interp_mode=interp_mode,
            )
            for _res, _output_dim in zip(res, output_dim)
        ])
    
    def forward(self, x, enable_act = True):
        out = torch.cat(
            [voxel(x, enable_act) for voxel in self.voxels],
            dim=-1
        )
        return out


class MultiResVoxelMLP(nn.Module):
    def __init__(
        self, 
        res, 
        input_dim=2, 
        hidden_dim_voxel=[1], 
        input_activation=nn.Sigmoid(),  # to [0, 1]
        interp_mode="bicubic",  # "bilinear" | "bicubic"
        net_depth=2, 
        net_width=32, 
        skip_layer=4, 
        output_dim=1,
    ):
        super().__init__()
        self.input_activation = input_activation
        self.voxels = nn.ModuleList([
            Voxel(
                res=_res, 
                input_dim=input_dim, 
                output_dim=_output_dim, 
                input_activation=input_activation, 
                interp_mode=interp_mode,
            )
            for _res, _output_dim in zip(res, hidden_dim_voxel)
        ])
        self.mlp = MLP(
            input_dim=sum(hidden_dim_voxel), 
            net_depth=net_depth, 
            net_width=net_width, 
            skip_layer=skip_layer, 
            output_dim=output_dim
        )
    
    def forward(self, x, enable_act = True):
        out = torch.cat(
            [voxel(x, enable_act) for voxel in self.voxels],
            dim=-1
        )
        out = self.mlp(out, enable_act = False)
        return out


def plot_canvas(
    image, points=None, color="red", markersize=400, dpi=50
):
    height, width = image.shape[:2]
    fig = plt.figure(
        figsize=(width / dpi, height / dpi), dpi=dpi
    )
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes(
        [0, 0, 1, 1],
        xlim=[0, width],
        ylim=[height, 0],
        aspect=1,
    )
    ax.axis("off")
    plt.imshow(image)

    if points is not None:
        points = points.reshape(-1, 2)
        plt.scatter(
            points[:, 0] * width,
            points[:, 1] * height,
            s=markersize,
            c=color,
        )

    canvas.draw()
    s, _ = canvas.print_to_buffer()
    image = np.frombuffer(s, np.uint8).reshape((height, width, 4))
    plt.close(fig)
    return image


def visualize_field(model):
    x = torch.stack(
        torch.meshgrid(
            torch.linspace(0, 1, 500),
            torch.linspace(0, 1, 500),
            indexing="ij"
        ),
        dim=-1
    )
    x = x.clone().detach().requires_grad_(True)
    y = model(x, enable_act=False)
    y.sum().backward()
    grad = x.grad
    return x, y, grad


def optim_input(
    model, x, x_gt, lr, max_steps, 
    device="cpu", draw=True, verbose=True, canvas=None, input_act=None,
    optim=torch.optim.SGD,
):  
    model = model.to(device)
    x = x.to(device)
    x_gt = x_gt.to(device)

    if canvas is None:
        field_x, field_y, field_grad = visualize_field(model)
        canvas = field_grad.abs().sum(dim=-1).cpu().detach().numpy()
    if input_act is None:
        input_act = model.input_activation

    x_opt = x.clone().detach().requires_grad_(True)
    with torch.no_grad():
        target = model(x_gt)
            
    optimizer = optim([x_opt], lr=lr)
    pbar = tqdm(range(max_steps))
    images = []
    for step in pbar:
        for param in optimizer.param_groups:
            param["lr"] = lr
        y = model(x_opt)
        loss = F.mse_loss(y, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        error = F.mse_loss(
            input_act(x_opt), 
            input_act(x_gt)
        )
        if verbose:
            pbar.set_description(
                f"step {step:07d} lr {lr:.4f}: loss {loss.data:.7f} "
                f"grad {x_opt.grad.abs().mean().data:.7f} "
                f"err {error.data: .7f} "
            )

        if draw:
            image = plot_canvas(
                canvas, 
                points=input_act(x_opt).cpu().detach().numpy(),
                color="yellow",
            )
            image = plot_canvas(
                image, 
                points=input_act(x_gt).cpu().detach().numpy(),
                color="red",
            )
            images.append(image[..., :3])

    return x_opt, error, images


def optim_model(
    model, x, target, lr, max_steps, 
    device="cpu", verbose=True, optim = torch.optim.SGD
):
    x = x.to(device)
    target = target.to(device)
    model = model.to(device)
            
    optimizer = optim(model.parameters(), lr=lr)
    pbar = tqdm(range(max_steps))
    for step in pbar:
        for param in optimizer.param_groups:
            param["lr"] = lr
        y = model(x, enable_act=False)
        loss = F.mse_loss(y, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if verbose:
            pbar.set_description(
                f"step {step:07d} lr {lr:.4f}: loss {loss.data:.7f} "
            )

    with torch.no_grad():
        return model(x)
