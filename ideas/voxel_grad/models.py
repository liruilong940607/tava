import torch
import torch.nn as nn
from tava.models.basic.posi_enc import PositionalEncoder


class MLP(nn.Module):
    def __init__(
        self, 
        input_dim: int = 2, 
        net_depth: int = 8, 
        net_width: int = 128, 
        skip_layer: int = 4, 
        output_dim: int = 1,
        input_activation = lambda x: x,
    ):
        super().__init__()
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

    def forward(self, x):
        x = self.input_activation(x)
        inputs = x
        for i in range(self.net_depth):
            x = self.input_layers[i](x)
            x = self.net_activation(x)
            if i % self.skip_layer == 0 and i > 0:
                x = torch.cat([x, inputs], dim=-1)
        return self.output_layer(x)


class PosiEncMLP(MLP):
    def __init__(
        self, 
        input_dim: int = 2, 
        net_depth: int = 8, 
        net_width: int = 128, 
        skip_layer: int = 4, 
        output_dim: int = 1,
    ):  
        posi_enc = PositionalEncoder(input_dim)
        super().__init__(
            input_dim = posi_enc.out_dim, 
            net_depth = net_depth, 
            net_width = net_width, 
            skip_layer = skip_layer, 
            output_dim = output_dim,
            input_activation = posi_enc,
        )

