import torch
from tava.models.basic.mlp import MLP as NerfMLP_old
from tava.reorg.models.modules import NerfMLP

x = torch.randn((1024, 3))
condition = torch.randn((1024, 12))

model = NerfMLP(input_dim=3, condition_dim=12)
sigma, rgb = model(x, condition)
print (sigma.shape, sigma.sum(), rgb.shape, rgb.sum())
print (model)

model = NerfMLP_old(input_dim=3, condition_dim=12)
print (model)
