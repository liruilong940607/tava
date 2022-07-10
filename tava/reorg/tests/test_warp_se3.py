import torch
from hydra import compose, initialize
from hydra.utils import instantiate

device = "cuda:0"
overrides = []

# create the cfg
with initialize(config_path="../configs/encoder"):
    cfg = compose(config_name="warp_dnerf")
print (cfg)

model = instantiate(cfg)
print (model)

x = torch.randn((1024, 3))
meta = {
    "timestamp": torch.randint(0, 9, size=(1024, 1)),
    "step": 1000
}
outputs = model(x, meta)
print (outputs["latent"].shape)
