import torch
from hydra import compose, initialize
from hydra.utils import instantiate

device = "cuda:0"
overrides = []

# create the cfg
with initialize(config_path="../configs/dataset"):
    cfg = compose(config_name="dnerf")
print (cfg)

dataset = instantiate(cfg, split="train")
print (dataset)

data = dataset[0]
print (data["timestamp"])
