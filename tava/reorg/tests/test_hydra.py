from hydra import compose, initialize
from hydra.utils import instantiate

overrides = []

# create the cfg
with initialize(config_path="../configs/model/encoder"):
    cfg = compose(config_name="warp_nerfies")

print (cfg)
cfg.warp_meta_encoder.num_embeddings = 10
model = instantiate(cfg)
