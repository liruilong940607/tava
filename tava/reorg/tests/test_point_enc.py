import torch
from tava.models.basic.posi_enc import IntegratedPositionalEncoder, PositionalEncoder
from tava.reorg.models.point_enc.annealed import (
    AnnealedIntegratedSinusoidalEncoder,
    AnnealedSinusoidalEncoder,
    IntegratedSinusoidalEncoder,
)
from tava.reorg.models.point_enc.base import (
    IntegratedSinusoidalEncoder,
    SinusoidalEncoder,
)
from tava.reorg.utils.schedules import LinearSchedule

x_dim = 3
min_deg = 1
max_deg = 10

x = torch.randn((1024, x_dim))
encoder = SinusoidalEncoder(x_dim, min_deg, max_deg, True)
latent = encoder(x)
print (latent.shape, latent.sum())
_encoder = PositionalEncoder(x_dim, min_deg, max_deg, True)
_latent = _encoder(x)
print (_latent.shape, _latent.sum())

x = (torch.randn((1024, x_dim)), torch.randn((1024, x_dim)).abs())
encoder = IntegratedSinusoidalEncoder(x_dim, min_deg, max_deg, True)
latent = encoder(x)
print (latent.shape, latent.sum())
_encoder = IntegratedPositionalEncoder(x_dim, min_deg, max_deg, True)
_latent = _encoder(x)
print (_latent.shape, _latent.sum())

x = torch.randn((1024, x_dim))
max_steps = 90_000
encoder = AnnealedSinusoidalEncoder(
    x_dim, min_deg, max_deg, True, 
    alpha_sched=LinearSchedule(min_deg, max_deg, max_steps)
)
latent = encoder(x, step=20_000)
print (latent.shape, latent.sum())

x = (torch.randn((1024, x_dim)), torch.randn((1024, x_dim)).abs())
max_steps = 90_000
encoder = AnnealedIntegratedSinusoidalEncoder(
    x_dim, min_deg, max_deg, True, 
    alpha_sched=LinearSchedule(min_deg, max_deg, max_steps)
)
latent = encoder(x, step=20_000)
print (latent.shape, latent.sum())
