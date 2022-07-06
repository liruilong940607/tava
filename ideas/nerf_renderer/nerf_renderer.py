import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

import snarf

# positions = torch.rand((1024, 3))
# grid = torch.rand((2^19, 2))

# positions_enc = snarf.grid_sample(positions, grid)
# print (positions_enc.shape)

torch.manual_seed(1234)
x_init = torch.rand((100000, 3)).float().to("cuda") * 0.1
x_jac = (2 * x_init - 1)[:, None, :] * torch.eye(3).to(x_init)

for _ in tqdm.tqdm(range(200)):
    x_root, success = snarf.root_finding(x_init, x_jac)
    torch.cuda.synchronize()

print (x_root.mean(), x_root.shape, success.shape, success.float().mean())
