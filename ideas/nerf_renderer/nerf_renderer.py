import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import snarf

positions = torch.rand((1024, 3))
grid = torch.rand((2^19, 2))


positions_enc = snarf.grid_sample(positions, grid)
print (positions_enc.shape)


