from typing import Any, Dict

import torch
import torch.nn as nn
from tava.reorg.models.encoder.abstract import AbstractEncoder


class EmbedEncoder(AbstractEncoder):
    """ A simple warpper for nn.Embedding """

    def __init__(self, num_embeddings: int, embedding_dim: int, **kwargs):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embed = nn.Embedding(num_embeddings, embedding_dim, **kwargs)

    @property
    def latent_dim(self) -> int:
        return self.embedding_dim

    def forward(self, x: torch.Tensor, meta: Dict = None) -> Dict:
        return {"latent": self.embed(x)}
