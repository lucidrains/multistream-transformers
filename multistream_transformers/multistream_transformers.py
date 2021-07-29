import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange

class MultistreamTransformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        num_tokens = 256,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()

    def forward(self, x):
        return x
