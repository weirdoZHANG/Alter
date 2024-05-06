from typing import List, Optional

import math
import torch
from torch import nn
from torch import Tensor


class GaussianFourierFeatureTransform(nn.Module):
    """
    https://github.com/ndahlquist/pytorch-fourier-feature-networks
    Given an input of size [..., alpha, dim], returns a tensor of size [..., alpha, layer_size].
    """
    def __init__(self, input_dim: int, layer_size: int, scales: List[int]):
        super().__init__()
        self.input_dim = input_dim
        self.layer_size = layer_size
        self.scales = scales

        n_scale_feats = layer_size // (2 * len(scales))
        assert n_scale_feats * 2 * len(scales) == layer_size, \
            f"layer_size: {layer_size} must be divisible by 2 * len(scales) = {2 * len(scales)}"
        B_size = (input_dim, n_scale_feats)
        B = torch.cat([torch.randn(B_size) * scale for scale in scales], dim=1)

        self.register_buffer('B', B)

    def forward(self, x: Tensor) -> Tensor:
        assert x.dim() >= 2, f"Expected 2 or more dimensional input (got {x.dim()}D input)"
        alpha, dim = x.shape[-2], x.shape[-1]

        assert dim == self.input_dim, \
            f"Expected input to have {self.input_dim} channels (got {dim} channels)"

        x = torch.einsum('... t n, n d -> ... t d', [x, self.B])
        x = 2 * math.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
