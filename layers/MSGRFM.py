from typing import List
import math
import torch
from torch import nn
from torch import Tensor


class MultiscaleGaussianRandomFourierMapping(nn.Module):
    """
    https://github.com/ndahlquist/pytorch-fourier-feature-networks
    Given an input of size [..., tau, dim], returns a tensor of size [..., tau, layer_size].
    """
    def __init__(self, input_dim: int, layer_size: int, scales: List[int]):
        super().__init__()
        self.input_dim = input_dim
        self._validate_layer_size(layer_size, scales)
        num_scales = len(scales)
        features_per_scale = layer_size // (2 * num_scales)
        basis_matrix_size = (input_dim, features_per_scale)
        basis_matrices = [torch.randn(basis_matrix_size) * scale for scale in scales]
        self.register_buffer('B', torch.cat(basis_matrices, dim=1))

    def _validate_layer_size(self, layer_size: int, scales: List[int]):
        num_scales = len(scales)
        expected_divisor = 2 * num_scales
        assert layer_size % expected_divisor == 0, \
            f"layer_size: {layer_size} must be divisible by 2 * len(scales) = {expected_divisor}"

    def _validate_input(self, x: Tensor):
        assert x.dim() >= 2, f"Expected 2 or more dimensional input (got {x.dim()}D input)"
        tau, dim = x.shape[-2], x.shape[-1]
        assert dim == self.input_dim, \
            f"Expected input to have {self.input_dim} channels (got {dim} channels)"

    def forward(self, x: Tensor) -> Tensor:
        self._validate_input(x)
        x = torch.einsum('... t n, n d -> ... t d', [x, self.B])
        x = 2 * math.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
