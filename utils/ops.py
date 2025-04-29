from typing import Optional, Tuple
import numpy as np
import torch
from torch import Tensor
from einops import reduce


def default_device() -> torch.device:
    """
    PyTorch default device is GPU when available, CPU otherwise.
    :return: Default device.
    """
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    mask = b == .0
    b[mask] = 1.
    result = a / b
    result[mask] = .0
    result[result != result] = .0
    result[result == np.inf] = .0
    return result

def scale(x: Tensor,
          scaling_factor: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    if scaling_factor is not None:
        x = x / scaling_factor
        return x, scaling_factor

    scaling_factor = reduce(torch.abs(x).data, 'b t d -> b 1 d', 'mean')
    scaling_factor[scaling_factor == 0.0] = 1.0
    x = x / scaling_factor
    return x, scaling_factor

def descale(forecast: Tensor, scaling_factor: Tensor) -> Tensor:
    return forecast * scaling_factor
