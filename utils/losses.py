from typing import Callable
import torch.nn.functional as F


def get_loss_fn(loss_name: str) -> Callable:
    return {'mse': F.mse_loss,
            'mae': F.l1_loss}[loss_name]
