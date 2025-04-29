import gin
import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange, repeat
from layers.strmlp import STRMLP
from layers.regressors import RidgeRegressor


@gin.configurable()
def alter(inputs_feats: int, layer_size: int, strinr_layers: int, scales: float, dropout: float):
    return Alter(inputs_feats, layer_size, strinr_layers, scales, dropout)


class Alter(nn.Module):
    def __init__(self, inputs_feats: int, layer_size: int, strinr_layers: int, scales: float, dropout: float):
        super().__init__()
        self.strmlp = STRMLP(in_feats=inputs_feats + 1, layers=strinr_layers, layer_size=layer_size,
                             scales=scales, dropout=dropout)
        self.adaptive_weights = RidgeRegressor()

    def forward(self, x: Tensor, x_inputs: Tensor, y_inputs: Tensor) -> Tensor:
        tgt_horizon_len = y_inputs.shape[1]
        batch_size, lookback_len, _ = x.shape
        coords = self.get_coords(lookback_len, tgt_horizon_len, device=x.device)
        if y_inputs.shape[-1] != 0:
            inputs = torch.cat([x_inputs, y_inputs], dim=1)
            coords = repeat(coords, '1 t 1 -> b t 1', b=inputs.shape[0])
            coords = torch.cat([coords, inputs], dim=-1)
            outputs_reprs = self.strmlp(coords)
        else:
            outputs_reprs = repeat(self.strmlp(coords), '1 t d -> b t d', b=batch_size)
        lookback_reprs = outputs_reprs[:, :-tgt_horizon_len]
        horizon_reprs = outputs_reprs[:, -tgt_horizon_len:]
        w, b = self.adaptive_weights(lookback_reprs, x)
        preds = self.forecast(horizon_reprs, w, b)
        return preds

    def forecast(self, inp: Tensor, w: Tensor, b: Tensor) -> Tensor:
        return torch.einsum('... d o, ... t d -> ... t o', [w, inp]) + b

    def get_coords(self, lookback_len: int, horizon_len: int, device=None) -> Tensor:
        coords = torch.linspace(0, 1, lookback_len + horizon_len, device=device)
        return rearrange(coords, 't -> 1 t 1')
