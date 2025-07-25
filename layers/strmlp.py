import torch.nn as nn
from torch import Tensor
from .CFF import CFF
from .egu import EGULayer
from .spatial_akconv import SpatialAKConv
from .random_inr import RandomINR
from .standard_inr import StandardINR


class STRINRLayer(nn.Module):
    def __init__(self, layer_size: int, dropout: float):
        super().__init__()
        self.iINR_f = StandardINR(layer_size, layer_size, dropout=dropout, isActivate=True)
        self.oINR_f = StandardINR(layer_size, layer_size, dropout=dropout, isActivate=False)
        self.sAK = SpatialAKConv(layer_size, dropout=dropout, isActivate=False)
        self.sAKINR = StandardINR(layer_size, layer_size, dropout=dropout, isActivate=True)
        self.egu = EGULayer(layer_size, layer_size, dropout=dropout, isActivate=False)
        self.eguINR = StandardINR(layer_size, layer_size, dropout, isActivate=True)

        self.iINR_b = StandardINR(layer_size, layer_size, dropout=dropout, isActivate=True)
        self.oINR_b = StandardINR(layer_size, layer_size, dropout=dropout, isActivate=False)
        self.random = RandomINR(layer_size, layer_size, dropout=dropout, isActivate=False)
        self.randomINR = StandardINR(layer_size, layer_size, dropout=dropout, isActivate=True)

        self.norm1 = nn.LayerNorm(layer_size)
        self.norm2 = nn.LayerNorm(layer_size)
        self.norm3 = nn.LayerNorm(layer_size)

    def forward(self, x: Tensor) -> Tensor:
        x_res = self.norm1(x)
        x_out = self.iINR_f(x)
        x_out = self.oINR_f(x_out)
        sAK = self.sAK(x)
        sAK = self.sAKINR(sAK)
        egu = self.egu(x)
        egu = self.eguINR(egu)
        y = self.norm2(x_res + x_out + sAK + egu)

        y_res = y.clone()
        y_out = self.iINR_b(y)
        y_out = self.oINR_b(y_out)
        random = self.random(y)
        random = self.randomINR(random)
        return self.norm3(y_res + y_out + random)


class STRMLP(nn.Module):
    def __init__(self, in_feats: int, layers: int, layer_size: int, scales: float, dropout: float):
        super().__init__()
        self.features = CFF(in_feats, layer_size, scales)

        layers = [STRINRLayer(layer_size, dropout) for _ in range(layers)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return self.layers(x)
