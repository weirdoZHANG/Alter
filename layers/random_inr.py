import torch
import torch.nn as nn
import torch.nn.functional as F
from .activation import act


class RandomINR(nn.Module):
    def __init__(self, input_size: int, output_size: int, dropout: float, isActivate: bool):
        super().__init__()
        self.isActivate = isActivate
        weight = torch.randn(output_size, input_size)
        self.register_buffer('weight', weight)
        self.register_buffer('bias', None)
        self.activation = act()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_size)

    def forward(self, x):
        if self.isActivate:
            return self.norm(self.dropout(self.activation(self.dropout(F.linear(x, self.weight, self.bias)))))
        else:
            return self.norm(self.dropout(F.linear(x, self.weight, self.bias)))