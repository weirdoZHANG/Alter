import torch.nn as nn
from torch import Tensor
from .activation import act


class StandardINR(nn.Module):
    def __init__(self, input_size: int, output_size: int, dropout: float, isActivate: bool):
        super().__init__()
        self.isActivate = isActivate
        self.fc = nn.Linear(input_size, output_size)
        self.activation = act()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_size)

    def forward(self, x: Tensor) -> Tensor:
        if self.isActivate:
            return self.norm(self.dropout(self.activation(self.dropout(self.fc(x)))))
        else:
            return self.norm(self.dropout(self.fc(x)))