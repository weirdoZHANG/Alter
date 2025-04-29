import torch.nn as nn
import gin


@gin.configurable()
def act(activation: str):
    return Act(activation)


class Act(nn.Module):
    def __init__(self, activation: str):
        super().__init__()
        self.activation = nn.ReLU() if activation == 'relu' else nn.GELU()

    def forward(self, x):
        return self.activation(x)