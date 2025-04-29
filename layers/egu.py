from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
from .activation import act


class EGULayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float, isActivate: bool):
        super().__init__()
        self.isActivate = isActivate
        self.egu = EGU(input_size, hidden_size, num_layers=1)
        self.activation = act()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        egu, _ = self.egu(x)
        if self.isActivate:
            return self.norm(self.dropout(self.activation(self.dropout(egu))))
        else:
            return self.norm(self.dropout(egu))


# Efficient Gated Unit
class EGUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(EGUCell, self).__init__()
        # Reset input weight matrix initialization
        self.only_x = nn.Linear(input_size, hidden_size)
        self.only_h = nn.Linear(hidden_size, hidden_size)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor, hidden: torch.Tensor):
        # Individual reset input
        only_x = self.only_x(x)
        x = self.silu(only_x)
        only_h = self.only_h(hidden)
        hidden = self.silu(only_h)

        # update output
        o_gate = torch.sigmoid(x + hidden)
        h_next = o_gate * x + (1 - o_gate) * hidden
        return h_next


class EGU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.,
                 batch_first: bool = True):
        super(EGU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Create multiple EGUCells, the first layer is special
        self.EGUs = nn.ModuleList([EGUCell(input_size, hidden_size)] +
                                   [EGUCell(hidden_size, hidden_size) for _ in range(num_layers - 1)])
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None):
        # state is hidden(h): [batch_size, hidden_size]
        if self.batch_first:
            batch_size, seq_len, _ = x.shape
        else:
            seq_len, batch_size, _ = x.shape
        # Need to initialize the hidden first layer
        if state is None:
            hidden = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
        else:
            (hidden) = state
            hidden = list(torch.unbind(hidden))
        # Collect output for each time step
        out = []
        for t in range(seq_len):
            # The input at the first moment is x itself
            inp = x[:, t, :] if self.batch_first else x[t, :, :]
            # Traverse n levels
            for layer in range(self.num_layers):
                hidden[layer] = self.EGUs[layer](inp, hidden[layer])
                # Use the output of the current layer as the input of the next layer
                inp = hidden[layer]
            # Collect last minute output
            out.append(hidden[-1])
        # Add all outputs together
        out = torch.stack(out, dim=1) if self.batch_first else torch.stack(out, dim=0)
        hidden = out[:, -1, :] if self.batch_first else out[-1, :, :]
        return self.dropout(out), hidden
