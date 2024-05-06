from typing import Optional
import gin
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange
import math
from models.modules.feature_transforms import GaussianFourierFeatureTransform


class StandardINR(nn.Module):
    def __init__(self, input_size: int, output_size: int,
                 dropout: float):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = nn.Linear(input_size, output_size).float().cuda()
        self.sinReLU = sinReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_size)

    def forward(self, x: Tensor) -> Tensor:
        out = self.linear(x)
        out = self.sinReLU(out)
        out = self.dropout(out)
        return self.norm(out)

class STRINRLayer(nn.Module):
    def __init__(self, in_channels: int, layer_size: int, multiple: int,
                 dropout: float):
        super().__init__()
        self.layer_size = layer_size
        self.multiple = multiple
        self.hidden_size = layer_size * multiple
        self.out_channels = in_channels * multiple
        self.iINR = StandardINR(layer_size, self.hidden_size, dropout).float().cuda()
        self.oINR = StandardINR(self.hidden_size, layer_size, dropout).float().cuda()
        self.sAKINR = StandardINR(self.out_channels, in_channels, dropout).float().cuda()
        self.eltmINR = StandardINR(self.hidden_size, layer_size, dropout).float().cuda()
        self.randomINR = StandardINR(self.hidden_size, layer_size, dropout).float().cuda()
        self.sAK = SpatialAKConv1d(in_channels, layer_size, multiple, dropout=dropout).float().cuda()
        self.eltm = ELTM(layer_size, multiple, num_layers=1, dropout=dropout).float().cuda()
        self.random = RandomINR(layer_size, multiple, dropout).float().cuda()
        self.norm = nn.LayerNorm(layer_size)

    def forward(self, x: Tensor) -> Tensor:
        x_res = x = self.norm(x)
        x_out = self.iINR(x)
        x_out = self.oINR(x_out)
        sAK = self.sAK(x)
        sAK = self.sAKINR(sAK)
        sAK = torch.transpose(sAK, 1, 2)
        eltm = self.eltm(x)
        eltm = self.eltmINR(eltm)
        x = x_res + x_out + sAK + eltm
        y_res = y = self.norm(x)
        y_out = self.iINR(y)
        y_out = self.oINR(y_out)
        y_random = self.random(y)
        y_random = self.randomINR(y_random)
        y = y_res + y_out + y_random
        return self.norm(y)

@gin.configurable()
def sinReLU(v: float):
    return SinReLU(v)

class SinReLU(nn.Module):
    def __init__(self, v: float):
        super().__init__()
        self.v = v

    def forward(self, x):
        return torch.where(x >= 0, (x + torch.sin(x)) * self.v, torch.zeros_like(x)).float().cuda()

    def gradient(self, x):
        return torch.where(x >= 0, (1 + torch.cos(x)) * self.v, torch.zeros_like(x)).float().cuda()

class SpatialAKConv1d(nn.Module):
    def __init__(self, in_channels: int, layer_size: int, multiple: int, dropout: float):
        super().__init__()
        self.out_channels = in_channels * multiple
        self.aKConv = AKConv(layer_size, multiple).float().cuda()
        self.sinReLU = sinReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.transpose(x, 1, 2)
        x = self.aKConv(x)
        x = self.sinReLU(x)
        x = self.dropout(x)
        return self.norm(x)

class AKConv(nn.Module):
    """
    based on
    ttps://blog.csdn.net/java1314777/article/details/134728612
    """

    def __init__(self, layer_size, multiple, stride=1):
        super(AKConv, self).__init__()
        self.num_param = multiple
        self.stride = stride
        self.p_conv = nn.Conv1d(layer_size, 2 * multiple, kernel_size=3, padding=1, stride=stride).float().cuda()
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        dtype = offset.data.type()
        N = offset.size(1) // 2
        p = self._get_p(offset, dtype)
        p = p.contiguous().permute(0, 2, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(1) - 1), torch.clamp(q_lt[..., N:], 0, x.size(2) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(1) - 1), torch.clamp(q_rb[..., N:], 0, x.size(2) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(1) - 1), torch.clamp(p[..., N:], 0, x.size(2) - 1)], dim=-1)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt
        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        return x_offset

    def _get_p_n(self, N, dtype):
        base_int = round(math.sqrt(self.num_param))
        row_number = self.num_param // base_int
        mod_number = self.num_param % base_int
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(0, row_number),
            torch.arange(0, base_int), indexing='xy')
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)
        if mod_number > 0:
            mod_p_n_x, mod_p_n_y = torch.meshgrid(
                torch.arange(row_number, row_number + 1),
                torch.arange(0, mod_number), indexing='xy')
            mod_p_n_x = torch.flatten(mod_p_n_x)
            mod_p_n_y = torch.flatten(mod_p_n_y)
            p_n_x, p_n_y = torch.cat((p_n_x, mod_p_n_x)), torch.cat((p_n_y, mod_p_n_y))
        p_n = torch.cat([p_n_x, p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1).type(dtype)
        return p_n

    def _get_p_0(self, h, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride),
            torch.arange(0, self.stride, self.stride), indexing='xy')
        p_0_x = torch.flatten(p_0_x).view(1, 1, h).repeat(1, N, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h).repeat(1, N, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0

    def _get_p(self, offset, dtype):
        N, h = offset.size(1) // 2, offset.size(2)
        p_n = self._get_p_n(N, dtype)
        p_0 = self._get_p_0(h, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, _ = q.size()
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)
        index = q[..., :N] + q[..., N:]
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1).contiguous().view(b, c, -1)
        index = index.clamp(min=0, max=x.shape[-1] - 1)
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, N)
        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        b, c, h, n = x_offset.size()
        x_offset = rearrange(x_offset, 'b c h n -> b c (h n)')
        return x_offset

# Efficient Long-Term Memory
class ELTMcell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(ELTMcell, self).__init__()
        # Reset input weight matrix initialization
        self.input_in = nn.Linear(input_size, hidden_size * 2, bias=False).float().cuda()
        self.hidden_in = nn.Linear(hidden_size, hidden_size * 2).float().cuda()
        self.only_x = nn.Linear(input_size, hidden_size, bias=False).float().cuda()
        self.only_h = nn.Linear(hidden_size, hidden_size, bias=False).float().cuda()

        # Update output weight matrix initialization
        self.input_out = nn.Linear(hidden_size, hidden_size, bias=False).float().cuda()
        self.hidden_out = nn.Linear(hidden_size, hidden_size, bias=False).float().cuda()
        self.candidate_hidden_out = nn.Linear(hidden_size, hidden_size).float().cuda()
        # Branch output parameter initialization
        self.alpha_p = nn.Linear(hidden_size, hidden_size, bias=False).float().cuda()
        self.beta_p = nn.Linear(hidden_size, hidden_size, bias=False).float().cuda()

    def forward(self, x: torch.Tensor, hidden: torch.Tensor):
        # Combined reset input
        input_gates = self.input_in(x) + self.hidden_in(hidden)
        i_gate, h_gate = input_gates.chunk(2, dim=-1)
        i_gate = torch.sigmoid(i_gate)  # Candidate reset input gate
        h_gate = torch.tanh(h_gate)  # Candidate hidden state
        ch_n = i_gate * h_gate  # Reset hidden state

        # Individual reset input
        only_x_gate = self.only_x(x)
        x = torch.tanh(only_x_gate)
        only_h_gate = self.only_h(hidden)
        hidden = torch.tanh(only_h_gate)

        # update output
        output_gates = self.input_out(x) + self.hidden_out(hidden) + self.candidate_hidden_out(ch_n)
        o_gate = torch.sigmoid(output_gates)  # Update output gate
        alpha = self.alpha_p(o_gate)  # Update individual input
        beta = self.beta_p(o_gate)  # Update individual past hidden state
        gamma = 1 - alpha - beta  # Update the current candidate hidden state
        h_next = alpha * x + beta * hidden + gamma * ch_n  # Final hidden state output
        return h_next

class ELTM(nn.Module):
    def __init__(self, layer_size: int, multiple: int, num_layers: int, dropout: float):
        super(ELTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = multiple * layer_size
        # Create multiple ELTMCells, the first layer is special
        self.ELTMs = nn.ModuleList([ELTMcell(layer_size, self.hidden_size)] +
                                   [ELTMcell(self.hidden_size, self.hidden_size) for _ in range(num_layers - 1)])
        self.sinReLU = sinReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.hidden_size)

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None):
        # x output shape: [seq_len. batch_size, input_size]
        # state is hidden(h): [batch_size, hidden_size]
        seq_len, batch_size = x.shape[:2]
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
            inp = x[t]
            # Traverse n levels
            for layer in range(self.num_layers):
                hidden[layer] = self.ELTMs[layer](inp, hidden[layer])
                # Use the output of the current layer as the input of the next layer
                inp = hidden[layer]
            # Collect last minute output
            out.append(hidden[-1])
        # Add all outputs together
        out = torch.stack(out, dim=0)
        out = self.sinReLU(out)
        out = self.dropout(out)
        # hidden = torch.stack(hidden)
        return self.norm(out)  # hidden


class RandomINR(nn.Module):
    def __init__(self, layer_size: int, multiple: int, dropout: float):
        super().__init__()
        self.hidden_size = layer_size * multiple
        # torch.manual_seed(seed=2021)
        weight = torch.randn(self.hidden_size, layer_size, requires_grad=False).float().cuda()
        self.register_buffer('weight', weight, persistent=True)
        self.register_buffer('bias', None)
        self.sinReLU = sinReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.hidden_size)

    def forward(self, x):
        x = F.linear(x, self.weight, self.bias)
        x = self.sinReLU(x)
        x = self.dropout(x)
        return self.norm(x)

class STRMLP(nn.Module):
    def __init__(self, in_channels: int, in_feats: int, layers: int, layer_size: int, scales: float, multiple: int,
                 dropout: float):
        super().__init__()
        self.features = nn.Linear(in_feats, layer_size).float().cuda() if layer_size == 0 \
            else GaussianFourierFeatureTransform(in_feats, layer_size, scales).float().cuda()

        layers = [STRINRLayer(in_channels, layer_size, multiple, dropout).float().cuda() for _ in range(layers)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return self.layers(x)

