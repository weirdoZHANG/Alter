import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
import math
from .activation import act


class SpatialAKConv(nn.Module):
    def __init__(self, layer_size: int, dropout: float, isActivate: bool):
        super().__init__()
        self.isActivate = isActivate
        self.aKConv = AKConv(layer_size)
        self.activation = act()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(layer_size)

    def forward(self, x: Tensor) -> Tensor:
        if self.isActivate:
            return self.norm(self.dropout(self.activation(self.dropout(self.aKConv(x.permute(0, 2, 1)).permute(0, 2, 1)))))
        else:
            return self.norm(self.dropout(self.aKConv(x.permute(0, 2, 1)).permute(0, 2, 1)))


class AKConv(nn.Module):
    """
    based on https://blog.csdn.net/java1314777/article/details/134728612
    """

    def __init__(self, layer_size, num_param=1):
        super(AKConv, self).__init__()
        self.num_param = num_param
        self.p_conv = nn.Conv1d(layer_size, 2 * num_param, kernel_size=3, padding=1, stride=1)
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
        x_offset = self._reshape_x_offset(x_offset)
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
            torch.arange(0, h, 1),
            torch.arange(0, 1, 1), indexing='xy')
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
    def _reshape_x_offset(x_offset):
        x_offset = rearrange(x_offset, 'b c h n -> b c (h n)')
        return x_offset


class DWConv(nn.Module):
    def __init__(self, c):
        super(DWConv, self).__init__()
        self.conv = nn.Conv1d(c, c, kernel_size=1, groups=c, bias=False)
        self.bn = nn.BatchNorm1d(c)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
