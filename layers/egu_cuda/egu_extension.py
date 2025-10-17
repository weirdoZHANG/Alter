import math
import torch
import torch.nn as nn
import torch.nn.init as init
import egu_impl_cuda


class EGU(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            num_layers=1,
            bias=True,
            batch_first=False,
            dropout=0.0,
            bidirectional=False,
            device=None,
            dtype=None
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_dir = 2 if bidirectional else 1

        self.layer_input_sizes = [input_size]
        for _ in range(1, num_layers):
            self.layer_input_sizes.append(hidden_size * self.num_dir)

        self.weight_ih_list = nn.ParameterList()
        self.weight_hh_list = nn.ParameterList()

        if bias:
            self.bias_ih_list = nn.ParameterList()
            self.bias_hh_list = nn.ParameterList()

        for layer in range(num_layers):
            curr_input_size = self.layer_input_sizes[layer]
            for _ in range(self.num_dir):
                w_ih = nn.Parameter(torch.empty(hidden_size, curr_input_size))
                w_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
                self.weight_ih_list.append(w_ih)
                self.weight_hh_list.append(w_hh)

                if bias:
                    b_ih = nn.Parameter(torch.empty(hidden_size))
                    b_hh = nn.Parameter(torch.empty(hidden_size))
                    self.bias_ih_list.append(b_ih)
                    self.bias_hh_list.append(b_hh)

        self.reset_parameters()

    def reset_parameters(self):
        """采用与RNN/LSTM*/GRU一致的权重初始化"""
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0

        for i in range(len(self.weight_ih_list)):
            init.uniform_(self.weight_ih_list[i], -stdv, stdv)
            init.uniform_(self.weight_hh_list[i], -stdv, stdv)

            if self.bias:
                init.zeros_(self.bias_ih_list[i])
                init.zeros_(self.bias_hh_list[i])

    def forward(self, input, hx=None):
        if hx is None:
            batch_size = input.size(0) if self.batch_first else input.size(1)
            hx = torch.zeros(
                self.num_layers * self.num_dir,
                batch_size,
                self.hidden_size,
                device=input.device,
                dtype=input.dtype
            )

        if input.is_cuda:
            output, hy = egu_impl_cuda.forward(
                input=input,
                hx=hx,
                weight_ih_list=self.weight_ih_list,
                weight_hh_list=self.weight_hh_list,
                bias_ih_list=self.bias_ih_list if self.bias else [],
                bias_hh_list=self.bias_hh_list if self.bias else [],
                num_layers=self.num_layers,
                dropout_p=self.dropout,
                training=self.training,
                bidirectional=self.bidirectional,
                batch_first=self.batch_first
            )
        else:
            raise RuntimeError(
                f"Here, the EGU module only supports CUDA devices, but got {input.device.type} device. "
                "Please move your input to a CUDA device with .to('cuda')"
            )

        return output, hy