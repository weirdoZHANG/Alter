from typing import Optional
import torch
from torch import Tensor
from torch import nn
from .activation import act


class RidgeRegressor(nn.Module):
    def __init__(self, lambda_init: Optional[float] = 0.):
        super().__init__()
        self._lambda = nn.Parameter(torch.as_tensor(lambda_init, dtype=torch.float))
        self.activation = act()

    def forward(self, reprs: Tensor, x: Tensor, reg_coeff: Optional[float] = None) -> Tensor:
        if reg_coeff is None:
            reg_coeff = self.reg_coeff()
        w, b = self.get_weights(reprs, x, reg_coeff)
        return w, b

    def get_weights(self, X: Tensor, Y: Tensor, reg_coeff: float) -> Tensor:
        batch_size, n_samples, n_dim = X.shape
        ones = torch.ones(batch_size, n_samples, 1, device=X.device)
        X = torch.concat([X, ones], dim=-1)
        if n_samples >= n_dim:
            A = torch.bmm(X.mT, X)
            A.diagonal(dim1=-2, dim2=-1).add_(reg_coeff)
            B = torch.bmm(X.mT, Y)
            weights = torch.linalg.solve(A, B)
        else:
            A = torch.bmm(X, X.mT)
            A.diagonal(dim1=-2, dim2=-1).add_(reg_coeff)
            weights = torch.bmm(X.mT, torch.linalg.solve(A, Y))
        return weights[:, :-1], weights[:, -1:]

    def reg_coeff(self) -> Tensor:
        return self.activation(self._lambda)
