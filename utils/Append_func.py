from torch import autograd

from utils.funcs import *
import torch
import torch.autograd as autograd
import torch.nn as nn

class Append_func(nn.Module):

    def __init__(self, coeff: float = 0.05, reg_type: str = '', stability: float = 0.02):
        super().__init__()
        self.coeff = float(coeff)
        self.reg_type = reg_type.strip()
        self.stability = float(stability)
    def forward(self, z, x, edge_index, norm_factor):

        if not self.reg_type or self.coeff == 0.0:
            return z

        if not z.requires_grad:
            z = z.clone().detach().requires_grad_(True)

        reg_loss = regularize(z, x, self.reg_type, edge_index, norm_factor)

        grad, = autograd.grad(
            outputs=reg_loss,
            inputs=z,
            create_graph=True,
            retain_graph=True,
        )

        grad_std = grad.std(dim=-1, keepdim=True).clamp(min=1e-6)
        adapt_coeff = self.coeff / (1.0 + grad_std.detach())

        smooth_term = self.stability * (z - z.detach())

        z_new = z - adapt_coeff * grad + smooth_term

        return z_new
