from utils.VariationalHidDropout import VariationalHidDropout
from utils.funcs import *
from utils.LayerNorm import LayerNorm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from utils.VariationalHidDropout import VariationalHidDropout
from utils.funcs import get_act

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Implicit_Func(nn.Module):

    def __init__(self, hidden_channel, middle_channel, alpha, norm, dropout, act,
                 double_linear=True, rescale=True, adaptive_gate=True):
        super().__init__()

        self.alpha = alpha
        self.rescale = rescale
        self.double_linear = double_linear
        self.adaptive_gate = adaptive_gate

        self.W = nn.Linear(hidden_channel, hidden_channel, bias=False)
        if double_linear:
            self.U = nn.Linear(hidden_channel, middle_channel, bias=True)

        self.act = get_act(act)
        self.norm = eval(norm)(middle_channel)
        self.drop = VariationalHidDropout(dropout)

        self.gate = nn.Parameter(torch.tensor(0.5))

        self.spectral_norm = nn.Parameter(torch.tensor(1.0))

        self.beta = nn.Parameter(torch.tensor(0.0))

    def _reset(self, z):
        self.drop.reset_mask(z)

    def forward(self, z, x, edge_index, norm_factor, batch):
        num_nodes = x.size(0)
        row, col = edge_index

        if self.rescale:
            degree = 1. / norm_factor
            degree[degree == float("inf")] = 0.
        else:
            degree = torch.ones_like(norm_factor)
        degree = degree.to(device)

        if self.adaptive_gate:
            gate_scale = torch.sigmoid(self.gate) * (
                (x.norm(p=2, dim=-1, keepdim=True) /
                 (z.norm(p=2, dim=-1, keepdim=True) + 1e-6))
            )
        else:
            gate_scale = torch.sigmoid(self.gate)

        if self.double_linear:
            msg = self.W(z) + degree * self.U(x)
        else:
            msg = self.W(z + degree * x)

        msg = msg * gate_scale

        msg = norm_factor * msg
        msg_diff = msg[row] - msg[col]

        if batch is not None:
            msg_diff = self.norm(self.act(msg_diff), batch[row])
        else:
            msg_diff = self.norm(self.act(msg_diff))

        pos_agg = scatter_add(msg_diff * norm_factor[row], row, dim=0, dim_size=num_nodes)
        neg_agg = scatter_add(msg_diff * norm_factor[col], col, dim=0, dim_size=num_nodes)
        delta_z = pos_agg - neg_agg

        delta_z = -self.spectral_norm * F.linear(delta_z, self.W.weight.t())

        res_mix = torch.sigmoid(self.beta)
        new_z = (1 - res_mix) * z + res_mix * delta_z

        z = self.alpha * self.drop(new_z) + (1 - self.alpha) * z
        return z
