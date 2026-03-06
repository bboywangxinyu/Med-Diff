import torch
from utils.Implicit_Func import Implicit_Func

import torch
import torch.nn as nn

class Implicit_Module(nn.Module):

    def __init__(self, hidden_channel, middle_channels, alpha, norm, dropout, act, double_linear, rescale,
                 tol=1e-4, smooth_beta=0.8):
        super().__init__()
        self.Fs = nn.ModuleList([
            Implicit_Func(hidden_channel, m, alpha, norm, dropout, act, double_linear, rescale)
            for m in middle_channels
        ])
        self.tol = tol
        self.smooth_beta = smooth_beta
        self.register_buffer("delta_avg", torch.tensor(0.0))

    def _reset(self, z):
        for func in self.Fs:
            func._reset(z)
        self.delta_avg.zero_()

    def forward(self, z, x, edge_index, norm_factor, batch):
        last_z = z
        for i, func in enumerate(self.Fs):
            new_z = func(z, x, edge_index, norm_factor, batch)

            diff = (new_z - last_z).norm(p=2)
            scale = last_z.norm(p=2) + 1e-6
            delta = (diff / scale).detach()

            self.delta_avg = self.smooth_beta * self.delta_avg + (1 - self.smooth_beta) * delta

            weight = torch.sigmoid((self.tol - self.delta_avg) * 100.0)

            z = weight * new_z + (1 - weight) * z

            last_z = z

        return z
