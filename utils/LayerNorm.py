from torch import Tensor
from torch.nn import Parameter
from torch_geometric.typing import OptTensor
from torch_scatter import scatter

from utils.funcs import degree
import torch

import torch
from torch import nn, Tensor
from torch_scatter import scatter
from torch_geometric.typing import OptTensor
from utils.funcs import degree

class LayerNorm(nn.Module):

    def __init__(self,
                 in_channels: int,
                 eps: float = 1e-5,
                 affine: bool = True,
                 adaptive_eps: bool = True,
                 residual_gate: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.eps_base = eps
        self.adaptive_eps = adaptive_eps
        self.residual_gate = residual_gate

        if affine:
            self.gamma_base = nn.Parameter(torch.ones(in_channels))
            self.beta_base = nn.Parameter(torch.zeros(in_channels))
        else:
            self.register_parameter("gamma_base", None)
            self.register_parameter("beta_base", None)

        self.dynamic_affine = nn.Sequential(
            nn.Linear(2, in_channels),
            nn.Sigmoid()
        )

        if residual_gate:
            self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: Tensor,
                batch: OptTensor = None,
                edge_index: OptTensor = None) -> Tensor:
        device = x.device
        N = x.size(0)

        try:
            if batch is None:
                mean = x.mean(dim=0, keepdim=True)
                var = x.var(dim=0, unbiased=False, keepdim=True)
                degree_feat = torch.ones(N, 1, device=device)
                fill_value = var.mean().item() if var.numel() > 0 else 1.0
                local_var = torch.full((N, 1), fill_value, device=device)

            else:
                batch_size = int(batch.max()) + 1
                deg = degree(batch, batch_size, dtype=x.dtype).clamp_(min=1)
                mean = scatter(x, batch, dim=0, dim_size=batch_size, reduce='mean')
                mean_batch = mean[batch]
                var = scatter((x - mean_batch) ** 2, batch, dim=0,
                              dim_size=batch_size, reduce='mean')
                var_batch = var[batch]

                mean = mean_batch
                degree_feat = deg[batch].view(-1, 1)

                fill_value = var.mean().item() if var.numel() > 0 else 1.0
                local_var = torch.full((N, 1), fill_value, device=device)

        except Exception as e:

            print(f"[WARN] GraphAdaptiveLayerNorm fallback due to: {e}")
            mean = x.mean(dim=0, keepdim=True)
            var = torch.var(x, dim=0, unbiased=False, keepdim=True)
            degree_feat = torch.ones(N, 1, device=device)
            fill_value = var.mean().item() if var.numel() > 0 else 1.0
            local_var = torch.full((N, 1), fill_value, device=device)

        eps = self.eps_base
        if self.adaptive_eps:
            eps = eps * (1 + torch.exp(-local_var.mean()))

        context = torch.cat([degree_feat.log1p(), local_var], dim=1)
        dyn_affine = self.dynamic_affine(context)
        gamma_dyn = self.gamma_base * (1 + 0.5 * dyn_affine)
        beta_dyn = self.beta_base + 0.5 * dyn_affine

        out = (x - mean) / torch.sqrt(var + eps)
        out = out * gamma_dyn + beta_dyn

        if self.residual_gate:
            out = out * self.gate + x * (1 - self.gate)

        out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        return out
