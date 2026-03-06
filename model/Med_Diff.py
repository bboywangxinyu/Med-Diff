import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from utils.Implicit_Module import Implicit_Module
from utils.Append_func import Append_func
from utils.MLP import MLP





class Med_Diff(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_layers: int,
                 alpha: float,
                 iter_nums: tuple,
                 dropout_imp: float = 0.2,
                 dropout_exp: float = 0.2,
                 norm: str = 'LayerNorm',
                 residual: bool = True,
                 rescale: bool = True,
                 act_imp: str = 'tanh',
                 act_exp: str = 'elu',
                 reg_coeff: float = 0.,
                 lambda_s: float = 1.0,
                 lambda_r: float = 1.0,
                 lambda_gc: float = 0.01,
                 lambda_lc: float = 0.02):  
        super().__init__()
        self.total_num, self.grad_num = iter_nums
        self.no_grad_num = self.total_num - self.grad_num
        self.residual = residual
        self.rescale = rescale
        self.dropout_imp = dropout_imp
        self.dropout_exp = dropout_exp
        self.extractor = nn.Linear(in_channels, hidden_channels)
        middle_channels = [hidden_channels] * num_layers
        self.implicit_module = Implicit_Module(
            hidden_channels, middle_channels,
            alpha, norm, dropout_imp, act_imp,
            double_linear=True, rescale=rescale
        )
        self.Append = Append_func(coeff=reg_coeff, reg_type='l2')
        self.decoder = nn.Linear(hidden_channels, in_channels)
        self.global_projector = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.PReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.readout = lambda z: z.mean(dim=0, keepdim=True)
        self.lambda_s = lambda_s
        self.lambda_r = lambda_r
        self.lambda_gc = lambda_gc
        self.lambda_lc = lambda_lc
        self.t_beta = 0.05
        self.epoch = 0  # 用于warm-up控制
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def structure_loss(self, z, edge_index):
        src, dst = edge_index
        diff = (z[src] - z[dst]).pow(2).sum(dim=1)
        return self.lambda_s * diff.mean()


    def recon_loss(self, z, x, gamma=2.0, lambda_uni=0.01):
        z_rec = self.decoder(z)
        z_norm = F.normalize(z_rec, dim=-1)
        x_norm = F.normalize(x, dim=-1)
        cosine_sim = (x_norm * z_norm).sum(dim=-1)          # 节点间余弦相似度
        sce_loss = (1 - cosine_sim).pow(gamma).mean()       # (1 - cos)^γ
        z_all = F.normalize(z, dim=-1)
        pairwise_d = torch.cdist(z_all, z_all, p=2).pow(2)
        uni_loss = torch.exp(-2 * pairwise_d).mean()
        total_loss = self.lambda_r * (sce_loss + lambda_uni * uni_loss)
        return total_loss

    def node_contrastive_loss(self, z_main, z_view, temperature: float = 0.2):
        z_main = F.normalize(z_main, dim=-1)
        z_view = F.normalize(z_view, dim=-1)
        sim_matrix = torch.mm(z_main, z_view.T) / temperature
        labels = torch.arange(sim_matrix.size(0), device=z_main.device)
        loss = F.cross_entropy(sim_matrix, labels)
        scale = min(1.0, self.epoch / 5.0)
        return scale * self.lambda_lc * loss

    def multiple_steps(self, iter_start, iter_num, z, x, edge_index, norm_factor, batch, t=None):
        total_node_contrast = 0.0
        epoch_ratio = min(self.epoch / 50.0, 1.0)
        time_beta = 0.05 * (1.0 - 0.3 * epoch_ratio)
        for step in range(iter_start, iter_start + iter_num):
            x_view = F.dropout(x, p=0.2, training=self.training)
            x_view = x_view + 0.02 * torch.randn_like(x_view)
            z_main = self.implicit_module(z, x, edge_index, norm_factor, batch)
            z_view = self.implicit_module(z, x_view, edge_index, norm_factor, batch)
            if self.training:
                total_node_contrast += self.node_contrastive_loss(z_main, z_view)
            if t is not None:
                t_embed = (t - t.min()) / (t.max() - t.min() + 1e-8)
                t_score = torch.median(t_embed)
                α_t = torch.sigmoid(0.8 * t_score + 0.4 * (1.0 - step / iter_num))
            else:
                α_t = torch.sigmoid(torch.tensor(1.0 - 0.5 * step / iter_num, device=z.device)) 
            z = α_t * z_main + (1 - α_t) * z_view
            if hasattr(self, 't_beta') and self.t_beta > 0:
                norm_factor = norm_factor * (1.0 + time_beta * torch.randn_like(norm_factor) * 0.01)
        self.loss_node_contrast = total_node_contrast / iter_num if self.training else 0.0
        return z
    def forward(self, x, edge_index, norm_factor, batch=None, t=None):
        x_raw = x.clone()
        x = F.dropout(x, self.dropout_exp, training=self.training)
        x = self.extractor(x)
        self.implicit_module._reset(x)
        z = torch.zeros_like(x)
        with torch.no_grad():
            z = self.multiple_steps(0, self.no_grad_num, z, x, edge_index, norm_factor, batch,t=t)
        new_z = self.multiple_steps(self.no_grad_num - 1, self.grad_num, z, x, edge_index, norm_factor, batch,t=t)
        z = norm_factor * new_z + x if self.residual else new_z
        if self.training:
            loss_s = self.structure_loss(z, edge_index)
            loss_r = self.recon_loss(z, x_raw)
            loss_nc = getattr(self, "loss_node_contrast", 0.0)
            loss = loss_s + loss_r + loss_nc
            return z, loss
        else:
            return z





@torch.enable_grad()
def regularize(z, x, reg_type, edge_index=None, norm_factor=None):
    z_reg = norm_factor * z

    if reg_type == 'Lap':  # Laplacian Regularization
        row, col = edge_index
        loss = scatter_add(((z_reg.index_select(0, row) - z_reg.index_select(0, col)) ** 2).sum(-1), col, dim=0,
                           dim_size=z.size(0))
        return loss.mean()

    elif reg_type == 'Dec':  # Feature Decorrelation
        zzt = torch.mm(z_reg.t(), z_reg)
        Dig = 1. / torch.sqrt(1e-8 + torch.diag(zzt, 0))
        z_new = torch.mm(z_reg, torch.diag(Dig))
        zzt = torch.mm(z_new.t(), z_new)
        zzt = zzt - torch.diag(torch.diag(zzt, 0))
        zzt = F.hardshrink(zzt, lambd=0.5)
        square_loss = F.mse_loss(zzt, torch.zeros_like(zzt))
        return square_loss

    else:
        raise NotImplementedError



class GradRegUpdate(nn.Module):
    def __init__(self, weight: float = 0.05, mode: str = "", stability: float = 0.02):
        super().__init__()
        self.weight = float(weight)
        self.mode = str(mode).strip()
        self.stability = float(stability)

    def forward(self, z, x, edge_index, norm_factor):
        if (not self.mode) or self.weight == 0.0:
            return z

        if not z.requires_grad:
            z = z.clone().detach().requires_grad_(True)

        reg_loss = regularize(z, x, self.mode, edge_index, norm_factor)
        (grad,) = autograd.grad(
            outputs=reg_loss,
            inputs=z,
            create_graph=True,
            retain_graph=True,
        )

        grad_std = grad.std(dim=-1, keepdim=True).clamp(min=1e-6)
        step = self.weight / (1.0 + grad_std.detach())

        z_new = z - step * grad + self.stability * (z - z.detach())
        return z_new
