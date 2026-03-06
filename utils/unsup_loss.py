
import torch
import torch.nn as nn
import torch.nn.functional as F

def build_norm_adj(edge_index, num_nodes, device):

    src, dst = edge_index
    A = torch.sparse_coo_tensor(
        torch.stack([src, dst]),
        torch.ones_like(src, dtype=torch.float32),
        (num_nodes, num_nodes),
        device=device
    )
    A = (A + A.transpose(0,1)).coalesce()
    deg = torch.sparse.sum(A, dim=1).to_dense() + 1e-6
    deg_inv_sqrt = deg.pow(-0.5)
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    A_norm = (D_inv_sqrt @ A.to_dense() @ D_inv_sqrt)
    return A_norm

def negsc_contrast(z, edge_index, tau=0.5, num_neg=5):
    src, dst = edge_index

    pos = F.cosine_similarity(z[src], z[dst]) / tau
    pos_loss = -torch.log(torch.exp(pos).mean() + 1e-8)

    neg_idx = torch.randint(0, z.size(0), (num_neg * src.size(0),), device=z.device)
    neg = F.cosine_similarity(z[src.repeat(num_neg)], z[neg_idx]) / tau
    neg_loss = torch.logsumexp(neg, dim=0).mean()
    return pos_loss + neg_loss

def structure_recon_loss(z, A_norm, sigmoid=True):
    S = z @ z.t()
    if sigmoid:
        S = torch.sigmoid(S)
        target = (A_norm > 0).float()
        return F.binary_cross_entropy(S, target)
    else:
        return F.mse_loss(S, A_norm)

def feature_recon_head(hidden_dim, in_dim):
    head = nn.Linear(hidden_dim, in_dim, bias=False)
    nn.init.xavier_uniform_(head.weight)
    return head

@torch.no_grad()
def ema_update(teacher, student, m=0.996):
    for t, s in zip(teacher.parameters(), student.parameters()):
        t.data = t.data * m + s.data * (1.0 - m)

def temporal_consistency_loss(z_student, z_teacher):
    z_s = F.normalize(z_student, p=2, dim=1)
    z_t = F.normalize(z_teacher.detach(), p=2, dim=1)
    return F.mse_loss(z_s, z_t)
