"""Train Med_Diff on NF-UNSW-NB15."""

import os
import sys
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
import gc
import random
import traceback
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import tqdm
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator, LastNeighborLoader
from torch_geometric.utils import to_undirected, degree
from model.Med_Diff import Med_Diff
from typing import Dict, Tuple, List
from typing import Optional, Tuple

def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@dataclass
class Config:
    seed: int = 42
    file_name: str = "NF-UNSW-NB15"
    data_dir: str = "./data"
    ckpt_dir: str = "NF-UNSW-NB15/checkpoints"

    train_ratio: float = 0.7
    val_ratio: float = 0.1

    batch_size: int = 1024
    lr: float = 1e-4
    weight_decay: float = 1e-4

    hidden_dim: int = 128
    dropout: float = 0.3558

    neighbor_size: int = 20
    tune_epochs: int = 100

    lr_max_iter: int = 2000
    lr_solver: str = "liblinear"
    lr_class_weight: str = "balanced"

    thr_metric: str = "f1_mac"
    thr_grid_n: int = 99

    use_scaler: bool = True

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(cfg: Config):
    data_path = os.path.join(cfg.data_dir, f"{cfg.file_name}.pt")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f" Data file {data_path} not found!")

    data_all = torch.load(data_path, weights_only=False)

    raw_labels_all = data_all["attack"].clone().cpu().numpy()

    if "attack" in data_all:
        data_all["attack"] = (data_all["attack"] != 2).long()

    return data_all, raw_labels_all

def split_train_val_test(datax, raw_lbls: np.ndarray, train_ratio: float, val_ratio: float):
    n = len(datax.t)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_d = datax[:n_train]
    val_d = datax[n_train:n_train + n_val]
    test_d = datax[n_train + n_val:]

    train_raw = raw_lbls[:n_train]
    val_raw = raw_lbls[n_train:n_train + n_val]
    test_raw = raw_lbls[n_train + n_val:]

    return train_d, val_d, test_d, train_raw, val_raw, test_raw

def compute_norm_and_edges(
    edge_index: torch.Tensor,
    num_nodes: Optional[int] = None,
    add_self_loops: bool = False,
    symmetric_cut: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:

    if num_nodes is None:

        num_nodes = int(edge_index.max().item()) + 1

    deg = degree(edge_index[0], num_nodes=num_nodes)

    if add_self_loops:
        deg = deg + 1.0

    if symmetric_cut:

        inv_sqrt = torch.rsqrt(deg)
        inv_sqrt = torch.nan_to_num(inv_sqrt, nan=0.0, posinf=0.0, neginf=0.0)

        undirected = to_undirected(edge_index, num_nodes=num_nodes)
        row, col = undirected
        keep = row < col
        edge_index_out = undirected[:, keep]
    else:

        inv_sqrt = torch.rsqrt(2.0 * deg)
        inv_sqrt = torch.nan_to_num(inv_sqrt, nan=0.0, posinf=0.0, neginf=0.0)
        edge_index_out = edge_index

    if inv_sqrt.dim() == 1:
        inv_sqrt = inv_sqrt.unsqueeze(-1)

    return inv_sqrt, edge_index_out

def remap_nodes(
    edge_index: torch.Tensor,
    mode: str = "encode",
    mapping: Dict[int, int] = None,
) -> Tuple[torch.Tensor, Dict[str, Dict[int, int]]]:

    if mode == "encode":
        src, dst = edge_index.tolist()

        unique_nodes: List[int] = sorted(set(src) | set(dst))

        encode_map: Dict[int, int] = {
            node_id: idx for idx, node_id in enumerate(unique_nodes)
        }
        decode_map: Dict[int, int] = {
            idx: node_id for node_id, idx in encode_map.items()
        }

        remapped_src = [encode_map[i] for i in src]
        remapped_dst = [encode_map[i] for i in dst]

        new_edge_index = torch.tensor(
            [remapped_src, remapped_dst],
            dtype=edge_index.dtype,
            device=edge_index.device,
        )

        return new_edge_index, {
            "encode": encode_map,
            "decode": decode_map,
        }

    elif mode == "decode":
        if mapping is None:
            raise ValueError("`mapping` must be provided in decode mode.")

        src, dst = edge_index.tolist()
        remapped_src = [mapping[i] for i in src]
        remapped_dst = [mapping[i] for i in dst]

        new_edge_index = torch.tensor(
            [remapped_src, remapped_dst],
            dtype=edge_index.dtype,
            device=edge_index.device,
        )

        return new_edge_index

    else:
        raise ValueError(f"Unsupported mode: {mode}")

def analyze_classification_detail(y_true_binary: np.ndarray, y_pred_binary: np.ndarray, y_raw_multiclass: np.ndarray, epoch: int):
    unique_classes = np.unique(y_raw_multiclass)
    stats = []
    for cls_id in unique_classes:
        idx = np.where(y_raw_multiclass == cls_id)[0]
        if len(idx) == 0:
            continue

        total = len(idx)
        sub_pred = y_pred_binary[idx]
        sub_true = y_true_binary[idx]

        correct = int(np.sum(sub_pred == sub_true))
        wrong = int(total - correct)
        acc = round(float(correct / total * 100.0), 2)

        is_attack = "Attack" if int(sub_true[0]) == 1 else "Benign"
        stats.append({
            "Raw_Label_ID": int(cls_id),
            "Type": is_attack,
            "Total": int(total),
            "Correct": correct,
            "Wrong": wrong,
            "Accuracy(%)": acc,
        })

    df = pd.DataFrame(stats).sort_values(by="Wrong", ascending=False)
    print(f"\n [Epoch {epoch}] Detailed Classification Report:")
    print(df.to_string(index=False))
    print("-" * 60)

def build_lr_probe(cfg: Config) -> LogisticRegression:
    return LogisticRegression(
        max_iter=cfg.lr_max_iter,
        class_weight=cfg.lr_class_weight,
        solver=cfg.lr_solver,
    )

def compute_metrics_from_prob(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Dict[str, Any]:
    y_pred = (y_prob >= thr).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1_bin = f1_score(y_true, y_pred, average="binary", zero_division=0)
    f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "acc": float(acc),
        "prec": float(prec),
        "rec": float(rec),
        "f1": float(f1_bin),
        "f1_mac": float(f1_mac),
        "cm": cm,
        "y_pred": y_pred,
    }

def search_best_threshold(cfg: Config, y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    best_t, best_s = 0.5, -1.0
    ts = np.linspace(0.01, 0.99, cfg.thr_grid_n)

    for t in ts:
        y_pred = (y_prob >= t).astype(int)
        if cfg.thr_metric == "f1_mac":
            s = f1_score(y_true, y_pred, average="macro", zero_division=0)
        elif cfg.thr_metric == "f1_bin":
            s = f1_score(y_true, y_pred, average="binary", zero_division=0)
        else:
            raise ValueError("cfg.thr_metric must be 'f1_mac' or 'f1_bin'")

        if s > best_s:
            best_s, best_t = float(s), float(t)

    return best_t, best_s

@torch.no_grad()
def extract_embeddings_no_rollout(med_diff, memory, neighbor_loader, assoc: torch.Tensor, device: torch.device, loader) -> np.ndarray:
    med_diff.eval()
    memory.eval()
    all_pairs = []

    for batch in loader:
        src, dst, t = batch.src.to(device), batch.dst.to(device), batch.t.to(device)

        n_id = torch.cat([src, dst]).unique()
        n_id, _, _ = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z_mem, _ = memory(n_id)
        ed, _ = remap_nodes(torch.stack((src, dst), dim=0))
        norm_factor, ed = compute_norm_and_edges(ed, num_nodes=len(z_mem), add_self_loops=False)

        out = med_diff(z_mem, ed.to(device), norm_factor.to(device), t=t)
        z = out[0] if isinstance(out, (tuple, list)) else out
	pair_emb = torch.cat([z[assoc[src]], z[assoc[dst]]], dim=0)
        
        all_pairs.append(pair_emb.cpu())

    return torch.cat(all_pairs, dim=0).numpy()

@torch.no_grad()
def extract_embeddings_with_rollout(med_diff, memory, neighbor_loader, assoc: torch.Tensor, device: torch.device, loader) -> np.ndarray:
    med_diff.eval()
    memory.eval()
    all_pairs = []

    for batch in loader:
        batch = batch.to(device)
        src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        n_id = torch.cat([src, dst]).unique()
        n_id, _, _ = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z_mem, _ = memory(n_id)
        ed, _ = remap_nodes(torch.stack((src, dst), dim=0))
        norm_factor, ed = compute_norm_and_edges(ed, num_nodes=len(z_mem), add_self_loops=False)

        out = med_diff(z_mem, ed.to(device), norm_factor.to(device), t=t)
        z = out[0] if isinstance(out, (tuple, list)) else out

        pair_emb = torch.cat([z[assoc[src]], z[assoc[dst]]], dim=0)
        all_pairs.append(pair_emb.cpu())

        memory.update_state(src, dst, t, msg)
        neighbor_loader.insert(src, dst)
        memory.detach()

    return torch.cat(all_pairs, dim=0).numpy()

def get_train_eval_embeddings(cfg: Config, med_diff, memory, neighbor_loader, assoc, device, train_loader, eval_loader):

    memory.reset_state()
    neighbor_loader.reset_state()
    Z_train = extract_embeddings_with_rollout(med_diff, memory, neighbor_loader, assoc, device, train_loader)
    Z_eval = extract_embeddings_no_rollout(med_diff, memory, neighbor_loader, assoc, device, eval_loader)
    return Z_train, Z_eval

def train_one_epoch(cfg: Config, med_diff, memory, neighbor_loader, assoc, device, train_loader, optimizer, epoch: int) -> float:
    med_diff.train()
    memory.train()

    memory.reset_state()
    neighbor_loader.reset_state()

    total_loss = 0.0
    pbar = tqdm.tqdm(
        train_loader,
        total=len(train_loader),
        desc=f"Manual Run | Ep {epoch}/{cfg.tune_epochs}",
        leave=False,
        dynamic_ncols=True,
        file=sys.stdout
    )

    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()

        src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        n_id = torch.cat([src, dst]).unique()
        n_id, _, _ = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z_mem, _ = memory(n_id)
        ed, _ = remap_nodes(torch.stack((src, dst), dim=0))
        norm_factor, ed = compute_norm_and_edges(ed, num_nodes=len(z_mem), add_self_loops=False)

        z, loss = med_diff(z_mem, ed.to(device), norm_factor.to(device), t=t)

        loss.backward()
        optimizer.step()
	total_loss += loss
        memory.update_state(src, dst, t, msg)
        neighbor_loader.insert(src, dst)
        memory.detach()

        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    pbar.close()
    return total_loss / max(1, len(train_loader))

def save_checkpoint(cfg: Config, path: str, epoch: int, best_val_f1: float, best_threshold: float, med_diff, memory, optimizer):
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_val_f1": float(best_val_f1),
            "best_threshold": float(best_threshold),
            "med_diff_state": med_diff.state_dict(),
            "memory_state": memory.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "hparams": {
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "hidden_dim": cfg.hidden_dim,
                "dropout": cfg.dropout,
                "train_ratio": cfg.train_ratio,
                "val_ratio": cfg.val_ratio,
                "neighbor_size": cfg.neighbor_size,

                "use_scaler": cfg.use_scaler,
                "thr_metric": cfg.thr_metric,
                "thr_grid_n": cfg.thr_grid_n,
            },
        },
        path
    )

def load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    return torch.load(path, map_location=device)

def run_once(cfg: Config):
    seed_all(cfg.seed)
    device = get_device()
    print(f" Using device: {device}")
    print(f" Dataset: {cfg.file_name}")
    print(f" Manual params: batch={cfg.batch_size}, lr={cfg.lr}, dim={cfg.hidden_dim}, dropout={cfg.dropout}")
    print(f" Val protocol: Thr metric={cfg.thr_metric}")

    data_all, raw_labels_all = load_data(cfg)
    num_nodes = int(max(data_all.src.max(), data_all.dst.max()) + 1)
    num_features = data_all.msg.size(-1)

    train_data, val_data, test_data, train_raw, val_raw, test_raw = split_train_val_test(
        data_all, raw_labels_all, cfg.train_ratio, cfg.val_ratio
    )

    train_labels = data_all["attack"][:len(train_data)].cpu().numpy()
    val_labels = data_all["attack"][len(train_data):len(train_data) + len(val_data)].cpu().numpy()
    test_labels = data_all["attack"][-len(test_data):].cpu().numpy()

    train_loader = TemporalDataLoader(train_data, batch_size=cfg.batch_size, num_workers=0)
    val_loader = TemporalDataLoader(val_data, batch_size=cfg.batch_size, num_workers=0)
    test_loader = TemporalDataLoader(test_data, batch_size=cfg.batch_size, num_workers=0)

    neighbor_loader = LastNeighborLoader(num_nodes, size=cfg.neighbor_size, device=device)
    memory_dim = time_dim = embedding_dim = 64

    memory = TGNMemory(
        num_nodes, num_features, memory_dim, time_dim,
        message_module=IdentityMessage(num_features, memory_dim, time_dim),
        aggregator_module=LastAggregator()
    ).to(device)

    med_diff = Med_Diff(
        in_channels=memory_dim,
        hidden_channels=cfg.hidden_dim,
        out_channels=cfg.hidden_dim,
        num_layers=2,
        alpha=0.2,
        iter_nums=(6, 2),
        dropout_imp=cfg.dropout,
        dropout_exp=cfg.dropout,
        lambda_s=1.0,
        lambda_r=1.0
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(med_diff.parameters())),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    assoc = torch.empty(num_nodes, dtype=torch.long)

    ckpt_path = os.path.join(cfg.ckpt_dir, f"best_{cfg.file_name}.pth")

    best_val_f1 = -1.0
    best_epoch = -1
    best_thr = 0.5

    for epoch in range(1, cfg.tune_epochs + 1):
        gc.collect()
        torch.cuda.empty_cache()

        avg_loss = train_one_epoch(cfg, med_diff, memory, neighbor_loader, assoc, device, train_loader, optimizer, epoch)

        Z_train, Z_val = get_train_eval_embeddings(cfg, med_diff, memory, neighbor_loader, assoc, device, train_loader, val_loader)

        if cfg.use_scaler:
            scaler = StandardScaler()
            Z_train = scaler.fit_transform(Z_train)
            Z_val = scaler.transform(Z_val)

        clf = build_lr_probe(cfg)
        clf.fit(Z_train, train_labels)

        y_prob_val = clf.predict_proba(Z_val)[:, 1]

        t_star, _ = search_best_threshold(cfg, val_labels, y_prob_val)
        val_metrics = compute_metrics_from_prob(val_labels, y_prob_val, thr=t_star)

        print(
            f"\n-> Ep {epoch} | "
            f"Loss={avg_loss:.4f} | "
            f"Val ACC={val_metrics['acc']:.4f} | "
            f"Val F1-mac={val_metrics['f1_mac']:.4f} | "
            f"thr={t_star:.2f}"
        )

        analyze_classification_detail(val_labels, val_metrics["y_pred"], val_raw, epoch)

        if val_metrics["f1_mac"] > best_val_f1:
            best_val_f1 = val_metrics["f1_mac"]
            best_epoch = epoch
            best_thr = t_star
            save_checkpoint(cfg, ckpt_path, epoch, best_val_f1, best_thr, med_diff, memory, optimizer)
            print(f" Saved BEST checkpoint @Ep {epoch} | best_val_f1={best_val_f1:.4f}, thr={best_thr:.2f} -> {ckpt_path}")

    print("\n" + "=" * 50)
    print(" Manual Run Finished")
    print(f" Best Val F1-Mac (over {cfg.tune_epochs} epochs): {best_val_f1:.4f} @Ep {best_epoch}")
    print(f" Best threshold (from VAL): {best_thr:.2f}")
    print("=" * 50)

    ckpt = load_checkpoint(ckpt_path, device=device)
    med_diff.load_state_dict(ckpt["med_diff_state"])
    memory.load_state_dict(ckpt["memory_state"])
    med_diff.eval()
    memory.eval()

    best_thr = float(ckpt.get("best_threshold", 0.5))

    memory.reset_state()
    neighbor_loader.reset_state()
    Z_train_best = extract_embeddings_with_rollout(med_diff, memory, neighbor_loader, assoc, device, train_loader)
    Z_test_best = extract_embeddings_no_rollout(med_diff, memory, neighbor_loader, assoc, device, test_loader)

    if cfg.use_scaler:
        scaler = StandardScaler()
        Z_train_best = scaler.fit_transform(Z_train_best)
        Z_test_best = scaler.transform(Z_test_best)

    clf = build_lr_probe(cfg)
    clf.fit(Z_train_best, train_labels)

    y_prob_test = clf.predict_proba(Z_test_best)[:, 1]
    test_metrics = compute_metrics_from_prob(test_labels, y_prob_test, thr=best_thr)

    print("\n" + "=" * 50)
    print(f" FINAL TEST (Best Val @Ep {ckpt['epoch']}, best_val_f1={ckpt['best_val_f1']:.4f})")
    print(f"[FINAL] using threshold={best_thr:.2f}")
    print(
        f"[FINAL] "
        f"ACC={test_metrics['acc']:.4f} | "
        f"Prec={test_metrics['prec']:.4f} | "
        f"Rec={test_metrics['rec']:.4f} | "
        f"F1={test_metrics['f1']:.4f} | "
        f"F1-mac={test_metrics['f1_mac']:.4f}"
    )
    print("[FINAL] Confusion Matrix:\n", test_metrics["cm"])
    print("=" * 50)

    del med_diff, memory, optimizer, clf
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    cfg = Config()
    try:
        run_once(cfg)
    except Exception as e:
        print(f" Run failed with error: {e}")
        traceback.print_exc()
        gc.collect()
        torch.cuda.empty_cache()
