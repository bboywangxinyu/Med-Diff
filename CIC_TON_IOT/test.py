"""Evaluate Med_Diff with LR probe and t-SNE visualization."""

from __future__ import annotations

import os
import sys
import gc
import time
import argparse
import traceback
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import psutil

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedShuffleSplit

from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator, LastNeighborLoader
from torch_geometric.utils import to_undirected, degree

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from model.Med_Diff import Med_Diff

def _bytes_to_mb(x: int) -> float:
    return float(x) / (1024.0 * 1024.0)

def get_process_rss_mb() -> float:
    p = psutil.Process(os.getpid())
    return _bytes_to_mb(p.memory_info().rss)

def cuda_mem_mb() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {"alloc": 0.0, "reserved": 0.0, "peak_alloc": 0.0, "peak_reserved": 0.0}
    torch.cuda.synchronize()
    return {
        "alloc": _bytes_to_mb(torch.cuda.memory_allocated()),
        "reserved": _bytes_to_mb(torch.cuda.memory_reserved()),
        "peak_alloc": _bytes_to_mb(torch.cuda.max_memory_allocated()),
        "peak_reserved": _bytes_to_mb(torch.cuda.max_memory_reserved()),
    }

class PerfTimer:
    def __init__(self, device: torch.device):
        self.device = device
        self.t0: float = 0.0
        self.t1: float = 0.0

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.t1 = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        return (self.t1 - self.t0) * 1000.0

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
    mapping: Optional[Dict[int, int]] = None,
) -> Tuple[torch.Tensor, Dict[str, Dict[int, int]]]:
    if mode == "encode":
        src, dst = edge_index.tolist()
        unique_nodes: List[int] = sorted(set(src) | set(dst))

        encode_map: Dict[int, int] = {nid: i for i, nid in enumerate(unique_nodes)}
        decode_map: Dict[int, int] = {i: nid for nid, i in encode_map.items()}

        src_m = [encode_map[i] for i in src]
        dst_m = [encode_map[i] for i in dst]

        new_edge_index = torch.tensor(
            [src_m, dst_m],
            dtype=edge_index.dtype,
            device=edge_index.device,
        )
        return new_edge_index, {"encode": encode_map, "decode": decode_map}

    if mode == "decode":
        if mapping is None:
            raise ValueError("`mapping` must be provided in decode mode.")
        src, dst = edge_index.tolist()
        src_m = [mapping[i] for i in src]
        dst_m = [mapping[i] for i in dst]
        new_edge_index = torch.tensor(
            [src_m, dst_m],
            dtype=edge_index.dtype,
            device=edge_index.device,
        )
        return new_edge_index, {}

    raise ValueError(f"Unsupported mode: {mode}")

@torch.no_grad()
def extract_pair_embeddings_no_rollout(med_diff, memory, neighbor_loader, assoc, device, loader) -> np.ndarray:
    med_diff.eval()
    memory.eval()
    all_pairs = []
    for batch in loader:
        src, dst, t = batch.src.to(device), batch.dst.to(device), batch.t.to(device)
        n_id = torch.cat([src, dst]).unique()
        n_id, _, _ = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)
        z_mem, _ = memory(n_id)
        ed, _ = remap_nodes(torch.stack((src, dst), dim=0), mode="encode")
        norm_factor, ed = compute_norm_and_edges(ed, num_nodes=len(z_mem), add_self_loops=False)
        out = med_diff(z_mem, ed.to(device), norm_factor.to(device), t=t)
        z = out[0] if isinstance(out, (tuple, list)) else out
        pair_emb = torch.cat([z[assoc[src]], z[assoc[dst]]], dim=-1)
        all_pairs.append(pair_emb.cpu())
    return torch.cat(all_pairs, dim=0).numpy()

@torch.no_grad()
def extract_pair_embeddings_with_rollout(
    med_diff: torch.nn.Module,
    memory: TGNMemory,
    neighbor_loader: LastNeighborLoader,
    assoc: torch.Tensor,
    device: torch.device,
    loader: TemporalDataLoader,
) -> np.ndarray:
    med_diff.eval()
    memory.eval()

    all_pairs: List[torch.Tensor] = []

    for batch in loader:
        batch = batch.to(device)
        src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        n_id = torch.cat([src, dst]).unique()
        n_id, _, _ = neighbor_loader(n_id)

        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z_mem, _ = memory(n_id)

        ed, _ = remap_nodes(torch.stack((src, dst), dim=0), mode="encode")
        norm_factor, ed = compute_norm_and_edges(ed, num_nodes=len(z_mem), add_self_loops=False)

        out = med_diff(z_mem, ed.to(device), norm_factor.to(device), t=t)
        z = out[0] if isinstance(out, (tuple, list)) else out

        pair_emb = torch.cat([z[assoc[src]], z[assoc[dst]]], dim=-1)
        all_pairs.append(pair_emb.cpu())

        memory.update_state(src, dst, t, msg)
        neighbor_loader.insert(src, dst)
        memory.detach()

    return torch.cat(all_pairs, dim=0).numpy()

def split_train_val_test(datax, train_ratio: float = 0.7, val_ratio: float = 0.1):
    n = len(datax.t)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_d = datax[:n_train]
    val_d = datax[n_train : n_train + n_val]
    test_d = datax[n_train + n_val :]
    return train_d, val_d, test_d

def resolve_pth_path(user_path: str) -> str:
    if user_path.endswith(".pth"):
        return user_path
    if os.path.exists(user_path + ".pth"):
        return user_path + ".pth"
    raise ValueError(f"--ckpt must be a .pth file (got: {user_path})")

def stratified_subsample_indices(y: np.ndarray, frac: float, seed: int = 42) -> np.ndarray:
    y = np.asarray(y)
    n = len(y)

    if frac >= 1.0:
        return np.arange(n)

    n_sub = max(2, int(n * frac))
    classes = np.unique(y)
    n_sub = max(n_sub, len(classes))

    if n_sub >= n:
        return np.arange(n)

    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_sub, random_state=seed)
    idx_sub, _ = next(sss.split(np.zeros(n), y))
    return idx_sub

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--neighbor_size", type=int, default=20)
    p.add_argument("--use_scaler", action="store_true")
    p.add_argument("--train_frac", type=float, default=0.01, help="fraction of TRAIN embeddings to fit LR (stratified)")
    p.add_argument("--thr", type=float, default=None, help="manual decision threshold. If not set, use ckpt best_threshold.")

    p.add_argument("--vis_sample", type=int, default=5000, help="Number of samples for t-SNE visualization")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("✅ device:", device)

    rss0 = get_process_rss_mb()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    cuda0 = cuda_mem_mb()
    print(f"🧠 Host RSS baseline: {rss0:.2f} MB")
    if device.type == "cuda":
        print(f"🎮 CUDA baseline (MB): alloc={cuda0['alloc']:.2f}, reserved={cuda0['reserved']:.2f}")

    ckpt_path = resolve_pth_path(args.ckpt)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_thr = float(ckpt.get("best_threshold", 0.5))
    thr = ckpt_thr if args.thr is None else float(args.thr)

    hparams = ckpt.get("hparams", {})
    hidden_dim = int(hparams.get("hidden_dim", 128))
    dropout = float(hparams.get("dropout", 0.2))

    print(f"✅ ckpt: {os.path.basename(ckpt_path)} | epoch={ckpt.get('epoch')} best_val_f1={ckpt.get('best_val_f1')}")
    print(f"🎯 threshold: {thr:.4f} ({'ckpt' if args.thr is None else 'manual'})")
    print(f"🧩 med_diff: hidden_dim={hidden_dim} dropout={dropout}")
    print(f"🧪 LR train frac (stratified): {args.train_frac}")

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data not found: {args.data}")

    data_all = torch.load(args.data, weights_only=False)
    if "attack" not in data_all:
        raise KeyError("Dataset has no 'attack' field.")
    data_all["attack"] = (data_all["attack"] != 0).long()

    num_nodes = int(max(data_all.src.max(), data_all.dst.max()).item()) + 1
    num_features = data_all.msg.size(-1)

    train_data, _, test_data = split_train_val_test(data_all, args.train_ratio, args.val_ratio)
    train_labels = data_all["attack"][: len(train_data)].cpu().numpy()
    test_labels = data_all["attack"][-len(test_data) :].cpu().numpy()

    train_loader = TemporalDataLoader(train_data, batch_size=args.batch_size, num_workers=0)
    test_loader = TemporalDataLoader(test_data, batch_size=args.batch_size, num_workers=0)

    neighbor_loader = LastNeighborLoader(num_nodes, size=args.neighbor_size, device=device)
    memory_dim = time_dim = 64

    memory = TGNMemory(
        num_nodes,
        num_features,
        memory_dim,
        time_dim,
        message_module=IdentityMessage(num_features, memory_dim, time_dim),
        aggregator_module=LastAggregator(),
    ).to(device)

    med_diff = Med_Diff(
        in_channels=memory_dim,
        hidden_channels=hidden_dim,
        out_channels=hidden_dim,
        num_layers=2,
        alpha=0.2,
        iter_nums=(6, 2),
        dropout_imp=dropout,
        dropout_exp=dropout,
        lambda_s=1.0,
        lambda_r=1.0,
    ).to(device)

    if "med_diff_state" not in ckpt or "memory_state" not in ckpt:
        raise KeyError("Checkpoint must contain keys: 'med_diff_state' and 'memory_state'.")

    med_diff.load_state_dict(ckpt["med_diff_state"])
    memory.load_state_dict(ckpt["memory_state"])

    memory.reset_state()
    neighbor_loader.reset_state()
    assoc = torch.empty(num_nodes, dtype=torch.long, device=device)

    print("⏳ Rolling out Training data (Updating Memory)...")
    Z_train_full = extract_pair_embeddings_with_rollout(med_diff, memory, neighbor_loader, assoc, device, train_loader)

    print("🧪 Extracting Test data (Static Memory)...")
    Z_test_full = extract_pair_embeddings_no_rollout(med_diff, memory, neighbor_loader, assoc, device, test_loader)

    print("\n🎨 Generating t-SNE visualization...")
    try:

        n_test_samples = len(test_labels)
        if n_test_samples > args.vis_sample:
            print(f"Sampling {args.vis_sample} out of {n_test_samples} for visualization...")

            vis_indices = np.random.RandomState(42).choice(n_test_samples, args.vis_sample, replace=False)
            Z_vis = Z_test_full[vis_indices]
            y_vis = test_labels[vis_indices]
        else:
            Z_vis = Z_test_full
            y_vis = test_labels

        print("Running t-SNE (this might take a moment)...")
        tsne = TSNE(n_components=2, init='pca', learning_rate='auto', perplexity=30, random_state=42)
        Z_tsne = tsne.fit_transform(Z_vis)

        plt.figure(figsize=(10, 8), dpi=150)

        normal_idx = np.where(y_vis == 0)[0]
        attack_idx = np.where(y_vis == 1)[0]

        plt.scatter(Z_tsne[attack_idx, 0], Z_tsne[attack_idx, 1],
                    c='red', marker='^', s=15, alpha=0.15, label='Anomaly (Attack)', edgecolors='none')

        plt.scatter(Z_tsne[normal_idx, 0], Z_tsne[normal_idx, 1],
                    c='green', marker='o', s=15, alpha=0.9, label='Normal', edgecolors='none')

        plt.title("t-SNE Visualization of Med_Diff Embeddings (No Classifier)", fontsize=14)
        plt.xlabel("Dim 1", fontsize=10)
        plt.ylabel("Dim 2", fontsize=10)
        plt.legend(fontsize=10, loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()

        ckpt_name = os.path.splitext(os.path.basename(args.ckpt))[0]
        output_fig_path = f"tsne_vis_{ckpt_name}.png"
        plt.savefig(output_fig_path)
        plt.close()
        print(f"✅ Visualization saved to: {output_fig_path}")

    except Exception as e:
        print(f"⚠️ Visualization failed: {e}")
        traceback.print_exc()

    rss_before_lr = get_process_rss_mb()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    idx_sub = stratified_subsample_indices(train_labels, frac=args.train_frac, seed=42)
    Z_train_sub = Z_train_full[idx_sub]
    y_train_sub = train_labels[idx_sub]

    if args.use_scaler:
        scaler = StandardScaler()

        scaler.fit(Z_train_full)
        Z_train_input = scaler.transform(Z_train_sub)
        Z_test_input = scaler.transform(Z_test_full)
    else:
        Z_train_input = Z_train_sub
        Z_test_input = Z_test_full

    clf = LogisticRegression(
        max_iter=4000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42
    )

    print(f"🚀 Fitting LR probe on {len(y_train_sub)} samples...")
    with PerfTimer(device) as tt:
        clf.fit(Z_train_input, y_train_sub)
    lr_fit_ms = tt.elapsed_ms

    with PerfTimer(device) as tt:
        y_prob_test = clf.predict_proba(Z_test_input)[:, 1]
    lr_pred_ms = tt.elapsed_ms

    rss_after_lr = get_process_rss_mb()
    cuda_after_lr = cuda_mem_mb()

    y_pred_test = (y_prob_test >= thr).astype(int)
    acc = accuracy_score(test_labels, y_pred_test)
    prec = precision_score(test_labels, y_pred_test, zero_division=0)
    rec = recall_score(test_labels, y_pred_test, zero_division=0)
    f1_bin = f1_score(test_labels, y_pred_test, average="binary", zero_division=0)
    f1_mac = f1_score(test_labels, y_pred_test, average="macro", zero_division=0)
    cm = confusion_matrix(test_labels, y_pred_test)

    print("\n⏱️ LR probe latency:")
    print(f"    - fit     : {lr_fit_ms:.2f} ms")
    print(f"    - predict : {lr_pred_ms:.2f} ms (N_test={len(test_labels)}, {lr_pred_ms/max(1,len(test_labels)):.6f} ms/sample)")

    print("\n🧠 Memory usage (LR section):")
    print(f"    - Host RSS before: {rss_before_lr:.2f} MB")
    print(f"    - Host RSS after : {rss_after_lr:.2f} MB (delta={rss_after_lr - rss_before_lr:.2f} MB)")
    if device.type == "cuda":
        print(f"    - CUDA current (MB): alloc={cuda_after_lr['alloc']:.2f}, reserved={cuda_after_lr['reserved']:.2f}")
        print(f"    - CUDA peak    (MB): peak_alloc={cuda_after_lr['peak_alloc']:.2f}, peak_reserved={cuda_after_lr['peak_reserved']:.2f}")

    print("\n" + "=" * 60)
    print("✅ FINAL TEST (LR on stratified subset of TRAIN embeddings)")
    print(f"[FINAL] thr={thr:.2f} | ACC={acc:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | F1={f1_bin:.4f} | F1-mac={f1_mac:.4f}")
    print("[FINAL] Confusion Matrix:\n", cm)
    print("=" * 60)

    del med_diff, memory, neighbor_loader, clf
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Failed: {e}")
        traceback.print_exc()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
