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

def compute_norm_and_edges(edge_index, num_nodes=None, add_self_loops=False, symmetric_cut=False):
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

def remap_nodes(edge_index, mode="encode", mapping=None):
    if mode == "encode":
        src, dst = edge_index.tolist()
        unique_nodes = sorted(set(src) | set(dst))
        encode_map = {nid: i for i, nid in enumerate(unique_nodes)}
        decode_map = {i: nid for nid, i in encode_map.items()}
        src_m = [encode_map[i] for i in src]
        dst_m = [encode_map[i] for i in dst]
        return torch.tensor([src_m, dst_m], dtype=edge_index.dtype, device=edge_index.device), {"encode": encode_map, "decode": decode_map}
    if mode == "decode":
        src, dst = edge_index.tolist()
        src_m = [mapping[i] for i in src]
        dst_m = [mapping[i] for i in dst]
        return torch.tensor([src_m, dst_m], dtype=edge_index.dtype, device=edge_index.device), {}
    raise ValueError(f"Unsupported mode: {mode}")

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

@torch.no_grad()
def extract_pair_embeddings_with_rollout(med_diff, memory, neighbor_loader, assoc, device, loader) -> np.ndarray:
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

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--neighbor_size", type=int, default=20)
    p.add_argument("--use_scaler", action="store_true")
    p.add_argument("--train_frac", type=float, default=0.01)
    p.add_argument("--thr", type=float, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("✅ device:", device)

    if not os.path.exists(args.ckpt): raise FileNotFoundError(args.ckpt)
    ckpt = torch.load(args.ckpt, map_location="cpu")

    if not os.path.exists(args.data): raise FileNotFoundError(args.data)
    data_all = torch.load(args.data, weights_only=False)
    data_all["attack"] = (data_all["attack"] != 2).long()

    num_nodes = int(max(data_all.src.max(), data_all.dst.max()).item()) + 1
    num_features = data_all.msg.size(-1)

    n = len(data_all.t)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    train_data = data_all[:n_train]
    test_data = data_all[n_train + n_val :]

    train_labels = data_all["attack"][:n_train].cpu().numpy()
    test_labels = data_all["attack"][n_train + n_val :].cpu().numpy()

    train_loader = TemporalDataLoader(train_data, batch_size=args.batch_size, num_workers=0)
    test_loader = TemporalDataLoader(test_data, batch_size=args.batch_size, num_workers=0)

    neighbor_loader = LastNeighborLoader(num_nodes, size=args.neighbor_size, device=device)
    memory = TGNMemory(num_nodes, num_features, 64, 64,
                       message_module=IdentityMessage(num_features, 64, 64),
                       aggregator_module=LastAggregator()).to(device)
    med_diff = Med_Diff(64, 128, 128, 2, 0.2, (6, 2)).to(device)

    med_diff.load_state_dict(ckpt["med_diff_state"])
    memory.load_state_dict(ckpt["memory_state"])

    memory.reset_state()
    neighbor_loader.reset_state()
    assoc = torch.empty(num_nodes, dtype=torch.long, device=device)

    print("⏳ Rolling out Training data...")
    Z_train_full = extract_pair_embeddings_with_rollout(med_diff, memory, neighbor_loader, assoc, device, train_loader)

    print("🧪 Extracting Test data...")
    Z_test_full = extract_pair_embeddings_no_rollout(med_diff, memory, neighbor_loader, assoc, device, test_loader)

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

    clf = LogisticRegression(max_iter=4000, class_weight="balanced", solver="liblinear", random_state=42)
    print(f"🚀 Fitting LR probe on {len(y_train_sub)} samples...")
    with PerfTimer(device) as tt:
        clf.fit(Z_train_input, y_train_sub)
    lr_fit_ms = tt.elapsed_ms

    with PerfTimer(device) as tt:
        y_prob_test = clf.predict_proba(Z_test_input)[:, 1]
    lr_pred_ms = tt.elapsed_ms

    rss_after_lr = get_process_rss_mb()
    cuda_after_lr = cuda_mem_mb()

    thr = (args.thr if args.thr is not None else 0.5)
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
