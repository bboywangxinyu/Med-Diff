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

from mpl_toolkits.mplot3d import Axes3D
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

def split_train_val_test(datax, train_ratio: float = 0.8, val_ratio: float = 0.1):
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
    n_sub = max(2, int(n * frac))
    classes = np.unique(y)
    n_sub = max(n_sub, len(classes))

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
    p.add_argument("--vis_sample_size", type=int, default=2000, help="Max samples to use for PCA visualization (to save time)")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(" device:", device)

    rss0 = get_process_rss_mb()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    cuda0 = cuda_mem_mb()
    print(f" Host RSS baseline: {rss0:.2f} MB")
    if device.type == "cuda":
        print(f" CUDA baseline (MB): alloc={cuda0['alloc']:.2f}, reserved={cuda0['reserved']:.2f}")

    ckpt_path = resolve_pth_path(args.ckpt)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_thr = float(ckpt.get("best_threshold", 0.5))
    thr = ckpt_thr if args.thr is None else float(args.thr)

    hparams = ckpt.get("hparams", {})
    hidden_dim = int(hparams.get("hidden_dim", 128))
    dropout = float(hparams.get("dropout", 0.2))

    print(f" ckpt: {os.path.basename(ckpt_path)} | epoch={ckpt.get('epoch')} best_val_f1={ckpt.get('best_val_f1')}")
    print(f" threshold: {thr:.4f} ({'ckpt' if args.thr is None else 'manual'})")
    print(f" med_diff: hidden_dim={hidden_dim} dropout={dropout}")
    print(f" LR train frac (stratified): {args.train_frac}")

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data not found: {args.data}")

    data_all = torch.load(args.data, weights_only=False)
    if "attack" not in data_all:
        raise KeyError("Dataset has no 'attack' field.")
    data_all["attack"] = (data_all["attack"] != 2).long()

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

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    rss_before_embed = get_process_rss_mb()
    with PerfTimer(device) as tt:
        Z_train_full = extract_pair_embeddings_with_rollout(med_diff, memory, neighbor_loader, assoc, device, train_loader)
    t_train_embed_ms = tt.elapsed_ms
    rss_after_train = get_process_rss_mb()

    with PerfTimer(device) as tt:

        Z_test = extract_pair_embeddings_no_rollout(med_diff, memory, neighbor_loader, assoc, device, test_loader)
    t_test_embed_ms = tt.elapsed_ms
    rss_after_test = get_process_rss_mb()
    cuda_after_embed = cuda_mem_mb()

    print("\n Generating t-SNE visualizations (2D and 3D)...")

    benign_all_indices = np.where(test_labels == 0)[0]
    attack_all_indices = np.where(test_labels == 1)[0]

    target_per_class = args.vis_sample_size // 2

    n_samples = min(target_per_class, len(benign_all_indices), len(attack_all_indices))

    print(f"  Sampling balanced 1:1 dataset for Vis: {n_samples} Benign + {n_samples} Attack (Total {n_samples*2})")

    benign_vis_idx = np.random.choice(benign_all_indices, n_samples, replace=False)
    attack_vis_idx = np.random.choice(attack_all_indices, n_samples, replace=False)

    vis_indices = np.concatenate([benign_vis_idx, attack_vis_idx])
    np.random.shuffle(vis_indices)

    Z_vis = Z_test[vis_indices]
    y_vis = test_labels[vis_indices]

    normal_idx = np.where(y_vis == 0)[0]
    attack_idx = np.where(y_vis == 1)[0]

    ckpt_name = os.path.splitext(os.path.basename(args.ckpt))[0]

    print("\n Generating t-SNE visualizations (2D and 3D)...")

    benign_all_indices = np.where(test_labels == 0)[0]
    attack_all_indices = np.where(test_labels == 1)[0]

    target_per_class = args.vis_sample_size // 2

    n_samples = min(target_per_class, len(benign_all_indices), len(attack_all_indices))

    print(f"  Sampling balanced 1:1 dataset for Vis: {n_samples} Benign + {n_samples} Attack (Total {n_samples*2})")

    benign_vis_idx = np.random.choice(benign_all_indices, n_samples, replace=False)
    attack_vis_idx = np.random.choice(attack_all_indices, n_samples, replace=False)

    vis_indices = np.concatenate([benign_vis_idx, attack_vis_idx])
    np.random.shuffle(vis_indices)

    Z_vis = Z_test[vis_indices]
    y_vis = test_labels[vis_indices]

    normal_idx = np.where(y_vis == 0)[0]
    attack_idx = np.where(y_vis == 1)[0]

    ckpt_name = os.path.splitext(os.path.basename(args.ckpt))[0]

    print("Computing 2D t-SNE (slower than PCA, please wait)...")

    plt.rcParams['font.family'] = 'serif'

    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'Liberation Serif', 'serif']

    plt.rcParams['axes.unicode_minus'] = False

    tsne_2d = TSNE(n_components=2, init='pca', learning_rate='auto', perplexity=30, random_state=42)
    Z_tsne_2d = tsne_2d.fit_transform(Z_vis)

    plt.figure(figsize=(8, 6), dpi=600)

    plt.scatter(Z_tsne_2d[attack_idx, 0], Z_tsne_2d[attack_idx, 1],
                c='#be4d59', marker='o', s=50, alpha=0.4, label='Attack', edgecolors='none')

    plt.scatter(Z_tsne_2d[normal_idx, 0], Z_tsne_2d[normal_idx, 1],
                c='#386e66', marker='o', s=50, alpha=0.4, label='Benign', edgecolors='none')

    plt.xlabel("")
    plt.ylabel("")

    plt.tick_params(axis='both', which='major', labelsize=22, direction='in')

    font_legend = {'family': 'serif', 'size': 24}

    plt.legend(loc='lower left', ncol=2, prop=font_legend, frameon=True, edgecolor='black', fancybox=False, markerscale=3.0)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()

    save_dir = "NF-UNSW-NB15"
    os.makedirs(save_dir, exist_ok=True)

    output_fig_path_2d = os.path.join(save_dir, f"tsne_2d_{ckpt_name}.png")
    plt.savefig(output_fig_path_2d, bbox_inches='tight')
    plt.close()
    print(f" 2D Visualization saved to: {output_fig_path_2d}")

    print("Computing 3D t-SNE (and generating a web page with 600 DPI export)...")
    import plotly.graph_objects as go

    tsne_3d = TSNE(n_components=3, init='pca', learning_rate='auto', perplexity=30, random_state=42)
    Z_tsne_3d = tsne_3d.fit_transform(Z_vis)

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=Z_tsne_3d[normal_idx, 0],
        y=Z_tsne_3d[normal_idx, 1],
        z=Z_tsne_3d[normal_idx, 2],
        mode='markers',
        name='Benign',
        marker=dict(
            size=2,
            color='#386e66',
            opacity=0.3,
            line=dict(width=0)
        )
    ))

    fig.add_trace(go.Scatter3d(
        x=Z_tsne_3d[attack_idx, 0],
        y=Z_tsne_3d[attack_idx, 1],
        z=Z_tsne_3d[attack_idx, 2],
        mode='markers',
        name='Attack',
        marker=dict(
            size=2,
            color='#be4d59',
            opacity=0.3,
            line=dict(width=0)
        )
    ))

    fig.update_layout(
        title={
            'text': "",
        },
        scene=dict(
            bgcolor='white',

            xaxis=dict(
                title='',
                showline=True,
                linewidth=2,
                linecolor='black',
                mirror=True,

                showbackground=True,
                backgroundcolor='#fbfbfc',

                gridcolor="#3c444c",
                gridwidth=2,

                tickfont=dict(size=14, color='black', family='Times New Roman'),
            ),

            yaxis=dict(
                title='',
                showline=True,
                linewidth=2,
                linecolor='black',
                mirror=True,

                showbackground=True,
                backgroundcolor='#fbfbfc',

                gridcolor="#3c444c",
                gridwidth=2,
                tickfont=dict(size=14, color='black', family='Times New Roman'),
            ),

            zaxis=dict(
                title='',
                showline=True,
                linewidth=2,
                linecolor='black',
                mirror=True,

                showbackground=True,
                backgroundcolor='#fbfbfc',

                gridcolor="#3c444c",
                gridwidth=2,

                tickfont=dict(size=14, color='black', family='Times New Roman'),
            ),

            aspectmode='cube'
        ),

        legend=dict(
            yanchor="bottom", y=0.15,
            xanchor="left", x=0.15,
            orientation="h",
            bgcolor="rgba(255,255,255,0)",
            font=dict(size=14, family='Times New Roman')
        ),

        margin=dict(l=0, r=0, b=0, t=0),
        font=dict(family="Times New Roman", size=18)
    )

    high_res_config = {
        'toImageButtonOptions': {
            'format': 'png',
            'filename': f'high_res_tsne_{ckpt_name}',
            'height': 1200,
            'width': 1200,
            'scale': 5
        },
        'displaylogo': False
    }

    os.makedirs(save_dir, exist_ok=True)

    output_html_path = os.path.join(save_dir, f"high_res_interactive_{ckpt_name}.html")

    fig.write_html(output_html_path, config=high_res_config, auto_open=False)

    print(f" [600 DPI config] Interactive web page saved: {output_html_path}")
    print(f" [600 DPI config] Interactive web page saved: {output_html_path}")
    print(" Steps: Download -> open in browser -> adjust angle -> click the camera icon in the top right.")
    print("  Note: After clicking the camera, the browser may lag for 1-2 seconds because it is generating a 6000x6000 image!")

    print("\n Embedding extraction latency:")
    print(f"    - train rollout: {t_train_embed_ms:.2f} ms")
    print(f"    - test  rollout: {t_test_embed_ms:.2f} ms")
    print(f"    - total embed  : {t_train_embed_ms + t_test_embed_ms:.2f} ms")

    print("\n Memory usage (embedding extraction):")
    print(f"    - Host RSS before: {rss_before_embed:.2f} MB")
    print(f"    - Host RSS after : {rss_after_test:.2f} MB (delta={rss_after_test - rss_before_embed:.2f} MB)")
    if device.type == "cuda":
        print("    - CUDA current (MB): "
              f"alloc={cuda_after_embed['alloc']:.2f}, reserved={cuda_after_embed['reserved']:.2f}")
        print("    - CUDA peak    (MB): "
              f"peak_alloc={cuda_after_embed['peak_alloc']:.2f}, peak_reserved={cuda_after_embed['peak_reserved']:.2f}")

    idx_sub = stratified_subsample_indices(train_labels, frac=args.train_frac, seed=42)
    Z_train = Z_train_full[idx_sub]
    y_train = train_labels[idx_sub]

    print("\n LR training samples (stratified subset):")
    print(f"    - {len(idx_sub)} / {len(train_labels)} = {len(idx_sub)/max(1,len(train_labels)):.4%}")
    uniq, cnt = np.unique(y_train, return_counts=True)
    print(f"    - label dist: {dict(zip(uniq.tolist(), cnt.tolist()))}")

    if args.use_scaler:
        scaler = StandardScaler()
        Z_train = scaler.fit_transform(Z_train)
        Z_test = scaler.transform(Z_test)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    rss_before_lr = get_process_rss_mb()
    clf = LogisticRegression(max_iter=4000, class_weight="balanced", solver="liblinear")

    with PerfTimer(device) as tt:
        clf.fit(Z_train, y_train)
    lr_fit_ms = tt.elapsed_ms

    with PerfTimer(device) as tt:
        y_prob_test = clf.predict_proba(Z_test)[:, 1]
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

    print("\n LR probe latency:")
    print(f"    - fit     : {lr_fit_ms:.2f} ms")
    print(f"    - predict : {lr_pred_ms:.2f} ms (N_test={len(test_labels)}, {lr_pred_ms/max(1,len(test_labels)):.6f} ms/sample)")
    print("\n Memory usage (LR section):")
    print(f"    - Host RSS before: {rss_before_lr:.2f} MB")
    print(f"    - Host RSS after : {rss_after_lr:.2f} MB (delta={rss_after_lr - rss_before_lr:.2f} MB)")
    if device.type == "cuda":
        print("    - CUDA current (MB): "
              f"alloc={cuda_after_lr['alloc']:.2f}, reserved={cuda_after_lr['reserved']:.2f}")
        print("    - CUDA peak    (MB): "
              f"peak_alloc={cuda_after_lr['peak_alloc']:.2f}, peak_reserved={cuda_after_lr['peak_reserved']:.2f}")

    print("\n" + "=" * 60)
    print(" FINAL TEST (LR on stratified subset of TRAIN embeddings)")
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
        print(f" Failed: {e}")
        traceback.print_exc()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
