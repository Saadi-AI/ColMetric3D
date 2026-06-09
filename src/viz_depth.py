#!/usr/bin/env python3
"""
Visualize EndoMetric3D metric depth predictions on C3VD.

Saves paper/repo-ready panels:
  RGB | GT depth (cm) | Pred depth (cm) | Absolute error (cm)

This script avoids the training parser conflict by parsing visualization
arguments first, then calling get_opts() only on remaining model/training args.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from checkpoint_utils import load_model_state_safely
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import open3d as o3d
    HAS_O3D = True
except Exception:
    HAS_O3D = False

from options import get_opts
from datasets import C3VDDataset
from networks.endometric3d_model import EndoMetric3DModel


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    v = str(v).lower().strip()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected")


def parse_eval_args() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="EndoMetric3D depth visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=True,
    )
    parser.add_argument("--c3vd_root", type=str, default=None)
    parser.add_argument("--c3vd_val_root", type=str, default=None)
    parser.add_argument("--c3vd_test_root", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints_endometric3d_joint_bf16")
    parser.add_argument("--ckpt", type=str, default="best.pth")
    parser.add_argument("--output_dir", type=str, default="results/qualitative_depth")
    parser.add_argument("--num_viz", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--shuffle", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_npy", type=str2bool, default=False)
    parser.add_argument("--save_pointcloud", type=str2bool, default=False)
    parser.add_argument("--dpi", type=int, default=250)
    parser.add_argument("--cmap", type=str, default="magma")
    parser.add_argument("--error_cmap", type=str, default="Reds")
    parser.add_argument("--min_depth_cm", type=float, default=0.1)
    parser.add_argument("--max_depth_cm", type=float, default=10.0)
    parser.add_argument("--error_vmax_cm", type=float, default=None)
    parser.add_argument("--amp", type=str2bool, default=True)

    # parse_known_args is important: any training/model args not listed above
    # are passed to options.get_opts().
    return parser.parse_known_args()


def safe_get_opts(remaining_args: List[str]) -> Dict[str, Any]:
    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0]] + remaining_args
        opt = get_opts()
    finally:
        sys.argv = old_argv
    return opt


def unnormalize_rgb(t: torch.Tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) -> np.ndarray:
    img = t.detach().float().cpu().numpy().transpose(1, 2, 0)
    img = img * np.asarray(std, dtype=np.float32) + np.asarray(mean, dtype=np.float32)
    return np.clip(img, 0.0, 1.0)


def load_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> None:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(str(ckpt_path), map_location=device)
    if isinstance(state, dict):
        if "model_state" in state:
            state_dict = state["model_state"]
        elif "model_state_dict" in state:
            state_dict = state["model_state_dict"]
        elif "model" in state:
            state_dict = state["model"]
        else:
            state_dict = state
    else:
        state_dict = state
    missing, unexpected, skipped, adapted = load_model_state_safely(model, state_dict, strict=False)
    print(f"[Checkpoint] loaded {ckpt_path}")
    print(f"[Checkpoint] missing={len(missing)} unexpected={len(unexpected)} skipped_shape_mismatch={len(skipped)} adapted_legacy_moe={len(adapted)}")
    if adapted:
        for k, msg in list(adapted.items())[:8]:
            print(f"  [adapt legacy] {k}: {msg}")
    if skipped:
        for k, (old_shape, new_shape) in list(skipped.items())[:8]:
            print(f"  [skip mismatch] {k}: ckpt={old_shape} model={new_shape}")


def compute_sample_metrics(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> Dict[str, float]:
    p = pred[valid].astype(np.float64)
    g = gt[valid].astype(np.float64)
    if p.size < 10:
        return {"abs_rel": float("nan"), "mae_cm": float("nan"), "rmse_cm": float("nan"), "delta1": float("nan")}
    abs_rel = np.mean(np.abs(p - g) / np.maximum(g, 1e-6))
    mae = np.mean(np.abs(p - g))
    rmse = np.sqrt(np.mean((p - g) ** 2))
    ratio = np.maximum(g / np.maximum(p, 1e-6), p / np.maximum(g, 1e-6))
    d1 = np.mean(ratio < 1.25)
    return {"abs_rel": float(abs_rel), "mae_cm": float(mae), "rmse_cm": float(rmse), "delta1": float(d1)}


def make_panel(
    rgb: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    err: np.ndarray,
    valid: np.ndarray,
    metrics: Dict[str, float],
    out_path: Path,
    cmap: str,
    error_cmap: str,
    min_depth: float,
    max_depth: float,
    error_vmax: float | None,
    dpi: int,
    title: str,
) -> None:
    gt_show = gt.copy()
    pred_show = pred.copy()
    err_show = err.copy()
    gt_show[~valid] = np.nan
    pred_show[~valid] = np.nan
    err_show[~valid] = np.nan

    # Use shared depth scale so GT/pred are visually comparable.
    if np.isfinite(gt_show).any():
        vmin = max(min_depth, float(np.nanpercentile(gt_show, 1)))
        vmax = min(max_depth, float(np.nanpercentile(gt_show, 99)))
        if vmax <= vmin:
            vmin, vmax = min_depth, max_depth
    else:
        vmin, vmax = min_depth, max_depth

    if error_vmax is None:
        evmax = float(np.nanpercentile(err_show, 95)) if np.isfinite(err_show).any() else 1.0
        evmax = max(evmax, 0.05)
    else:
        evmax = float(error_vmax)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(title, fontsize=13)

    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[0].axis("off")

    im1 = axes[1].imshow(gt_show, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title("GT Depth (cm)")
    axes[1].axis("off")

    im2 = axes[2].imshow(pred_show, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[2].set_title("EndoMetric3D (cm)")
    axes[2].axis("off")

    im3 = axes[3].imshow(err_show, cmap=error_cmap, vmin=0, vmax=evmax)
    axes[3].set_title(f"Abs Error (cm)\nAbsRel={metrics['abs_rel']:.3f}, MAE={metrics['mae_cm']:.3f}")
    axes[3].axis("off")

    cbar1 = fig.colorbar(im2, ax=axes[1:3], orientation="horizontal", fraction=0.046, pad=0.08)
    cbar1.set_label("Depth (cm)")
    cbar2 = fig.colorbar(im3, ax=axes[3], orientation="horizontal", fraction=0.046, pad=0.08)
    cbar2.set_label("Error (cm)")

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_pointcloud(rgb: np.ndarray, depth_cm: np.ndarray, K: np.ndarray, valid: np.ndarray, out_path: Path) -> None:
    if not HAS_O3D:
        print("[Open3D] not installed; skipping pointcloud")
        return
    h, w = depth_cm.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    z = depth_cm / 100.0  # cm -> m
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (xx - cx) * z / fx
    y = (yy - cy) * z / fy
    pts = np.stack([x, y, z], axis=-1)[valid]
    cols = rgb[valid]
    if pts.shape[0] == 0:
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
    o3d.io.write_point_cloud(str(out_path), pcd)


def main() -> None:
    args, remaining = parse_eval_args()
    opt = safe_get_opts(remaining)

    # Override options with visualization args.
    for key in ["c3vd_root", "c3vd_val_root", "c3vd_test_root", "batch_size", "num_workers", "min_depth_cm", "max_depth_cm"]:
        val = getattr(args, key, None)
        if val is not None:
            opt[key] = val
    opt["amp"] = args.amp

    if args.split == "test":
        root = args.c3vd_test_root or opt.get("c3vd_test_root") or opt.get("c3vd_val_root")
    elif args.split == "val":
        root = args.c3vd_val_root or opt.get("c3vd_val_root")
    else:
        root = args.c3vd_root or opt.get("c3vd_root")
    if root is None:
        raise ValueError("Dataset root is missing. Pass --c3vd_test_root/--c3vd_val_root/--c3vd_root.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    print(f"[Dataset] split={args.split} root={root}")

    ds = C3VDDataset(root, opt, split=args.split)
    n = len(ds)
    indices = list(range(n))
    if args.shuffle:
        random.shuffle(indices)
    indices = indices[: min(args.num_viz, n)]
    ds_viz = Subset(ds, indices)
    loader = DataLoader(ds_viz, batch_size=1, shuffle=False, num_workers=args.num_workers)

    model = EndoMetric3DModel(opt).to(device)
    ckpt_path = Path(args.ckpt_dir) / args.ckpt
    load_checkpoint(model, ckpt_path, device)
    model.eval()

    out_root = Path(args.output_dir)
    panel_dir = out_root / "panels"
    npy_dir = out_root / "npy"
    pc_dir = out_root / "pointclouds"
    panel_dir.mkdir(parents=True, exist_ok=True)
    if args.save_npy:
        npy_dir.mkdir(parents=True, exist_ok=True)
    if args.save_pointcloud:
        pc_dir.mkdir(parents=True, exist_ok=True)

    metrics_all = []
    use_amp = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="viz-depth")):
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(inputs)

            pred = outputs["metric_depth"].float()
            gt = inputs["depth_gt"].float()
            if pred.shape[-2:] != gt.shape[-2:]:
                pred = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)

            rgb_np = unnormalize_rgb(inputs["image"][0])
            gt_np = gt[0, 0].detach().cpu().numpy()
            pred_np = pred[0, 0].detach().cpu().numpy()
            err_np = np.abs(pred_np - gt_np)
            valid = np.isfinite(gt_np) & (gt_np > args.min_depth_cm) & (gt_np < args.max_depth_cm)

            metrics = compute_sample_metrics(pred_np, gt_np, valid)
            metrics["index"] = int(indices[i])
            # These fields are available if your dataset returns them; otherwise fallback.
            seq = batch.get("seq_name", ["unknown"])[0] if isinstance(batch, dict) else "unknown"
            metrics["seq_name"] = str(seq)
            metrics_all.append(metrics)

            name = f"sample_{i:03d}_idx{indices[i]:05d}_{seq}"
            title = f"{name} | AbsRel={metrics['abs_rel']:.3f}, RMSE={metrics['rmse_cm']:.3f} cm, δ1={metrics['delta1']:.3f}"
            make_panel(
                rgb_np,
                gt_np,
                pred_np,
                err_np,
                valid,
                metrics,
                panel_dir / f"{name}.png",
                args.cmap,
                args.error_cmap,
                args.min_depth_cm,
                args.max_depth_cm,
                args.error_vmax_cm,
                args.dpi,
                title,
            )

            if args.save_npy:
                np.save(npy_dir / f"{name}_pred_cm.npy", pred_np.astype(np.float32))
                np.save(npy_dir / f"{name}_gt_cm.npy", gt_np.astype(np.float32))
                np.save(npy_dir / f"{name}_error_cm.npy", err_np.astype(np.float32))

            if args.save_pointcloud:
                K = inputs["K"][0].detach().float().cpu().numpy() if "K" in inputs else np.eye(3, dtype=np.float32)
                save_pointcloud(rgb_np, pred_np, K, valid, pc_dir / f"{name}_pred.ply")

    with open(out_root / "visualization_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_all, f, indent=2)

    print(f"[Saved] panels: {panel_dir}")
    print(f"[Saved] metrics: {out_root / 'visualization_metrics.json'}")
    if args.save_npy:
        print(f"[Saved] npy: {npy_dir}")
    if args.save_pointcloud:
        print(f"[Saved] pointclouds: {pc_dir}")


if __name__ == "__main__":
    main()
