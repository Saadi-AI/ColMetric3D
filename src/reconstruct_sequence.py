#!/usr/bin/env python3
"""Metric 3D reconstruction from EndoMetric3D depth + pose checkpoints.

This script is intentionally independent of GT pose. It runs the model on adjacent
pairs, integrates the network-predicted relative poses, backprojects predicted
metric depth, and exports:
  * predicted depths (.npy and preview .png)
  * relative and global camera poses (.npy/.json)
  * fused colored point cloud (.ply, no Open3D required)
  * optional Open3D TSDF mesh if Open3D is installed and --tsdf is used

Recommended final usage:
  --ablation pose_full
  --depth_checkpoint checkpoints_c3vd/full_d2_msec/epoch_006.pth
  --pose_checkpoint  checkpoints_c3vd/full_pose_refine_from_msec/best.pth

All internal metric depth and pose translations are centimeters. PLY coordinates
are exported in centimeters by default to match the paper metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

try:
    import open3d as o3d  # optional
except Exception:  # pragma: no cover
    o3d = None

from options import get_opts
from datasets import C3VDDataset
from networks import EndoMetric3DModel


def move_to_device(batch: Dict, device: torch.device) -> Dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def _state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ["model_state", "model_state_dict", "state_dict", "model"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def _resolve_ckpt(path_or_name: str, ckpt_dir: str = "") -> Path:
    p = Path(path_or_name)
    candidates = []
    if path_or_name:
        candidates.append(p)
        if ckpt_dir and not p.is_absolute():
            candidates.append(Path(ckpt_dir) / path_or_name)
    if ckpt_dir:
        candidates.extend([Path(ckpt_dir) / "best.pth", Path(ckpt_dir) / "latest.pth"])
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Could not find checkpoint. Checked: {[str(c) for c in candidates]}")


def load_depth_pose_weights(model: torch.nn.Module, depth_checkpoint: str, pose_checkpoint: str = "", ckpt_dir: str = "") -> Dict:
    """Load depth checkpoint, then optionally overwrite pose-related tensors from pose checkpoint."""
    report = {}
    depth_p = _resolve_ckpt(depth_checkpoint, ckpt_dir)
    depth_ckpt = torch.load(depth_p, map_location="cpu")
    dsd = _state_dict(depth_ckpt)
    missing, unexpected = model.load_state_dict(dsd, strict=False)
    print(f"[Depth checkpoint] {depth_p} missing={len(missing)} unexpected={len(unexpected)}")
    report["depth_checkpoint"] = str(depth_p)
    report["depth_load_missing"] = len(missing)
    report["depth_load_unexpected"] = len(unexpected)

    copied = []
    if pose_checkpoint:
        pose_p = _resolve_ckpt(pose_checkpoint, "")
        pose_ckpt = torch.load(pose_p, map_location="cpu")
        psd = _state_dict(pose_ckpt)
        cur = model.state_dict()
        update = {}
        prefixes = ("pose_head", "artifact_bridge")
        for k, v in psd.items():
            if k in cur and any(k.startswith(pref) for pref in prefixes) and tuple(v.shape) == tuple(cur[k].shape):
                update[k] = v
                copied.append(k)
        cur.update(update)
        model.load_state_dict(cur, strict=True)
        print(f"[Pose checkpoint] {pose_p} copied_pose_tensors={len(copied)}")
        report["pose_checkpoint"] = str(pose_p)
        report["pose_tensors_copied"] = len(copied)
    return report


def invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def integrate_pose(C_w_t: np.ndarray, T_t_to_next_cm: np.ndarray, mode: str = "inv") -> np.ndarray:
    if mode == "inv":
        return C_w_t @ invert_T(T_t_to_next_cm)
    if mode == "direct":
        return C_w_t @ T_t_to_next_cm
    raise ValueError(mode)


def tensor_rgb01_to_uint8(img01: torch.Tensor) -> np.ndarray:
    arr = img01.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (arr * 255.0).round().astype(np.uint8)


def save_depth_preview(depth_cm: np.ndarray, out_png: Path, max_depth_cm: float) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    d = np.nan_to_num(depth_cm, nan=0.0, posinf=max_depth_cm, neginf=0.0)
    valid = d[d > 0]
    if valid.size:
        lo, hi = np.percentile(valid, [2, 98])
    else:
        lo, hi = 0.0, max_depth_cm
    if hi <= lo:
        hi = max_depth_cm
    x = np.clip((d - lo) / (hi - lo + 1e-6), 0, 1)
    Image.fromarray((x * 255).astype(np.uint8)).save(out_png)


def backproject_points(depth_cm: np.ndarray, rgb_u8: np.ndarray, K: np.ndarray, C_w_c: np.ndarray,
                       stride: int = 4, min_depth_cm: float = 0.05, max_depth_cm: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    H, W = depth_cm.shape
    ys = np.arange(0, H, stride)
    xs = np.arange(0, W, stride)
    xv, yv = np.meshgrid(xs, ys)
    z = depth_cm[yv, xv]
    valid = np.isfinite(z) & (z > min_depth_cm) & (z < max_depth_cm)
    if not np.any(valid):
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)
    x = xv[valid].astype(np.float64)
    y = yv[valid].astype(np.float64)
    zc = z[valid].astype(np.float64)
    X = (x - K[0, 2]) / K[0, 0] * zc
    Y = (y - K[1, 2]) / K[1, 1] * zc
    pts_c = np.stack([X, Y, zc, np.ones_like(zc)], axis=1)
    pts_w = (C_w_c @ pts_c.T).T[:, :3]
    cols = rgb_u8[yv[valid], xv[valid], :]
    return pts_w.astype(np.float32), cols.astype(np.uint8)


def write_ply_ascii(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert points.shape[0] == colors.shape[0]
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sequence", default="", help="C3VD sequence folder name, e.g. trans_t4_a. Empty = all sequences in dataset order.")
    parser.add_argument("--depth_checkpoint", default="", help="Best metric-depth checkpoint, e.g. full_d2_msec/epoch_006.pth")
    parser.add_argument("--pose_checkpoint", default="", help="Optional pose checkpoint. Pose tensors overwrite depth checkpoint pose tensors.")
    parser.add_argument("--ckpt", default="", help="Backward-compatible single checkpoint name if depth_checkpoint is omitted.")
    parser.add_argument("--output_dir", default="results/reconstruction/c3vd_sequence")
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--point_stride", type=int, default=4)
    parser.add_argument("--max_points", type=int, default=2000000)
    parser.add_argument("--integration_mode", default="inv", choices=["inv", "direct"])
    parser.add_argument("--save_every_depth", type=int, default=10)
    parser.add_argument("--tsdf", action="store_true", help="Optional Open3D TSDF mesh. Requires open3d.")
    parser.add_argument("--depth_trunc_m", type=float, default=0.12)
    parser.add_argument("--tsdf_voxel_m", type=float, default=0.0015)
    parser.add_argument("--tsdf_sdf_trunc_m", type=float, default=0.006)
    args, remaining = parser.parse_known_args()

    opt = get_opts(remaining)
    if str(opt.get("dataset", "c3vd")).lower() != "c3vd":
        raise NotImplementedError("reconstruct_sequence.py currently supports C3VD. Use predicted depths/poses export for other datasets.")
    opt["use_triplet_pose"] = False
    opt["batch_size"] = 1
    opt["num_workers"] = int(opt.get("num_workers", 2))

    depth_checkpoint = args.depth_checkpoint or args.ckpt
    if not depth_checkpoint:
        depth_checkpoint = str(Path(opt["ckpt_dir"]) / "best.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16
    use_amp = device.type == "cuda"
    out_dir = Path(args.output_dir)
    depth_dir = out_dir / "pred_depth"
    out_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    root = Path(opt.get("c3vd_test_root") or opt.get("c3vd_val_root") or opt.get("c3vd_root"))
    ds = C3VDDataset(str(root), opt, split="test", target_seq=args.sequence or None)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=opt["num_workers"], pin_memory=True)
    print(f"[Dataset] root={root} sequence={args.sequence or 'ALL'} pairs={len(ds)}")

    model = EndoMetric3DModel(opt).to(device)
    load_report = load_depth_pose_weights(model, depth_checkpoint, args.pose_checkpoint, opt.get("ckpt_dir", ""))
    model.eval()

    C_w_c = np.eye(4, dtype=np.float64)
    trajectory = [C_w_c.copy()]
    rel_poses = []
    all_points: List[np.ndarray] = []
    all_colors: List[np.ndarray] = []
    frame_ids = []

    # Optional TSDF setup.
    tsdf_vol = None
    intr_o3d = None
    if args.tsdf and o3d is not None:
        tsdf_vol = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=float(args.tsdf_voxel_m),
            sdf_trunc=float(args.tsdf_sdf_trunc_m),
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )
    elif args.tsdf:
        print("[WARN] --tsdf requested but Open3D is not installed. Skipping TSDF mesh.")

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="reconstruct")):
            if args.max_frames > 0 and i >= args.max_frames:
                break
            if i % max(1, args.stride) != 0:
                continue
            batch = move_to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
                outputs = model(batch)
            if "metric_depth" not in outputs or "pose_matrix" not in outputs:
                raise RuntimeError("Model output needs both metric_depth and pose_matrix. Use --ablation pose_full and load depth+pose checkpoints.")
            depth_cm = F.interpolate(outputs["metric_depth"], size=batch["image_01"].shape[-2:], mode="bilinear", align_corners=False)[0, 0].float().cpu().numpy()
            rgb = tensor_rgb01_to_uint8(batch["image_01"][0])
            K = batch["K"][0].float().cpu().numpy()
            fid = int(batch["frame_idx"][0].detach().cpu().item()) if "frame_idx" in batch else i
            frame_ids.append(fid)

            pts, cols = backproject_points(depth_cm, rgb, K, C_w_c, stride=max(1, args.point_stride),
                                           min_depth_cm=float(opt.get("min_depth_cm", 0.05)), max_depth_cm=float(opt.get("max_depth_cm", 10.0)))
            if pts.size:
                all_points.append(pts); all_colors.append(cols)
            if i % max(1, args.save_every_depth) == 0:
                np.save(depth_dir / f"depth_cm_{i:05d}_frame{fid}.npy", depth_cm.astype(np.float32))
                save_depth_preview(depth_cm, depth_dir / f"depth_preview_{i:05d}_frame{fid}.png", float(opt.get("max_depth_cm", 10.0)))

            if tsdf_vol is not None:
                H, W = depth_cm.shape
                if intr_o3d is None:
                    intr_o3d = o3d.camera.PinholeCameraIntrinsic(W, H, float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
                depth_m = (depth_cm * 0.01).astype(np.float32)
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(rgb), o3d.geometry.Image(depth_m), depth_scale=1.0,
                    depth_trunc=float(args.depth_trunc_m), convert_rgb_to_intensity=False)
                # Open3D expects camera extrinsic world->camera.
                tsdf_vol.integrate(rgbd, intr_o3d, np.linalg.inv(C_w_c))

            T_rel = outputs["pose_matrix"][0].float().cpu().numpy().astype(np.float64)
            rel_poses.append(T_rel)
            C_w_c = integrate_pose(C_w_c, T_rel, mode=args.integration_mode)
            trajectory.append(C_w_c.copy())

    if all_points:
        points = np.concatenate(all_points, axis=0)
        colors = np.concatenate(all_colors, axis=0)
        if args.max_points > 0 and points.shape[0] > args.max_points:
            rng = np.random.default_rng(123)
            idx = rng.choice(points.shape[0], size=args.max_points, replace=False)
            points, colors = points[idx], colors[idx]
        write_ply_ascii(out_dir / "fused_point_cloud_cm.ply", points, colors)
        print(f"[Saved] {out_dir / 'fused_point_cloud_cm.ply'} points={points.shape[0]}")
    else:
        print("[WARN] No valid 3D points were produced.")

    if tsdf_vol is not None:
        mesh = tsdf_vol.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(out_dir / "tsdf_mesh_m.ply"), mesh)
        print(f"[Saved] {out_dir / 'tsdf_mesh_m.ply'}")

    traj = np.stack(trajectory, axis=0)
    rel = np.stack(rel_poses, axis=0) if rel_poses else np.empty((0, 4, 4))
    np.save(out_dir / "trajectory_c2w_cm.npy", traj)
    np.save(out_dir / "pred_rel_poses_cm.npy", rel)
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump({
            "root": str(root), "sequence": args.sequence, "num_frames": len(trajectory), "num_rel_poses": len(rel_poses),
            "frame_ids": frame_ids, "integration_mode": args.integration_mode, "load_report": load_report,
            "coordinates": "PLY and trajectory translations are in centimeters unless filename says _m.",
        }, f, indent=2)
    print(f"[Saved] {out_dir}")


if __name__ == "__main__":
    main()
