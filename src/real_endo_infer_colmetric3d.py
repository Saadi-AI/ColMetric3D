#!/usr/bin/env python3
"""
Run ColMetric3D / EndoMetric3D qualitative inference on real colonoscopy frames
without ground-truth depth.

What it does:
  1) reads image frames from a folder
  2) optionally undistorts with OpenCV calibration K + distortion coefficients
  3) crops the valid undistorted ROI / removes black borders
  4) resizes to model input resolution and updates K accordingly
  5) predicts metric depth + reliability mask
  6) optionally predicts pairwise pose for consecutive frames
  7) saves visualization panels, .npy depth maps, and optional point clouds

Place this script in your project root, next to options.py and networks/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from checkpoint_utils import load_model_state_safely
import torch.nn.functional as F
from PIL import Image
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
from networks.endometric3d_model import EndoMetric3DModel


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower().strip()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected")


def parse_args():
    p = argparse.ArgumentParser("ColMetric3D real-endoscopy qualitative inference")
    p.add_argument("--frames_dir", type=str, required=True, help="Folder containing extracted real colonoscopy frames.")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--ckpt_dir", type=str, required=True)
    p.add_argument("--ckpt", type=str, default="best.pth")

    # calibration: use either JSON or direct CLI values
    p.add_argument("--calib_json", type=str, default=None,
                   help='JSON with keys "K" and "dist". K is 3x3; dist is [k1,k2,p1,p2,k3,...].')
    p.add_argument("--K", type=float, nargs=9, default=None,
                   help="Original camera matrix values: fx 0 cx  0 fy cy  0 0 1")
    p.add_argument("--dist", type=float, nargs="*", default=None,
                   help="OpenCV distortion coefficients, e.g. k1 k2 p1 p2 k3.")

    # preprocessing
    p.add_argument("--undistort", type=str2bool, default=True)
    p.add_argument("--alpha", type=float, default=0.0,
                   help="OpenCV getOptimalNewCameraMatrix alpha. 0 removes black borders, 1 keeps full FOV.")
    p.add_argument("--crop_valid_roi", type=str2bool, default=True,
                   help="Crop the valid ROI returned by OpenCV after undistortion.")
    p.add_argument("--auto_crop_black", type=str2bool, default=False,
                   help="Additional threshold-based black-border crop after undistortion.")
    p.add_argument("--height", type=int, default=476)
    p.add_argument("--width", type=int, default=588)
    p.add_argument("--max_frames", type=int, default=-1)
    p.add_argument("--stride", type=int, default=1)

    # model/options
    p.add_argument("--amp", type=str2bool, default=True)
    p.add_argument("--min_depth_cm", type=float, default=0.05)
    p.add_argument("--max_depth_cm", type=float, default=10.0)
    p.add_argument("--run_pose", type=str2bool, default=False,
                   help="Also run pairwise PoseNet on consecutive frames if your checkpoint has pose weights.")
    p.add_argument("--save_npy", type=str2bool, default=True)
    p.add_argument("--save_pointcloud", type=str2bool, default=False)
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument("--device", type=str, default="cuda")

    # Remaining args are passed to your original options.py, e.g. --use_moe true --use_ada_depth true
    args, remaining = p.parse_known_args()
    return args, remaining


def load_calibration(args):
    if args.calib_json:
        data = json.loads(Path(args.calib_json).read_text())
        K = np.array(data["K"], dtype=np.float32).reshape(3, 3)
        dist = np.array(data.get("dist", data.get("distortion", [])), dtype=np.float32).reshape(-1)
        return K, dist
    if args.K is None:
        raise ValueError("Provide either --calib_json or --K.")
    K = np.array(args.K, dtype=np.float32).reshape(3, 3)
    dist = np.array(args.dist if args.dist is not None else [], dtype=np.float32).reshape(-1)
    return K, dist


def load_checkpoint(model, ckpt_path, device):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(str(ckpt_path), map_location=device)
    if isinstance(state, dict):
        sd = state.get("model_state", state.get("model_state_dict", state.get("model", state)))
    else:
        sd = state
    missing, unexpected, skipped, adapted = load_model_state_safely(model, sd, strict=False)
    print(f"[Checkpoint] loaded: {ckpt_path}")
    print(f"[Checkpoint] missing={len(missing)} unexpected={len(unexpected)} skipped_shape_mismatch={len(skipped)} adapted_legacy_moe={len(adapted)}")
    if adapted:
        for k, msg in list(adapted.items())[:8]:
            print(f"  [adapt legacy] {k}: {msg}")
    if skipped:
        for k, (old_shape, new_shape) in list(skipped.items())[:8]:
            print(f"  [skip mismatch] {k}: ckpt={old_shape} model={new_shape}")
    if len(missing) > 0:
        print("  first missing:", missing[:5])
    if len(unexpected) > 0:
        print("  first unexpected:", unexpected[:5])


def list_frames(frames_dir: Path, stride: int, max_frames: int) -> List[Path]:
    paths = sorted([p for p in frames_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])
    paths = paths[::max(1, stride)]
    if max_frames and max_frames > 0:
        paths = paths[:max_frames]
    if not paths:
        raise FileNotFoundError(f"No images found in {frames_dir}")
    return paths


def crop_black_border(img_bgr: np.ndarray, K: np.ndarray, thr: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = gray > thr
    ys, xs = np.where(mask)
    if len(xs) < 100 or len(ys) < 100:
        return img_bgr, K
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    cropped = img_bgr[y0:y1, x0:x1]
    K2 = K.copy()
    K2[0, 2] -= x0
    K2[1, 2] -= y0
    return cropped, K2


def undistort_and_resize(
    img_bgr: np.ndarray,
    K_orig: np.ndarray,
    dist: np.ndarray,
    out_hw: Tuple[int, int],
    undistort: bool = True,
    alpha: float = 0.0,
    crop_valid_roi: bool = True,
    auto_crop_black: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns RGB uint8 image resized to out_hw and the corresponding updated K.
    The returned K should be passed to the model.
    """
    h0, w0 = img_bgr.shape[:2]
    K_work = K_orig.astype(np.float32).copy()
    img_work = img_bgr

    if undistort and dist is not None and len(dist) > 0:
        newK, roi = cv2.getOptimalNewCameraMatrix(K_orig, dist, (w0, h0), alpha, (w0, h0))
        img_work = cv2.undistort(img_bgr, K_orig, dist, None, newK)
        K_work = newK.astype(np.float32)

        if crop_valid_roi:
            x, y, w, h = roi
            if w > 10 and h > 10:
                img_work = img_work[y:y+h, x:x+w]
                K_work[0, 2] -= x
                K_work[1, 2] -= y

    if auto_crop_black:
        img_work, K_work = crop_black_border(img_work, K_work)

    h1, w1 = img_work.shape[:2]
    out_h, out_w = out_hw
    sx = out_w / float(w1)
    sy = out_h / float(h1)

    img_resized = cv2.resize(img_work, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    K_resized = K_work.copy()
    K_resized[0, 0] *= sx
    K_resized[0, 1] *= sx
    K_resized[0, 2] *= sx
    K_resized[1, 1] *= sy
    K_resized[1, 2] *= sy

    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    return img_rgb, K_resized.astype(np.float32)


def to_model_tensor(img_rgb: np.ndarray, device: torch.device):
    img01 = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_norm = (img01 - mean) / std
    return img_norm.unsqueeze(0).to(device), img01.unsqueeze(0).to(device)


def colorize_depth(depth_cm: np.ndarray, vmin: float, vmax: float, cmap_name: str = "magma"):
    d = np.clip((depth_cm - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    cmap = plt.get_cmap(cmap_name)
    rgb = (cmap(d)[..., :3] * 255).astype(np.uint8)
    return rgb


def save_panel(img_rgb, depth_cm, reliability, out_path, min_depth, max_depth, title=""):
    depth_vis = colorize_depth(depth_cm, min_depth, max_depth, "magma")
    rel = np.clip(reliability, 0, 1)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(img_rgb); ax[0].set_title("Input frame"); ax[0].axis("off")
    ax[1].imshow(depth_vis); ax[1].set_title(r"Predicted metric depth $D_{\mathrm{MSEC}}$"); ax[1].axis("off")
    ax[2].imshow(rel, cmap="gray", vmin=0, vmax=1); ax[2].set_title(r"Reliability mask $M_r$"); ax[2].axis("off")
    if title:
        fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_pointcloud(img_rgb, depth_cm, K, out_path, min_depth_cm, max_depth_cm):
    if not HAS_O3D:
        print("[Open3D] not installed; skipping point cloud.")
        return
    h, w = depth_cm.shape
    valid = np.isfinite(depth_cm) & (depth_cm > min_depth_cm) & (depth_cm < max_depth_cm)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    z = depth_cm / 100.0
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (xx - cx) * z / fx
    y = (yy - cy) * z / fy
    pts = np.stack([x, y, z], axis=-1)[valid]
    cols = (img_rgb.astype(np.float32) / 255.0)[valid]
    if pts.shape[0] == 0:
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
    o3d.io.write_point_cloud(str(out_path), pcd)


def main():
    args, remaining = parse_args()

    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0]] + remaining
        opt = get_opts()
    finally:
        sys.argv = old_argv

    opt["min_depth_cm"] = args.min_depth_cm
    opt["max_depth_cm"] = args.max_depth_cm
    opt["amp"] = args.amp

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"[Device] {device}")

    K_orig, dist = load_calibration(args)
    print("[Calibration] K original:\n", K_orig)
    print("[Calibration] dist:", dist)

    model = EndoMetric3DModel(opt).to(device)
    load_checkpoint(model, Path(args.ckpt_dir) / args.ckpt, device)
    model.eval()

    frames = list_frames(Path(args.frames_dir), args.stride, args.max_frames)

    out_root = Path(args.output_dir)
    panel_dir = out_root / "panels"
    depth_dir = out_root / "depth_npy"
    rel_dir = out_root / "reliability_npy"
    prep_dir = out_root / "preprocessed_frames"
    pcd_dir = out_root / "pointclouds"
    pose_dir = out_root / "pose"
    for d in [panel_dir, depth_dir, rel_dir, prep_dir]:
        d.mkdir(parents=True, exist_ok=True)
    if args.save_pointcloud:
        pcd_dir.mkdir(parents=True, exist_ok=True)
    if args.run_pose:
        pose_dir.mkdir(parents=True, exist_ok=True)

    use_amp = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16
    out_hw = (args.height, args.width)

    last = None
    pose_records = []

    with torch.no_grad():
        for i, path in enumerate(tqdm(frames, desc="real inference")):
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"[Skip] failed to read {path}")
                continue

            img_rgb, K_resized = undistort_and_resize(
                bgr, K_orig, dist, out_hw,
                undistort=args.undistort,
                alpha=args.alpha,
                crop_valid_roi=args.crop_valid_roi,
                auto_crop_black=args.auto_crop_black,
            )

            image, image01 = to_model_tensor(img_rgb, device)
            K_t = torch.from_numpy(K_resized).float().unsqueeze(0).to(device)

            inputs = {"image": image, "image_01": image01, "K": K_t}
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(inputs)

            depth = out["metric_depth"].float()[0, 0].detach().cpu().numpy()
            reliability = out.get("tissue_mask", torch.ones_like(out["metric_depth"])).float()[0, 0].detach().cpu().numpy()

            stem = f"{i:05d}_{path.stem}"
            cv2.imwrite(str(prep_dir / f"{stem}.png"), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            if args.save_npy:
                np.save(depth_dir / f"{stem}_depth_cm.npy", depth.astype(np.float32))
                np.save(rel_dir / f"{stem}_reliability.npy", reliability.astype(np.float32))
                np.save(out_root / f"{stem}_K_resized.npy", K_resized.astype(np.float32))

            save_panel(
                img_rgb, depth, reliability,
                panel_dir / f"{stem}_panel.png",
                args.min_depth_cm, args.max_depth_cm,
                title=str(path.name),
            )

            if args.save_pointcloud:
                save_pointcloud(
                    img_rgb, depth, K_resized,
                    pcd_dir / f"{stem}.ply",
                    args.min_depth_cm, args.max_depth_cm,
                )

            if args.run_pose and last is not None:
                # PoseNet predicts from anchor last -> current, using predicted metric depth and reliability.
                pair_inputs = {
                    "image": last["image"],
                    "image_01": last["image01"],
                    "image_next": image,
                    "K": last["K"],
                }
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    pair_out = model(pair_inputs)
                if "pose_matrix" in pair_out:
                    T = pair_out["pose_matrix"].float()[0].detach().cpu().numpy()
                    np.save(pose_dir / f"{last['stem']}_to_{stem}_T.npy", T.astype(np.float32))
                    pose_records.append({"from": last["path"], "to": str(path), "T_npy": f"{last['stem']}_to_{stem}_T.npy"})

            last = {"image": image, "image01": image01, "K": K_t, "stem": stem, "path": str(path)}

    if args.run_pose:
        (pose_dir / "pose_pairs.json").write_text(json.dumps(pose_records, indent=2), encoding="utf-8")

    meta = {
        "frames_dir": str(args.frames_dir),
        "ckpt": str(Path(args.ckpt_dir) / args.ckpt),
        "height": args.height,
        "width": args.width,
        "undistort": args.undistort,
        "alpha": args.alpha,
        "crop_valid_roi": args.crop_valid_roi,
        "auto_crop_black": args.auto_crop_black,
        "K_original": K_orig.tolist(),
        "distortion": dist.tolist(),
    }
    (out_root / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[Done] Saved panels to: {panel_dir}")


if __name__ == "__main__":
    main()
