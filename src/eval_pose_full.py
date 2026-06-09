#!/usr/bin/env python3
"""
Robust pose evaluation for EndoMetric3D / C3VD.

Key fixes vs earlier scripts:
  * CLI-safe: eval-only args do not conflict with options.py training parser.
  * Loads pose_max_translation_cm from checkpoint options if present, or from CLI.
  * Uses the model forward path first, falling back to pose_head only if needed.
  * Uses the same C3VD pose row/column convention fix used in training.
  * Reports native relative pose errors, raw APE, SE(3)-aligned APE, Sim(3)-aligned APE, and scale.
  * Saves raw predicted relative poses separately; never overwrites them.

Example:
python src/eval_pose_full.py \
  --c3vd_test_root datasets/c3vd_dataset/c3vd_test/ \
  --ckpt_dir checkpoints_endometric3d_joint_bf16 \
  --ckpt best.pth \
  --pose_max_translation_cm 0.10 \
  --run_name posenet_best \
  --save_plots
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------
# Constants from your C3VD undistortion preprocessing
# --------------------------------------------------------------------------------------
C3VD_K = np.array([
    [769.8074, -0.1412, 675.2264],
    [0.0,      769.7205, 548.9035],
    [0.0,      0.0,      1.0],
], dtype=np.float32)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --------------------------------------------------------------------------------------
# CLI handling
# --------------------------------------------------------------------------------------
def str2bool(x):
    if isinstance(x, bool):
        return x
    return str(x).lower() in {"1", "true", "yes", "y", "on"}


def parse_eval_args() -> Tuple[argparse.Namespace, List[str]]:
    p = argparse.ArgumentParser("EndoMetric3D pose evaluation", add_help=True)
    p.add_argument("--c3vd_test_root", type=str, required=True)
    p.add_argument("--ckpt_dir", type=str, required=True)
    p.add_argument("--ckpt", type=str, default="best.pth")
    p.add_argument("--output_dir", type=str, default="results/pose_eval",
                   help="Base output directory. With --run_name, files go to output_dir/run_name.")
    p.add_argument("--run_name", type=str, default=None,
                   help="Optional run folder name, e.g. posenet_epoch30. Keeps plots/metrics separated per run.")
    p.add_argument("--timestamp_output", type=str2bool, default=False,
                   help="Append YYYYmmdd_HHMMSS to the run folder to avoid overwriting older evals.")
    p.add_argument("--height", type=int, default=476)
    p.add_argument("--width", type=int, default=588)
    p.add_argument("--num_workers", type=int, default=0)  # kept for compatibility
    p.add_argument("--sequences", nargs="*", default=None,
                   help="Optional list of sequence names. If omitted, all sequences in root are evaluated.")
    p.add_argument("--max_pairs", type=int, default=-1,
                   help="Limit number of frame pairs per sequence for quick debugging. -1 = all.")
    p.add_argument("--sample_step", type=int, default=1,
                   help="Evaluate every Nth adjacent pair. Usually 1.")
    p.add_argument("--pose_source_unit", type=str, default="mm", choices=["mm", "cm", "m"])
    p.add_argument("--pose_max_translation_cm", type=float, default=None,
                   help="IMPORTANT: set this to the same value used for training if checkpoint does not store it.")
    p.add_argument("--amp", type=str2bool, default=True)
    p.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32", "auto"])
    p.add_argument("--save_plots", action="store_true")
    p.add_argument("--save_pred_npy", type=str2bool, default=True)
    p.add_argument("--integration_mode", type=str, default="inv", choices=["inv", "direct", "both"],
                   help="How to integrate a predicted T_t_to_t1. inv is correct for point-transform convention.")
    p.add_argument("--debug_pairs", type=int, default=3,
                   help="Print first N predicted/GT relative translations for the first sequence.")
    args, unknown = p.parse_known_args()
    return args, unknown


def _slugify_run_name(text: str) -> str:
    safe = []
    for ch in str(text):
        if ch.isalnum() or ch in {"-", "_", "."}:
            safe.append(ch)
        else:
            safe.append("_")
    out = "".join(safe).strip("._")
    return out or "pose_eval_run"


def resolve_output_dir(args: argparse.Namespace) -> Path:
    """Create a deterministic per-run output folder.

    Default behavior changed intentionally: if the user keeps
    --output_dir results/pose_eval, this script now writes into
    results/pose_eval/<ckpt_dir>__<ckpt_stem>/ so different checkpoints
    do not overwrite each other's plots/metrics.
    """
    base = Path(args.output_dir)
    default_base = Path("results/pose_eval")

    if args.run_name is not None:
        run_name = _slugify_run_name(args.run_name)
        out_dir = base / run_name
    elif base == default_base:
        run_name = _slugify_run_name(f"{Path(args.ckpt_dir).name}__{Path(args.ckpt).stem}")
        out_dir = base / run_name
    else:
        # User gave an exact output_dir; keep it exact for backwards compatibility.
        run_name = _slugify_run_name(base.name)
        out_dir = base

    if args.timestamp_output:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = out_dir.parent / f"{out_dir.name}__{stamp}"
        run_name = out_dir.name

    args.run_name = run_name
    args.output_dir = str(out_dir)
    return out_dir


def build_opt_for_model(eval_args: argparse.Namespace, train_unknown: List[str], ckpt_state: Optional[dict]) -> Dict:
    """Build model options for evaluation without silently losing checkpoint-critical values.

    IMPORTANT BUG FIX:
    The previous version used opt.setdefault(k, v) for checkpoint options. Because
    options.py already supplies defaults such as pose_max_translation_cm=5.0, the
    checkpoint value pose_max_translation_cm=0.10 from pose fine-tuning was ignored.
    That made evaluation translations about 50x too large. Here checkpoint options
    override defaults first; CLI overrides only when explicitly provided.
    """
    try:
        from options import get_opts
        old_argv = sys.argv[:]
        sys.argv = [old_argv[0]] + train_unknown
        opt = get_opts()
        sys.argv = old_argv
        if not isinstance(opt, dict):
            opt = vars(opt)
    except Exception as e:
        print(f"[WARN] Could not call options.get_opts cleanly ({e}). Using minimal opt dict.")
        opt = {}

    ckpt_opt = None
    if isinstance(ckpt_state, dict):
        for k in ["opt", "opts", "options", "config", "args"]:
            if k in ckpt_state and isinstance(ckpt_state[k], dict):
                ckpt_opt = ckpt_state[k]
                break
    if ckpt_opt is not None:
        print("[Eval Opt] loaded model-critical options from checkpoint")
        opt.update(ckpt_opt)  # checkpoint values MUST override parser defaults

    # Eval/runtime overrides.
    opt["c3vd_test_root"] = eval_args.c3vd_test_root
    opt["ckpt_dir"] = eval_args.ckpt_dir
    opt["height"] = eval_args.height
    opt["width"] = eval_args.width
    opt["image_height"] = eval_args.height
    opt["image_width"] = eval_args.width
    opt["image_size"] = (eval_args.height, eval_args.width)
    opt["num_workers"] = eval_args.num_workers
    opt["pose_source_unit"] = eval_args.pose_source_unit

    # Critical: only CLI should override checkpoint cap. Otherwise preserve ckpt opt.
    if eval_args.pose_max_translation_cm is not None:
        opt["pose_max_translation_cm"] = float(eval_args.pose_max_translation_cm)
        print(f"[Eval Opt] CLI override pose_max_translation_cm={opt['pose_max_translation_cm']}")
    elif "pose_max_translation_cm" not in opt:
        opt["pose_max_translation_cm"] = 0.10
        print("[WARN] pose_max_translation_cm not found in checkpoint/options; using 0.10 cm.")

    # Safe architecture defaults if options.py/checkpoint were incomplete.
    opt.setdefault("backbone_model_name", "depth-anything/Depth-Anything-V2-Base-hf")
    opt.setdefault("backbone_mode", "la_svft")
    opt.setdefault("lasvft_rank", 8)
    opt.setdefault("lasvft_alpha", 8.0)
    opt.setdefault("embed_dim", 768)
    opt.setdefault("use_moe", True)
    opt.setdefault("num_experts", 4)
    opt.setdefault("moe_top_k", 2)
    opt.setdefault("moe_hidden_dim", 256)
    opt.setdefault("use_artifact_mask", True)
    opt.setdefault("min_depth_cm", 0.05)
    opt.setdefault("max_depth_cm", 10.0)
    opt.setdefault("dav2_output_is_inverse_depth", False)
    opt.setdefault("freeze_vit_backbone", True)
    opt.setdefault("freeze_dav2_decode_head", False)
    opt.setdefault("train_pose_head", True)
    opt.setdefault("pose_use_metric_depth", True)
    opt.setdefault("pose_use_intrinsics", True)
    opt.setdefault("pose_input_channels", 9 if bool(opt.get("pose_use_metric_depth", True)) else 7)
    opt.setdefault("pose_mag_bias_init", -2.0)
    opt.setdefault("pose_detach_depth_for_pose", False)
    opt.setdefault("canonical_f_norm", 0.57)
    opt.setdefault("hard_focal_scaling", False)
    return opt


# --------------------------------------------------------------------------------------
# Geometry utilities
# --------------------------------------------------------------------------------------
def c3vd_pose_line_to_c2w(line: str, line_idx: int, unit: str = "mm") -> Tuple[int, np.ndarray]:
    vals = [float(x) for x in line.replace(",", " ").split()]
    if len(vals) == 17:
        idx = int(vals[0])
        mat_vals = vals[1:]
    elif len(vals) == 16:
        idx = line_idx
        mat_vals = vals
    else:
        raise ValueError(f"Unexpected pose line with {len(vals)} values: {line[:100]}")

    T_raw = np.array(mat_vals, dtype=np.float64).reshape(4, 4)

    # C3VD pose.txt commonly stores translation in the last row; convert to standard column translation.
    last_col_zero = np.allclose(T_raw[:3, 3], 0.0, atol=1e-8)
    last_row_has_translation = np.linalg.norm(T_raw[3, :3]) > 1e-8
    if last_col_zero and last_row_has_translation:
        T = T_raw.T.copy()
    else:
        T = T_raw.copy()

    if unit == "mm":
        T[:3, 3] /= 10.0  # mm -> cm
    elif unit == "m":
        T[:3, 3] *= 100.0  # m -> cm
    elif unit == "cm":
        pass
    else:
        raise ValueError(unit)
    return idx, T


def load_c3vd_poses(pose_path: Path, unit: str = "mm") -> Dict[int, np.ndarray]:
    poses: Dict[int, np.ndarray] = {}
    with open(pose_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            idx, T = c3vd_pose_line_to_c2w(line, i, unit=unit)
            poses[idx] = T
    return poses


def relative_t_to_next_from_c2w(C_t: np.ndarray, C_next: np.ndarray) -> np.ndarray:
    # Point transform from camera-t coordinates to camera-next coordinates.
    return np.linalg.inv(C_next) @ C_t


def so3_angle_deg(R_err: np.ndarray) -> float:
    val = (np.trace(R_err) - 1.0) / 2.0
    val = np.clip(val, -1.0, 1.0)
    return float(np.degrees(np.arccos(val)))


def invert_T(T: np.ndarray) -> np.ndarray:
    Rm = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = Rm.T
    out[:3, 3] = -Rm.T @ t
    return out


def integrate_rel_poses(rel_poses: List[np.ndarray], mode: str = "inv") -> List[np.ndarray]:
    # Global C2W poses anchored at identity.
    Cs = [np.eye(4, dtype=np.float64)]
    C = np.eye(4, dtype=np.float64)
    for T in rel_poses:
        if mode == "inv":
            # T maps points from camera t to camera t+1, so C_{t+1} = C_t @ inv(T).
            C = C @ invert_T(T)
        elif mode == "direct":
            C = C @ T
        else:
            raise ValueError(mode)
        Cs.append(C.copy())
    return Cs


def anchor_gt_poses(c2w_list: List[np.ndarray]) -> List[np.ndarray]:
    C0_inv = np.linalg.inv(c2w_list[0])
    return [C0_inv @ C for C in c2w_list]


def umeyama_align(src: np.ndarray, dst: np.ndarray, with_scale: bool) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Align src Nx3 to dst Nx3. Returns aligned src, scale, R, t."""
    assert src.shape == dst.shape and src.shape[1] == 3
    n = src.shape[0]
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    X = src - mu_s
    Y = dst - mu_d
    cov = (Y.T @ X) / n
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[-1, -1] = -1
    Rm = U @ D @ Vt
    if with_scale:
        var = np.mean(np.sum(X * X, axis=1))
        scale = float(np.trace(np.diag(S) @ D) / (var + 1e-12))
    else:
        scale = 1.0
    t = mu_d - scale * (Rm @ mu_s)
    aligned = (scale * (Rm @ src.T)).T + t
    return aligned, scale, Rm, t


def rmse_xyz(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


# --------------------------------------------------------------------------------------
# Image and model helpers
# --------------------------------------------------------------------------------------
def resize_K(K: np.ndarray, old_hw: Tuple[int, int], new_hw: Tuple[int, int]) -> np.ndarray:
    old_h, old_w = old_hw
    new_h, new_w = new_hw
    sx = new_w / float(old_w)
    sy = new_h / float(old_h)
    K2 = K.copy().astype(np.float32)
    K2[0, 0] *= sx
    K2[0, 1] *= sx
    K2[0, 2] *= sx
    K2[1, 1] *= sy
    K2[1, 2] *= sy
    return K2


def load_image_tensor(path: Path, height: int, width: int) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """Return normalized tensor, raw [0,1] tensor, and original H/W.

    The model's artifact/glare prior expects image_01, not ImageNet-normalized
    values. Earlier eval passed only normalized images, so the model fell back to
    img.clamp(0,1), producing a different tissue mask than training.
    """
    img = Image.open(path).convert("RGB")
    old_hw = (img.height, img.width)
    img = img.resize((width, height), Image.BILINEAR)
    arr01 = np.asarray(img).astype(np.float32) / 255.0
    arr_norm = (arr01 - IMAGENET_MEAN) / IMAGENET_STD
    ten = torch.from_numpy(arr_norm).permute(2, 0, 1).unsqueeze(0).float()
    ten01 = torch.from_numpy(arr01).permute(2, 0, 1).unsqueeze(0).float()
    return ten, ten01, old_hw


def find_sequences(root: Path, requested: Optional[List[str]] = None) -> List[Path]:
    seqs = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if requested is not None and d.name not in requested:
            continue
        if (d / "color").exists() and (d / "pose.txt").exists():
            seqs.append(d)
    return seqs


def checkpoint_state_dict(state):
    if not isinstance(state, dict):
        return state
    for k in ["model_state", "model_state_dict", "state_dict", "model"]:
        if k in state and isinstance(state[k], dict):
            return state[k]
    return state


def set_pose_translation_cap(model, cap_cm: float):
    """Best effort: update common attr names in pose head/model."""
    for obj in [model, getattr(model, "pose_head", None)]:
        if obj is None:
            continue
        for name in ["pose_max_translation_cm", "max_translation_cm", "max_trans_cm", "translation_scale", "max_translation"]:
            if hasattr(obj, name):
                try:
                    setattr(obj, name, float(cap_cm))
                    print(f"[Eval Debug] set {obj.__class__.__name__}.{name} = {cap_cm}")
                except Exception:
                    pass


def get_pose_translation_cap(model, opt: Dict) -> Dict[str, object]:
    out = {"opt_pose_max_translation_cm": opt.get("pose_max_translation_cm", None)}
    for prefix, obj in [("model", model), ("pose_head", getattr(model, "pose_head", None))]:
        if obj is None:
            continue
        for name in ["pose_max_translation_cm", "max_translation_cm", "max_trans_cm", "translation_scale", "max_translation"]:
            if hasattr(obj, name):
                val = getattr(obj, name)
                if torch.is_tensor(val):
                    val = val.detach().cpu().item()
                out[f"{prefix}.{name}"] = val
    return out


def maybe_extract_pose(outputs: Dict) -> Optional[torch.Tensor]:
    candidates = [
        "pose_matrix", "pose", "T", "T_pred", "T_t_to_t1", "T_t_to_next", "pose_forward", "T_forward"
    ]
    for k in candidates:
        if k in outputs and torch.is_tensor(outputs[k]):
            T = outputs[k]
            if T.ndim == 3 and T.shape[-2:] == (4, 4):
                return T
            if T.ndim == 4 and T.shape[-2:] == (4, 4):
                # e.g. [B, N, 4, 4], take first temporal output
                return T[:, 0]
    return None


def predict_pair_pose(model, img_t: torch.Tensor, img_next: torch.Tensor, K: torch.Tensor,
                      device: torch.device, autocast_dtype, use_amp: bool,
                      img_t_01: Optional[torch.Tensor] = None, img_next_01: Optional[torch.Tensor] = None) -> np.ndarray:
    img_t = img_t.to(device, non_blocking=True)
    img_next = img_next.to(device, non_blocking=True)
    K = K.to(device, non_blocking=True)
    if img_t_01 is not None:
        img_t_01 = img_t_01.to(device, non_blocking=True)
    if img_next_01 is not None:
        img_next_01 = img_next_01.to(device, non_blocking=True)

    inputs = {
        "image": img_t,
        "image_next": img_next,
        "K": K,
    }
    if img_t_01 is not None:
        inputs["image_01"] = img_t_01
    if img_next_01 is not None:
        inputs["image_next_01"] = img_next_01

    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=(use_amp and device.type == "cuda")):
            outputs = model(inputs)
        T = maybe_extract_pose(outputs if isinstance(outputs, dict) else {})

        if T is None:
            # Fallback path. Use model-provided artifact/tissue mask if possible.
            mask = None
            if isinstance(outputs, dict):
                for mk in ["tissue_mask", "pose_mask", "attention_mask", "mask"]:
                    if mk in outputs and torch.is_tensor(outputs[mk]):
                        mask = outputs[mk]
                        break
            if mask is None:
                mask = torch.ones((img_t.shape[0], 1, img_t.shape[2], img_t.shape[3]), device=device, dtype=img_t.dtype)

            if not hasattr(model, "pose_head"):
                raise RuntimeError("Could not find pose in model outputs and model has no pose_head fallback.")

            out = model.pose_head(
                img_t,
                img_next,
                mask,
                depth_t=outputs.get("metric_depth") if isinstance(outputs, dict) else None,
                depth_next=outputs.get("metric_depth_next") if isinstance(outputs, dict) else None,
                K=inputs.get("K"),
            )
            if isinstance(out, tuple):
                T = out[0]
            elif isinstance(out, dict):
                T = maybe_extract_pose(out)
            else:
                T = out

        if T is None or not torch.is_tensor(T):
            raise RuntimeError("Could not extract predicted pose matrix from model.")
        T_np = T[0].detach().float().cpu().numpy().astype(np.float64)
    return T_np


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------
def save_relative_diagnostic_plots(seq_name: str, pred_norms: List[float], gt_norms: List[float],
                                   trans_errs: List[float], rot_errs: List[float],
                                   scale_logs: List[float], args: argparse.Namespace) -> None:
    if not args.save_plots:
        return
    plot_dir = Path(args.output_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(pred_norms))

    fig = plt.figure(figsize=(8, 4))
    plt.plot(x, gt_norms, label="GT step norm")
    plt.plot(x, pred_norms, label="Pred step norm")
    plt.xlabel("Pair index")
    plt.ylabel("Translation step norm (cm)")
    plt.title(f"{seq_name}: predicted vs GT step scale")
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / f"{seq_name}_step_norms.png", dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 4))
    plt.plot(x, trans_errs, label="Relative translation error (cm)")
    plt.xlabel("Pair index")
    plt.ylabel("Error (cm)")
    plt.title(f"{seq_name}: relative translation error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / f"{seq_name}_rel_trans_error.png", dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 4))
    plt.plot(x, rot_errs, label="Relative rotation error (deg)")
    plt.xlabel("Pair index")
    plt.ylabel("Error (degrees)")
    plt.title(f"{seq_name}: relative rotation error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / f"{seq_name}_rel_rot_error.png", dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 4))
    plt.plot(x, scale_logs, label="|log(pred_step / gt_step)|")
    plt.xlabel("Pair index")
    plt.ylabel("Absolute log scale error")
    plt.title(f"{seq_name}: per-pair pose scale error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / f"{seq_name}_pose_scale_log.png", dpi=200)
    plt.close(fig)


def eval_sequence(seq_dir: Path, model, opt: Dict, args: argparse.Namespace, device: torch.device,
                  autocast_dtype) -> Dict[str, float]:
    color_dir = seq_dir / "color"
    frames = sorted([p for p in color_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if len(frames) < 2:
        raise RuntimeError(f"Not enough frames in {seq_dir}")

    poses_abs_dict = load_c3vd_poses(seq_dir / "pose.txt", unit=args.pose_source_unit)

    pair_indices = list(range(0, len(frames) - 1, max(1, args.sample_step)))
    if args.max_pairs is not None and args.max_pairs > 0:
        pair_indices = pair_indices[: args.max_pairs]

    pred_rels: List[np.ndarray] = []
    gt_rels: List[np.ndarray] = []
    gt_abs_used: List[np.ndarray] = []

    debug_left = args.debug_pairs
    for i in tqdm(pair_indices, desc=f"pose {seq_dir.name}", leave=False):
        j = i + 1
        if i not in poses_abs_dict or j not in poses_abs_dict:
            continue
        img_t, img_t_01, old_hw = load_image_tensor(frames[i], args.height, args.width)
        img_n, img_n_01, _ = load_image_tensor(frames[j], args.height, args.width)
        K_np = resize_K(C3VD_K, old_hw=old_hw, new_hw=(args.height, args.width))
        K = torch.from_numpy(K_np).unsqueeze(0).float()

        T_pred = predict_pair_pose(model, img_t, img_n, K, device, autocast_dtype, args.amp, img_t_01=img_t_01, img_next_01=img_n_01)
        C_t = poses_abs_dict[i]
        C_n = poses_abs_dict[j]
        T_gt = relative_t_to_next_from_c2w(C_t, C_n)

        pred_rels.append(T_pred)
        gt_rels.append(T_gt)
        if len(gt_abs_used) == 0:
            gt_abs_used.append(C_t)
        gt_abs_used.append(C_n)

        if debug_left > 0:
            print(f"[PairDebug {seq_dir.name} {i}->{j}] pred_t={T_pred[:3,3]} norm={np.linalg.norm(T_pred[:3,3]):.6f} | "
                  f"gt_t={T_gt[:3,3]} norm={np.linalg.norm(T_gt[:3,3]):.6f}")
            debug_left -= 1

    if not pred_rels:
        raise RuntimeError(f"No valid pairs evaluated for {seq_dir}")

    # Relative metrics against T_t_to_t1 convention.
    trans_errs = []
    trans_mae = []
    rot_errs = []
    scale_logs = []
    pred_norms = []
    gt_norms = []
    for Tp, Tg in zip(pred_rels, gt_rels):
        tp = Tp[:3, 3]
        tg = Tg[:3, 3]
        trans_errs.append(float(np.linalg.norm(tp - tg)))
        trans_mae.append(float(np.mean(np.abs(tp - tg))))
        rot_errs.append(so3_angle_deg(Tg[:3, :3].T @ Tp[:3, :3]))
        pn = float(np.linalg.norm(tp))
        gn = float(np.linalg.norm(tg))
        pred_norms.append(pn)
        gt_norms.append(gn)
        scale_logs.append(float(abs(np.log((pn + 1e-9) / (gn + 1e-9)))))

    save_relative_diagnostic_plots(seq_dir.name, pred_norms, gt_norms, trans_errs, rot_errs, scale_logs, args)

    def traj_metrics(mode: str) -> Dict[str, float]:
        pred_C = integrate_rel_poses(pred_rels, mode=mode)
        gt_C = anchor_gt_poses(gt_abs_used[: len(pred_C)])
        pred_xyz = np.array([C[:3, 3] for C in pred_C], dtype=np.float64)
        gt_xyz = np.array([C[:3, 3] for C in gt_C], dtype=np.float64)

        raw = rmse_xyz(pred_xyz, gt_xyz)
        pred_se3, se3_scale, _, _ = umeyama_align(pred_xyz, gt_xyz, with_scale=False)
        pred_sim3, sim3_scale, _, _ = umeyama_align(pred_xyz, gt_xyz, with_scale=True)
        ape_se3 = rmse_xyz(pred_se3, gt_xyz)
        ape_sim3 = rmse_xyz(pred_sim3, gt_xyz)

        if args.save_plots:
            plot_dir = Path(args.output_dir) / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            axes[0].plot(gt_xyz[:, 0], gt_xyz[:, 2], "k--", label="GT")
            axes[0].plot(pred_xyz[:, 0], pred_xyz[:, 2], "-", alpha=0.45, label=f"Pred raw ({mode})")
            axes[0].plot(pred_se3[:, 0], pred_se3[:, 2], "-", label=f"Pred SE3 no-scale ({mode})")
            axes[0].plot(pred_sim3[:, 0], pred_sim3[:, 2], "-", label=f"Pred Sim3 scaled ({mode})")
            axes[0].set_title(f"{seq_dir.name} X-Z | Sim3 scale={sim3_scale:.3f}")
            axes[0].set_xlabel("X (cm)")
            axes[0].set_ylabel("Z (cm)")
            axes[0].axis("equal")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(fontsize=8)
            axes[1].plot(gt_xyz[:, 1], gt_xyz[:, 2], "k--", label="GT")
            axes[1].plot(pred_xyz[:, 1], pred_xyz[:, 2], "-", alpha=0.45, label=f"Pred raw ({mode})")
            axes[1].plot(pred_se3[:, 1], pred_se3[:, 2], "-", label=f"Pred SE3 no-scale ({mode})")
            axes[1].plot(pred_sim3[:, 1], pred_sim3[:, 2], "-", label=f"Pred Sim3 scaled ({mode})")
            axes[1].set_title(f"{seq_dir.name} Y-Z | APE SE3={ape_se3:.3f} cm")
            axes[1].set_xlabel("Y (cm)")
            axes[1].set_ylabel("Z (cm)")
            axes[1].axis("equal")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(plot_dir / f"{seq_dir.name}_trajectory_{mode}.png", dpi=200)
            plt.close(fig)

        return {
            f"traj/{mode}/ape_raw_cm": raw,
            f"traj/{mode}/ape_se3_cm_no_scale": ape_se3,
            f"traj/{mode}/ape_sim3_cm_scaled": ape_sim3,
            f"traj/{mode}/sim3_scale": sim3_scale,
        }

    out = {
        "sequence": seq_dir.name,
        "rel/rel_trans_cm": float(np.mean(trans_errs)),
        "rel/rel_trans_mae_cm": float(np.mean(trans_mae)),
        "rel/rel_rot_deg": float(np.mean(rot_errs)),
        "rel/pose_scale_log": float(np.mean(scale_logs)),
        "rel/pred_step_norm_cm": float(np.mean(pred_norms)),
        "rel/gt_step_norm_cm": float(np.mean(gt_norms)),
        "rel/pred_nonzero": float(np.mean(np.array(pred_norms) > 1e-8)),
        "traj/num_pairs": float(len(pred_rels)),
    }


    cap = float(opt.get("pose_max_translation_cm", 0.0) or 0.0)
    if cap > 0 and float(np.mean(pred_norms)) > (np.sqrt(3.0) * cap + 1e-4):
        print(f"[WARN] Mean predicted step norm {float(np.mean(pred_norms)):.6f} cm exceeds theoretical cap sqrt(3)*{cap:.4f}. Check pose_max_translation_cm loading.")

    if args.integration_mode in {"inv", "both"}:
        out.update(traj_metrics("inv"))
    if args.integration_mode in {"direct", "both"}:
        out.update(traj_metrics("direct"))

    if args.save_pred_npy:
        pred_dir = Path(args.output_dir) / "pred_poses" / seq_dir.name
        pred_dir.mkdir(parents=True, exist_ok=True)
        np.save(pred_dir / f"{seq_dir.name}_pred_rel_poses.npy", np.stack(pred_rels, axis=0))
        np.save(pred_dir / f"{seq_dir.name}_gt_rel_poses.npy", np.stack(gt_rels, axis=0))

    return out


def summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = [k for k in rows[0].keys() if k != "sequence"]
    summary = {}
    for k in keys:
        vals = np.array([float(r[k]) for r in rows if k in r and np.isfinite(float(r[k]))], dtype=np.float64)
        if vals.size == 0:
            continue
        summary[k] = float(vals.mean())
        if vals.size > 1:
            summary[k + "/ci95"] = float(1.96 * vals.std(ddof=1) / math.sqrt(vals.size))
    return summary


def main():
    args, unknown = parse_eval_args()
    root = Path(args.c3vd_test_root)
    ckpt_path = Path(args.ckpt_dir) / args.ckpt
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu")
    opt = build_opt_for_model(args, unknown, state if isinstance(state, dict) else None)
    out_dir = resolve_output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    (out_dir / "pred_poses").mkdir(parents=True, exist_ok=True)

    # Import model after opt construction.
    try:
        from networks import EndoMetric3DModel
    except Exception:
        from networks.endometric3d_model import EndoMetric3DModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    print(f"[Checkpoint] {ckpt_path}")
    print(f"[Output] {args.output_dir}")
    print(f"[Eval Opt] height={args.height} width={args.width} pose_max_translation_cm={opt.get('pose_max_translation_cm')}")

    if not bool(opt.get("train_pose_head", False)):
        raise RuntimeError(
            "This checkpoint/options say train_pose_head=False. It is a depth-only checkpoint. "
            "Train with --ablation pose_full or joint_full before running pose evaluation."
        )

    model = EndoMetric3DModel(opt).to(device)
    set_pose_translation_cap(model, float(opt.get("pose_max_translation_cm", 0.10)))
    sd = checkpoint_state_dict(state)
    current = model.state_dict()
    filtered = {}
    skipped = {}
    for k, v in sd.items():
        if k in current and tuple(v.shape) != tuple(current[k].shape):
            skipped[k] = (tuple(v.shape), tuple(current[k].shape))
            continue
        filtered[k] = v
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"[Checkpoint] loaded missing={len(missing)} unexpected={len(unexpected)} skipped_shape_mismatch={len(skipped)}")
    if skipped:
        for k, (old_shape, new_shape) in list(skipped.items())[:12]:
            print(f"  [skip mismatch] {k}: ckpt={old_shape} model={new_shape}")
    print("[Eval Debug] pose cap attrs:", get_pose_translation_cap(model, opt))
    model.eval()

    if args.amp_dtype == "auto":
        autocast_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    elif args.amp_dtype == "bf16":
        autocast_dtype = torch.bfloat16
    elif args.amp_dtype == "fp16":
        autocast_dtype = torch.float16
    else:
        autocast_dtype = torch.float32
        args.amp = False

    seqs = find_sequences(root, args.sequences)
    print(f"[Sequences] {len(seqs)} from {root}")
    if not seqs:
        raise RuntimeError(f"No C3VD sequences found in {root}")

    # out_dir was resolved before model creation so plots/metrics are grouped per run.
    rows = []
    for seq in seqs:
        print(f"\n[Sequence] {seq.name}")
        row = eval_sequence(seq, model, opt, args, device, autocast_dtype)
        rows.append(row)
        for k, v in row.items():
            if k != "sequence":
                print(f"  {k}={v:.6f}")

    summary = summarize(rows)
    print("\n" + "=" * 88)
    print("Pose Evaluation Summary")
    print("Primary CMIG-comparable metrics: Sim(3)-aligned APE/RPE-style trajectory metrics + scale.")
    print("Strict metric diagnostic: SE(3) no-scale APE and pred/GT step norm ratio.")
    for k in sorted(summary):
        print(f"  {k:42s}: {summary[k]:.6f}")
    print("=" * 88)

    # Save run manifest and JSON/CSV.
    manifest = {
        "run_name": args.run_name,
        "checkpoint": str(ckpt_path),
        "output_dir": str(out_dir),
        "c3vd_test_root": args.c3vd_test_root,
        "sequences": [s.name for s in seqs],
        "eval_args": vars(args),
        "pose_max_translation_cm_effective": opt.get("pose_max_translation_cm", None),
        "pose_use_metric_depth": opt.get("pose_use_metric_depth", None),
        "pose_use_intrinsics": opt.get("pose_use_intrinsics", None),
    }
    with open(out_dir / "eval_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with open(out_dir / "pose_metrics_by_sequence.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with open(out_dir / "pose_metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    all_keys = ["sequence"] + sorted({k for r in rows for k in r.keys() if k != "sequence"})
    with open(out_dir / "pose_metrics_by_sequence.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(out_dir / "pose_metrics_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k in sorted(summary):
            w.writerow([k, summary[k]])

    print(f"[Saved] {out_dir}")


if __name__ == "__main__":
    main()