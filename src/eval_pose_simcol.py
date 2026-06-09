#!/usr/bin/env python3
"""Pose evaluation for EndoMetric3D on SimCol/SyntheticColon.

This is the SimCol counterpart of eval_pose_full.py. It uses datasets_simcol.SimColDataset
so it follows the same SimCol split, intrinsics, pose-unit, pose-layout and pose-convention
settings used during training.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def str2bool(x):
    if isinstance(x, bool):
        return x
    return str(x).lower() in {"1", "true", "yes", "y", "on"}


def parse_args():
    p = argparse.ArgumentParser("EndoMetric3D SimCol pose evaluation", add_help=True)
    p.add_argument("--simcol_root", type=str, required=True)
    p.add_argument("--eval_split", type=str, default="test", choices=["train", "val", "test", "test_iii", "all"])
    p.add_argument("--ckpt_dir", type=str, required=True)
    p.add_argument("--ckpt", type=str, default="best.pth")
    p.add_argument("--output_dir", type=str, default="results/pose_eval_simcol")
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--height", type=int, default=476)
    p.add_argument("--width", type=int, default=476)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_pairs", type=int, default=-1)
    p.add_argument("--pose_max_translation_cm", type=float, default=None)
    p.add_argument("--amp", type=str2bool, default=True)
    p.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32", "auto"])
    p.add_argument("--integration_mode", type=str, default="inv", choices=["inv", "direct", "both"])
    p.add_argument("--debug_pairs", type=int, default=5)
    p.add_argument("--save_plots", action="store_true")
    p.add_argument("--save_pred_npy", type=str2bool, default=True)
    args, unknown = p.parse_known_args()
    return args, unknown


def checkpoint_state_dict(state):
    if not isinstance(state, dict):
        return state
    for k in ["model_state", "model_state_dict", "state_dict", "model"]:
        if k in state and isinstance(state[k], dict):
            return state[k]
    return state


def checkpoint_opt(state) -> Optional[dict]:
    if not isinstance(state, dict):
        return None
    for k in ["opt", "opts", "options", "config", "args"]:
        if k in state and isinstance(state[k], dict):
            return state[k]
    return None


def build_opt(args, unknown, state):
    from options import get_opts
    # Always force dataset=simcol unless user already supplied it.
    if "--dataset" not in unknown:
        unknown = ["--dataset", "simcol"] + list(unknown)
    opt = get_opts(unknown)
    ckopt = checkpoint_opt(state)
    if ckopt is not None:
        print("[Eval Opt] loaded model-critical options from checkpoint")
        opt.update(ckopt)

    # Runtime/eval overrides.
    opt["dataset"] = "simcol"
    opt["simcol_root"] = args.simcol_root
    opt["height"] = args.height
    opt["width"] = args.width
    opt["image_height"] = args.height
    opt["image_width"] = args.width
    opt["image_size"] = (args.height, args.width)
    opt["batch_size"] = args.batch_size
    opt["num_workers"] = args.num_workers
    opt["use_triplet_pose"] = True
    opt["train_pose_head"] = True
    opt.setdefault("pose_use_metric_depth", True)
    opt.setdefault("pose_use_intrinsics", True)
    opt.setdefault("pose_input_channels", 9 if bool(opt.get("pose_use_metric_depth", True)) else 7)
    opt.setdefault("max_depth_cm", 20.0)
    opt.setdefault("simcol_depth_scale_cm", 20.0)
    if args.pose_max_translation_cm is not None:
        opt["pose_max_translation_cm"] = float(args.pose_max_translation_cm)
        print(f"[Eval Opt] CLI override pose_max_translation_cm={opt['pose_max_translation_cm']}")
    elif "pose_max_translation_cm" not in opt:
        opt["pose_max_translation_cm"] = 1.0
        print("[WARN] pose_max_translation_cm missing; using 1.0 cm for SimCol.")
    return opt


def set_pose_translation_cap(model, cap_cm: float):
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


def maybe_extract_pose(outputs: Dict) -> Optional[torch.Tensor]:
    for k in ["pose_matrix", "pose", "T", "T_pred", "T_t_to_t1", "T_t_to_next", "pose_forward", "T_forward"]:
        if k in outputs and torch.is_tensor(outputs[k]):
            T = outputs[k]
            if T.ndim == 3 and T.shape[-2:] == (4, 4):
                return T
            if T.ndim == 4 and T.shape[-2:] == (4, 4):
                return T[:, 0]
    return None


def move_to_device(batch: Dict, device: torch.device) -> Dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def predict_pose(model, batch, device, autocast_dtype, use_amp: bool) -> np.ndarray:
    batch = move_to_device(batch, device)
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=(use_amp and device.type == "cuda")):
            outputs = model(batch)
        T = maybe_extract_pose(outputs if isinstance(outputs, dict) else {})
        if T is None:
            if not hasattr(model, "pose_head"):
                raise RuntimeError("Could not find pose in outputs and model has no pose_head fallback.")
            mask = None
            if isinstance(outputs, dict):
                for mk in ["tissue_mask", "pose_mask", "attention_mask", "mask"]:
                    if mk in outputs and torch.is_tensor(outputs[mk]):
                        mask = outputs[mk]
                        break
            if mask is None:
                mask = torch.ones((batch["image"].shape[0], 1, batch["image"].shape[2], batch["image"].shape[3]), device=device)
            out = model.pose_head(
                batch["image"], batch["image_next"], mask,
                depth_t=outputs.get("metric_depth") if isinstance(outputs, dict) else None,
                depth_next=outputs.get("metric_depth_next") if isinstance(outputs, dict) else None,
                K=batch.get("K"),
            )
            T = maybe_extract_pose(out) if isinstance(out, dict) else (out[0] if isinstance(out, tuple) else out)
        if T is None or not torch.is_tensor(T):
            raise RuntimeError("Could not extract predicted pose matrix.")
        return T[0].detach().float().cpu().numpy().astype(np.float64)


def so3_angle_deg(R_err: np.ndarray) -> float:
    v = (np.trace(R_err) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(v, -1.0, 1.0))))


def invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def integrate_rel_poses(rel_poses: List[np.ndarray], mode: str = "inv") -> List[np.ndarray]:
    Cs = [np.eye(4, dtype=np.float64)]
    C = np.eye(4, dtype=np.float64)
    for T in rel_poses:
        if mode == "inv":
            C = C @ invert_T(T)
        elif mode == "direct":
            C = C @ T
        else:
            raise ValueError(mode)
        Cs.append(C.copy())
    return Cs


def umeyama_align(src: np.ndarray, dst: np.ndarray, with_scale: bool):
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
    R = U @ D @ Vt
    if with_scale:
        var = np.mean(np.sum(X * X, axis=1))
        scale = float(np.trace(np.diag(S) @ D) / (var + 1e-12))
    else:
        scale = 1.0
    t = mu_d - scale * (R @ mu_s)
    aligned = (scale * (R @ src.T)).T + t
    return aligned, scale


def rmse_xyz(a, b) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def plot_sequence(run_dir: Path, seq: str, pred_rels: List[np.ndarray], gt_rels: List[np.ndarray], mode: str):
    pred_C = integrate_rel_poses(pred_rels, mode=mode)
    gt_C = integrate_rel_poses(gt_rels, mode=mode)
    pred_xyz = np.array([C[:3, 3] for C in pred_C], dtype=np.float64)
    gt_xyz = np.array([C[:3, 3] for C in gt_C], dtype=np.float64)
    pred_se3, _ = umeyama_align(pred_xyz, gt_xyz, with_scale=False)

    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(gt_xyz[:, 0], gt_xyz[:, 2], "k--", label="GT")
    axes[0].plot(pred_se3[:, 0], pred_se3[:, 2], "r-", label=f"Pred SE3 ({mode})")
    axes[0].set_title(f"{seq} X-Z")
    axes[0].set_xlabel("X (cm)"); axes[0].set_ylabel("Z (cm)"); axes[0].axis("equal"); axes[0].grid(True, alpha=0.3); axes[0].legend()
    axes[1].plot(gt_xyz[:, 1], gt_xyz[:, 2], "k--", label="GT")
    axes[1].plot(pred_se3[:, 1], pred_se3[:, 2], "b-", label=f"Pred SE3 ({mode})")
    axes[1].set_title(f"{seq} Y-Z")
    axes[1].set_xlabel("Y (cm)"); axes[1].set_ylabel("Z (cm)"); axes[1].axis("equal"); axes[1].grid(True, alpha=0.3); axes[1].legend()
    fig.tight_layout()
    fig.savefig(plot_dir / f"{seq}_trajectory_{mode}.png", dpi=200)
    plt.close(fig)

    pred_norms = np.array([np.linalg.norm(T[:3, 3]) for T in pred_rels])
    gt_norms = np.array([np.linalg.norm(T[:3, 3]) for T in gt_rels])
    fig = plt.figure(figsize=(10, 4))
    plt.plot(gt_norms, label="GT step norm")
    plt.plot(pred_norms, label="Pred step norm")
    plt.title(f"{seq} step norms")
    plt.xlabel("pair index"); plt.ylabel("cm"); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    fig.savefig(plot_dir / f"{seq}_step_norms.png", dpi=200)
    plt.close(fig)


def sequence_metrics(seq: str, pred_rels: List[np.ndarray], gt_rels: List[np.ndarray], run_dir: Path, args) -> Dict[str, float]:
    trans_errs, trans_mae, rot_errs, scale_logs, pred_norms, gt_norms = [], [], [], [], [], []
    for Tp, Tg in zip(pred_rels, gt_rels):
        tp, tg = Tp[:3, 3], Tg[:3, 3]
        pn, gn = float(np.linalg.norm(tp)), float(np.linalg.norm(tg))
        pred_norms.append(pn); gt_norms.append(gn)
        trans_errs.append(float(np.linalg.norm(tp - tg)))
        trans_mae.append(float(np.mean(np.abs(tp - tg))))
        rot_errs.append(so3_angle_deg(Tg[:3, :3].T @ Tp[:3, :3]))
        scale_logs.append(float(abs(np.log((pn + 1e-9) / (gn + 1e-9)))))

    out = {
        "sequence": seq,
        "rel/rel_trans_cm": float(np.mean(trans_errs)),
        "rel/rel_trans_mae_cm": float(np.mean(trans_mae)),
        "rel/rel_rot_deg": float(np.mean(rot_errs)),
        "rel/pose_scale_log": float(np.mean(scale_logs)),
        "rel/pred_step_norm_cm": float(np.mean(pred_norms)),
        "rel/gt_step_norm_cm": float(np.mean(gt_norms)),
        "rel/pred_nonzero": float(np.mean(np.array(pred_norms) > 1e-8)),
        "traj/num_pairs": float(len(pred_rels)),
    }

    def add_traj(mode: str):
        pred_C = integrate_rel_poses(pred_rels, mode=mode)
        gt_C = integrate_rel_poses(gt_rels, mode=mode)
        pred_xyz = np.array([C[:3, 3] for C in pred_C], dtype=np.float64)
        gt_xyz = np.array([C[:3, 3] for C in gt_C], dtype=np.float64)
        pred_se3, _ = umeyama_align(pred_xyz, gt_xyz, with_scale=False)
        pred_sim3, scale = umeyama_align(pred_xyz, gt_xyz, with_scale=True)
        out[f"traj/{mode}/ape_raw_cm"] = rmse_xyz(pred_xyz, gt_xyz)
        out[f"traj/{mode}/ape_se3_cm_no_scale"] = rmse_xyz(pred_se3, gt_xyz)
        out[f"traj/{mode}/ape_sim3_cm_scaled"] = rmse_xyz(pred_sim3, gt_xyz)
        out[f"traj/{mode}/sim3_scale"] = scale
        if args.save_plots:
            plot_sequence(run_dir, seq, pred_rels, gt_rels, mode)

    if args.integration_mode in {"inv", "both"}:
        add_traj("inv")
    if args.integration_mode in {"direct", "both"}:
        add_traj("direct")
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
    args, unknown = parse_args()
    ckpt_path = Path(args.ckpt_dir) / args.ckpt
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    opt = build_opt(args, unknown, state if isinstance(state, dict) else None)

    from datasets_simcol import SimColDataset
    try:
        from networks import EndoMetric3DModel
    except Exception:
        from networks.endometric3d_model import EndoMetric3DModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name.strip() or f"{Path(args.ckpt_dir).name}_{args.eval_split}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Device] {device}")
    print(f"[Checkpoint] {ckpt_path}")
    print(f"[Output] {run_dir}")
    print(f"[Eval Opt] split={args.eval_split} height={args.height} width={args.width} pose_max_translation_cm={opt.get('pose_max_translation_cm')}")

    ds = SimColDataset(args.simcol_root, opt, split=args.eval_split)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    print(f"[Dataset] SimCol split={args.eval_split} samples={len(ds)}")

    model = EndoMetric3DModel(opt).to(device)
    set_pose_translation_cap(model, float(opt.get("pose_max_translation_cm", 1.0)))
    current = model.state_dict()
    sd = checkpoint_state_dict(state)
    filtered = {}
    skipped = {}
    for k, v in sd.items():
        if k in current and tuple(v.shape) != tuple(current[k].shape):
            skipped[k] = (tuple(v.shape), tuple(current[k].shape))
            continue
        filtered[k] = v
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"[Checkpoint] loaded missing={len(missing)} unexpected={len(unexpected)} skipped_shape_mismatch={len(skipped)}")
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

    pred_by_seq: Dict[str, List[np.ndarray]] = {}
    gt_by_seq: Dict[str, List[np.ndarray]] = {}
    printed = 0
    for ni, batch in enumerate(tqdm(loader, desc="eval-pose-simcol")):
        if args.max_pairs is not None and args.max_pairs > 0 and ni >= args.max_pairs:
            break
        seq = batch.get("seq_name", ["unknown"])[0] if isinstance(batch.get("seq_name"), list) else str(batch.get("seq_name", "unknown"))
        T_pred = predict_pose(model, batch, device, autocast_dtype, args.amp)
        T_gt = batch["pose_gt"][0].detach().cpu().numpy().astype(np.float64)
        pred_by_seq.setdefault(seq, []).append(T_pred)
        gt_by_seq.setdefault(seq, []).append(T_gt)
        if printed < args.debug_pairs:
            print(f"[PairDebug {seq}] pred_t={T_pred[:3,3]} norm={np.linalg.norm(T_pred[:3,3]):.6f} | "
                  f"gt_t={T_gt[:3,3]} norm={np.linalg.norm(T_gt[:3,3]):.6f}")
            printed += 1

    rows = []
    for seq in sorted(pred_by_seq.keys()):
        if not pred_by_seq[seq]:
            continue
        row = sequence_metrics(seq, pred_by_seq[seq], gt_by_seq[seq], run_dir, args)
        rows.append(row)
        print(f"\n[Sequence] {seq}")
        for k, v in row.items():
            if k != "sequence":
                print(f"  {k}={v:.6f}")
        if args.save_pred_npy:
            pred_dir = run_dir / "pred_poses" / seq
            pred_dir.mkdir(parents=True, exist_ok=True)
            np.save(pred_dir / f"{seq}_pred_rel_poses.npy", np.stack(pred_by_seq[seq], axis=0))
            np.save(pred_dir / f"{seq}_gt_rel_poses.npy", np.stack(gt_by_seq[seq], axis=0))

    if not rows:
        raise RuntimeError("No SimCol pose pairs were evaluated.")
    summary = summarize(rows)
    print("\n" + "=" * 88)
    print("SimCol Pose Evaluation Summary")
    print("Native metric diagnostics: pred/GT step norm, SE(3) no-scale APE, Sim(3) scale.")
    for k in sorted(summary):
        print(f"  {k:42s}: {summary[k]:.6f}")
    print("=" * 88)

    (run_dir / "eval_manifest.json").write_text(json.dumps({"ckpt": str(ckpt_path), "eval_split": args.eval_split, "opt_subset": {k: opt.get(k) for k in ["simcol_pose_unit", "simcol_pose_convention", "simcol_pose_layout", "simcol_rotation_mode", "simcol_quat_format", "simcol_depth_scale_cm", "pose_max_translation_cm", "moe_prototype_path"]}}, indent=2), encoding="utf-8")
    (run_dir / "pose_metrics_by_sequence.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (run_dir / "pose_metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    all_keys = ["sequence"] + sorted({k for r in rows for k in r.keys() if k != "sequence"})
    with (run_dir / "pose_metrics_by_sequence.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader(); w.writerows(rows)
    with (run_dir / "pose_metrics_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["metric", "value"])
        for k in sorted(summary):
            w.writerow([k, summary[k]])
    print(f"[Saved] {run_dir}")


if __name__ == "__main__":
    main()
