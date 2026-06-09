from __future__ import annotations

from typing import Dict, Optional
import math
import torch


def _nan_dict(keys):
    return {k: float("nan") for k in keys}


def compute_boundary_f1(
    pred: torch.Tensor,
    gt: torch.Tensor,
    min_depth: float = 0.05,
    max_depth: float = 10.0,
) -> float:
    """Depth-discontinuity F1. pred/gt are [B,1,H,W] in cm."""
    pred = pred.float().clamp(min=1e-6)
    gt = gt.float().clamp(min=1e-6)
    valid = (gt > min_depth) & (gt < max_depth) & (pred > min_depth) & torch.isfinite(gt) & torch.isfinite(pred)

    valid_x = valid[:, :, :, :-1] & valid[:, :, :, 1:]
    valid_y = valid[:, :, :-1, :] & valid[:, :, 1:, :]
    if valid_x.sum() < 10 or valid_y.sum() < 10:
        return float("nan")

    gt_rx = torch.maximum(gt[:, :, :, :-1] / gt[:, :, :, 1:], gt[:, :, :, 1:] / gt[:, :, :, :-1])
    gt_ry = torch.maximum(gt[:, :, :-1, :] / gt[:, :, 1:, :], gt[:, :, 1:, :] / gt[:, :, :-1, :])
    pr_rx = torch.maximum(pred[:, :, :, :-1] / pred[:, :, :, 1:], pred[:, :, :, 1:] / pred[:, :, :, :-1])
    pr_ry = torch.maximum(pred[:, :, :-1, :] / pred[:, :, 1:, :], pred[:, :, 1:, :] / pred[:, :, :-1, :])

    thresholds = torch.linspace(1.05, 1.15, 10, device=pred.device)
    score = 0.0
    n = 0
    for th in thresholds:
        gt_bx, pr_bx = (gt_rx > th) & valid_x, (pr_rx > th) & valid_x
        gt_by, pr_by = (gt_ry > th) & valid_y, (pr_ry > th) & valid_y
        tp = (gt_bx & pr_bx).sum() + (gt_by & pr_by).sum()
        fp = (~gt_bx & pr_bx).sum() + (~gt_by & pr_by).sum()
        fn = (gt_bx & ~pr_bx).sum() + (gt_by & ~pr_by).sum()
        precision = tp.float() / (tp + fp + 1e-7)
        recall = tp.float() / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        score += f1.item()
        n += 1
    return score / max(n, 1)


def mean_log_scale(
    pred: torch.Tensor,
    gt: torch.Tensor,
    min_depth: float = 0.05,
    max_depth: float = 10.0,
) -> float:
    valid = (gt > min_depth) & (gt < max_depth) & (pred > min_depth) & torch.isfinite(gt) & torch.isfinite(pred)
    if valid.sum() < 10:
        return float("nan")
    return torch.abs(torch.log(gt[valid].clamp(min=1e-6) / pred[valid].clamp(min=1e-6))).mean().item()


def _compute_depth_errors_single(
    pred: torch.Tensor,
    gt: torch.Tensor,
    min_depth: float,
    max_depth: float,
    scale_pred: bool,
) -> Dict[str, float]:
    """Per-image depth metrics. pred/gt are [1,H,W] or [H,W] in cm."""
    pred = pred.float()
    gt = gt.float()
    valid = (gt > min_depth) & (gt < max_depth) & (pred > min_depth) & torch.isfinite(gt) & torch.isfinite(pred)
    if valid.sum() < 10:
        return _nan_dict(["abs_rel", "sq_rel", "rmse", "rmse_log", "mae_cm", "a1", "a2", "a3", "mls", "scale", "boundary_f1"])

    p = pred[valid]
    g = gt[valid]
    scale = 1.0
    if scale_pred:
        scale = (g.median() / p.median().clamp(min=1e-7)).item()
    p = p * scale

    thresh = torch.maximum(g / p.clamp(min=1e-7), p / g.clamp(min=1e-7))
    abs_rel = torch.mean(torch.abs(g - p) / g).item()
    sq_rel = torch.mean((g - p) ** 2 / g).item()
    rmse = torch.sqrt(torch.mean((g - p) ** 2)).item()
    rmse_log = torch.sqrt(torch.mean((torch.log(g.clamp(min=1e-6)) - torch.log(p.clamp(min=1e-6))) ** 2)).item()
    mae_cm = torch.mean(torch.abs(g - p)).item()
    a1 = torch.mean((thresh < 1.25).float()).item()
    a2 = torch.mean((thresh < 1.25 ** 2).float()).item()
    a3 = torch.mean((thresh < 1.25 ** 3).float()).item()

    pred_scaled = pred.unsqueeze(0) * scale if pred.ndim == 3 else pred.view(1, 1, *pred.shape[-2:]) * scale
    gt4 = gt.unsqueeze(0) if gt.ndim == 3 else gt.view(1, 1, *gt.shape[-2:])
    mls = mean_log_scale(pred_scaled, gt4, min_depth, max_depth)
    bf1 = compute_boundary_f1(pred_scaled, gt4, min_depth, max_depth)

    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "rmse_log": rmse_log,
        "mae_cm": mae_cm,
        "a1": a1,
        "a2": a2,
        "a3": a3,
        "mls": mls,
        "scale": scale,
        "boundary_f1": bf1,
    }


def compute_depth_errors(
    pred: torch.Tensor,
    gt: torch.Tensor,
    min_depth: float = 0.05,
    max_depth: float = 10.0,
    scale_pred: bool = False,
) -> Dict[str, float]:
    """
    Depth metrics averaged per image.

    Native metric evaluation: scale_pred=False. This is your main result.
    Median-scaled evaluation: scale_pred=True. Use only for comparison with relative-depth SSL baselines.
    """
    if pred.ndim == 3:
        pred = pred.unsqueeze(1)
    if gt.ndim == 3:
        gt = gt.unsqueeze(1)
    b = min(pred.shape[0], gt.shape[0])
    sums, counts = {}, {}
    for i in range(b):
        m = _compute_depth_errors_single(pred[i], gt[i], min_depth, max_depth, scale_pred)
        for k, v in m.items():
            if isinstance(v, float) and math.isnan(v):
                continue
            sums[k] = sums.get(k, 0.0) + float(v)
            counts[k] = counts.get(k, 0) + 1
    return {k: sums[k] / max(counts[k], 1) for k in sums}


def compute_both_depth_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    min_depth: float = 0.05,
    max_depth: float = 10.0,
) -> Dict[str, float]:
    native = compute_depth_errors(pred, gt, min_depth, max_depth, scale_pred=False)
    scaled = compute_depth_errors(pred, gt, min_depth, max_depth, scale_pred=True)
    out = {f"native/{k}": v for k, v in native.items()}
    out.update({f"scaled/{k}": v for k, v in scaled.items()})
    return out


def rotation_geodesic_deg(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    R_err = torch.matmul(R_pred.transpose(-1, -2), R_gt)
    tr = R_err[..., 0, 0] + R_err[..., 1, 1] + R_err[..., 2, 2]
    cos = ((tr - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.rad2deg(torch.acos(cos))


def compute_pose_errors(
    pred_pose: torch.Tensor,
    gt_pose: torch.Tensor,
    has_pose: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Relative-pose metrics. Poses are [B,4,4], translation in cm."""
    if pred_pose is None or gt_pose is None:
        return {}
    pred_pose = pred_pose.float()
    gt_pose = gt_pose.float()
    if has_pose is not None:
        mask = has_pose.bool().view(-1)
        pred_pose = pred_pose[mask]
        gt_pose = gt_pose[mask]
    if pred_pose.numel() == 0:
        return {}

    t_pred = pred_pose[:, :3, 3]
    t_gt = gt_pose[:, :3, 3]
    trans_vec_err = torch.linalg.norm(t_pred - t_gt, dim=-1)
    trans_rmse = torch.sqrt(torch.mean(trans_vec_err ** 2)).item()
    trans_mae = torch.mean(trans_vec_err).item()
    trans_comp_l1 = torch.mean(torch.abs(t_pred - t_gt)).item()

    r_deg = rotation_geodesic_deg(pred_pose[:, :3, :3], gt_pose[:, :3, :3])
    rot_deg = torch.mean(r_deg).item()

    pred_norm = torch.linalg.norm(t_pred, dim=-1).clamp(min=1e-7)
    gt_norm = torch.linalg.norm(t_gt, dim=-1).clamp(min=1e-7)
    scale_log = torch.mean(torch.abs(torch.log(pred_norm / gt_norm))).item()

    return {
        "rel_trans_cm": trans_rmse,
        "rel_trans_mae_cm": trans_mae,
        "rel_trans_l1_comp_cm": trans_comp_l1,
        "rel_rot_deg": rot_deg,
        "pose_scale_log": scale_log,
    }


class MetricAccumulator:
    def __init__(self):
        self._sums: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def update(self, metrics: Dict[str, float], n: int = 1):
        for k, v in metrics.items():
            if v is None:
                continue
            try:
                fv = float(v)
            except Exception:
                continue
            if math.isnan(fv) or math.isinf(fv):
                continue
            self._sums[k] = self._sums.get(k, 0.0) + fv * n
            self._counts[k] = self._counts.get(k, 0) + n

    def compute(self) -> Dict[str, float]:
        return {k: self._sums[k] / max(1, self._counts.get(k, 1)) for k in sorted(self._sums.keys())}
