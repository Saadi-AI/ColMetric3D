from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict

import torch
from checkpoint_utils import load_model_state_safely
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from options import get_opts
from datasets import C3VDDataset
from datasets_simcol import SimColDataset
try:
    from networks import EndoMetric3DModel
except ImportError:
    from networks.endometric3d_model import EndoMetric3DModel
from metrics import compute_both_depth_metrics, MetricAccumulator


def move_to_device(batch: Dict, device: torch.device) -> Dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def load_model_checkpoint(model: torch.nn.Module, ckpt_dir: str, checkpoint: str, device: torch.device):
    ckpt_dir = Path(ckpt_dir)
    candidates = []
    if checkpoint:
        candidates.append(Path(checkpoint))
    candidates += [
        ckpt_dir / "best.pth",
        ckpt_dir / "latest.pth",
        ckpt_dir / "best_checkpoint.pth",
        ckpt_dir / "latest_checkpoint.pth",
    ]
    ckpt_path = next((p for p in candidates if p.exists()), None)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found. Checked: {[str(p) for p in candidates]}")
    state = torch.load(ckpt_path, map_location=device)
    state_dict = state.get("model_state", state.get("model_state_dict", state.get("model", state)))
    missing, unexpected, skipped, adapted = load_model_state_safely(model, state_dict, strict=False)
    print(f"[Checkpoint] loaded {ckpt_path}")
    print(f"[Checkpoint] missing={len(missing)} unexpected={len(unexpected)} skipped_shape_mismatch={len(skipped)} adapted_legacy_moe={len(adapted)}")
    if adapted:
        for k, msg in list(adapted.items())[:8]:
            print(f"  [adapt legacy] {k}: {msg}")
    if skipped:
        for k, (old_shape, new_shape) in list(skipped.items())[:8]:
            print(f"  [skip mismatch] {k}: ckpt={old_shape} model={new_shape}")
    return ckpt_path


def print_depth_table(results: Dict[str, float]):
    def g(k):
        return results.get(k, float("nan"))

    print("\n" + "=" * 88)
    print("EndoMetric3D Depth Evaluation")
    print("=" * 88)
    print("PRIMARY: Native metric evaluation, NO median scaling / NO post-hoc scale correction")
    print(f"  AbsRel       : {g('native/abs_rel'):.5f}")
    print(f"  SqRel        : {g('native/sq_rel'):.5f}")
    print(f"  RMSE (cm)    : {g('native/rmse'):.5f}")
    print(f"  MAE  (cm)    : {g('native/mae_cm'):.5f}")
    print(f"  RMSElog      : {g('native/rmse_log'):.5f}")
    print(f"  MeanLogScale : {g('native/mls'):.5f}")
    print(f"  delta1       : {g('native/a1'):.5f}")
    print(f"  delta2       : {g('native/a2'):.5f}")
    print(f"  delta3       : {g('native/a3'):.5f}")
    print(f"  Boundary-F1  : {g('native/boundary_f1'):.5f}")

    print("\nSECONDARY: Per-image median-scaled evaluation, for relative-depth baselines only")
    print(f"  AbsRel       : {g('scaled/abs_rel'):.5f}")
    print(f"  SqRel        : {g('scaled/sq_rel'):.5f}")
    print(f"  RMSE (cm)    : {g('scaled/rmse'):.5f}")
    print(f"  MAE  (cm)    : {g('scaled/mae_cm'):.5f}")
    print(f"  RMSElog      : {g('scaled/rmse_log'):.5f}")
    print(f"  delta1       : {g('scaled/a1'):.5f}")
    print(f"  Median scale : {g('scaled/scale'):.5f}  # should be near 1 if native scale is good")
    print("=" * 88)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--eval_split", default="test", choices=["train", "val", "test", "test_iii", "all"])
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--save_json", default="results/depth_metrics.json")
    parser.add_argument("--save_csv", default="results/depth_metrics.csv")
    script_args, remaining = parser.parse_known_args()

    opt = get_opts(remaining)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    if str(opt.get("dataset", "c3vd")).lower() == "simcol":
        root = opt["simcol_root"]
        ds = SimColDataset(root, opt, split=script_args.eval_split)
    else:
        if script_args.eval_split == "train":
            root = opt["c3vd_root"]
        elif script_args.eval_split == "val":
            root = opt["c3vd_val_root"]
        else:
            root = opt.get("c3vd_test_root") or opt["c3vd_val_root"]
            if not Path(root).exists():
                print(f"[WARN] test root not found ({root}); using val root")
                root = opt["c3vd_val_root"]
        ds = C3VDDataset(root, opt, split=script_args.eval_split if script_args.eval_split in {"train", "val", "test"} else "test")

    loader = DataLoader(ds, batch_size=opt["batch_size"], shuffle=False, num_workers=opt["num_workers"], pin_memory=True)
    print(f"[Dataset] dataset={opt.get('dataset')} split={script_args.eval_split} root={root} samples={len(ds)}")

    model = EndoMetric3DModel(opt).to(device)
    load_model_checkpoint(model, opt["ckpt_dir"], script_args.checkpoint, device)
    model.eval()

    meter = MetricAccumulator()
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval-depth"):
            batch = move_to_device(batch, device)
            outputs = model(batch)
            if "depth_gt" not in batch or "metric_depth" not in outputs:
                continue
            pred = F.interpolate(outputs["metric_depth"], size=batch["depth_gt"].shape[-2:], mode="bilinear", align_corners=False)
            gt = batch["depth_gt"]
            has_depth = batch.get("has_depth", torch.ones(gt.shape[0], dtype=torch.bool, device=device)).bool()
            if not has_depth.any():
                continue
            metrics = compute_both_depth_metrics(pred[has_depth], gt[has_depth], opt["min_depth_cm"], opt["max_depth_cm"])
            meter.update(metrics, n=int(has_depth.sum().item()))

    results = meter.compute()
    print_depth_table(results)

    out_json = Path(script_args.save_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[Saved] {out_json}")

    out_csv = Path(script_args.save_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in sorted(results.items()):
            writer.writerow([k, v])
    print(f"[Saved] {out_csv}")


if __name__ == "__main__":
    main()