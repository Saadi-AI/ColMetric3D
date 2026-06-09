from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import EndoDAV2Backbone
from .moe_layer import SparseMoEResidualLayer
from .moe_router import PhysicalGlarePrior
from .pose_net import PoseNet


class IntrinsicsCalibrator(nn.Module):
    """Learned intrinsics-conditioned calibration for metric depth.

    K is used as conditioning input. Hard focal scaling is optional and disabled
    by default for supervised C3VD metric-depth training.
    """

    def __init__(self, canonical_f_norm: float = 0.57, hard_focal_scaling: bool = False):
        super().__init__()
        self.canonical_f_norm = float(canonical_f_norm)
        self.hard_focal_scaling = bool(hard_focal_scaling)

        self.mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        self.base_log_gain = nn.Parameter(torch.tensor(1.0).log())
        self.base_bias = nn.Parameter(torch.tensor(0.05))

    def forward(self, rel_depth: torch.Tensor, K: Optional[torch.Tensor]):
        b, _, h, w = rel_depth.shape

        if K is None:
            kvec = torch.zeros(b, 4, device=rel_depth.device, dtype=rel_depth.dtype)
            focal_factor = torch.ones(b, 1, 1, 1, device=rel_depth.device, dtype=rel_depth.dtype)
        else:
            fx, fy = K[:, 0, 0], K[:, 1, 1]
            cx, cy = K[:, 0, 2], K[:, 1, 2]

            kvec = torch.stack(
                [fx / float(w), fy / float(h), cx / float(w), cy / float(h)],
                dim=1,
            ).to(rel_depth.dtype)

            f_norm = 0.5 * (kvec[:, 0] + kvec[:, 1]).view(b, 1, 1, 1)

            if self.hard_focal_scaling:
                focal_factor = f_norm / max(self.canonical_f_norm, 1e-6)
            else:
                focal_factor = torch.ones_like(f_norm)

        delta = self.mlp(kvec).view(b, 2, 1, 1)

        gain = torch.exp(self.base_log_gain + 0.1 * delta[:, 0:1]) * focal_factor
        bias = F.softplus(self.base_bias + 0.1 * delta[:, 1:2])

        metric = F.softplus(rel_depth * gain + bias)
        return metric, gain, bias
    
class AdaptiveDepthAdjustment(nn.Module):
    """ADA-style local depth distribution refinement.

    Inspired by ColonFMGC's local depth-distribution adjustment, but adapted to
    the current DAv2/SG-MoE pipeline. Around each relative-depth estimate, the
    head predicts K local multiplicative bin offsets and probabilities:

        D_refined = sum_k softmax(logits_k) * D_base * exp(offset_k)

    This makes the SG-MoE/decoder more expressive than a single residual and is
    especially useful for folds, far-depth, and high-gradient regions.
    """

    def __init__(self, embed_dim: int = 768, hidden_dim: int = 192, num_bins: int = 8, max_log_offset: float = 0.35):
        super().__init__()
        self.num_bins = int(num_bins)
        self.max_log_offset = float(max_log_offset)
        self.stem = nn.Sequential(
            nn.Conv2d(embed_dim + 1, hidden_dim, 3, padding=1),
            nn.GroupNorm(min(16, hidden_dim), hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(min(16, hidden_dim), hidden_dim),
            nn.GELU(),
        )
        self.logit_head = nn.Conv2d(hidden_dim, self.num_bins, 1)
        self.offset_head = nn.Conv2d(hidden_dim, self.num_bins, 1)
        self.uncertainty_head = nn.Conv2d(hidden_dim, 1, 1)
        nn.init.zeros_(self.logit_head.weight)
        nn.init.zeros_(self.logit_head.bias)
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        nn.init.zeros_(self.uncertainty_head.weight)
        nn.init.constant_(self.uncertainty_head.bias, -2.0)

    def forward(self, features: torch.Tensor, base_rel_depth: torch.Tensor) -> Dict[str, torch.Tensor]:
        if base_rel_depth.shape[-2:] != features.shape[-2:]:
            base_rel_depth = F.interpolate(base_rel_depth, size=features.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([features, base_rel_depth], dim=1)
        h = self.stem(x)
        logits = self.logit_head(h)
        probs = F.softmax(logits, dim=1)
        log_offsets = self.max_log_offset * torch.tanh(self.offset_head(h))
        centers = base_rel_depth.clamp(min=1e-6) * torch.exp(log_offsets)
        refined = (probs * centers).sum(dim=1, keepdim=True)
        # Relative variance gives a useful uncertainty/reliability proxy.
        var = (probs * (centers - refined).pow(2)).sum(dim=1, keepdim=True)
        rel_var = var / refined.pow(2).clamp(min=1e-6)
        uncertainty = F.softplus(self.uncertainty_head(h)) + rel_var
        entropy = -(probs * probs.clamp(min=1e-8).log()).sum(dim=1, keepdim=True)
        return {
            "refined_rel_depth": refined,
            "ada_bin_probs": probs,
            "ada_bin_centers": centers,
            "ada_uncertainty": uncertainty,
            "ada_entropy": entropy,
        }


class ArtifactBridge(nn.Module):
    def __init__(self, num_experts: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(num_experts + 1, 16, 1), nn.GELU(), nn.Conv2d(16, 1, 1), nn.Sigmoid())
        nn.init.zeros_(self.net[-2].weight)
        nn.init.constant_(self.net[-2].bias, -2.0)

    def forward(self, route_probs: torch.Tensor, route_entropy: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([route_probs, route_entropy], dim=1))


class EndoMetric3DModel(nn.Module):
    def __init__(self, opt: Dict):
        super().__init__()
        self.opt = opt
        self.backbone = EndoDAV2Backbone(opt)
        self.moe_layer = SparseMoEResidualLayer(opt)
        self.use_ada_depth = bool(opt.get("use_ada_depth", False))
        self.ada_depth = AdaptiveDepthAdjustment(
            embed_dim=int(opt.get("embed_dim", 768)),
            hidden_dim=int(opt.get("ada_hidden_dim", 192)),
            num_bins=int(opt.get("ada_num_bins", 8)),
            max_log_offset=float(opt.get("ada_max_log_offset", 0.35)),
        )
        self.calibrator = IntrinsicsCalibrator(
            canonical_f_norm=float(opt.get("canonical_f_norm", 0.57)),
            hard_focal_scaling=bool(opt.get("hard_focal_scaling", False)),
        )
        self.artifact_bridge = ArtifactBridge(int(opt.get("num_experts", 4)))
        self.glare_prior = PhysicalGlarePrior(opt.get("glare_brightness_thr", 0.78), opt.get("glare_saturation_thr", 0.55))
        # Single final pose implementation: metric-depth-conditioned PoseNet.
        self.pose_head = PoseNet(
            max_translation_cm=float(opt.get("pose_max_translation_cm", 0.10)),
            imagenet_init=bool(opt.get("pose_imagenet_init", True)),
            rot_scale_rad=float(opt.get("pose_rot_scale_rad", 0.05)),
            input_channels=int(opt.get("pose_input_channels", 9)),
            use_metric_depth=bool(opt.get("pose_use_metric_depth", True)),
            use_intrinsics=bool(opt.get("pose_use_intrinsics", True)),
            use_tissue_mask=bool(opt.get("pose_use_tissue_mask", True)),
            max_depth_cm=float(opt.get("max_depth_cm", 10.0)),
            detach_depth=bool(opt.get("pose_detach_depth_for_pose", False)),
            mag_bias_init=float(opt.get("pose_mag_bias_init", -2.0)),
        )
        self.pose_head_returns_aux = True
        self.depth_residual_scale = float(opt.get("depth_residual_scale", 0.10))
        self.msec_max_log_scale = float(opt.get("msec_max_log_scale", 0.20))
        self.msec_max_shift = float(opt.get("msec_max_shift", 0.10))
        self.msec_residual_scale = float(opt.get("msec_residual_scale", 0.05))
        self.min_depth_cm = float(opt.get("min_depth_cm", 0.05))
        self.max_depth_cm = float(opt.get("max_depth_cm", 10.0))
        self.use_artifact_mask = bool(opt.get("use_artifact_mask", True))

    def _run_pose_head(
        self,
        img_t: torch.Tensor,
        img_s: torch.Tensor,
        tissue_t: torch.Tensor,
        prefix: str,
        outputs: Dict[str, torch.Tensor],
        depth_t: Optional[torch.Tensor] = None,
        depth_s: Optional[torch.Tensor] = None,
        K: Optional[torch.Tensor] = None,
    ):
        pose_out = self.pose_head(img_t, img_s, tissue_t, depth_t=depth_t, depth_next=depth_s, K=K)
        if isinstance(pose_out, tuple) and len(pose_out) == 3:
            pose_matrix, pose_vec, aux = pose_out
            for k, v in aux.items():
                outputs[f"{k}_{prefix}"] = v
        else:
            pose_matrix, pose_vec = pose_out
        return pose_matrix, pose_vec

    def _metric_depth_from_features(self, feat: torch.Tensor, rel_depth_backbone: torch.Tensor, img: torch.Tensor, K: Optional[torch.Tensor]):
        """Predict native metric depth.

        Pipeline:
            DAv2 relative prior -> optional ADA local-bin refinement ->
            optional MSEC/SG-MoE local scale-shift calibration -> global intrinsics calibrator.

        The MoE experts do not predict full depth. They predict dense metric-scale
        calibration fields: local log-scale, local shift, and a small residual. This
        makes the expert module directly relevant to native metric scale instead of
        being a generic residual head.
        """
        rel_low = F.interpolate(rel_depth_backbone, size=feat.shape[-2:], mode="bilinear", align_corners=False).clamp(min=1e-6)

        ada_outputs = {}
        refined_low = rel_low
        if self.use_ada_depth:
            ada_outputs = self.ada_depth(feat, refined_low)
            refined_low = ada_outputs["refined_rel_depth"].clamp(min=1e-6)

        moe_out, route = self.moe_layer(feat, return_route=True)
        # MSEC mode: experts output [delta_log_scale, delta_shift, delta_residual].
        if moe_out.shape[1] >= 3 and bool(self.opt.get("use_moe", False)):
            delta_log_scale = self.msec_max_log_scale * torch.tanh(moe_out[:, 0:1])
            delta_shift = self.msec_max_shift * torch.tanh(moe_out[:, 1:2])
            delta_resid = self.msec_residual_scale * torch.tanh(moe_out[:, 2:3])
            refined_low = (refined_low * torch.exp(delta_log_scale) + delta_shift + delta_resid).clamp(min=1e-6)
            residual_for_log = delta_resid
            ada_outputs["msec_delta_log_scale"] = delta_log_scale
            ada_outputs["msec_delta_shift"] = delta_shift
            ada_outputs["msec_delta_residual"] = delta_resid
        else:
            # Legacy residual mode / disabled MoE.
            residual = moe_out[:, :1]
            if bool(self.opt.get("use_moe", False)):
                refined_low = (refined_low + self.depth_residual_scale * residual).clamp(min=1e-6)
            residual_for_log = residual

        refined = F.interpolate(refined_low, size=img.shape[-2:], mode="bilinear", align_corners=False)
        metric, gain, bias = self.calibrator(refined, K)
        metric = metric.clamp(min=self.min_depth_cm, max=self.max_depth_cm)
        residual_up = F.interpolate(residual_for_log, size=img.shape[-2:], mode="bilinear", align_corners=False)
        return metric, route, gain, bias, residual_up, ada_outputs

    def _artifact_and_tissue_masks(self, route: Dict[str, torch.Tensor], img_01: torch.Tensor, out_hw: Tuple[int, int], ada_uncertainty: Optional[torch.Tensor] = None):
        probs = route["all_probs"]
        entropy = route["entropy"]
        learned_art = self.artifact_bridge(probs, entropy)
        learned_art = F.interpolate(learned_art, size=out_hw, mode="bilinear", align_corners=False)
        physical = self.glare_prior(img_01)
        if physical.shape[-2:] != out_hw:
            physical = F.interpolate(physical, size=out_hw, mode="bilinear", align_corners=False)
        if self.use_artifact_mask:
            artifact = torch.maximum(learned_art, physical)
        else:
            artifact = torch.zeros_like(physical)
        tissue = (1.0 - artifact).clamp(0.0, 1.0)
        if ada_uncertainty is not None and bool(self.opt.get("use_depth_uncertainty_for_pose", True)):
            unc = F.interpolate(ada_uncertainty, size=out_hw, mode="bilinear", align_corners=False)
            reliability = torch.exp(-unc.clamp(min=0.0, max=5.0))
            tissue = tissue * reliability
        return artifact, tissue, learned_art, physical

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        img_t = inputs["image"]
        img_next = inputs.get("image_next")
        img_prev = inputs.get("image_prev")
        img_t_01 = inputs.get("image_01", img_t.clamp(0.0, 1.0))
        K = inputs.get("K")

        bb = self.backbone(img_t, img_next)
        depth_t, route_t, gain_t, bias_t, residual_t, ada_t = self._metric_depth_from_features(bb["feat_t"], bb["rel_depth_t"], img_t, K)
        ada_unc_t = ada_t.get("ada_uncertainty") if ada_t else None
        artifact_t, tissue_t, learned_art_t, physical_glare_t = self._artifact_and_tissue_masks(route_t, img_t_01, img_t.shape[-2:], ada_unc_t)
        outputs: Dict[str, torch.Tensor] = {
            "metric_depth": depth_t,
            "depth_gain": gain_t,
            "depth_bias": bias_t,
            "depth_residual": residual_t,
            "lb_loss": route_t["lb_loss"],
            "moe_probs": route_t["all_probs"],
            "moe_logits": route_t.get("logits", route_t["all_probs"]),
            "moe_entropy": route_t.get("entropy", torch.zeros_like(route_t["all_probs"][:, :1])),
            "artifact_mask": artifact_t,
            "tissue_mask": tissue_t,
            "learned_artifact_mask": learned_art_t,
            "physical_glare_prior": physical_glare_t,
        }
        if ada_t:
            for k, v in ada_t.items():
                outputs[k] = v if v.shape[-2:] == img_t.shape[-2:] else F.interpolate(v, size=img_t.shape[-2:], mode="bilinear", align_corners=False)
        if "proto_target" in route_t:
            outputs["moe_proto_target"] = route_t["proto_target"]
        if "proto_logits" in route_t:
            outputs["moe_proto_logits"] = route_t["proto_logits"]

        lb_terms = [route_t["lb_loss"]]
        if img_next is not None:
            if "feat_next" in bb:
                depth_next, route_next, _, _, _, _ = self._metric_depth_from_features(bb["feat_next"], bb["rel_depth_next"], img_next, K)
                outputs["metric_depth_next"] = depth_next
                lb_terms.append(route_next["lb_loss"])
            if bool(self.opt.get("train_pose_head", True)):
                pose_matrix, pose_vec = self._run_pose_head(
                    img_t, img_next, tissue_t, "next", outputs,
                    depth_t=depth_t, depth_s=outputs.get("metric_depth_next"), K=K,
                )
                outputs["pose_matrix"] = pose_matrix
                outputs["pose"] = pose_vec

        if img_prev is not None:
            bb_prev = self.backbone(img_prev)
            depth_prev, route_prev, _, _, _, _ = self._metric_depth_from_features(bb_prev["feat_t"], bb_prev["rel_depth_t"], img_prev, K)
            outputs["metric_depth_prev"] = depth_prev
            lb_terms.append(route_prev["lb_loss"])
            if bool(self.opt.get("train_pose_head", True)):
                pose_prev, pose_prev_vec = self._run_pose_head(
                    img_t, img_prev, tissue_t, "prev", outputs,
                    depth_t=depth_t, depth_s=outputs.get("metric_depth_prev"), K=K,
                )
                outputs["pose_matrix_prev"] = pose_prev
                outputs["pose_prev"] = pose_prev_vec

        if len(lb_terms) > 1:
            outputs["lb_loss"] = torch.stack([x if x.ndim > 0 else x.reshape(()) for x in lb_terms]).mean()
        return outputs

    def trainable_parameter_groups(self, base_lr: float):
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        new_params = []
        for m in [self.moe_layer, self.ada_depth, self.calibrator, self.artifact_bridge, self.pose_head]:
            new_params += [p for p in m.parameters() if p.requires_grad]
        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": base_lr * float(self.opt.get("backbone_lr_mult", 0.25)), "name": "backbone"})
        if new_params:
            groups.append({"params": new_params, "lr": base_lr, "name": "heads_pose_msec_ada"})
        return groups