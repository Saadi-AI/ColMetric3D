from __future__ import annotations

from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models



class CBAM(nn.Module):
    """Compact channel-spatial attention used inside PoseNet."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        ch = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        x = x * ch
        avg_sp = x.mean(dim=1, keepdim=True)
        max_sp = x.amax(dim=1, keepdim=True)
        sp = torch.sigmoid(self.spatial(torch.cat([avg_sp, max_sp], dim=1)))
        return x * sp


def axis_angle_to_matrix(axisangle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle vectors [B,3] to rotation matrices [B,3,3]."""
    angle = torch.linalg.norm(axisangle, dim=1, keepdim=True).clamp(min=1e-9)
    axis = axisangle / angle
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    ca = torch.cos(angle[:, 0])
    sa = torch.sin(angle[:, 0])
    C = 1.0 - ca
    Rm = torch.zeros(axisangle.shape[0], 3, 3, device=axisangle.device, dtype=axisangle.dtype)
    Rm[:, 0, 0] = ca + x * x * C
    Rm[:, 0, 1] = x * y * C - z * sa
    Rm[:, 0, 2] = x * z * C + y * sa
    Rm[:, 1, 0] = y * x * C + z * sa
    Rm[:, 1, 1] = ca + y * y * C
    Rm[:, 1, 2] = y * z * C - x * sa
    Rm[:, 2, 0] = z * x * C - y * sa
    Rm[:, 2, 1] = z * y * C + x * sa
    Rm[:, 2, 2] = ca + z * z * C
    return Rm


class PoseNet(nn.Module):
    """Metric-depth-anchored, reliability-aware SE(3) pose network.

    The head predicts a relative transform in centimeters from an RGB pair,
    optional metric-depth pair, tissue/reliability mask, and normalized camera
    intrinsics. Translation remains direction/magnitude decoupled:

        t = m * normalize(v)

    The metric depth maps are the scale anchor; K is supplied as a compact geometric conditioning vector.
    """

    def __init__(
        self,
        max_translation_cm: float = 0.10,
        imagenet_init: bool = True,
        rot_scale_rad: float = 0.05,
        input_channels: int = 9,
        use_metric_depth: bool = True,
        use_intrinsics: bool = True,
        use_tissue_mask: bool = True,
        max_depth_cm: float = 10.0,
        detach_depth: bool = False,
        mag_bias_init: float = -2.0,
    ):
        super().__init__()
        self.max_translation_cm = float(max_translation_cm)
        self.rot_scale_rad = float(rot_scale_rad)
        self.use_metric_depth = bool(use_metric_depth)
        self.use_intrinsics = bool(use_intrinsics)
        self.use_tissue_mask = bool(use_tissue_mask)
        self.max_depth_cm = float(max_depth_cm)
        self.detach_depth = bool(detach_depth)
        self.mag_bias_init = float(mag_bias_init)
        self.input_channels = int(input_channels)

        try:
            weights = models.ResNet18_Weights.DEFAULT if imagenet_init else None
            self.backbone = models.resnet18(weights=weights)
        except Exception:
            self.backbone = models.resnet18(weights=None)

        old_conv = self.backbone.conv1
        new_conv = nn.Conv2d(
            self.input_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
            if old_conv.weight.shape[1] == 3:
                # Keep ImageNet priors for the two RGB frames. Extra geometry/mask
                # channels start near zero so the first updates are stable.
                copy_rgb = min(3, self.input_channels)
                new_conv.weight[:, :copy_rgb].copy_(old_conv.weight[:, :copy_rgb])
                if self.input_channels >= 6:
                    new_conv.weight[:, 3:6].copy_(old_conv.weight)
                if self.input_channels > 6:
                    new_conv.weight[:, 6:].mul_(0.01)
        self.backbone.conv1 = new_conv
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.cbam = CBAM(512)

        intr_dim = 32 if self.use_intrinsics else 0
        if self.use_intrinsics:
            self.intrinsics_mlp = nn.Sequential(
                nn.Linear(4, 32),
                nn.LayerNorm(32),
                nn.GELU(),
                nn.Linear(32, 32),
                nn.GELU(),
            )
        else:
            self.intrinsics_mlp = None

        self.shared = nn.Sequential(
            nn.Linear(in_features + intr_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.05),
        )
        self.rot_head = nn.Linear(256, 3)
        self.dir_head = nn.Linear(256, 3)
        self.mag_head = nn.Linear(256, 1)
        
        # Identity-ish start, but not near-zero translation collapse. With the
        # default C3VD cap 0.10 cm and bias -2, magnitude starts around 0.012 cm,
        # close to observed adjacent-frame C3VD motion.
        nn.init.zeros_(self.rot_head.weight)
        nn.init.zeros_(self.rot_head.bias)
        nn.init.xavier_uniform_(self.dir_head.weight, gain=0.01)
        nn.init.zeros_(self.dir_head.bias)
        nn.init.zeros_(self.mag_head.weight)
        nn.init.constant_(self.mag_head.bias, self.mag_bias_init)

    def _depth_channel(self, depth: Optional[torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
        b, _, h, w = ref.shape
        if depth is None or (not self.use_metric_depth):
            return torch.zeros((b, 1, h, w), device=ref.device, dtype=ref.dtype)
        if self.detach_depth:
            depth = depth.detach()
        if depth.shape[-2:] != (h, w):
            depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=False)
        depth = depth.to(device=ref.device, dtype=ref.dtype).clamp(min=1e-4, max=max(self.max_depth_cm, 1e-4))
        # Log compression keeps centimeter-valued depth numerically comparable to RGB.
        return torch.log1p(depth) / torch.log1p(torch.tensor(self.max_depth_cm, device=ref.device, dtype=ref.dtype))

    def _k_vector(self, K: Optional[torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
        b, _, h, w = ref.shape
        if K is None or (not self.use_intrinsics):
            return torch.zeros((b, 4), device=ref.device, dtype=ref.dtype)
        K = K.to(device=ref.device, dtype=ref.dtype)
        fx, fy = K[:, 0, 0], K[:, 1, 1]
        cx, cy = K[:, 0, 2], K[:, 1, 2]
        return torch.stack([fx / float(w), fy / float(h), cx / float(w), cy / float(h)], dim=1)

    def _encode(
        self,
        img_t: torch.Tensor,
        img_next: torch.Tensor,
        tissue_mask: torch.Tensor,
        depth_t: Optional[torch.Tensor] = None,
        depth_next: Optional[torch.Tensor] = None,
        K: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if tissue_mask.shape[-2:] != img_t.shape[-2:]:
            tissue_mask = F.interpolate(tissue_mask, size=img_t.shape[-2:], mode="bilinear", align_corners=False)
        tissue_mask = tissue_mask.to(device=img_t.device, dtype=img_t.dtype).clamp(0.0, 1.0)
        if not self.use_tissue_mask:
            tissue_mask = torch.ones_like(tissue_mask)
        if self.use_metric_depth:
            d_t = self._depth_channel(depth_t, img_t)
            d_n = self._depth_channel(depth_next, img_t)
            x = torch.cat([img_t, img_next, d_t, d_n, tissue_mask], dim=1)
        else:
            x = torch.cat([img_t, img_next, tissue_mask], dim=1)
        if x.shape[1] != self.input_channels:
            # Backward compatibility for checkpoints/experiments with a different stack.
            if x.shape[1] > self.input_channels:
                x = x[:, : self.input_channels]
            else:
                pad = torch.zeros(x.shape[0], self.input_channels - x.shape[1], *x.shape[-2:], device=x.device, dtype=x.dtype)
                x = torch.cat([x, pad], dim=1)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.cbam(x)
        x = self.backbone.avgpool(x)
        visual = torch.flatten(x, 1)
        if self.use_intrinsics and self.intrinsics_mlp is not None:
            visual = torch.cat([visual, self.intrinsics_mlp(self._k_vector(K, img_t))], dim=1)
        return visual

    def forward(
        self,
        img_t: torch.Tensor,
        img_next: torch.Tensor,
        tissue_mask: torch.Tensor,
        depth_t: Optional[torch.Tensor] = None,
        depth_next: Optional[torch.Tensor] = None,
        K: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        feat = self.shared(self._encode(img_t, img_next, tissue_mask, depth_t=depth_t, depth_next=depth_next, K=K))
        axisangle = self.rot_scale_rad * torch.tanh(self.rot_head(feat))
        direction_raw = self.dir_head(feat)
        direction = F.normalize(direction_raw, dim=-1, eps=1e-6)
        magnitude = self.max_translation_cm * torch.sigmoid(self.mag_head(feat))
        translation = direction * magnitude
        pose_vec = torch.cat([axisangle, translation], dim=1)
        pose_mat = self.pose_vec2mat(axisangle, translation)
        aux = {
            "pose_mu": pose_vec,
            "pose_dir_pred": direction,
            "pose_mag_pred": magnitude.squeeze(1),
            "pose_uses_metric_depth": torch.ones((), device=img_t.device, dtype=img_t.dtype) if self.use_metric_depth else torch.zeros((), device=img_t.device, dtype=img_t.dtype),
            "pose_uses_tissue_mask": torch.ones((), device=img_t.device, dtype=img_t.dtype) if self.use_tissue_mask else torch.zeros((), device=img_t.device, dtype=img_t.dtype),
        }
        return pose_mat, pose_vec, aux

    @staticmethod
    def pose_vec2mat(axisangle: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
        b = axisangle.shape[0]
        T = torch.eye(4, device=axisangle.device, dtype=axisangle.dtype).unsqueeze(0).repeat(b, 1, 1)
        T[:, :3, :3] = axis_angle_to_matrix(axisangle)
        T[:, :3, 3] = translation
        return T
