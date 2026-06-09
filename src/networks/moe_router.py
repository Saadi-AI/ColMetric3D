from __future__ import annotations

import torch
import torch.nn as nn


class PhysicalGlarePrior(nn.Module):
    """Non-trainable soft prior for specular highlight / saturation regions.

    This is not used as a standalone semantic expert. It regularizes the learned
    artifact mask and down-weights unreliable regions for pose estimation and diagnostics
    regions.
    """

    def __init__(self, brightness_thr: float = 0.78, saturation_thr: float = 0.55):
        super().__init__()
        self.brightness_thr = float(brightness_thr)
        self.saturation_thr = float(saturation_thr)

    def forward(self, img_01: torch.Tensor) -> torch.Tensor:
        # img_01 [B,3,H,W] in [0,1]
        img = img_01.clamp(0.0, 1.0)
        maxc = img.max(dim=1, keepdim=True).values
        minc = img.min(dim=1, keepdim=True).values
        brightness = img.mean(dim=1, keepdim=True)
        saturation = maxc - minc
        bright_score = ((brightness - self.brightness_thr) / max(1e-6, 1.0 - self.brightness_thr)).clamp(0.0, 1.0)
        sat_score = ((saturation - self.saturation_thr) / max(1e-6, 1.0 - self.saturation_thr)).clamp(0.0, 1.0)
        white_score = ((maxc - 0.80) / 0.20).clamp(0.0, 1.0)
        return torch.maximum(bright_score, torch.maximum(sat_score, white_score))
