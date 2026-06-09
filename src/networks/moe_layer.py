"""Sparse residual Mixture-of-Experts for metric depth.

Two modes are supported:
  1) unstructured: learned content router only
  2) feature_guided: learned content router with a *frozen prototype teacher*

Important: DAv2/LA-SVFT is NOT frozen by this module. Only the prototype centers
used for early router guidance are frozen. The router still receives the current
DAv2 feature map and learns a content-driven gate. The prototype teacher decays
during training, so the final router is learned from the task loss.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAdaptiveVisionRouter(nn.Module):
    def __init__(self, embed_dim: int, num_experts: int, top_k: int, lb_loss_coeff: float = 1e-2):
        super().__init__()
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.lb_loss_coeff = float(lb_loss_coeff)
        hidden = max(16, int(embed_dim) // 4)
        groups = min(16, hidden)
        self.gate = nn.Sequential(
            nn.Conv2d(embed_dim, hidden, 1, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, self.num_experts, 1, bias=True),
        )

    def _load_balancing_loss(self, probs: torch.Tensor) -> torch.Tensor:
        # Switch-style load balance between hard top-1 usage and soft probability mass.
        top1 = probs.argmax(dim=1)  # [B,H,W]
        f_i = F.one_hot(top1, num_classes=self.num_experts).float().mean(dim=(0, 1, 2))
        p_i = probs.mean(dim=(0, 2, 3))
        return self.lb_loss_coeff * self.num_experts * torch.sum(f_i * p_i)

    def forward(
        self,
        features: torch.Tensor,
        proto_logits: Optional[torch.Tensor] = None,
        proto_bias: float = 0.0,
        proto_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        learned_logits = self.gate(features)

        logits = learned_logits
        if proto_logits is not None and float(proto_bias) != 0.0:
            logits = logits + float(proto_bias) * proto_logits.to(dtype=logits.dtype)

        if self.training:
            logits = logits + 0.01 * torch.randn_like(logits)

        probs = F.softmax(logits, dim=1)  # [B,E,H,W]
        k = min(self.top_k, self.num_experts)
        top_weights, top_idx = probs.topk(k=k, dim=1)
        top_weights = top_weights / top_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        entropy = -(probs * probs.clamp(min=1e-8).log()).sum(dim=1, keepdim=True)

        out = {
            "logits": logits,
            "learned_logits": learned_logits,
            "all_probs": probs,
            "top_weights": top_weights,
            "top_idx": top_idx,
            "entropy": entropy,
            "lb_loss": self._load_balancing_loss(probs),
        }
        if proto_logits is not None:
            out["proto_logits"] = proto_logits
        if proto_target is not None:
            out["proto_target"] = proto_target
        return out


class DepthExpert(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, out_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(min(16, hidden_dim), hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(min(16, hidden_dim), hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, int(out_channels), 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SparseMoEResidualLayer(nn.Module):
    def __init__(self, opt: Dict):
        super().__init__()
        self.opt = opt
        self.enabled = bool(opt.get("use_moe", True))
        self.num_experts = int(opt.get("num_experts", 4))
        self.top_k = int(opt.get("moe_top_k", 2))
        self.routing_mode = str(opt.get("moe_routing", "unstructured")).lower()
        self.proto_bias = float(opt.get("moe_proto_bias", 0.0))
        self.proto_temp = float(opt.get("moe_proto_temp", 10.0))

        embed_dim = int(opt.get("embed_dim", 768))
        hidden_dim = int(opt.get("moe_hidden_dim", 256))
        self.output_channels = 3 if str(opt.get("moe_output", "msec")).lower() == "msec" else 1
        self.router = SelfAdaptiveVisionRouter(embed_dim, self.num_experts, self.top_k, opt.get("lb_loss_coeff", 1e-2))
        self.experts = nn.ModuleList([DepthExpert(embed_dim, hidden_dim, out_channels=self.output_channels) for _ in range(self.num_experts)])
        self.single_head = DepthExpert(embed_dim, hidden_dim, out_channels=self.output_channels)
        self.register_buffer("expert_usage", torch.zeros(self.num_experts))

        self.has_feature_prototypes = False
        proto_path = str(opt.get("moe_prototype_path", "") or "")
        if self.routing_mode == "feature_guided":
            if not proto_path:
                print("[MoE WARN] feature_guided requested but --moe_prototype_path is empty; using learned router only. For the proposed ablation, build prototypes first.")
            else:
                self._load_feature_prototypes(proto_path, embed_dim)

    def _load_feature_prototypes(self, path: str, embed_dim: int) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"MoE prototype file not found: {p}")
        data = np.load(str(p))
        centers = data["centers"] if "centers" in data else data["prototypes"]
        centers = np.asarray(centers, dtype=np.float32)
        if centers.ndim != 2:
            raise ValueError(f"Expected centers [E,C], got {centers.shape}")
        if centers.shape[0] != self.num_experts:
            raise ValueError(f"Prototype expert count {centers.shape[0]} != num_experts {self.num_experts}")
        if centers.shape[1] != embed_dim:
            raise ValueError(f"Prototype dim {centers.shape[1]} != embed_dim {embed_dim}")
        centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)
        self.register_buffer("feature_prototypes", torch.from_numpy(centers))
        self.has_feature_prototypes = True
        print(f"[MoE] loaded frozen router-teacher prototypes from {p} shape={centers.shape}. DAv2 remains trainable if enabled by options.")

    def reset_expert_usage(self) -> None:
        self.expert_usage.zero_()

    def get_expert_usage_fractions(self) -> torch.Tensor:
        return self.expert_usage / self.expert_usage.sum().clamp(min=1.0)

    def _prototype_logits(self, features: torch.Tensor):
        if not self.has_feature_prototypes:
            return None, None
        # cosine similarity between patch feature and frozen prototype centers
        f = F.normalize(features.float(), dim=1, eps=1e-6)
        p = F.normalize(self.feature_prototypes.float(), dim=1, eps=1e-6)
        proto_logits = torch.einsum("bchw,ec->behw", f, p) * self.proto_temp
        proto_target = proto_logits.argmax(dim=1).long()  # [B,H,W]
        return proto_logits.to(dtype=features.dtype), proto_target

    def forward(self, features: torch.Tensor, return_route: bool = True):
        b, _, h, w = features.shape
        if not self.enabled:
            # True no-MoE/no-residual baseline: backbone relative depth goes directly to calibrator.
            residual = torch.zeros(b, self.output_channels, h, w, device=features.device, dtype=features.dtype)
            dummy_probs = torch.full((b, self.num_experts, h, w), 1.0 / self.num_experts, device=features.device, dtype=features.dtype)
            route = {
                "logits": torch.zeros_like(dummy_probs),
                "learned_logits": torch.zeros_like(dummy_probs),
                "all_probs": dummy_probs,
                "top_weights": dummy_probs[:, :1],
                "top_idx": torch.zeros(b, 1, h, w, device=features.device, dtype=torch.long),
                "entropy": torch.ones(b, 1, h, w, device=features.device, dtype=features.dtype),
                "lb_loss": torch.zeros((), device=features.device, dtype=features.dtype),
            }
            return (residual, route) if return_route else residual

        proto_logits, proto_target = self._prototype_logits(features) if self.routing_mode == "feature_guided" else (None, None)
        route = self.router(features, proto_logits=proto_logits, proto_bias=self.proto_bias, proto_target=proto_target)

        weights = route["top_weights"]
        indices = route["top_idx"]
        expert_outputs = torch.stack([expert(features) for expert in self.experts], dim=1)  # [B,E,Cout,H,W]
        combined = torch.zeros(b, self.output_channels, h, w, device=features.device, dtype=features.dtype)
        for k in range(indices.shape[1]):
            idx_k = indices[:, k:k+1]  # [B,1,H,W]
            w_k = weights[:, k:k+1]
            gather_idx = idx_k.unsqueeze(2).expand(-1, -1, self.output_channels, -1, -1)
            selected = expert_outputs.gather(dim=1, index=gather_idx).squeeze(1)
            combined = combined + selected * w_k
            if self.training:
                with torch.no_grad():
                    for e in range(self.num_experts):
                        self.expert_usage[e] += (idx_k == e).sum().float()
        return (combined, route) if return_route else combined