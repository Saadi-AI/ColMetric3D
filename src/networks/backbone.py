"""Depth Anything V2 backbone with LA-SVFT adaptation.

The key idea is not to make DAv2 metric by itself. DAv2 provides strong
structure features; metric scale is learned from C3VD depth supervision and
intrinsics-conditioned calibration in the full EndoMetric3D model.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation


class AdapterLinearBase(nn.Module):
    def __init__(self, base_linear: nn.Linear):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.weight = nn.Parameter(base_linear.weight.detach().clone(), requires_grad=False)
        self.bias = None
        if base_linear.bias is not None:
            self.bias = nn.Parameter(base_linear.bias.detach().clone(), requires_grad=False)


class LoRALinear(AdapterLinearBase):
    """Standard LoRA: W_eff = W0 + B A."""
    def __init__(self, base_linear: nn.Linear, rank: int = 8, alpha: float = 1.0):
        super().__init__(base_linear)
        self.rank = min(int(rank), self.in_features, self.out_features)
        self.alpha = float(alpha)
        self.A = nn.Parameter(torch.empty(self.rank, self.in_features))
        self.B = nn.Parameter(torch.zeros(self.out_features, self.rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def delta_weight(self) -> torch.Tensor:
        return (self.B @ self.A) * (self.alpha / max(1, self.rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.delta_weight().to(dtype=self.weight.dtype, device=self.weight.device)
        return F.linear(x, self.weight + delta, self.bias)


class DoRALinear(LoRALinear):
    """DoRA-style weight decomposition: LoRA updates direction, magnitude is trainable."""
    def __init__(self, base_linear: nn.Linear, rank: int = 8, alpha: float = 1.0):
        super().__init__(base_linear, rank=rank, alpha=alpha)
        with torch.no_grad():
            mag = torch.linalg.norm(self.weight.detach().float(), dim=1)
        self.magnitude = nn.Parameter(mag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight + self.delta_weight().to(dtype=self.weight.dtype, device=self.weight.device)
        direction = F.normalize(w.float(), dim=1).to(dtype=w.dtype)
        w_eff = direction * self.magnitude.to(dtype=w.dtype, device=w.device).view(-1, 1)
        return F.linear(x, w_eff, self.bias)


class SVFTLinear(AdapterLinearBase):
    """SVD/SVFT adapter using frozen singular vectors and a trainable diagonal band.

    svft mode uses only a diagonal matrix M. la_svft mode uses a small banded
    M (diagonal + first upper/lower diagonals), allowing local interaction
    between nearby singular directions. Both preserve the pretrained function at
    initialization because all adapter parameters start at zero.
    """
    def __init__(self, base_linear: nn.Linear, rank: int = 8, alpha: float = 1.0, banded: bool = False):
        super().__init__(base_linear)
        self.rank = min(int(rank), self.in_features, self.out_features)
        self.alpha = float(alpha)
        self.banded = bool(banded)
        with torch.no_grad():
            U, S, Vh = torch.linalg.svd(self.weight.detach().cpu().float(), full_matrices=False)
            self.register_buffer("U_r", U[:, : self.rank].contiguous())
            self.register_buffer("V_r", Vh[: self.rank, :].T.contiguous())
            self.register_buffer("S_r", S[: self.rank].contiguous())
        self.diag = nn.Parameter(torch.zeros(self.rank))
        if self.banded and self.rank > 1:
            self.upper = nn.Parameter(torch.zeros(self.rank - 1))
            self.lower = nn.Parameter(torch.zeros(self.rank - 1))
        else:
            self.register_buffer("upper", torch.zeros(0))
            self.register_buffer("lower", torch.zeros(0))

    def delta_weight(self) -> torch.Tensor:
        dtype = self.diag.dtype
        M = torch.diag(self.diag)
        if self.banded and self.rank > 1:
            M = M + torch.diag(self.upper, diagonal=1) + torch.diag(self.lower, diagonal=-1)
        delta = self.U_r.to(dtype=dtype, device=self.diag.device) @ M @ self.V_r.T.to(dtype=dtype, device=self.diag.device)
        return delta * (self.alpha / max(1, self.rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.delta_weight().to(dtype=self.weight.dtype, device=self.weight.device)
        return F.linear(x, self.weight + delta, self.bias)


# Backward-compatible class name used by older checkpoints.
class LASVFTLinear(SVFTLinear):
    def __init__(self, base_linear: nn.Linear, rank: int = 8, banded: bool = True, alpha: float = 1.0):
        super().__init__(base_linear, rank=rank, alpha=alpha, banded=banded)


def make_adapter_linear(base_linear: nn.Linear, adapter_type: str, rank: int, alpha: float) -> nn.Module:
    adapter_type = str(adapter_type).lower()
    if adapter_type == "lora":
        return LoRALinear(base_linear, rank=rank, alpha=alpha)
    if adapter_type == "dora":
        return DoRALinear(base_linear, rank=rank, alpha=alpha)
    if adapter_type == "svft":
        return SVFTLinear(base_linear, rank=rank, alpha=alpha, banded=False)
    if adapter_type in {"la_svft", "lasvft"}:
        return SVFTLinear(base_linear, rank=rank, alpha=alpha, banded=True)
    raise ValueError(f"Unknown adapter_type: {adapter_type}")


def replace_linear_with_adapter(module: nn.Module, rank: int, target_keywords: Iterable[str], adapter_type: str, alpha: float = 1.0) -> int:
    count = 0
    keywords = [k.lower() for k in target_keywords]
    for name, child in list(module.named_children()):
        lname = name.lower()
        if isinstance(child, nn.Linear) and any(k in lname for k in keywords):
            setattr(module, name, make_adapter_linear(child, adapter_type=adapter_type, rank=rank, alpha=alpha))
            count += 1
        else:
            count += replace_linear_with_adapter(child, rank=rank, target_keywords=keywords, adapter_type=adapter_type, alpha=alpha)
    return count


def replace_linear_with_lasvft(module: nn.Module, rank: int, target_keywords: Iterable[str], alpha: float = 1.0) -> int:
    return replace_linear_with_adapter(module, rank=rank, target_keywords=target_keywords, adapter_type="la_svft", alpha=alpha)


class EndoDAV2Backbone(nn.Module):
    """Depth Anything V2 feature/depth wrapper.

    Returns a DAv2 relative-depth prior and final ViT patch features. The full
    EndoMetric3D model turns these into metric depth using supervised C3VD
    metric labels and camera intrinsics.
    """

    def __init__(self, opt: Dict):
        super().__init__()
        self.opt = opt
        model_name = opt.get("backbone_model_name", "depth-anything/Depth-Anything-V2-Base-hf")
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
        self.vit_backbone = getattr(self.model, "backbone", None)
        if self.vit_backbone is None:
            raise RuntimeError("Could not find .backbone in the HuggingFace Depth Anything model.")

        # Optional PEFT adapter. Main method uses LA-SVFT.
        adapter_type = str(opt.get("adapter_type", opt.get("backbone_mode", "la_svft"))).lower()
        if adapter_type in {"frozen", "none", "no_svft"}:
            adapter_type = "none"
        self.adapter_type = adapter_type
        self.num_adapter_layers = 0
        if adapter_type != "none":
            keywords = opt.get("svft_target_keywords", ["query", "key", "value", "dense", "fc1", "fc2", "proj", "qkv"])
            self.num_adapter_layers = replace_linear_with_adapter(
                self.vit_backbone,
                rank=int(opt.get("adapter_rank", opt.get("lasvft_rank", 8))),
                target_keywords=keywords,
                adapter_type=adapter_type,
                alpha=float(opt.get("adapter_alpha", opt.get("lasvft_alpha", 8.0))),
            )
            print(f"[EndoDAV2] {adapter_type} adapted {self.num_adapter_layers} linear layers")

        # Freeze ViT base, then unfreeze only adapter parameters. DPT decode head remains trainable by default.
        if opt.get("freeze_vit_backbone", True):
            for p in self.vit_backbone.parameters():
                p.requires_grad_(False)
            trainable_names = ["diag", "upper", "lower", "A", "B", "magnitude"]
            for n, p in self.vit_backbone.named_parameters():
                if any(k in n for k in trainable_names):
                    p.requires_grad_(True)

        if opt.get("freeze_dav2_decode_head", False):
            for n, p in self.model.named_parameters():
                if not n.startswith("backbone"):
                    p.requires_grad_(False)

        self.predicted_depth_is_inverse = bool(opt.get("dav2_output_is_inverse_depth", False))

    def _convert_dav2_depth(self, predicted_depth: torch.Tensor) -> torch.Tensor:
        # HF DAv2 predicted_depth is used directly by default. Some custom checkpoints may output inverse depth.
        d = predicted_depth.unsqueeze(1) if predicted_depth.ndim == 3 else predicted_depth
        d = d.float().clamp(min=1e-6)
        if self.predicted_depth_is_inverse:
            d = 1.0 / d
        # Normalize per image for stable residual/calibration learning; metric scale is learned downstream.
        if self.opt.get("normalize_dav2_prior", True):
            med = d.flatten(1).median(dim=1).values.view(-1, 1, 1, 1).clamp(min=1e-6)
            d = d / med
        return d

    @staticmethod
    def _tokens_to_feature_map(hidden: torch.Tensor, img_hw: Optional[tuple[int, int]] = None, patch_size: int = 14) -> torch.Tensor:
        """Convert ViT patch tokens to [B,C,Hf,Wf].

        The old implementation assumed a square token grid. That works for
        518x518 (37x37 patches), but fails for MIA-style C3VD resize
        476x588, where the patch grid is rectangular: 34x42 = 1428 tokens
        plus one CLS token -> 1429 tokens.
        """
        if hidden.ndim != 3:
            raise RuntimeError(f"Expected hidden state [B,N,C], got {tuple(hidden.shape)}")
        b, n, c = hidden.shape

        # Preferred path: use the actual input image size to infer the rectangular
        # DINOv2/DepthAnything patch grid. For C3VD 476x588 -> 34x42.
        if img_hw is not None:
            ih, iw = int(img_hw[0]), int(img_hw[1])
            gh, gw = ih // int(patch_size), iw // int(patch_size)
            expected = gh * gw
            if n == expected + 1:      # CLS + patches
                tokens = hidden[:, 1:, :]
            elif n == expected:        # patches only
                tokens = hidden
            else:
                # Some HF configs may still return square-ish/padded tokens. Fall back below.
                tokens = None
            if tokens is not None and tokens.shape[1] == expected:
                return tokens.permute(0, 2, 1).reshape(b, c, gh, gw).contiguous()

        # Backward-compatible fallback for square inputs such as 518x518.
        sq_n = int(math.sqrt(n))
        if sq_n * sq_n == n:
            tokens = hidden
            h = w = sq_n
        else:
            tokens = hidden[:, 1:, :]
            n2 = tokens.shape[1]
            sq_n2 = int(math.sqrt(n2))
            if sq_n2 * sq_n2 != n2:
                msg = (
                    f"Could not reshape {n} tokens to a square feature map. "
                    f"If using non-square input such as C3VD 476x588, make sure "
                    f"_tokens_to_feature_map receives img_hw and patch_size=14. "
                    f"Token count after CLS removal={n2}."
                )
                raise RuntimeError(msg)
            h = w = sq_n2
        return tokens.permute(0, 2, 1).reshape(b, c, h, w).contiguous()

    def _forward_one(self, img: torch.Tensor, prefix: str) -> Dict[str, torch.Tensor]:
        outputs = self.model(img, output_hidden_states=True, return_dict=True)
        rel_depth = self._convert_dav2_depth(outputs.predicted_depth)
        hidden_states = outputs.hidden_states
        patch_size = int(self.opt.get("dav2_patch_size", 14))
        feat = self._tokens_to_feature_map(hidden_states[-1], img_hw=tuple(img.shape[-2:]), patch_size=patch_size)
        return {f"rel_depth_{prefix}": rel_depth, f"feat_{prefix}": feat}

    def forward(self, img_t: torch.Tensor, img_next: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        out = self._forward_one(img_t, "t")
        if img_next is not None:
            out.update(self._forward_one(img_next, "next"))
        return out