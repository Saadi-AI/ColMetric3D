from __future__ import annotations

from typing import Dict, Tuple
import torch


def _is_legacy_moe_out_key(k: str) -> bool:
    return (
        k.startswith("moe_layer.experts.") and k.endswith(("net.6.weight", "net.6.bias"))
    ) or k in {
        "moe_layer.single_head.net.6.weight",
        "moe_layer.single_head.net.6.bias",
    }


def adapt_legacy_residual_moe_tensor(k: str, v: torch.Tensor, target: torch.Tensor):
    """Adapt old 1-channel residual MoE heads to the new 3-channel MSEC heads.

    Old residual head output: [residual]
    New MSEC head output: [delta_log_scale, delta_shift, delta_residual]

    The old residual weights are copied into channel 2. Scale and shift channels
    are left at the current model initialization, usually zero. This lets older
    ADA/residual-MoE checkpoints initialize the residual part of MSEC safely.
    """
    if not _is_legacy_moe_out_key(k):
        return None
    if tuple(v.shape) == tuple(target.shape):
        return v
    if v.ndim == target.ndim and v.shape[0] == 1 and target.shape[0] == 3 and tuple(v.shape[1:]) == tuple(target.shape[1:]):
        out = target.detach().clone()
        out[2:3].copy_(v.detach())
        return out
    return None


def filter_compatible_state_dict(
    state: Dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]], Dict[str, str]]:
    """Prepare a checkpoint state dict for safe loading.

    - Keeps tensors with matching shapes.
    - Adapts legacy 1-channel residual MoE/MSEC output heads to the new 3-channel
      MSEC heads when possible.
    - Drops incompatible tensors that cannot be adapted.

    This avoids PyTorch size-mismatch errors while preserving useful old weights.
    """
    current = model.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    skipped: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]] = {}
    adapted: Dict[str, str] = {}

    for k, v in state.items():
        if k in current and tuple(v.shape) != tuple(current[k].shape):
            v_adapted = adapt_legacy_residual_moe_tensor(k, v, current[k])
            if v_adapted is not None:
                filtered[k] = v_adapted
                adapted[k] = f"{tuple(v.shape)} -> {tuple(current[k].shape)}; copied old residual to MSEC residual channel"
            else:
                skipped[k] = (tuple(v.shape), tuple(current[k].shape))
            continue
        filtered[k] = v
    return filtered, skipped, adapted


def load_model_state_safely(model: torch.nn.Module, state: Dict[str, torch.Tensor], strict: bool = False):
    filtered, skipped, adapted = filter_compatible_state_dict(state, model)
    missing, unexpected = model.load_state_dict(filtered, strict=strict)
    return missing, unexpected, skipped, adapted
