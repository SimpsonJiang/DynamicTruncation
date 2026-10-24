"""First-crossing imagination truncation driven by value-head disagreement."""

import torch
from torch import nn


class ThresholdEMA(nn.Module):
    """Use the previous EMA for gating, then update it with the current quantile."""

    def __init__(self, decay=0.999):
        super().__init__()
        if not 0.0 <= decay < 1.0:
            raise ValueError("dynamic_trunc.threshold_ema must be in [0, 1)")
        self.decay = float(decay)
        self.register_buffer("value", torch.zeros((), dtype=torch.float32))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def forward(self, raw):
        raw = raw.detach().reshape(()).to(device=self.value.device, dtype=self.value.dtype)
        previous = self.value.clone()
        used = torch.where(self.initialized, previous, raw)
        updated = torch.where(self.initialized, self.decay * previous + (1.0 - self.decay) * raw, raw)
        self.value.copy_(updated)
        self.initialized.fill_(True)
        return used


@torch.no_grad()
def build_truncation_mask(uncertainty, threshold, min_horizon=8):
    """Keep a forced prefix, then only steps before the first threshold crossing.

    uncertainty and the returned boolean mask have shape (trajectories, H, 1).
    Early crossings still stop the suffix after the forced prefix.
    """
    if uncertainty.ndim != 3 or uncertainty.shape[-1] != 1 or uncertainty.shape[1] < 1:
        raise ValueError("uncertainty must have shape (trajectories, H, 1) with H >= 1")
    if min_horizon < 0:
        raise ValueError("dynamic_trunc.min_horizon must be non-negative")
    below = (uncertainty <= threshold) & torch.isfinite(uncertainty)
    active = below.to(torch.int32).cumprod(dim=1).bool()
    prefix = torch.arange(uncertainty.shape[1], device=uncertainty.device) < min_horizon
    return active | prefix[None, :, None]


@torch.no_grad()
def normalized_mask(mask):
    """Preserve mean loss scale, returning zeros when all positions are masked."""
    mask = mask.float()
    mean = mask.mean()
    scaled = mask / mean.clamp_min(torch.finfo(mask.dtype).eps)
    return torch.where(mean > 0, scaled, torch.zeros_like(mask))


@torch.no_grad()
def mask_to_last(mask):
    """Stop return recursion before an untrusted outgoing step, retaining bootstrap."""
    if mask.ndim != 3 or mask.shape[-1] != 1:
        raise ValueError("mask must have shape (trajectories, H, 1)")
    trajectories, horizon, _ = mask.shape
    last = torch.ones(trajectories, horizon + 1, 1, dtype=torch.float32, device=mask.device)
    last[:, 0] = 0.0
    last[:, 1:horizon] = (~mask[:, 1:]).float()
    return last
