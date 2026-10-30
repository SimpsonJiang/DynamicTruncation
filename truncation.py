"""Value-ensemble uncertainty and first-crossing imagination truncation."""

import torch
from torch import nn


@torch.no_grad()
def cal_ens_uncertainty(imag_value_dist, imag_value, metric, *, beta=0.1):
    """Return detached float32 uncertainty with shape (B*T, H, 1).

    imag_value_dist is the list of K online value distributions with batch shape
    (B*T, H+1). imag_value contains *all* K heads' scalar predictions, with shape
    (K, B*T, H+1, 1), not the two-head bootstrap mean. The last state is excluded.
    Only inputs needed by the selected metric are evaluated. beta affects UCB only.
    """
    if metric == "wasser":
        return _imag_wasserstein(imag_value_dist)
    if metric in ("cramer", "sqrt_cramer"):
        return _imag_cramer(imag_value_dist, take_sqrt=metric == "sqrt_cramer")
    if metric == "ucb":
        return _imag_ucb(imag_value, beta=beta)
    if metric == "sigma":
        return _imag_sigma(imag_value)
    raise ValueError(f"Unknown uncertainty metric: {metric!r}")


@torch.no_grad()
def _imag_ucb(imag_value, *, beta=0.1):
    """Mean + beta * sample std over all heads, excluding the bootstrap state."""
    values = imag_value.detach()[:, :, :-1].to(dtype=torch.float32)
    if values.shape[0] == 1:
        return values[0]
    std, mean = torch.std_mean(values, dim=0, unbiased=True)
    return mean + beta * std


@torch.no_grad()
def _imag_sigma(imag_value):
    """Sample std over all heads, excluding the bootstrap state."""
    values = imag_value.detach()[:, :, :-1].to(dtype=torch.float32)
    if values.shape[0] == 1:
        return torch.zeros_like(values[0])
    return torch.std(values, dim=0, unbiased=True)


@torch.no_grad()
def _imag_wasserstein(value_dists):
    """Per-state mean pairwise W1 on raw return bins, excluding bootstrap."""
    if len(value_dists) == 1:
        return torch.zeros_like(value_dists[0].probs[:, :-1, :1], dtype=torch.float32)
    cdfs = [dist.probs.detach()[:, :-1].to(dtype=torch.float32).cumsum(dim=-1)[..., :-1] for dist in value_dists]
    widths = value_dists[0].bins.detach().to(dtype=torch.float32).diff()
    total = torch.zeros_like(cdfs[0][..., :1])
    for i in range(len(cdfs)):
        for j in range(i + 1, len(cdfs)):
            total = total + ((cdfs[i] - cdfs[j]).abs() * widths).sum(dim=-1, keepdim=True)
    return total / (len(cdfs) * (len(cdfs) - 1) // 2)


@torch.no_grad()
def _imag_cramer(value_dists, take_sqrt=False):
    """Mean pairwise squared Cramer score, optionally rooted before averaging."""
    if len(value_dists) == 1:
        return torch.zeros_like(value_dists[0].probs[:, :-1, :1], dtype=torch.float32)
    cdfs = [dist.probs.detach()[:, :-1].to(dtype=torch.float32).cumsum(dim=-1)[..., :-1] for dist in value_dists]
    widths = value_dists[0].bins.detach().to(dtype=torch.float32).diff()
    total = torch.zeros_like(cdfs[0][..., :1])
    for i in range(len(cdfs)):
        for j in range(i + 1, len(cdfs)):
            distance = ((cdfs[i] - cdfs[j]).square() * widths).sum(dim=-1, keepdim=True)
            if take_sqrt:
                distance = distance.clamp_min(0.0).sqrt()
            total = total + distance
    return total / (len(cdfs) * (len(cdfs) - 1) // 2)


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
