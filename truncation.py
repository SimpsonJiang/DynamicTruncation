"""Value-ensemble uncertainty and first-crossing imagination truncation."""

import math

import torch
from torch import nn

from distributions import Quantile, TwoHot, symlog


@torch.no_grad()
def cal_ens_uncertainty(imag_value_dist, imag_value, metric, *, beta=0.1, distance_space="raw"):
    """Return detached float32 uncertainty with shape (B*T, H, 1).

    imag_value_dist is the list of K online value distributions with batch shape
    (B*T, H+1). imag_value contains *all* K heads' scalar predictions, with shape
    (K, B*T, H+1, 1), not the two-head bootstrap mean. The last state is excluded.
    Only inputs needed by the selected metric are evaluated. beta affects UCB only.
    distance_space selects raw or symlog bins for W1 and both Cramer scores.
    """
    if metric != "wasser" and imag_value_dist is not None and any(isinstance(d, Quantile) for d in imag_value_dist):
        raise ValueError("Quantile ensembles support only the wasser uncertainty metric")
    if metric == "wasser":
        return _imag_wasserstein(imag_value_dist, space=distance_space)
    if metric in ("cramer", "sqrt_cramer"):
        return _imag_cramer(imag_value_dist, take_sqrt=metric == "sqrt_cramer", space=distance_space)
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


def _distance_bin_widths(value_dist, space):
    """Integrate CDF differences in the selected return coordinate system."""
    if space not in ("raw", "symlog"):
        raise ValueError("model.distance_space must be 'raw' or 'symlog'")
    bins = value_dist.bins.detach().to(dtype=torch.float64)
    if space == "symlog":
        bins = symlog(bins)
    return bins.diff()


@torch.no_grad()
def _imag_wasserstein(value_dists, *, space="raw"):
    """Mean pairwise W1 on raw or symlog bins; float64 math, float32 output."""
    if any(isinstance(dist, Quantile) for dist in value_dists):
        categorical = [dist for dist in value_dists if isinstance(dist, TwoHot)]
        quantile = [dist for dist in value_dists if isinstance(dist, Quantile)]
        if len(value_dists) != 2 or len(categorical) != 1 or len(quantile) != 1:
            raise ValueError("Mixed W1 requires exactly one TwoHot and one Quantile distribution")
        return _categorical_quantile_wasserstein(categorical[0], quantile[0], space=space)
    widths = _distance_bin_widths(value_dists[0], space)
    if len(value_dists) == 1:
        return torch.zeros_like(value_dists[0].probs[:, :-1, :1], dtype=torch.float32)
    # Recompute probabilities: casting float32 softmax output retains mass errors
    # that the large raw-return bin gaps can amplify into spurious distances.
    cdfs = [
        dist.logits.detach()[:, :-1].to(dtype=torch.float64).softmax(dim=-1).cumsum(dim=-1)[..., :-1]
        for dist in value_dists
    ]
    total = torch.zeros_like(cdfs[0][..., :1])
    for i in range(len(cdfs)):
        for j in range(i + 1, len(cdfs)):
            total = total + ((cdfs[i] - cdfs[j]).abs() * widths).sum(dim=-1, keepdim=True)
    return (total / (len(cdfs) * (len(cdfs) - 1) // 2)).to(dtype=torch.float32)


@torch.no_grad()
def _categorical_quantile_wasserstein(categorical, quantile, *, space="raw"):
    """Exact CDF integration on the union of supports, with bounded working memory.

    At most 2048 trajectory/time rows are sorted together. No C-by-Q tensor is
    constructed. Training quantile order is untouched; this path is detached.
    """
    if space not in ("raw", "symlog"):
        raise ValueError("model.distance_space must be 'raw' or 'symlog'")
    if categorical.logits.shape[:-1] != quantile.quantiles.shape[:-1]:
        raise ValueError("Categorical and quantile batch shapes must match")
    logits = categorical.logits.detach()[:, :-1]
    atoms = quantile.quantiles.detach()[:, :-1]
    batch_shape = logits.shape[:-1]
    bins = categorical.bins.detach().to(torch.float64)
    if space == "symlog":
        bins = symlog(bins)
    logits = logits.reshape(-1, logits.shape[-1])
    atoms = atoms.reshape(-1, atoms.shape[-1])
    pieces = []
    for cat_logits, quantiles in zip(logits.split(2048), atoms.split(2048)):
        probs = cat_logits.to(torch.float64).softmax(dim=-1)
        quantiles = quantiles.to(torch.float64)
        if space == "symlog":
            quantiles = symlog(quantiles)
        quantiles = quantiles.sort(dim=-1).values.contiguous()
        support = torch.cat([bins.expand(probs.shape[0], -1), quantiles], dim=-1).sort(dim=-1).values
        left = support[:, :-1].contiguous()
        widths = support.diff(dim=-1)
        zero = torch.zeros_like(probs[:, :1])
        prefix = torch.cat([zero, probs.cumsum(dim=-1)], dim=-1)
        survival = torch.cat([probs.flip(-1).cumsum(-1).flip(-1), zero], dim=-1)
        # Sum from the closer endpoint: exactly empty far tails stay exactly empty.
        cdf = torch.where(prefix <= 0.5, prefix, 1.0 - survival)
        cat_index = torch.searchsorted(bins, left, right=True)
        cat_cdf = cdf.gather(-1, cat_index)
        quantile_cdf = torch.searchsorted(quantiles, left, right=True).to(torch.float64) / quantiles.shape[-1]
        distance = ((cat_cdf - quantile_cdf).abs() * widths).sum(dim=-1, keepdim=True)
        pieces.append(distance.to(torch.float32))
    return torch.cat(pieces, dim=0).reshape(*batch_shape, 1)


@torch.no_grad()
def _imag_cramer(value_dists, take_sqrt=False, *, space="raw"):
    """Mean pairwise squared Cramer score, optionally rooted before averaging.

    Integrate on raw or symlog bins. Softmax, CDF, integration, and optional roots
    use float64; output is float32.
    """
    widths = _distance_bin_widths(value_dists[0], space)
    if len(value_dists) == 1:
        return torch.zeros_like(value_dists[0].probs[:, :-1, :1], dtype=torch.float32)
    cdfs = [
        dist.logits.detach()[:, :-1].to(dtype=torch.float64).softmax(dim=-1).cumsum(dim=-1)[..., :-1]
        for dist in value_dists
    ]
    total = torch.zeros_like(cdfs[0][..., :1])
    for i in range(len(cdfs)):
        for j in range(i + 1, len(cdfs)):
            distance = ((cdfs[i] - cdfs[j]).square() * widths).sum(dim=-1, keepdim=True)
            if take_sqrt:
                distance = distance.clamp_min(0.0).sqrt()
            total = total + distance
    return (total / (len(cdfs) * (len(cdfs) - 1) // 2)).to(dtype=torch.float32)


class AdvantageScaleEMA(nn.Module):
    """Normalize full-horizon advantages using a separate, lagged absolute quantile.

    Nonfinite entries are excluded from the quantile. An entirely invalid batch
    leaves both buffers unchanged; invalid normalized advantages remain invalid.
    """

    def __init__(self, quantile=0.9, decay=0.99, eps=1e-6):
        super().__init__()
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("dynamic_trunc.advantage.scale_quantile must be in [0, 1]")
        if not 0.0 <= decay < 1.0:
            raise ValueError("dynamic_trunc.advantage.scale_ema must be in [0, 1)")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("dynamic_trunc.advantage.scale_eps must be finite and positive")
        self.quantile = float(quantile)
        self.decay = float(decay)
        self.eps = float(eps)
        self.register_buffer("value", torch.zeros((), dtype=torch.float32))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def forward(self, advantage):
        advantage = advantage.detach().float()
        finite = torch.isfinite(advantage)
        raw = torch.nanquantile(torch.where(finite, advantage.abs(), torch.nan).flatten(), self.quantile)
        valid = torch.isfinite(raw)
        safe_raw = torch.where(valid, raw, torch.zeros_like(raw))
        previous = self.value.clone()
        scale = torch.where(self.initialized, previous, safe_raw).clamp_min(self.eps)
        updated = torch.where(self.initialized, self.decay * previous + (1 - self.decay) * safe_raw, safe_raw)
        self.value.copy_(torch.where(valid, updated, previous))
        self.initialized.logical_or_(valid)
        return advantage / scale, scale


@torch.no_grad()
def advantage_truncation_score(uncertainty, advantage, *, mode="positive_only", max_factor=2.0):
    """Rescale nonnegative uncertainty by bounded, dimensionless advantage.

    Both inputs have shape (B*T, H, 1). off/max_factor=1 are exact identity
    paths, including when advantage is invalid. Otherwise invalid inputs cross
    the threshold; build_truncation_mask still enforces its minimum prefix.
    """
    if mode not in ("off", "positive_only", "signed"):
        raise ValueError("dynamic_trunc.advantage.mode must be 'off', 'positive_only' or 'signed'")
    if not math.isfinite(max_factor) or max_factor < 1:
        raise ValueError("dynamic_trunc.advantage.max_factor must be finite and >= 1")
    uncertainty = uncertainty.detach().float()
    if mode == "off" or max_factor == 1:
        return uncertainty
    advantage = advantage.detach().float()
    valid = torch.isfinite(uncertainty) & (uncertainty >= 0) & torch.isfinite(advantage)
    shaped = advantage.clamp_min(0) if mode == "positive_only" else advantage
    score = uncertainty / (max_factor ** shaped.tanh())
    return torch.where(valid, score, torch.full_like(score, torch.inf))


@torch.no_grad()
def advantage_drop_fractions(advantage, weight, mask):
    """Fractions of weighted positive/negative full advantage removed by mask.

    Ignore nonfinite inputs for diagnostics, and return zero for zero mass.
    """
    advantage, weight = advantage.detach().float(), weight.detach().float()
    valid = torch.isfinite(advantage) & torch.isfinite(weight)
    advantage = torch.where(valid, advantage, torch.zeros_like(advantage))
    weight = torch.where(valid, weight.clamp_min(0), torch.zeros_like(weight))
    fractions = []
    for mass in (advantage.clamp_min(0), (-advantage).clamp_min(0)):
        mass = mass * weight
        total = mass.sum()
        dropped = torch.where(mask, torch.zeros_like(mass), mass).sum()
        fractions.append(dropped / torch.where(total > 0, total, torch.ones_like(total)))
    return fractions[0], fractions[1]


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
