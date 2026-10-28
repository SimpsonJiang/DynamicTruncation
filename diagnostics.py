"""Detached measurements that do not contribute to the training objective."""

import torch


@torch.no_grad()
def imagined_entropy_metrics(entropy, mask=None):
    """Summarize (starts, H, 1) entropies; bootstrap is excluded by the caller.

    The valid mean weights retained positions equally, without continuation or
    discount weights. An empty mask has a defined value of zero.
    """
    entropy = entropy.detach().float()
    if mask is None:
        valid_mean = entropy.mean()
    else:
        valid_mean = torch.where(mask, entropy, 0.0).sum() / mask.sum().clamp_min(1)
    return {
        "action_entropy_valid": valid_mean,
        "imag_action_entropy": entropy.mean(dim=0),
    }
