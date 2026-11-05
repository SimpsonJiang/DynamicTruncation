"""Run with python -m ens_toy_example.run; CPU only, one thread."""

import argparse
import copy
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.stats import rankdata, spearmanr
from torch import nn

from distributions import symexp_twohot
from networks import MLPHead

REGIONS = ("id", "gap", "extrapolation")


def truth(x):
    """Known conditional Uniform(mu(x)-scale(x), mu(x)+scale(x))."""
    return 2 * np.sin(2 * x) + 0.4 * x, 0.35 + 0.15 * np.cos(2 * x) ** 2


def region_masks(x):
    return {"id": (np.abs(x) >= 0.3) & (np.abs(x) <= 1), "gap": np.abs(x) < 0.3, "extrapolation": np.abs(x) > 1}


def distance_to_data(x):
    return np.maximum(np.maximum(0.3 - np.abs(x), np.abs(x) - 1), 0)


def dataset(n, rng):
    x = rng.uniform(0.3, 1, n) * rng.choice([-1, 1], n)
    mean, scale = truth(x)
    y = mean + rng.uniform(-1, 1, n) * scale
    return x.astype(np.float32), y.astype(np.float32)


def features(x):
    # Constant coordinate prevents first-layer RMSNorm erasing scalar magnitude
    # when biases start at zero. Same fixed, non-learned features for all heads.
    x = torch.as_tensor(x, dtype=torch.float32)
    return torch.stack((x, torch.ones_like(x)), dim=-1)


def make_ensemble(seed, width):
    config = OmegaConf.create(
        {
            "act": "SiLU",
            "symlog_inputs": False,
            "device": "cpu",
            "layers": 3,
            "units": width,
            "name": "value",
            "outscale": 0.0,
            "shape": [255],
            "dist": {"name": "symexp_twohot", "bin_num": 255},
        }
    )
    heads = []
    for i in range(5):
        torch.manual_seed(seed * 100 + i)
        heads.append(MLPHead(config, 2))
    return nn.ModuleList(heads)


def logits_of(heads, x):
    return torch.stack([h.last(h.mlp(x)) for h in heads])


def twohot(y, bins):
    """Vectorized scalar encoding; tested against repository TwoHot.log_prob."""
    y = y.clamp(bins[0], bins[-1])
    upper = torch.searchsorted(bins, y).clamp(1, len(bins) - 1)
    lower = upper - 1
    fraction = (y - bins[lower]) / (bins[upper] - bins[lower])
    labels = torch.zeros(*y.shape, len(bins), dtype=y.dtype)
    labels.scatter_add_(-1, lower.unsqueeze(-1), (1 - fraction).unsqueeze(-1))
    labels.scatter_add_(-1, upper.unsqueeze(-1), fraction.unsqueeze(-1))
    return labels


def projected_truth(x, bins):
    """Exact expected two-hot labels for the known uniform conditional law."""
    mean, scale = truth(x)
    a, b = (mean - scale)[:, None], (mean + scale)[:, None]
    left, right = bins[:-1][None], bins[1:][None]
    lo, hi = np.maximum(a, left), np.minimum(b, right)
    span = np.maximum(hi - lo, 0)
    extra = span * (right - (lo + hi) / 2) / (right - left) / (b - a)
    cdf = np.clip((left - a) / (b - a), 0, 1) + extra
    return np.diff(np.concatenate([np.zeros((len(x), 1)), cdf, np.ones((len(x), 1))], axis=1), axis=1)


def discrete_w1(p, q, bins):
    return (np.abs(np.cumsum(p - q, axis=-1)[..., :-1]) * np.diff(bins)).sum(-1)


def uniform_w1(p, bins, x):
    """Exact CDF integration against continuous uniform truth, including tails."""
    mean, scale = truth(x)
    a, b = mean - scale, mean + scale
    edges = np.sort(np.concatenate([np.broadcast_to(bins, (len(x), len(bins))), a[:, None], b[:, None]], 1), 1)
    left, right = edges[:, :-1], edges[:, 1:]
    index = np.searchsorted(bins, (left + right) / 2, side="right")
    cdf = np.concatenate([np.zeros((len(x), 1)), np.cumsum(p, axis=-1)], 1)
    # Roundoff at the final CDF must not create a fictitious infinite tail.
    cdf[:, -1] = 1.0
    pred = np.take_along_axis(cdf, index, axis=1)
    v0 = pred - np.clip((left - a[:, None]) / (b - a)[:, None], 0, 1)
    v1 = pred - np.clip((right - a[:, None]) / (b - a)[:, None], 0, 1)
    area = (np.abs(v0) + np.abs(v1)) / 2
    cross = v0 * v1 < 0
    area[cross] = (v0[cross] ** 2 + v1[cross] ** 2) / (2 * np.abs(v1[cross] - v0[cross]))
    return (area * (right - left)).sum(-1)


def score_arrays(logits, bins, x):
    probs = torch.as_tensor(logits).double().softmax(-1).numpy()
    cdf = probs.cumsum(-1)[..., :-1]
    widths = np.diff(bins)
    pairs = np.stack([np.abs(cdf[i] - cdf[j]) * widths for i in range(5) for j in range(i + 1, 5)])
    u = pairs.sum(-1).mean(0)
    mixture = probs.mean(0)
    projected = projected_truth(x, bins)
    means = (probs * bins).sum(-1)
    tail = (bins[:-1] < -5) | (bins[1:] > 5)
    quantiles = np.stack([bins[(probs.cumsum(-1) >= q).argmax(-1)] for q in (0.1, 0.5, 0.9)], -1)
    return {
        "probs": probs,
        "quantiles": quantiles,
        "means": means,
        "w1": u,
        "sigma": means.std(0, ddof=1),
        "error": uniform_w1(mixture, bins, x),
        "projected_error": discrete_w1(mixture, projected, bins),
        "projection_floor": uniform_w1(projected, bins, x),
        "mean_error": np.abs(means.mean(0) - truth(x)[0]),
        "w1_tail_fraction": pairs[..., tail].sum(-1).mean(0) / np.maximum(u, 1e-30),
    }


def correlation(a, b):
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(spearmanr(a, b).statistic)


def auroc(positive, negative):
    n, m = len(positive), len(negative)
    ranks = rankdata(np.concatenate([positive, negative]))
    return float((ranks[:n].sum() - n * (n + 1) / 2) / (n * m))


def risk_curve(error, score):
    # Average the entire tied group at a boundary, avoiding ordering artifacts.
    order = np.argsort(score, kind="stable")
    values, starts, counts = np.unique(score[order], return_index=True, return_counts=True)
    del values
    tied_means = np.repeat(np.add.reduceat(error[order], starts) / counts, counts)
    return np.cumsum(tied_means) / np.arange(1, len(error) + 1)


def summarize(x, arrays, threshold):
    masks = region_masks(x)
    result = {}
    for name, mask in masks.items():
        error, score = arrays["error"][mask], arrays["w1"][mask]
        distance = distance_to_data(x[mask])
        risk = risk_curve(error, score)
        result[name] = {
            "n": int(mask.sum()),
            "w1_mean": float(score.mean()),
            "w1_median": float(np.median(score)),
            "true_error_mean": float(error.mean()),
            "projected_error_mean": float(arrays["projected_error"][mask].mean()),
            "projection_floor_mean": float(arrays["projection_floor"][mask].mean()),
            "mean_error_mean": float(arrays["mean_error"][mask].mean()),
            "spearman_w1_error": correlation(score, error),
            "spearman_sigma_error": correlation(arrays["sigma"][mask], error),
            "flagged_by_id_q95": float((score > threshold).mean()),
            "risk_at_half_coverage": float(risk[max(0, len(risk) // 2 - 1)]),
            "unfiltered_risk": float(error.mean()),
            "aurc_w1": float(risk.mean()),
            "aurc_distance": float(risk_curve(error, distance).mean()),
            "aurc_oracle": float(risk_curve(error, error).mean()),
            "tail_fraction_median": float(np.median(arrays["w1_tail_fraction"][mask])),
        }
        if name != "id":
            result[name]["ood_detection_auroc"] = auroc(score, arrays["w1"][masks["id"]])
            result[name]["w1_ratio_to_id"] = float(score.mean() / arrays["w1"][masks["id"]].mean())
    return result


def plot_run(out, x, arrays, train_x, train_y, mode, seed, threshold):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean, scale = truth(x)
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.6), sharex=True, sharey=True, constrained_layout=True)
    for i, ax in enumerate(axes):
        ax.axvspan(-1, -0.3, color="green", alpha=0.08)
        ax.axvspan(0.3, 1, color="green", alpha=0.08)
        ax.fill_between(x, mean - 0.8 * scale, mean + 0.8 * scale, color="black", alpha=0.12, label="true 10-90%")
        ax.scatter(train_x[::8], train_y[::8], s=3, color="black", alpha=0.25)
        for j, style in enumerate(("--", "-", "--")):
            ax.plot(x, arrays["quantiles"][i, :, j], style, color=f"C{i}", lw=1.4)
        ax.plot(x, mean, "k:", lw=1)
        ax.set(title=f"Categorical {i + 1}", xlabel="x", xlim=(-2, 2), ylim=(-4, 4))
    axes[0].set_ylabel("y; predictive quantiles 10 / 50 / 90%")
    fig.suptitle(f"{mode}, ensemble seed {seed}; green = training support; tails outside view")
    fig.savefig(out / "distributions.png", dpi=170)
    fig.savefig(out / "distributions.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    ax = axes[0, 0]
    for key, label in (
        ("w1", "ensemble disagreement W1"),
        ("error", "true distribution error W1"),
        ("projection_floor", "representation floor"),
    ):
        ax.semilogy(x, np.maximum(arrays[key], 1e-12), label=label)
    ax.axhline(threshold, color="grey", ls=":", label="ID calibration q95")
    ax.legend(fontsize=8)
    ax.set(xlabel="x", ylabel="W1 (log scale)")
    for name, mask in region_masks(x).items():
        axes[0, 1].scatter(arrays["w1"][mask], arrays["error"][mask], s=9, label=name)
    axes[0, 1].set(xscale="log", yscale="log", xlabel="disagreement W1", ylabel="true distribution error W1")
    axes[0, 1].legend()
    axes[1, 0].plot(x, arrays["w1_tail_fraction"])
    axes[1, 0].set(xlabel="x", ylabel="fraction of W1 outside [-5, 5]", ylim=(-0.02, 1.02))
    ood = ~region_masks(x)["id"]
    err = arrays["error"][ood]
    coverage = np.arange(1, len(err) + 1) / len(err)
    for score, label in ((arrays["w1"][ood], "W1"), (distance_to_data(x[ood]), "distance to data"), (err, "oracle")):
        axes[1, 1].plot(coverage, risk_curve(err, score), label=label)
    axes[1, 1].axhline(err.mean(), ls="--", color="grey", label="random (expected)")
    axes[1, 1].set(xlabel="retained OOD fraction", ylabel="mean retained true error", yscale="log")
    axes[1, 1].legend(fontsize=8)
    fig.suptitle(f"{mode}, ensemble seed {seed}; true error is against continuous ground truth")
    fig.savefig(out / "diagnostics.png", dpi=170)
    fig.savefig(out / "diagnostics.pdf")
    plt.close(fig)


def run_one(args, root, mode, seed, data):
    out = root / f"{mode}_s{seed}"
    out.mkdir()
    train_x, train_y, val_x, val_y = data
    heads = make_ensemble(seed, args.width)
    bins = symexp_twohot(torch.zeros(1, 255), 255).bins if mode == "repository" else torch.linspace(-5, 5, 255)
    train_features, val_features = features(train_x), features(val_x)
    targets, val_targets = twohot(torch.from_numpy(train_y), bins), twohot(torch.from_numpy(val_y), bins)
    optimizer = torch.optim.Adam(heads.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.batch_seed)
    best_loss, best_step, best_state = float("inf"), 0, None
    history = []
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        idx = torch.randint(len(train_x), (args.batch_size,), generator=generator)
        logits = logits_of(heads, train_features[idx])
        member_losses = -(targets[idx] * logits.log_softmax(-1)).sum(-1).mean(-1)
        optimizer.zero_grad(set_to_none=True)
        # Sum independent losses: each member gets its standalone Adam gradient.
        member_losses.sum().backward()
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            with torch.no_grad():
                val_loss = -(val_targets * logits_of(heads, val_features).log_softmax(-1)).sum(-1).mean(-1)
            loss = float(val_loss.mean())
            if loss < best_loss:
                best_loss, best_step = loss, step
                best_state = copy.deepcopy(heads.state_dict())
            history.append(
                {
                    "step": step,
                    "train_member_nll": member_losses.detach().tolist(),
                    "validation_member_nll": val_loss.tolist(),
                }
            )
            (out / "training.json").write_text(json.dumps(history, indent=2, allow_nan=False) + "\n")
            torch.save(
                {
                    "state_dict": best_state,
                    "args": vars(args),
                    "support": mode,
                    "seed": seed,
                    "best_step": best_step,
                    "completed_step": step,
                    "complete": step == args.steps,
                },
                out / "model.pt",
            )
            print(
                f"{mode} seed={seed} step={step} val_nll={loss:.5f} best_step={best_step} "
                f"elapsed={time.monotonic()-started:.1f}s",
                flush=True,
            )
    heads.load_state_dict(best_state)
    x = np.linspace(-2, 2, 801)
    with torch.no_grad():
        logits = logits_of(heads, features(x))
        calibration_logits = logits_of(heads, val_features)
    z = bins.double().numpy()
    arrays = score_arrays(logits, z, x)
    cal = score_arrays(calibration_logits, z, val_x.astype(np.float64))
    threshold = float(np.quantile(cal["w1"], 0.95))
    result = {
        "support": mode,
        "seed": seed,
        "member_seeds": [seed * 100 + i for i in range(5)],
        "best_step": best_step,
        "validation_nll": best_loss,
        "threshold": threshold,
        "seconds": time.monotonic() - started,
        "regions": summarize(x, arrays, threshold),
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (out / "training.json").write_text(json.dumps(history, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(out / "predictions.npz", x=x, bins=z, logits=logits.numpy(), **arrays)
    torch.save({"state_dict": best_state, "args": vars(args), "support": mode, "seed": seed}, out / "model.pt")
    plot_run(out, x, arrays, train_x, train_y, mode, seed, threshold)
    return result


def aggregate_report(root, results):
    lines = [
        "# Five categorical heads: measured results",
        "",
        "Independent initialization repeats on one fixed dataset; mean ± SD across ensemble seeds.",
        "These are descriptive statistics, not confidence intervals or independent grid-point samples.",
        "",
        "| Support | Region | Disagreement W1 | True error W1 | "
        "Spearman(disagreement,error) | OOD AUROC | Risk@50% / full |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for mode in dict.fromkeys(r["support"] for r in results):
        runs = [r for r in results if r["support"] == mode]
        for region in REGIONS:

            def fmt(key):
                values = [r["regions"][region].get(key) for r in runs]
                if any(v is None for v in values):
                    return "n/a"
                return f"{np.mean(values):.4g} ± {np.std(values, ddof=1) if len(values)>1 else 0:.2g}"

            ratios = [
                r["regions"][region]["risk_at_half_coverage"] / r["regions"][region]["unfiltered_risk"] for r in runs
            ]
            lines.append(
                f"| {mode} | {region} | {fmt('w1_mean')} | {fmt('true_error_mean')} | "
                f"{fmt('spearman_w1_error')} | {fmt('ood_detection_auroc')} | {np.mean(ratios):.3f} |"
            )
    lines += [
        "",
        "Lower risk ratio is better; >1 means keeping low-disagreement points increases error.",
        "AUROC measures OOD detection, not error detection; Spearman is reported within each region.",
        "",
        "## Per-run evidence",
        "",
    ]
    for r in results:
        name = f"{r['support']}_s{r['seed']}"
        lines.append(
            f"- [{name}]({name}/metrics.json): selected step {r['best_step']}; "
            f"[quantiles]({name}/distributions.png), [diagnostics]({name}/diagnostics.png)."
        )
    lines += [
        "",
        "See the code-directory README for protocol, limitations and reproduction commands.",
        "No RL, shared learned feature, bootstrap target, slow-value regularization or dynamic truncation is modeled.",
    ]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    (root / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--supports", nargs="+", choices=["repository", "bounded"], default=["repository", "bounded"])
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--validation-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-seed", type=int, default=20261104)
    parser.add_argument("--batch-seed", type=int, default=42)
    args = parser.parse_args()
    for name in ("steps", "eval_every", "width", "batch_size", "train_size", "validation_size"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("lr must be positive and finite")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.supports)) != len(args.supports):
        parser.error("duplicate seeds/supports")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(args.data_seed)
    data = (*dataset(args.train_size, rng), *dataset(args.validation_size, rng))
    np.savez_compressed(root / "data.npz", train_x=data[0], train_y=data[1], val_x=data[2], val_y=data[3])
    source_files = [Path(__file__), Path("networks.py"), Path("distributions.py"), Path("tools.py")]
    protocol = {
        "args": vars(args),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "data_sha256": hashlib.sha256((root / "data.npz").read_bytes()).hexdigest(),
        "threads": torch.get_num_threads(),
        "device": "cpu",
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "feature": "[x, 1]",
        "distribution": "Uniform(2sin(2x)+0.4x +/- (0.35+0.15cos(2x)^2))",
        "training_support": [[-1, -0.3], [0.3, 1]],
        "evaluation_support": [-2, 2],
        "selection": "common checkpoint minimizing mean validation NLL; ID-only validation",
        "status": "running",
    }
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    results = []
    for mode in args.supports:
        for seed in args.seeds:
            results.append(run_one(args, root, mode, seed, data))
            aggregate_report(root, results)
    protocol["status"] = "complete"
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")


if __name__ == "__main__":
    main()
