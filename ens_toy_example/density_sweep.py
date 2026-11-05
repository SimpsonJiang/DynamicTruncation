"""Smooth, nonuniform coverage: dense / medium / sparse / empty / sparse / empty."""

import argparse
import copy
import csv
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from distributions import symexp_twohot
from ens_toy_example.compare_quantile import (
    GROUPS,
    LABELS,
    distances,
    pinball,
    sha256,
    uniform_error,
)
from ens_toy_example.run import (
    correlation,
    features,
    logits_of,
    make_ensemble,
    truth,
    twohot,
)

REGIONS = (
    ("dense", 0.0, 1.0),
    ("medium", 1.0, 1.5),
    ("sparse_1", 1.5, 2.1),
    ("empty_gap", 2.1, 2.5),
    ("sparse_2", 2.5, 3.2),
    ("empty_tail", 3.2, 4.2),
)
NORMALIZER = 0.675 + 0.1075 + 0.024 + 0.028
COLORS = ("#badcb9", "#e7dfad", "#f4ceb0", "#dddddd", "#f4ceb0", "#dddddd")


def masks(x):
    x = np.asarray(x)
    return {name: (x >= a) & ((x < b) if i < len(REGIONS) - 1 else (x <= b)) for i, (name, a, b) in enumerate(REGIONS)}


def sampling_density(x):
    """Unnormalized C1 density on the evaluation domain [0,4.2]."""
    x = np.asarray(x, dtype=np.float64)
    p = np.zeros_like(x)
    for a, b, high, low in ((0, 1, 1, 0.35), (1, 1.5, 0.35, 0.08), (1.5, 2.1, 0.08, 0)):
        active = (x >= a) & (x < b)
        t = (x[active] - a) / (b - a)
        smoothstep = 3 * t**2 - 2 * t**3
        p[active] = high + (low - high) * smoothstep
    active = (x >= 2.5) & (x < 3.2)
    p[active] = 0.08 * np.sin(np.pi * (x[active] - 2.5) / 0.7) ** 2
    return p


def sample_data(n, rng):
    """Exact rejection sampling, then the unchanged conditional uniform target."""
    kept, remaining = [], n
    while remaining:
        candidates = rng.uniform(0, 4.2, max(1024, remaining * 6))
        accepted = candidates[rng.random(len(candidates)) < sampling_density(candidates)][:remaining]
        kept.append(accepted)
        remaining -= len(accepted)
    x = np.concatenate(kept).astype(np.float32)
    mean, scale = truth(x)
    y = (mean + scale * rng.uniform(-1, 1, n)).astype(np.float32)
    return x, y


def coverage(train_x, val_x):
    train_masks, val_masks = masks(train_x), masks(val_x)
    areas = (0.675, 0.1075, 0.024, 0, 0.028, 0)
    return [
        {
            "name": name,
            "start": a,
            "end": b,
            "width": b - a,
            "train_count": int(train_masks[name].sum()),
            "validation_count": int(val_masks[name].sum()),
            "train_count_per_x_unit": float(train_masks[name].sum() / (b - a)),
            "expected_train_count": len(train_x) * area / NORMALIZER,
        }
        for (name, a, b), area in zip(REGIONS, areas)
    ]


def decorate_regions(ax, labels=False):
    for (name, a, b), color in zip(REGIONS, COLORS):
        ax.axvspan(a, b, color=color, alpha=0.25, zorder=0)
        ax.axvline(a, color="grey", lw=0.6, ls=":")
        if labels:
            ax.text(
                (a + b) / 2,
                0.96,
                name.replace("_", " "),
                ha="center",
                va="top",
                transform=ax.get_xaxis_transform(),
                fontsize=9,
            )
    ax.set_xlim(0, 4.2)


def draw_density(ax, data):
    grid = np.linspace(0, 4.2, 1681)
    ax.hist(
        data["train_x"],
        bins=np.linspace(0, 4.2, 85),
        density=True,
        color="grey",
        alpha=0.35,
        label="actual training histogram (density)",
    )
    ax.plot(grid, sampling_density(grid) / NORMALIZER, "k", lw=2, label="smooth sampling density")
    decorate_regions(ax, labels=True)
    ax.set(ylabel="p(x)", ylim=(0, 1.6))
    ax.legend(loc="center right", fontsize=8)


def plot_density(root, data):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 3.6), constrained_layout=True)
    draw_density(ax, data)
    ax.set(xlabel="x", title="Smooth coverage with unequal region widths; no samples in the two empty regions")
    fig.savefig(root / "density_profile.png", dpi=180)
    fig.savefig(root / "density_profile.pdf")
    plt.close(fig)


def train_family(config, seed, family, data, grid, root):
    folder = root / f"{family}_s{seed}"
    folder.mkdir()
    model = make_ensemble(seed + (10000 if family == "quantile" else 0), config.width)
    bins = symexp_twohot(torch.zeros(1, 255), 255).bins if family == "repository" else torch.linspace(-5, 5, 255)
    tx, vx = features(data["train_x"]), features(data["val_x"])
    ty, vy = torch.from_numpy(data["train_y"]), torch.from_numpy(data["val_y"])
    if family != "quantile":
        ty, vy = twohot(ty, bins), twohot(vy, bins)

    def loss_fn(output, labels):
        if family == "quantile":
            return pinball(output, labels)
        return -(labels * output.log_softmax(-1)).sum(-1).mean(-1)

    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    batch_rng = torch.Generator().manual_seed(config.batch_seed)
    history, best_state, best_step, best = [], None, 0, float("inf")
    start = time.monotonic()
    for step in range(1, config.steps + 1):
        idx = torch.randint(len(tx), (config.batch_size,), generator=batch_rng)
        loss = loss_fn(logits_of(model, tx[idx]), ty[idx])
        opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        opt.step()
        if step % config.eval_every == 0 or step == config.steps:
            with torch.no_grad():
                val_loss = loss_fn(logits_of(model, vx), vy)
            if float(val_loss.mean()) < best:
                best, best_step, best_state = float(val_loss.mean()), step, copy.deepcopy(model.state_dict())
            history.append(
                {"step": step, "train_member_loss": loss.detach().tolist(), "validation_member_loss": val_loss.tolist()}
            )
            (folder / "training.json").write_text(json.dumps(history, indent=2) + "\n")
            torch.save(
                {
                    "state_dict": best_state,
                    "best_step": best_step,
                    "completed_step": step,
                    "family": family,
                    "seed": seed,
                    "config": {**vars(config), "output": str(config.output)},
                },
                folder / "model.pt",
            )
            print(
                f"{family} s{seed} step={step} val={val_loss.mean():.6f} best={best_step} "
                f"seconds={time.monotonic()-start:.1f}",
                flush=True,
            )
    model.load_state_dict(best_state)
    with torch.no_grad():
        raw = logits_of(model, features(grid))
    if family == "quantile":
        prediction = raw.double().numpy()
        atoms = prediction.transpose(1, 0, 2).reshape(len(grid), -1)
        mass = np.ones_like(atoms)
        np.savez_compressed(folder / "predictions.npz", x=grid, atoms=prediction)
    else:
        prediction = raw.double().softmax(-1).numpy()
        atoms, mass = bins.double().numpy(), prediction.mean(0)
        np.savez_compressed(folder / "predictions.npz", x=grid, probs=prediction, bins=atoms, logits=raw.numpy())
    errors = {"full": uniform_error(atoms, mass, grid), "central": uniform_error(np.clip(atoms, -5, 5), mass, grid)}
    np.savez_compressed(folder / "errors.npz", **errors)
    stats = {
        "family": family,
        "seed": seed,
        "best_step": best_step,
        "validation_loss": best,
        "seconds": time.monotonic() - start,
        "regions": {
            name: {view: float(e[mask].mean()) for view, e in errors.items()} for name, mask in masks(grid).items()
        },
    }
    if family == "quantile":
        crossing = (np.diff(prediction, axis=-1) < 0).mean(axis=(0, 2))
        stats["crossing_fraction"] = {name: float(crossing[mask].mean()) for name, mask in masks(grid).items()}
    (folder / "metrics.json").write_text(json.dumps(stats, indent=2, allow_nan=False) + "\n")
    return stats


def compare_seed(root, seed, support):
    q = np.load(root / f"quantile_s{seed}" / "predictions.npz")
    c = np.load(root / f"{support}_s{seed}" / "predictions.npz")
    x = q["x"]
    np.testing.assert_array_equal(x, c["x"])
    ce = np.load(root / f"{support}_s{seed}" / "errors.npz")
    qe = np.load(root / f"quantile_s{seed}" / "errors.npz")
    folder = root / f"comparison_{support}_s{seed}"
    folder.mkdir()
    result = {"seed": seed, "support": support, "views": {}}
    for view in ("full", "central"):
        scores, pairs = distances(c["probs"], c["bins"], q["atoms"], central=view == "central")
        np.savez_compressed(folder / f"{view}.npz", x=x, **scores, **pairs)
        stats = {}
        for name, mask in masks(x).items():
            stats[name] = {group: float(score[mask].mean()) for group, score in scores.items()}
            stats[name].update(
                {
                    "categorical_error": float(ce[view][mask].mean()),
                    "quantile_error": float(qe[view][mask].mean()),
                    "between_greater_than_both_fraction": float(
                        (
                            (scores["between"] > scores["categorical_within"])
                            & (scores["between"] > scores["quantile_within"])
                        )[mask].mean()
                    ),
                }
            )
        result["views"][view] = {
            "regions": stats,
            "density_spearman": {group: correlation(sampling_density(x), score) for group, score in scores.items()},
        }
    (folder / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def plot_results(root, data, results, seeds):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def draw_scores(ax, support, view):
        arrays = [np.load(root / f"comparison_{support}_s{s}" / f"{view}.npz") for s in seeds]
        x = arrays[0]["x"]
        for key, label, color in zip(GROUPS, LABELS, ("C1", "C0", "C4")):
            values = np.stack([a[key] for a in arrays])
            ax.plot(x, values.mean(0), color=color, label=label, lw=1.8)
            ax.fill_between(x, values.min(0), values.max(0), color=color, alpha=0.12)
        decorate_regions(ax)
        ax.set(ylabel="Mean pairwise W1", title=f"{support} categorical bins; {view} W1")
        ax.legend(loc="upper left", fontsize=8)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    draw_density(axes[0], data)
    draw_scores(axes[1], "bounded", "full")
    draw_scores(axes[2], "repository", "central")
    axes[-1].set_xlabel("x; two empty regions are shaded grey")
    fig.suptitle("Gradually varying coverage: lines = initialization mean, bands = observed min-max")
    fig.savefig(root / "density_comparison.png", dpi=180)
    fig.savefig(root / "density_comparison.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    for ax, support in zip(axes, ("repository", "bounded")):
        draw_scores(ax, support, "full")
        ax.set_yscale("log")
    axes[-1].set_xlabel("x")
    fig.savefig(root / "full_w1.png", dpi=180)
    fig.savefig(root / "full_w1.pdf")
    plt.close(fig)

    # Preselected first repeat; no cherry-picking based on appearance.
    seed = seeds[0]
    q = np.load(root / f"quantile_s{seed}" / "predictions.npz")
    x, qa = q["x"], np.sort(q["atoms"], axis=-1)
    mean, scale = truth(x)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True, constrained_layout=True)
    for row, support in enumerate(("repository", "bounded")):
        cat = np.load(root / f"{support}_s{seed}" / "predictions.npz")
        cdf = cat["probs"].cumsum(-1)
        for col, tau in enumerate((0.5, 0.9)):
            ax = axes[row, col]
            decorate_regions(ax)
            ax.fill_between(x, mean - scale, mean + scale, color="black", alpha=0.1, label="true random range")
            ax.plot(x, mean + (2 * tau - 1) * scale, "k--", label="true quantile")
            ax.scatter(data["train_x"][::8], data["train_y"][::8], s=2, color="grey", alpha=0.3)
            for i in range(5):
                ax.plot(
                    x,
                    cat["bins"][(cdf[i] >= tau).argmax(-1)],
                    color="C1",
                    alpha=0.65,
                    label="categorical (5)" if i == 0 else None,
                )
                ax.plot(
                    x,
                    qa[i, :, int(np.ceil(tau * 255)) - 1],
                    color="C0",
                    alpha=0.65,
                    label="quantile (5)" if i == 0 else None,
                )
            ax.set(title=f"{support}: {tau:g} quantile", xlabel="x", ylabel="y", ylim=(-4, 5))
            ax.legend(loc="upper left", fontsize=8)
    fig.savefig(root / "predictions.png", dpi=180)
    fig.savefig(root / "predictions.pdf")
    plt.close(fig)

    rows = []
    for record in results:
        for view, statistics in record["views"].items():
            for name, values in statistics["regions"].items():
                rows.append(
                    {"support": record["support"], "seed": record["seed"], "view": view, "region": name, **values}
                )
    with (root / "region_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(root, results, counts):
    lines = [
        "# Smooth density experiment",
        "",
        "Fixed data, three initialization repeats by default.",
        "Coverage is continuously differentiable at internal region boundaries. Region widths are unequal.",
        "",
        "| Region | Interval | Train n | Validation n | Train n / x-unit |",
        "|---|---|---:|---:|---:|",
    ]
    for c in counts:
        lines.append(
            f"| {c['name']} | [{c['start']},{c['end']}] | {c['train_count']} | "
            f"{c['validation_count']} | {c['train_count_per_x_unit']:.2f} |"
        )
    for support in ("bounded", "repository"):
        for view in ("full", "central"):
            lines += [
                "",
                f"## {support} / {view}",
                "",
                "Uniform-in-x region averages, mean ± sample SD across initialization repeats.",
                "",
                "| Region | C-C | Q-Q | C-Q | C true W1 | Q true W1 | C-Q > both fraction |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
            subset = [r for r in results if r["support"] == support]
            for name, _, _ in REGIONS:
                values = []
                for key in (*GROUPS, "categorical_error", "quantile_error", "between_greater_than_both_fraction"):
                    v = [r["views"][view]["regions"][name][key] for r in subset]
                    values.append(f"{np.mean(v):.5g} ± {np.std(v, ddof=1) if len(v)>1 else 0:.3g}")
                lines.append(f"| {name} | " + " | ".join(values) + " |")
    lines += [
        "",
        "![Coverage and disagreement](density_comparison.png)",
        "",
        "![Predictions](predictions.png)",
        "",
        "Density, x location, target shape and gap/extrapolation geometry co-vary; "
        "this is not a causal density ablation.",
        "Validation uses the same sampling density, contains no empty-region samples, and favors dense regions.",
        "Empty regions are reported separately; sparse regions are in-support, not strictly out-of-support.",
        "Central W1 integrates only [-5,5]; full W1 retains all categorical tails. No probability renormalization.",
        "No joint mixed-ensemble RL training or uncertainty calibration guarantee is implied.",
    ]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
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
    if args.lr <= 0 or not np.isfinite(args.lr) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Invalid lr or duplicate seeds")
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(args.data_seed)
    tx, ty = sample_data(args.train_size, rng)
    vx, vy = sample_data(args.validation_size, rng)
    data = {"train_x": tx, "train_y": ty, "val_x": vx, "val_y": vy}
    counts = coverage(tx, vx)
    np.savez_compressed(root / "data.npz", **data)
    (root / "coverage.json").write_text(json.dumps(counts, indent=2) + "\n")
    config = {**vars(args), "output": str(root)}
    source_names = [
        __file__,
        "ens_toy_example/run.py",
        "ens_toy_example/compare_quantile.py",
        "networks.py",
        "distributions.py",
        "tools.py",
    ]
    protocol = {
        "status": "running",
        "args": config,
        "regions": REGIONS,
        "density_normalizer": NORMALIZER,
        "data_sha256": sha256(root / "data.npz"),
        "source_sha256": {s: sha256(s) for s in source_names},
        "device": "cpu",
        "threads": 1,
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "torch": torch.__version__,
        "selection": "ID validation at the same nonuniform sampling density",
        "evaluation": "uniform grid in [0,4.2], uniform-x averages within each region",
        "initialization": "categorical=100*seed+i; quantile=100*(seed+10000)+i",
        "truth": "unchanged Uniform(m(x)-s(x),m(x)+s(x)); m=2sin(2x)+0.4x; s=0.35+0.15cos(2x)^2",
    }
    shutil.copyfile(__file__, root / "source_density_sweep.py")
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    plot_density(root, data)
    grid = np.linspace(0, 4.2, 841)
    results, training = [], []
    for seed in args.seeds:
        for family in ("bounded", "quantile", "repository"):
            training.append(train_family(args, seed, family, data, grid, root))
            (root / "training_summary.json").write_text(json.dumps(training, indent=2, allow_nan=False) + "\n")
        for support in ("bounded", "repository"):
            results.append(compare_seed(root, seed, support))
        (root / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
        report(root, results, counts)
    plot_results(root, data, results, args.seeds)
    protocol["status"] = "complete"
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")


if __name__ == "__main__":
    main()
