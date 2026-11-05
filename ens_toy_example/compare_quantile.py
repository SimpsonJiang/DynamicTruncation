"""Train five quantile members and compare C-C, Q-Q and C-Q exact W1."""

import argparse
import ast
import copy
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from ens_toy_example.run import (
    auroc,
    features,
    logits_of,
    make_ensemble,
    region_masks,
    truth,
)

GROUPS = ("categorical_within", "quantile_within", "between")
LABELS = ("C-C: within categorical (10 pairs)", "Q-Q: within quantile (10 pairs)", "C-Q: between (25 pairs)")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinball(prediction, target):
    """Per-member loss for fixed midpoint quantiles; no Huber approximation."""
    tau = (torch.arange(prediction.shape[-1], device=prediction.device) + 0.5) / prediction.shape[-1]
    residual = target[..., None] - prediction
    return torch.maximum(tau * residual, (tau - 1) * residual).mean(dim=(-1, -2))


def weighted_w1(a, pa, b, pb):
    """Exact batched 1D W1 with arbitrary, possibly duplicate atom locations."""
    a, pa = np.broadcast_arrays(np.asarray(a, dtype=np.float64), np.asarray(pa, dtype=np.float64))
    b, pb = np.broadcast_arrays(np.asarray(b, dtype=np.float64), np.asarray(pb, dtype=np.float64))
    pa, pb = pa / pa.sum(-1, keepdims=True), pb / pb.sum(-1, keepdims=True)
    atoms = np.concatenate([a, b], axis=-1)
    signed_mass = np.concatenate([pa, -pb], axis=-1)
    order = np.argsort(atoms, axis=-1, kind="stable")
    atoms = np.take_along_axis(atoms, order, axis=-1)
    difference = np.take_along_axis(signed_mass, order, axis=-1).cumsum(-1)[..., :-1]
    return (np.abs(difference) * np.diff(atoms, axis=-1)).sum(-1)


def uniform_error(atoms, mass, x):
    """Exact W1 to Uniform truth through inverse-CDF integration."""
    atoms, mass = np.broadcast_arrays(np.asarray(atoms, dtype=np.float64), np.asarray(mass, dtype=np.float64))
    order = np.argsort(atoms, axis=-1, kind="stable")
    atoms, mass = np.take_along_axis(atoms, order, -1), np.take_along_axis(mass, order, -1)
    mass = mass / mass.sum(-1, keepdims=True)
    right = mass.cumsum(-1)
    left = right - mass
    mean, scale = truth(x)
    u = atoms - (mean[:, None] + scale[:, None] * (2 * left - 1))
    v = atoms - (mean[:, None] + scale[:, None] * (2 * right - 1))
    area = (np.abs(u) + np.abs(v)) / 2
    crossing = u * v < 0
    area[crossing] = (u[crossing] ** 2 + v[crossing] ** 2) / (2 * np.abs(v[crossing] - u[crossing]))
    return (area * mass).sum(-1)


def distances(categorical, bins, quantiles, central=False):
    """Return all 10/10/25 distances, then arithmetic means on equal footing."""
    bins = np.clip(bins, -5, 5) if central else bins
    q = np.sort(np.clip(quantiles, -5, 5) if central else quantiles, axis=-1)
    cc = np.stack(
        [
            (np.abs((categorical[i] - categorical[j]).cumsum(-1)[..., :-1]) * np.diff(bins)).sum(-1)
            for i in range(5)
            for j in range(i + 1, 5)
        ]
    )
    # Equal atom weights permit exact quantile-matching W1, after sorting.
    qq = np.stack([np.abs(q[i] - q[j]).mean(-1) for i in range(5) for j in range(i + 1, 5)])
    cq = np.stack(
        [
            weighted_w1(bins, categorical[i], q[j], np.full_like(q[j], 1 / q.shape[-1]))
            for i in range(5)
            for j in range(5)
        ]
    )
    return dict(zip(GROUPS, (cc.mean(0), qq.mean(0), cq.mean(0)))), {"cc_pairs": cc, "qq_pairs": qq, "cq_pairs": cq}


def train_quantiles(config, seed, data, output):
    output.mkdir()
    # Disjoint initialization streams from categorical; architecture/count unchanged.
    ensemble = make_ensemble(seed + 10000, config["width"])
    train_x, val_x = features(data["train_x"]), features(data["val_x"])
    train_y, val_y = torch.from_numpy(data["train_y"]), torch.from_numpy(data["val_y"])
    optimizer = torch.optim.Adam(ensemble.parameters(), lr=config["lr"])
    batches = torch.Generator().manual_seed(config["batch_seed"])
    best_loss, best_step, state = float("inf"), 0, None
    history = []
    start = time.monotonic()
    for step in range(1, config["steps"] + 1):
        idx = torch.randint(len(train_x), (config["batch_size"],), generator=batches)
        loss = pinball(logits_of(ensemble, train_x[idx]), train_y[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.sum().backward()
        optimizer.step()
        if step % config["eval_every"] == 0 or step == config["steps"]:
            with torch.no_grad():
                validation = pinball(logits_of(ensemble, val_x), val_y)
            if float(validation.mean()) < best_loss:
                best_loss, best_step = float(validation.mean()), step
                state = copy.deepcopy(ensemble.state_dict())
            history.append(
                {"step": step, "train_pinball": loss.detach().tolist(), "validation_pinball": validation.tolist()}
            )
            (output / "training.json").write_text(json.dumps(history, indent=2) + "\n")
            torch.save(
                {
                    "state_dict": state,
                    "best_step": best_step,
                    "completed_step": step,
                    "config": config,
                    "ensemble_seed": seed,
                },
                output / "model.pt",
            )
            print(
                f"quantile seed={seed} step={step} val={validation.mean():.6f} "
                f"best={best_step} seconds={time.monotonic()-start:.1f}",
                flush=True,
            )
    ensemble.load_state_dict(state)
    x = np.linspace(-2, 2, 801)
    with torch.no_grad():
        q = logits_of(ensemble, features(x)).double().numpy()
    sorted_q = np.sort(q, axis=-1)
    mixture = np.sort(sorted_q.transpose(1, 0, 2).reshape(len(x), -1), axis=-1)
    error = uniform_error(mixture, np.ones_like(mixture), x)
    crossing = (np.diff(q, axis=-1) < 0).mean(axis=(0, 2))
    masks = region_masks(x)
    stats = {
        "seed": seed,
        "member_seeds": [(seed + 10000) * 100 + i for i in range(5)],
        "best_step": best_step,
        "validation_pinball": best_loss,
        "seconds": time.monotonic() - start,
        "regions": {
            region: {
                "mixture_true_w1": float(error[mask].mean()),
                "adjacent_quantile_crossing_fraction": float(crossing[mask].mean()),
            }
            for region, mask in masks.items()
        },
    }
    (output / "metrics.json").write_text(json.dumps(stats, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(output / "predictions.npz", x=x, quantiles=q, sorted_atoms=sorted_q, true_error=error)
    return q, stats


def compare_run(source, root, support, seed, q):
    cat_path = source / f"{support}_s{seed}" / "predictions.npz"
    cat = np.load(cat_path)
    x, p, bins = cat["x"], cat["probs"], cat["bins"]
    out = root / f"{support}_s{seed}"
    out.mkdir()
    regions = region_masks(x)
    result = {"support": support, "seed": seed, "categorical_predictions_sha256": sha256(cat_path), "views": {}}
    for central in (False, True):
        view = "central" if central else "full"
        scores, pairs = distances(p, bins, q, central=central)
        np.savez_compressed(out / f"{view}.npz", x=x, **scores, **pairs)
        stats = {}
        for region, mask in regions.items():
            stats[region] = {}
            for name, score in scores.items():
                baseline = score[regions["id"]].mean()
                stats[region][name] = {
                    "mean": float(score[mask].mean()),
                    "ood_over_id": float(score[mask].mean() / max(baseline, 1e-30)),
                    "excess_over_id": float(score[mask].mean() - baseline),
                    "ood_auroc": auroc(score[mask], score[regions["id"]]) if region != "id" else None,
                }
            stats[region]["between_greater_than_both_fraction"] = float(
                ((scores["between"] > scores["categorical_within"]) & (scores["between"] > scores["quantile_within"]))[
                    mask
                ].mean()
            )
        result["views"][view] = stats
    (out / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def plot_comparison(root, source, results, seeds, supports):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(supports), 2, figsize=(13, 4.4 * len(supports)), squeeze=False, constrained_layout=True
    )
    for row, support in enumerate(supports):
        for col, view in enumerate(("full", "central")):
            ax = axes[row, col]
            arrays = [np.load(root / f"{support}_s{seed}" / f"{view}.npz") for seed in seeds]
            x = arrays[0]["x"]
            for name, label, color in zip(GROUPS, LABELS, ("C1", "C0", "C4")):
                values = np.stack([a[name] for a in arrays])
                ax.plot(x, values.mean(0), label=label, color=color)
                ax.fill_between(x, values.min(0), values.max(0), color=color, alpha=0.12)
            ax.axvspan(-1, -0.3, color="green", alpha=0.07)
            ax.axvspan(0.3, 1, color="green", alpha=0.07)
            title = "full W1" if view == "full" else "W1 over [-5,5] only (diagnostic)"
            ax.set(title=f"Categorical: {support} bins; {title}", xlabel="x", ylabel="Mean pairwise W1", yscale="log")
            ax.legend(fontsize=8, loc="lower left")
    fig.suptitle(f"Within-family vs cross-family disagreement; mean and observed range over {len(seeds)} repeats")
    fig.savefig(root / "uncertainty_comparison.png", dpi=180)
    fig.savefig(root / "uncertainty_comparison.pdf")
    plt.close(fig)

    if "bounded" in supports:
        fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
        arrays = [np.load(root / f"bounded_s{seed}" / "full.npz") for seed in seeds]
        x = arrays[0]["x"]
        maximum = 0.0
        for name, label, color in zip(GROUPS, LABELS, ("C1", "C0", "C4")):
            values = np.stack([a[name] for a in arrays])
            maximum = max(maximum, float(values.max()))
            ax.plot(x, values.mean(0), label=label, color=color, lw=2)
            ax.fill_between(x, values.min(0), values.max(0), color=color, alpha=0.12)
        ax.axvspan(-1, -0.3, color="green", alpha=0.07)
        ax.axvspan(0.3, 1, color="green", alpha=0.07)
        ax.set(
            title="Five categorical vs five quantile members: bounded categorical bins",
            xlabel="x (green = training support; white = OOD)",
            ylabel="Mean pairwise W1 disagreement",
            ylim=(0, maximum * 1.2),
        )
        ax.legend(loc="upper center", ncol=3, fontsize=8)
        fig.savefig(root / "bounded_disagreement.png", dpi=180)
        fig.savefig(root / "bounded_disagreement.pdf")
        plt.close(fig)

    # First predetermined seed, no selection based on appearance or outcome.
    seed = seeds[0]
    q = np.load(root / f"quantile_s{seed}" / "predictions.npz")
    x, atoms = q["x"], q["sorted_atoms"]
    mean, scale = truth(x)
    data = np.load(source / "data.npz")
    fig, axes = plt.subplots(
        len(supports), 2, figsize=(13, 4.4 * len(supports)), squeeze=False, constrained_layout=True
    )
    for row, support in enumerate(supports):
        cat = np.load(source / f"{support}_s{seed}" / "predictions.npz")
        for col, tau in enumerate((0.5, 0.9)):
            ax = axes[row, col]
            ax.fill_between(x, mean - scale, mean + scale, color="grey", alpha=0.15, label="full true support")
            ax.plot(x, mean + (2 * tau - 1) * scale, "k--", lw=2, label=f"true {tau:g} quantile")
            ax.scatter(data["train_x"][::16], data["train_y"][::16], s=2, c="grey", alpha=0.3)
            for i in range(5):
                ax.plot(
                    x,
                    cat["quantiles"][i, :, 1 if col == 0 else 2],
                    color="C1",
                    alpha=0.6,
                    label="categorical: 5 members" if i == 0 else None,
                )
                ax.plot(
                    x,
                    atoms[i, :, int(np.ceil(tau * atoms.shape[-1])) - 1],
                    color="C0",
                    alpha=0.6,
                    label="quantile: 5 members" if i == 0 else None,
                )
            ax.axvspan(-1, -0.3, color="green", alpha=0.07)
            ax.axvspan(0.3, 1, color="green", alpha=0.07)
            ax.set(title=f"{support} categorical vs quantile: {tau:g} quantile", xlabel="x", ylabel="y", ylim=(-4, 4))
            ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Predictive distributions, seed {seed}; shaded band = full random target range")
    fig.savefig(root / "prediction_comparison.png", dpi=180)
    fig.savefig(root / "prediction_comparison.pdf")
    plt.close(fig)

    lines = [
        "# Categorical / quantile ensemble disagreement",
        "",
        "Each entry is mean ± sample SD over initialization repeats; shared fixed data, not confidence intervals.",
        "C-C and Q-Q average 10 unordered pairs each; C-Q averages all 25 cross pairs. Same W1 units.",
        "",
        "| Categorical bins | W1 view | Region | C-C | Q-Q | C-Q | Fraction of x with C-Q > both (mean) |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for support in supports:
        subset = [r for r in results if r["support"] == support]
        for view in ("full", "central"):
            for region in region_masks(x):
                records = [r["views"][view][region] for r in subset]
                values = []
                for name in GROUPS:
                    v = [r[name]["mean"] for r in records]
                    values.append(f"{np.mean(v):.5g} ± {np.std(v, ddof=1) if len(v)>1 else 0:.3g}")
                fraction = np.mean([r["between_greater_than_both_fraction"] for r in records])
                lines.append(f"| {support} | {view} | {region} | " + " | ".join(values) + f" | {fraction:.3f} |")
    lines += [
        "",
        "![Disagreement](uncertainty_comparison.png)",
        "",
        "![Predictions](prediction_comparison.png)",
        "",
        "Full W1 includes the categorical long tails. Central integration clips both families' atoms to [-5,5].",
        "Between-family disagreement can include representation bias even on ID data; "
        "larger is not automatically better.",
        "Only independently trained homogeneous ensembles are compared; this is not mixed-ensemble RL training.",
        "Detailed OOD/ID ratios, ID-subtracted scores and OOD AUROC are in summary.json.",
    ]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categorical-root", type=Path, default=Path("logdir/ens_toy_example/run_20261104_v2"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, root = args.categorical_root, args.output
    prior = json.loads((source / "protocol.json").read_text())
    if prior["status"] != "complete" or sha256(source / "data.npz") != prior["data_sha256"]:
        raise ValueError("Categorical experiment incomplete or data hash changed")
    for name in ("networks.py", "distributions.py", "tools.py"):
        if sha256(name) != prior["source_sha256"][name]:
            raise ValueError(f"Architecture dependency changed: {name}")
    if ast.dump(ast.parse((source / "source_run.py").read_text())) != ast.dump(
        ast.parse(Path("ens_toy_example/run.py").read_text())
    ):
        raise ValueError("Shared toy code changed since categorical experiment")
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    config = prior["args"]
    seeds, supports = config["seeds"], config["supports"]
    data = np.load(source / "data.npz")
    protocol = {
        "status": "running",
        "categorical_root": str(source.resolve()),
        "inherited_config": config,
        "data_sha256": sha256(source / "data.npz"),
        "loss": "exact pinball; midpoint tau=(k+0.5)/255",
        "quantile_init_seed_offset": 10000,
        "quantile_output": "255 raw learned atom locations, equal weights",
        "crossing": "No sorting during training; sorted as equal-mass atoms for distance evaluation",
        "selection": "ID validation mean pinball loss; same search frequency/budget as categorical NLL selection",
        "device": "cpu",
        "threads": 1,
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "torch": torch.__version__,
        "source_sha256": {str(Path(__file__)): sha256(__file__)},
    }
    shutil.copyfile(__file__, root / "source_compare_quantile.py")
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    results, q_stats = [], []
    for seed in seeds:
        q, stats = train_quantiles(config, seed, data, root / f"quantile_s{seed}")
        q_stats.append(stats)
        for support in supports:
            results.append(compare_run(source, root, support, seed, q))
        (root / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
        (root / "quantile_summary.json").write_text(json.dumps(q_stats, indent=2, allow_nan=False) + "\n")
    plot_comparison(root, source, results, seeds, supports)
    protocol["status"] = "complete"
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")


if __name__ == "__main__":
    main()
