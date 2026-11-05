"""Extract quantile-only disagreement plots and tables from saved density runs."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

from ens_toy_example.density_sweep import decorate_regions, draw_density, masks

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot(root):
    protocol = json.loads((root / "protocol.json").read_text())
    if protocol["status"] != "complete":
        raise ValueError("Density experiment must be complete")
    seeds = protocol["args"]["seeds"]
    curves, central_curves, grid = [], [], None
    for seed in seeds:
        for view, destination in (("full", curves), ("central", central_curves)):
            with np.load(root / f"comparison_bounded_s{seed}" / f"{view}.npz") as saved:
                if grid is not None:
                    np.testing.assert_array_equal(grid, saved["x"])
                grid = saved["x"].copy()
                curve = saved["quantile_within"].copy()
                np.testing.assert_allclose(curve, saved["qq_pairs"].mean(0))
                assert np.isfinite(curve).all()
                destination.append(curve)
            # Both categorical comparisons use exactly the same quantile members.
            with np.load(root / f"comparison_repository_s{seed}" / f"{view}.npz") as other:
                np.testing.assert_array_equal(grid, other["x"])
                np.testing.assert_allclose(curve, other["quantile_within"])
    values, central = np.stack(curves), np.stack(central_curves)
    sd = values.std(0, ddof=1) if len(seeds) > 1 else np.zeros_like(grid)
    regions = masks(grid)
    region_names = np.full(len(grid), "", dtype=object)
    for name, mask in regions.items():
        region_names[mask] = name
    assert np.all(region_names != "")

    with (root / "density_qq_only.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["x", "region", *[f"qq_s{s}" for s in seeds], "qq_mean", "qq_sd", "qq_min", "qq_max", "qq_central_mean"]
        )
        writer.writerows(
            zip(grid, region_names, *values, values.mean(0), sd, values.min(0), values.max(0), central.mean(0))
        )
    coverage = json.loads((root / "coverage.json").read_text())
    with (root / "density_qq_regions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "region",
                "start",
                "end",
                "train_count",
                "train_count_per_x_unit",
                *[f"qq_s{s}" for s in seeds],
                "qq_mean",
                "qq_sd",
            ]
        )
        for region in coverage:
            means = values[:, regions[region["name"]]].mean(1)
            writer.writerow(
                [region[k] for k in ("name", "start", "end", "train_count", "train_count_per_x_unit")]
                + list(means)
                + [means.mean(), means.std(ddof=1) if len(seeds) > 1 else 0]
            )
    np.savez_compressed(
        root / "density_qq_only.npz",
        x=grid,
        seeds=seeds,
        qq_by_seed=values,
        qq_central_by_seed=central,
        qq_mean=values.mean(0),
        qq_sd=sd,
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, constrained_layout=True)
    with np.load(root / "data.npz") as data:
        draw_density(axes[0], data)
    ax = axes[1]
    ax.plot(grid, values.mean(0), color="C0", lw=2, label="Q-Q: mean of 10 member pairs")
    ax.fill_between(
        grid, values.min(0), values.max(0), color="C0", alpha=0.2, label="range across initialization repeats"
    )
    decorate_regions(ax)
    ax.set(
        title="Quantile ensemble: full W1",
        xlabel="x; grey regions have no training samples",
        ylabel="Q-Q disagreement (W1)",
        ylim=(0, None),
    )
    ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(f"Quantile ensemble disagreement only; {len(seeds)} initialization repeats")
    for extension in ("png", "pdf"):
        fig.savefig(root / f"density_qq_only.{extension}", dpi=180)
    plt.close(fig)
    print(f"Exported Q-Q plot, {len(grid)} point rows and {len(coverage)} region rows.")
    print(f"Max full/central score difference: {np.max(np.abs(values-central)):.3g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("logdir/ens_toy_example/density_20261104"))
    plot(parser.parse_args().root)
