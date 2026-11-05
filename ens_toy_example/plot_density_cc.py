"""Plot C-C only from the saved density experiment, without training."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

from ens_toy_example.density_sweep import decorate_regions, draw_density

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot(root):
    protocol = json.loads((root / "protocol.json").read_text())
    if protocol["status"] != "complete":
        raise ValueError("Density experiment must be complete")
    seeds = protocol["args"]["seeds"]
    with np.load(root / "data.npz") as data:
        for filename, specs in (
            ("density_cc_only", (("bounded", "full"), ("repository", "central"))),
            ("full_w1_cc_only", (("repository", "full"), ("bounded", "full"))),
        ):
            fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
            draw_density(axes[0], data)
            for ax, (support, view) in zip(axes[1:], specs):
                curves = []
                grid = None
                for seed in seeds:
                    with np.load(root / f"comparison_{support}_s{seed}" / f"{view}.npz") as saved:
                        if grid is not None:
                            np.testing.assert_array_equal(grid, saved["x"])
                        grid = saved["x"].copy()
                        curve = saved["categorical_within"].copy()
                        np.testing.assert_allclose(curve, saved["cc_pairs"].mean(0))
                        curves.append(curve)
                values = np.stack(curves)
                ax.plot(grid, values.mean(0), color="C1", lw=2, label="C-C: mean of 10 member pairs")
                ax.fill_between(
                    grid,
                    values.min(0),
                    values.max(0),
                    color="C1",
                    alpha=0.2,
                    label="range across initialization repeats",
                )
                decorate_regions(ax)
                description = "full W1" if view == "full" else "W1 over [-5,5] only (diagnostic)"
                ax.set(title=f"Categorical: {support} bins; {description}", ylabel="C-C disagreement (W1)")
                ax.set_ylim(bottom=0)
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
                ax.legend(loc="upper left", fontsize=8)
            axes[-1].set_xlabel("x; grey regions have no training samples")
            fig.suptitle(f"Categorical ensemble disagreement only; {len(seeds)} initialization repeats")
            for extension in ("png", "pdf"):
                fig.savefig(root / f"{filename}.{extension}", dpi=180)
            plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("logdir/ens_toy_example/density_20261104"))
    plot(parser.parse_args().root)
