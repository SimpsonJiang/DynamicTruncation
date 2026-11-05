"""Additional central-range diagnostics on saved predictions; no retraining."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

from ens_toy_example.run import correlation, region_masks, risk_curve, truth, uniform_w1

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def analyze(root):
    runs = json.loads((root / "summary.json").read_text())
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for run in runs:
        name = f"{run['support']}_s{run['seed']}"
        saved = np.load(root / name / "predictions.npz")
        x, p, bins = saved["x"], saved["probs"], saved["bins"]
        clipped_bins = np.clip(bins, -5, 5)
        cdfs = p.cumsum(-1)[..., :-1]
        score = np.mean(
            [(np.abs(cdfs[i] - cdfs[j]) * np.diff(clipped_bins)).sum(-1) for i in range(5) for j in range(i + 1, 5)],
            axis=0,
        )
        error = uniform_w1(p.mean(0), clipped_bins, x)
        regions = {}
        for region, mask in region_masks(x).items():
            e, u = error[mask], score[mask]
            risk = risk_curve(e, u)
            regions[region] = {
                "central_w1_mean": float(u.mean()),
                "central_error_mean": float(e.mean()),
                "rho_central_w1_central_error": correlation(u, e),
                "rho_raw_w1_central_error": correlation(saved["w1"][mask], e),
                "rho_raw_w1_mean_error": correlation(saved["w1"][mask], saved["mean_error"][mask]),
                "risk_at_half_over_full": float(risk[len(risk) // 2 - 1] / e.mean()),
            }
        exact_tail_fraction = 1 - score / np.maximum(saved["w1"], 1e-30)
        tail_mass = p[..., np.abs(bins) > 5].sum(-1).mean(0)
        rows.append(
            {
                "name": name,
                "regions": regions,
                "exact_tail_fraction_median": {
                    region: float(np.median(exact_tail_fraction[mask])) for region, mask in region_masks(x).items()
                },
                "outside_5_probability_mean": {
                    region: float(tail_mass[mask].mean()) for region, mask in region_masks(x).items()
                },
            }
        )
        row = 0 if run["support"] == "repository" else 1
        axes[row, 0].plot(x, score, label=f"seed {run['seed']}")
        axes[row, 1].plot(x, error)
        axes[row, 2].plot(x, saved["mean_error"])
        for col, label in enumerate(
            ("W1 integrated over [-5,5]", "True error integrated over [-5,5]", "Absolute mean error")
        ):
            axes[row, col].set(title=f"{run['support']}: {label}", xlabel="x", yscale="log")
            if run is next(r for r in runs if r["support"] == run["support"]):
                axes[row, col].axvspan(-1, -0.3, color="green", alpha=0.08)
                axes[row, col].axvspan(0.3, 1, color="green", alpha=0.08)
        np.savez_compressed(root / name / "central_diagnostics.npz", x=x, central_w1=score, central_error=error)
    for row in range(2):
        axes[row, 0].legend()
    fig.suptitle("Central-range diagnostics: same trained predictions, no renormalization or retraining")
    fig.savefig(root / "central_diagnostics.png", dpi=180)
    fig.savefig(root / "central_diagnostics.pdf")
    plt.close(fig)
    (root / "central_diagnostics.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")

    lines = [
        "# Central-range diagnostic (post-training)",
        "",
        "Integrate CDF differences over [-5,5], equivalent to clipping atoms to that interval; no renormalization.",
        "This is a diagnostic, not the repository's raw-return W1 or a proposed replacement metric.",
        "Added after the first completed run showed tail-dominated raw W1; all trained models are unchanged.",
        "",
        "| Run | Region | Central W1 | Central true error | rho(central W1, central error) | "
        "rho(raw W1, central error) | rho(raw W1, mean error) | Central risk@50% / full |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        for region, metrics in row["regions"].items():
            vals = ["n/a" if v is None else f"{v:.4g}" for v in metrics.values()]
            lines.append(f"| {row['name']} | {region} | " + " | ".join(vals) + " |")
    (root / "CENTRAL.md").write_text("\n".join(lines) + "\n")

    # Deterministic first seed display, not a seed selected for a desired result.
    first_seed = runs[0]["seed"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row, support in enumerate(("repository", "bounded")):
        saved = np.load(root / f"{support}_s{first_seed}" / "predictions.npz")
        x, quantiles = saved["x"], saved["quantiles"]
        mean, scale = truth(x)
        for col, quantile_index in enumerate((1, 2)):
            ax = axes[row, col]
            ax.axvspan(-1, -0.3, color="green", alpha=0.08)
            ax.axvspan(0.3, 1, color="green", alpha=0.08)
            target = mean if quantile_index == 1 else mean + 0.8 * scale
            ax.plot(x, target, "k--", label="truth", lw=2)
            for i in range(5):
                ax.plot(x, quantiles[i, :, quantile_index], label=f"head {i+1}", alpha=0.8)
            ax.set(title=f"{support}: {'median' if col == 0 else '90% quantile'}", xlabel="x", ylabel="y", ylim=(-4, 4))
            if col == 0:
                ax.legend(fontsize=8)
    fig.suptitle(f"Five initializations, identical training data; ensemble seed {first_seed}")
    fig.savefig(root / "ensemble_overlay.png", dpi=180)
    fig.savefig(root / "ensemble_overlay.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    analyze(parser.parse_args().output)
