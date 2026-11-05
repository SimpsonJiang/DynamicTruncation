"""Reproduce the exp vs direct-power comparison without launching training jobs.

Run from the repository root:
mkdir -p .cache/adv_gate_exp_vs_pow/tmp .cache/adv_gate_exp_vs_pow/inductor
TMPDIR="$PWD/.cache/adv_gate_exp_vs_pow/tmp" \
TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/adv_gate_exp_vs_pow/inductor" \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
  .venv/bin/python analysis_outputs/adv_gate_exp_vs_pow/benchmark.py
"""

import json
import math
import platform
import statistics
import sys
from pathlib import Path

import torch
from torch.utils.benchmark import Timer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from truncation import build_truncation_mask


def score(u, a, mode, factor, power):
    u, a = u.detach().float(), a.detach().float()
    if factor == 1:
        return u
    valid = torch.isfinite(u) & (u >= 0) & torch.isfinite(a)
    shaped = a.clamp_min(0) if mode == "positive_only" else a
    if power:
        result = u / (factor ** shaped.tanh())
    else:
        result = u * torch.exp(-math.log(factor) * shaped.tanh())
    return torch.where(valid, result, torch.full_like(result, torch.inf))


def compare_values():
    rows = []
    boundary_example = None
    for mode in ("positive_only", "signed"):
        for factor in (1.0, 1.01, 1.5, 2.0, 4.0, 10.0):
            row = dict(mode=mode, factor=factor, elements=0, bitwise_different=0,
                       mask_elements_different=0, max_abs_difference=0.0,
                       max_rel_difference=0.0, exp_max_rel_vs_fp64=0.0,
                       pow_max_rel_vs_fp64=0.0)
            for seed in range(10):
                generator = torch.Generator().manual_seed(seed)
                # Nonnegative symlog W1 is bounded by the [-20,20] support width.
                u = torch.exp(torch.rand((1024, 15, 1), generator=generator) * math.log(40 / 1e-8) + math.log(1e-8))
                a = torch.randn(u.shape, generator=generator) * 2
                u.flatten()[::101] = 0
                a.flatten()[::113] = 100
                a.flatten()[::127] = -100
                exp = score(u, a, mode, factor, False)
                power = score(u, a, mode, factor, True)
                shaped = a.double().clamp_min(0) if mode == "positive_only" else a.double()
                reference = u.double() / (factor ** shaped.tanh())
                denom = reference.abs().clamp_min(1e-300)
                diff = (exp.double() - power.double()).abs()
                row["elements"] += u.numel()
                row["bitwise_different"] += int((exp != power).sum())
                for key, value in (
                    ("max_abs_difference", diff.max().item()),
                    ("max_rel_difference", (diff / denom).max().item()),
                    ("exp_max_rel_vs_fp64", ((exp.double() - reference).abs() / denom).max().item()),
                    ("pow_max_rel_vs_fp64", ((power.double() - reference).abs() / denom).max().item()),
                ):
                    row[key] = max(row[key], value)
                threshold = torch.quantile(u[:, 0].flatten(), 0.95)
                mask_exp = build_truncation_mask(exp, threshold, 8)
                mask_pow = build_truncation_mask(power, threshold, 8)
                row["mask_elements_different"] += int((mask_exp != mask_pow).sum())
                if factor == 2 and boundary_example is None and (power != exp).any():
                    index = (power != exp).flatten().nonzero()[0, 0]
                    test_u, test_a = torch.zeros(1, 15, 1), torch.zeros(1, 15, 1)
                    test_u[0, 8, 0], test_a[0, 8, 0] = u.flatten()[index], a.flatten()[index]
                    left = score(test_u, test_a, mode, factor, False)
                    right = score(test_u, test_a, mode, factor, True)
                    boundary = torch.minimum(left[0, 8, 0], right[0, 8, 0])
                    boundary_example = dict(
                        mode=mode, factor=factor, u=test_u[0, 8, 0].item(), a=test_a[0, 8, 0].item(),
                        threshold=boundary.item(), exp_score=left[0, 8, 0].item(),
                        pow_score=right[0, 8, 0].item(),
                        exp_kept=int(build_truncation_mask(left, boundary, 8).sum()),
                        pow_kept=int(build_truncation_mask(right, boundary, 8).sum()),
                        note="Constructed threshold at the smaller rounded score; not sampled from an experiment.",
                    )
            rows.append(row)
    return rows, boundary_example


def compare_timing():
    rows = []
    for mode in ("positive_only", "signed"):
        def exp_fn(u, a):
            return score(u, a, mode, 2.0, False)

        def pow_fn(u, a):
            return score(u, a, mode, 2.0, True)

        for backend in ("eager", "inductor"):
            fns = [exp_fn, pow_fn]
            if backend == "inductor":
                fns = [torch.compile(fn, fullgraph=True) for fn in fns]
            for trajectories in (8, 1024):
                torch.manual_seed(20261105)
                u = torch.rand(trajectories, 15, 1)
                a = torch.randn_like(u)
                outputs = [fn(u, a) for fn in fns]  # Compilation excluded from timing.
                times = [[], []]
                for repeat in range(5):
                    for i in (range(2) if repeat % 2 == 0 else reversed(range(2))):
                        result = Timer("fn(u, a)", globals=dict(fn=fns[i], u=u, a=a), num_threads=1)
                        times[i].append(result.blocked_autorange(min_run_time=0.2).median * 1e6)
                row = dict(mode=mode, backend=backend, shape=list(u.shape), factor=2.0,
                           exp_median_us=statistics.median(times[0]),
                           pow_median_us=statistics.median(times[1]),
                           exp_runs_us=times[0], pow_runs_us=times[1],
                           output_max_abs_diff=(outputs[0] - outputs[1]).abs().max().item())
                row["pow_over_exp"] = row["pow_median_us"] / row["exp_median_us"]
                rows.append(row)
                print(json.dumps(row), flush=True)
    return rows


def main():
    torch.set_num_threads(1)
    result = dict(torch=torch.__version__, python=platform.python_version(),
                  device="cpu", threads=1, cuda_available=torch.cuda.is_available(),
                  dtype="float32", note="Synthetic inputs; no RL training experiments launched.")
    result["numerical"], result["boundary_example"] = compare_values()
    destination = Path(__file__).with_name("results.json")
    destination.write_text(json.dumps(result, indent=2) + "\n")
    result["timing"] = compare_timing()
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print("Saved", destination, flush=True)


if __name__ == "__main__":
    main()
