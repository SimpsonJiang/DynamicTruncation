"""Stage 2: compare full, float64-core and float32-core restores to untouched DMC trajectories."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .common import load_config, new_directory, provenance, statistics, write_json
from .environment import TASKS, ProbeEnv, observation_vector, state_errors, tensor

MODES = ("full", "core64", "core32")
FULL_METRICS = ("integration_max_abs", "observation_max_abs", "reward_abs", "done_mismatch")


def max_abs(array):
    return float(np.max(np.abs(array))) if np.size(array) else 0.0


def record(env, obs):
    return {
        "core": env.core_state(),
        "integration": env.integration_state(),
        "observation": observation_vector(obs).copy(),
        "obs_fields": {k: np.asarray(v).copy() for k, v in obs.items() if not k.startswith("is_")},
        "contacts": int(env.physics.data.ncon),
        "done": bool(env.finished),
    }


def errors(env, obs, reference, schema, reward=0.0, reference_reward=0.0):
    prediction, target = env.core_state(), reference["core"]
    nq, nv = schema["nq"], schema["nv"]
    result = state_errors(prediction, target, schema)
    result.update(
        core_max_abs=max_abs(prediction - target),
        qpos_max_abs=max_abs(prediction[:nq] - target[:nq]),
        qvel_max_abs=max_abs(prediction[nq : nq + nv] - target[nq : nq + nv]),
        integration_max_abs=max_abs(env.integration_state() - reference["integration"]),
        observation_max_abs=max_abs(observation_vector(obs) - reference["observation"]),
        observation_rmse=float(np.sqrt(np.mean((observation_vector(obs) - reference["observation"]) ** 2))),
        reward_abs=abs(reward - reference_reward),
        done_mismatch=float(env.finished != reference["done"]),
        contact_count_difference=abs(int(env.physics.data.ncon) - reference["contacts"]),
    )
    if schema["na"]:
        result["act_max_abs"] = max_abs(prediction[nq + nv :] - target[nq + nv :])
    for key, value in reference["obs_fields"].items():
        result[f"obs_{key}_max_abs"] = max_abs(np.asarray(obs[key]) - value)
    if not np.isfinite(list(result.values())).all():
        raise ValueError("Nonfinite restoration comparison")
    return result


def build_reference(config, seed, points, hold, horizon):
    """Generate once, never restore the reference environment, never auto-reset."""
    env = ProbeEnv(config, seed)
    try:
        if any(t < 0 or t >= env.limit for t in points):
            raise ValueError(f"Snapshot indices must lie in [0,{env.limit})")
        length = min(max(points) + horizon, env.limit)
        rng = np.random.default_rng(np.random.SeedSequence([seed, hold]))
        controls = rng.uniform(-1, 1, ((length + hold - 1) // hold, *env.action_space.shape))
        actions = np.repeat(controls, hold, axis=0)[:length]
        refs = [record(env, env.observation(first=True))]
        rewards, snapshots = [], {}
        for t, action in enumerate(actions):
            if t in points:
                snapshots[t] = env.snapshot()
            obs, reward, done = env.step(action)
            refs.append(record(env, obs))
            rewards.append(reward)
            if done:
                break
        return refs, snapshots, actions[: len(rewards)], np.asarray(rewards), env.schema()
    finally:
        env.close()


def save_reference(path, seed, hold, refs, snapshots, actions, rewards, schema):
    torch.save(
        {
            "seed": seed,
            "hold": hold,
            "schema": schema,
            "snapshots": snapshots,
            "actions": tensor(actions),
            "rewards": tensor(rewards),
            "cores": tensor(np.stack([r["core"] for r in refs])),
            "integration": tensor(np.stack([r["integration"] for r in refs])),
            "observations": tensor(np.stack([r["observation"] for r in refs])),
            "obs_fields": {k: tensor(np.stack([r["obs_fields"][k] for r in refs])) for k in refs[0]["obs_fields"]},
            "contacts": torch.tensor([r["contacts"] for r in refs]),
            "done": torch.tensor([r["done"] for r in refs]),
        },
        path,
    )


def run_task(config, seeds, points, holds, horizon, output, full_atol):
    output = new_directory(output)
    rows, cases, failures, coverage = [], [], [], []
    with (output / "metrics.jsonl").open("x") as stream:
        for seed in seeds:
            for hold in holds:
                refs, snapshots, actions, rewards, schema = build_reference(config, seed, points, hold, horizon)
                save_reference(
                    output / f"reference_s{seed}_hold{hold}.pt", seed, hold, refs, snapshots, actions, rewards, schema
                )
                coverage.append({
                    "seed": seed,
                    "hold": hold,
                    "steps": len(rewards),
                    "contact_steps": sum(r["contacts"] > 0 for r in refs),
                    "max_contacts": max(r["contacts"] for r in refs),
                    "reference_reward_sum": float(rewards.sum()),
                    "reference_qvel_norm_max": max(
                        float(np.linalg.norm(r["core"][schema["nq"] : schema["nq"] + schema["nv"]])) for r in refs
                    ),
                })
                env = ProbeEnv(config, seed)
                try:
                    for start in points:
                        if start not in snapshots:
                            failures.append({
                                "seed": seed,
                                "hold": hold,
                                "start": start,
                                "mode": "reference",
                                "error": "Reference terminated before requested snapshot",
                            })
                            continue
                        snapshot = snapshots[start]
                        for mode in MODES:
                            case_rows = []
                            key = {
                                "seed": seed,
                                "hold": hold,
                                "start": start,
                                "mode": mode,
                                "initial_contacts": refs[start]["contacts"],
                            }
                            try:
                                core = None
                                if mode != "full":
                                    core = snapshot["core"].numpy().copy()
                                    if mode == "core32":
                                        core = core.astype(np.float32).astype(np.float64)
                                obs = env.restore(snapshot, core)
                                cumulative_error = 0.0
                                for depth in range(min(horizon, len(actions) - start) + 1):
                                    reward = reference_reward = 0.0
                                    if depth:
                                        obs, reward, _ = env.step(actions[start + depth - 1])
                                        reference_reward = float(rewards[start + depth - 1])
                                        cumulative_error += reward - reference_reward
                                    metrics = errors(env, obs, refs[start + depth], schema, reward, reference_reward)
                                    metrics["cumulative_return_abs"] = abs(cumulative_error)
                                    row = {**key, "depth": depth, "metrics": metrics}
                                    rows.append(row)
                                    case_rows.append(row)
                                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                                    if env.finished or refs[start + depth]["done"]:
                                        break
                            except (ValueError, RuntimeError, ArithmeticError) as exc:
                                failures.append({**key, "completed_depth": len(case_rows) - 1, "error": str(exc)})
                            cases.append({
                                **key,
                                "completed_depth": len(case_rows) - 1,
                                "expected_depth": min(horizon, len(actions) - start),
                                "initial": case_rows[0]["metrics"] if case_rows else {},
                                "peak": (
                                    {k: max(r["metrics"][k] for r in case_rows) for k in case_rows[0]["metrics"]}
                                    if case_rows
                                    else {}
                                ),
                            })
                finally:
                    env.close()
                stream.flush()
                print(f"{config.task}: seed={seed}, action_hold={hold}, snapshots={len(snapshots)}", flush=True)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["mode"], row["depth"])].append(row["metrics"])
    full_cases = [c for c in cases if c["mode"] == "full"]
    summary = {
        "task": str(config.task),
        "schema": schema,
        "provenance": provenance(),
        "seeds": seeds,
        "points": points,
        "action_holds": holds,
        "horizon": horizon,
        "action_repeat": int(config.action_repeat),
        "time_limit": int(config.time_limit),
        "reference": "Untouched original environment trajectories; no restore applied to reference",
        "requested_origins": len(seeds) * len(points) * len(holds),
        "cases": cases,
        "coverage": coverage,
        "failures": failures,
        "full_atol": full_atol,
        "full_passed": (
            len(full_cases) == len(seeds) * len(points) * len(holds)
            and not any(f["mode"] in ("full", "reference") for f in failures)
            and all(
                c["completed_depth"] == c["expected_depth"] and all(c["peak"][k] <= full_atol for k in FULL_METRICS)
                for c in full_cases
            )
        ),
        "by_mode": {
            mode: {
                "initial": statistics([c["initial"] for c in cases if c["mode"] == mode and c["initial"]]),
                "peak_per_origin": statistics([c["peak"] for c in cases if c["mode"] == mode and c["peak"]]),
                "by_depth": {
                    str(depth): statistics(values) for (m, depth), values in sorted(grouped.items()) if m == mode
                },
            }
            for mode in MODES
        },
        "core_accepted": None,
        "limitations": [
            "Random control trajectories, not trained-policy or imagined-state distributions.",
            "True qpos/qvel/act are supplied; no decoder is trained or assessed.",
            "core32 isolates floating-point conversion, not learned decoder error.",
            "Episode-level context and known time/control are restored, not predicted.",
            "Same-trajectory origins are correlated; summary quantiles are descriptive, not confidence intervals.",
        ],
    }
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, required=True, help="Use saved environment configs only; no model loaded"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--points", nargs="+", type=int, required=True)
    parser.add_argument("--holds", nargs="+", type=int, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--full-atol", type=float, default=1e-9)
    args = parser.parse_args()
    if (
        args.horizon < 1
        or any(s < 0 for s in args.seeds)
        or any(p < 0 for p in args.points)
        or any(h < 1 for h in args.holds)
        or not np.isfinite(args.full_atol)
        or args.full_atol <= 0
    ):
        parser.error("Invalid seed/point/hold/horizon/tolerance")
    for name in ("tasks", "seeds", "points", "holds"):
        values = getattr(args, name)
        if len(set(values)) != len(values):
            parser.error(f"Duplicate {name}")
    torch.set_num_threads(1)
    manifest = json.loads(args.manifest.read_text())
    configs = {}
    for task in args.tasks:
        entries = [e for e in manifest["entries"] if e["task"] == task and e["model_seed"] == 200]
        if len(entries) != 1:
            raise ValueError(f"Expected one environment config for {task}")
        configs[task] = load_config(entries[0]).env
    output = new_directory(args.output)
    write_json(
        output / "protocol.json",
        {
            "provenance": provenance(),
            "manifest": str(args.manifest),
            "tasks": args.tasks,
            "seeds": args.seeds,
            "points": args.points,
            "holds": args.holds,
            "horizon": args.horizon,
            "full_atol": args.full_atol,
            "device": "cpu",
            "decoder_used": False,
            "checkpoint_loaded": False,
            "modes": list(MODES),
            "sampling": "uniform normalized actions in [-1,1], held for 1 or more policy steps",
        },
    )
    results = {}
    for task, config in configs.items():
        report = run_task(
            config, args.seeds, sorted(args.points), args.holds, args.horizon, output / task, args.full_atol
        )
        results[task] = {
            "full_passed": report["full_passed"],
            "failures": len(report["failures"]),
            "origins": report["requested_origins"],
        }
    write_json(output / "status.json", {"completed": True, "tasks": results, "core_accepted": None})
    print(json.dumps(results, indent=2), flush=True)
    if not all(r["full_passed"] for r in results.values()):
        raise SystemExit("Full-restore baseline failed; inspect reports before interpreting core results")


if __name__ == "__main__":
    main()
