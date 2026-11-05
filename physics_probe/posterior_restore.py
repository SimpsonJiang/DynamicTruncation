"""Matched-action posterior-state restoration with explicit oracle controls and failure accounting."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .common import (
    load_config,
    new_directory,
    provenance,
    sha256,
    statistics,
    validate_render_device,
    write_json,
)
from .environment import ProbeEnv, observation_vector
from .pipeline import load_probe, load_shard
from .prediction import metric_vectors


def observe_record(env, obs, reward=0.0):
    proprio = {k: v for k, v in obs.items() if k != "image"}
    result = {
        "core": env.core_state(),
        "obs": observation_vector(proprio).astype(np.float64),
        "fields": {k: np.asarray(v, dtype=np.float64) for k, v in proprio.items() if not k.startswith("is_")},
        "reward": float(reward),
        "done": bool(env.finished),
        "time": float(env.physics.data.time),
        "contacts": int(env.physics.data.ncon),
    }
    if "image" in obs:
        result["image"] = np.asarray(obs["image"]).copy()
    return result


def trace_reference(env, shard, start, horizon):
    """Reproduce recorded controls; verify available future states against the untouched collection."""
    obs = env.restore(shard["snapshots"][start])
    records = [observe_record(env, obs)]
    check = {
        "stored_core_max_abs": 0.0,
        "stored_observation_max_abs": 0.0,
        "stored_reward_max_abs": 0.0,
        "done_mismatches": 0,
        "checked_states": 0,
    }
    actions = shard["actions"][start : start + horizon].numpy()
    for depth in range(len(actions) + 1):
        if depth:
            obs, reward, _ = env.step(actions[depth - 1])
            records.append(observe_record(env, obs, reward))
            check["stored_reward_max_abs"] = max(
                check["stored_reward_max_abs"], abs(reward - float(shard["rewards"][start + depth - 1]))
            )
            check["done_mismatches"] += int(env.finished != bool(shard["done"][start + depth - 1]))
        index = start + depth
        if index < len(shard["cores"]):
            check["stored_core_max_abs"] = max(
                check["stored_core_max_abs"], float(np.max(np.abs(records[-1]["core"] - shard["cores"][index].numpy())))
            )
            check["stored_observation_max_abs"] = max(
                check["stored_observation_max_abs"],
                float(np.max(np.abs(records[-1]["obs"] - shard["observations"][index].numpy()))),
            )
            check["checked_states"] += 1
            if "image" in records[-1] and "images" in shard:
                difference = records[-1]["image"].astype(np.int16) - shard["images"][index].numpy().astype(np.int16)
                check["stored_image_max_abs"] = max(check.get("stored_image_max_abs", 0), int(np.abs(difference).max()))
        if env.finished:
            break
    # Stored rewards/observations are float32, while the live reward/core remain float64.
    check["passed"] = (
        check["stored_core_max_abs"] <= 1e-9
        and check["stored_observation_max_abs"] <= 1e-6
        and check["stored_reward_max_abs"] <= 1e-6
        and check["done_mismatches"] == 0
    )
    if "stored_image_max_abs" in check:
        check["image_within_one_gray_level"] = check["stored_image_max_abs"] <= 1
        # Raster differences are separate from authoritative state/reward reference validation.
    return records, actions[: len(records) - 1], check


def step_metrics(current, reference, schema):
    vectors = metric_vectors(current["core"][None], reference["core"][None], schema)
    result = {k: float(v[0]) for k, v in vectors.items()}
    delta = current["obs"] - reference["obs"]
    result.update(
        observation_rmse=float(np.sqrt(np.mean(delta**2))),
        observation_max_abs=float(np.max(np.abs(delta))),
        reward_abs=abs(current["reward"] - reference["reward"]),
        done_mismatch=float(current["done"] != reference["done"]),
        time_abs=abs(current["time"] - reference["time"]),
        contact_count_difference=abs(current["contacts"] - reference["contacts"]),
    )
    for key, value in current["fields"].items():
        result[f"obs_{key}_max_abs"] = float(np.max(np.abs(value - reference["fields"][key])))
    if "image" in reference:
        difference = (current["image"].astype(np.float64) - reference["image"].astype(np.float64)) / 255.0
        result["image_rmse_01"] = float(np.sqrt(np.mean(difference**2)))
        result["image_max_abs_01"] = float(np.abs(difference).max())
    if not np.isfinite(list(result.values())).all():
        raise ValueError("Nonfinite restoration metric")
    return result


def trace_candidate(env, snapshot, core, references, actions, schema, control_initial=None):
    rows, failure = [], None
    current_return = reference_return = 0.0
    phase = "restore"
    try:
        obs = env.restore(snapshot, core)
        for depth, reference in enumerate(references):
            reward = 0.0
            if depth:
                phase = "step"
                obs, reward, _ = env.step(actions[depth - 1])
            current = observe_record(env, obs, reward)
            phase = "metrics"
            metrics = step_metrics(current, reference, schema)
            current_return += reward
            reference_return += reference["reward"]
            metrics.update(
                candidate_return=current_return,
                reference_return=reference_return,
                cumulative_return_abs=abs(current_return - reference_return),
                cumulative_return_signed=current_return - reference_return,
            )
            if depth == 0 and control_initial is not None:
                difference = current["obs"] - control_initial
                metrics["observation_vs_core_control_rmse"] = float(np.sqrt(np.mean(difference**2)))
            rows.append({"depth": depth, "metrics": metrics})
            if metrics["time_abs"] > 1e-8:
                failure = {
                    "phase": "step",
                    "depth": depth,
                    "error": "Unexpected simulator time; possible automatic reset",
                }
                break
            if current["done"] != reference["done"]:
                failure = {"phase": "termination", "depth": depth, "error": "Candidate/reference termination differ"}
                break
            if current["done"]:
                break
    except (ValueError, RuntimeError, ArithmeticError) as exc:
        failure = {"phase": phase, "depth": len(rows), "error": f"{type(exc).__name__}: {exc}"}
    return rows, failure


def summarize_cases(cases, metric_rows, sources, origins):
    grouped = defaultdict(list)
    for row in metric_rows:
        grouped[(row["source"], row["depth"])].append(row["metrics"])
    summary = {}
    for source in sources:
        selected = [c for c in cases if c["source"] == source]
        failures = [c for c in selected if c["failure"] is not None]
        summary[source] = {
            "attempted": len(selected),
            "planned": origins,
            "failed": len(failures),
            "failure_rate": len(failures) / len(selected) if selected else None,
            "missing_origins": origins - len(selected),
            "full_horizon_completed": sum(
                c["failure"] is None and c["completed_depth"] == c["expected_depth"] for c in selected
            ),
            "by_depth": {
                str(depth): {
                    "available_origins": len(values),
                    "planned_origins": origins,
                    "metrics": statistics(values),
                }
                for (kind, depth), values in sorted(grouped.items())
                if kind == source
            },
        }
    return summary


@torch.no_grad()
def evaluate_model(
    source_root, entry_id, output, points, horizon, sources, device="cpu", checkpoint_group="VE_wasser_symlog_f64_H15"
):
    source_root = Path(source_root)
    source_protocol = json.loads((source_root / "protocol.json").read_text())
    if source_protocol.get("checkpoint_group") != checkpoint_group:
        raise ValueError(f"This stage-4 campaign is restricted to {checkpoint_group}")
    if (
        entry_id not in source_protocol["entries"]
        or not sources
        or not set(sources) <= {"features", "observations", "images"}
    ):
        raise ValueError("Invalid model or decoder source")
    if horizon < 1 or not points or min(points) < 0 or len(set(points)) != len(points):
        raise ValueError("Invalid rollout sampling specification")
    base = source_root / entry_id
    models, checkpoints = {}, {}
    meta = split = path = None
    for source in sources:
        checkpoint = base / source / "decoder.pt"
        path, current_meta, saved, model = load_probe(base / "data", checkpoint, device)
        if saved["source"] != source or model.dtype != torch.float64:
            raise ValueError("Wrong decoder source or precision")
        if Path(current_meta["entry"]["run"]).parent.name != checkpoint_group:
            raise ValueError("Dataset checkpoint group differs from requested group")
        if meta is not None and (meta != current_meta or split != saved["split"]):
            raise ValueError("Controls must use exactly the same dataset and episode split")
        meta, split = current_meta, saved["split"]
        models[source], checkpoints[source] = model, sha256(checkpoint)
    output = new_directory(output)
    config = load_config(meta["entry"])
    include_image = meta.get("policy_modality") == "vision"
    if include_image:
        validate_render_device()
    origin_count = len(split["test"]) * len(points)
    cases, rows, reference_checks, reference_failures = [], [], [], []
    with (output / "metrics.jsonl").open("x") as stream:
        for episode in split["test"]:
            shard = load_shard(path, meta["episodes"][episode])
            if shard["cores"].dtype != torch.float64 or max(points) >= len(shard["cores"]):
                raise ValueError("Episode lacks requested states or original float64 labels")
            predictions = {
                source: model(shard[source][points].to(device)).cpu().numpy() for source, model in models.items()
            }
            reference_env = ProbeEnv(config.env, shard["episode_seed"], include_image=include_image)
            candidate_env = ProbeEnv(config.env, shard["episode_seed"], include_image=include_image)
            try:
                for index, start in enumerate(points):
                    identity = {"episode_id": episode, "episode_seed": shard["episode_seed"], "start": start}
                    try:
                        references, actions, check = trace_reference(reference_env, shard, start, horizon)
                        reference_checks.append({**identity, **check})
                        if not check["passed"]:
                            raise ValueError("Full reference does not reproduce original collected trajectory")
                    except (ValueError, RuntimeError, ArithmeticError) as exc:
                        reference_failures.append({**identity, "error": str(exc)})
                        continue
                    snapshot = shard["snapshots"][start]
                    initial_control = None
                    for source in ["core64_control", *sources]:
                        core = (
                            shard["cores"][start].numpy() if source == "core64_control" else predictions[source][index]
                        )
                        branch, failure = trace_candidate(
                            candidate_env, snapshot, core, references, actions, meta["schema"], initial_control
                        )
                        if source == "core64_control" and branch:
                            # Recompute the same control observation independently of its final rollout state.
                            control_obs = candidate_env.restore(snapshot, core)
                            initial_control = observe_record(candidate_env, control_obs)["obs"]
                        key = {**identity, "source": source}
                        for row in branch:
                            record = {**key, **row}
                            rows.append(record)
                            stream.write(json.dumps(record, allow_nan=False) + "\n")
                        cases.append(
                            {
                                **key,
                                "expected_depth": len(actions),
                                "completed_depth": len(branch) - 1,
                                "failure": failure,
                                "initial_contacts": references[0]["contacts"],
                            }
                        )
                stream.flush()
                print(
                    f"{entry_id}: test episode {episode}, origins={len(points)}, completed episodes="
                    f"{split['test'].index(episode) + 1}/{len(split['test'])}",
                    flush=True,
                )
            finally:
                reference_env.close()
                candidate_env.close()
    report = {
        "kind": "stage4_posterior_same_action_restore",
        "checkpoint_group": checkpoint_group,
        "policy_modality": source_protocol.get("modality", "proprio"),
        "observation_metrics": "proprio scalar observations; not image pixels",
        "image_metrics": "image_rmse_01: RGB normalized to [0,1]; no acceptance threshold" if include_image else None,
        "entry": meta["entry"],
        "decoder_sha256": checkpoints,
        "dataset_sha256": sha256(path / "dataset.json"),
        "sources": sources,
        "dtype": "float64",
        "points": points,
        "horizon": horizon,
        "test_episodes": split["test"],
        "planned_origins": origin_count,
        "reference_checks": reference_checks,
        "reference_failures": reference_failures,
        "reference_passed": len(reference_checks) == origin_count and not reference_failures,
        "summary": summarize_cases(cases, rows, ["core64_control", *sources], origin_count),
        "cases": cases,
        "provenance": provenance(),
        "scientific_acceptance": None,
        "notes": [
            "No decoder training, imagined features, policy feedback, or replay insertion.",
            "All branches execute the same recorded actions; evaluation is open-loop.",
            (
                "Initial observation error includes the known sensor-cache discrepancy; a same-time core control is"
                " included."
            ),
            "Failed/missing origins are reported separately; later-depth averages cover only available branches.",
            (
                "Absolute coordinates are unchanged; periodic angles/quaternion orientation have separate"
                " geometry-aware metrics."
            ),
            "Reported execution completion is not scientific success.",
        ],
    }
    write_json(output / "report.json", report)
    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--entry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--points", nargs="+", type=int, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--sources", nargs="+", choices=["features", "observations", "images"], required=True)
    parser.add_argument("--checkpoint-group", default="VE_wasser_symlog_f64_H15")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    evaluate_model(
        args.source_root,
        args.entry,
        args.output,
        args.points,
        args.horizon,
        args.sources,
        checkpoint_group=args.checkpoint_group,
    )


if __name__ == "__main__":
    main()
