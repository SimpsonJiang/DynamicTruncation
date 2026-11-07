"""Fixed imagined-depth decoding and one-step same-action restoration diagnostics."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from truncation import (
    advantage_truncation_score,
    build_truncation_mask,
    cal_ens_uncertainty,
)

from .common import (
    load_agent,
    load_config,
    new_directory,
    provenance,
    sha256,
    statistics,
    validate_device,
    validate_render_device,
    write_json,
)
from .environment import ProbeEnv
from .pipeline import load_probe, load_shard
from .posterior_restore import observe_record, step_metrics
from .prediction import metric_vectors


@torch.no_grad()
def frozen_truncation_diagnostics(agent, features):
    """Use saved EMA values without calling their mutating forward methods."""
    distributions = [head(features) for head in agent.value]
    values = torch.stack([dist.mode() for dist in distributions])
    uncertainty = cal_ens_uncertainty(
        distributions, values, agent.uncer_metric, beta=agent.ucb_beta, distance_space=agent.distance_space
    )
    if agent.trunc_kappa is None or not bool(agent.trunc_kappa.initialized):
        raise ValueError("Require the saved initialized truncation threshold")
    threshold = agent.trunc_kappa.value.detach().clone()
    score, scale, pair = uncertainty, None, None
    if agent.adv_scale is not None:
        if not bool(agent.adv_scale.initialized):
            raise ValueError("Saved advantage normalization is uninitialized")
        scale = agent.adv_scale.value.detach().clone().clamp_min(agent.adv_scale.eps)
        if agent.mixed_value_ensemble:
            value = values.mean(0)
            pair = list(range(len(distributions)))
        else:
            indices = torch.randperm(len(distributions), device=features.device)[:2]
            value = values.index_select(0, indices).mean(0)
            pair = indices.cpu().tolist()
        reward = agent._frozen_reward(features).mode()
        cont = agent._frozen_cont(features).mean
        full_return = agent._lambda_return(
            torch.zeros_like(cont), 1 - cont, reward, value, value, 1 - 1 / agent.horizon, agent.lamb
        )
        advantage = (full_return - value[:, :-1]).detach().float() / scale
        score = advantage_truncation_score(
            uncertainty, advantage, mode=agent.adv_trunc_mode, max_factor=agent.adv_trunc_max_factor
        )
    mask = build_truncation_mask(score, threshold, agent.dynamic_trunc_min_horizon)
    return {
        "threshold": float(threshold),
        "advantage_scale": float(scale) if scale is not None else None,
        "advantage_mode": agent.adv_trunc_mode,
        "bootstrap_heads": pair,
        "distance_space": agent.distance_space,
        "critic_count": len(distributions),
        "uncertainty": uncertainty[0, :, 0].cpu().tolist(),
        "score": score[0, :, 0].cpu().tolist(),
        "mask": mask[0, :, 0].cpu().tolist(),
        "retained_steps": int(mask[0].sum()),
        "threshold_policy": "fixed checkpoint EMAs; never updated; training warmup not replayed",
    }


def reference_rollout(env, snapshot, actions, depths):
    """Index d is after exactly actions[0:d], before actions[d]."""
    obs = env.restore(snapshot)
    records = [observe_record(env, obs)]
    snapshots = {0: env.snapshot()} if 0 in depths else {}
    for index, action in enumerate(actions, 1):
        if env.finished:
            break
        obs, reward, _ = env.step(action)
        records.append(observe_record(env, obs, reward))
        if index in depths:
            snapshots[index] = env.snapshot()
    return records, snapshots


def one_step_branch(env, snapshot, core, before, after, action, schema):
    """Capture initial error and exactly one successor, without a posterior update."""
    rows, final_snapshot, failure = [], None, None
    phase = "restore"
    try:
        obs = env.restore(snapshot, core)
        rows.append({"phase": 0, "metrics": step_metrics(observe_record(env, obs), {**before, "reward": 0.0}, schema)})
        phase = "step"
        obs, reward, _ = env.step(action)
        current = observe_record(env, obs, reward)
        metrics = step_metrics(current, after, schema)
        rows.append({"phase": 1, "metrics": metrics})
        if metrics["time_abs"] > 1e-8 or metrics["done_mismatch"]:
            raise ValueError("Simulator time or termination differs; possible reset")
        final_snapshot = env.snapshot()
    except (ValueError, RuntimeError, ArithmeticError) as exc:
        failure = {"phase": phase, "error": f"{type(exc).__name__}: {exc}"}
    return rows, final_snapshot, failure


def summarize(rows, cases, depths, planned):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["imag_depth"], row["source"], row["phase"])].append(row["metrics"])
    result = []
    for depth in depths:
        for source in ("features", "core64_control"):
            selected = [case for case in cases if case["imag_depth"] == depth and case["source"] == source]
            for phase in (0, 1):
                values = grouped[(depth, source, phase)]
                result.append(
                    {
                        "imag_depth": depth,
                        "source": source,
                        "phase": phase,
                        "planned_origins": planned,
                        "attempted": len(selected),
                        "failed": sum(case["failure"] is not None for case in selected),
                        "available_origins": len(values),
                        "metrics": statistics(values) if values else {},
                    }
                )
    return result


@torch.no_grad()
def evaluate(
    source_root,
    entry_id,
    output,
    checkpoint_group,
    points=(0, 100, 200, 300, 400, 450),
    depths=(8, 10, 12, 14),
    device="cuda:0",
    seed=71317,
    test_limit=None,
    example_start=100,
):
    validate_device(device)
    source_root = Path(source_root).resolve()
    protocol = json.loads((source_root / "protocol.json").read_text())
    if protocol["checkpoint_group"] != checkpoint_group or entry_id not in protocol["entries"]:
        raise ValueError("Group/entry differs from requested campaign")
    if sorted(set(depths)) != list(depths) or min(depths) < 1 or max(depths) >= 15:
        raise ValueError("Choose distinct increasing imagined depths 1..14")
    if not points or len(set(points)) != len(points) or min(points) < 0 or example_start not in points:
        raise ValueError("Invalid origin selection")
    checkpoint = source_root / entry_id / "features/decoder.pt"
    path, meta, saved, decoder = load_probe(source_root / entry_id / "data", checkpoint, device)
    decoder_digest = sha256(checkpoint)
    prediction_report = json.loads((source_root / entry_id / "test_features.json").read_text())
    if prediction_report["checkpoint_sha256"] != decoder_digest:
        raise ValueError("Feature decoder changed since its held-out prediction test")
    if saved["source"] != "features" or decoder.dtype != torch.float64:
        raise ValueError("Require the held-out float64 feature decoder")
    if Path(meta["entry"]["run"]).parent.name != checkpoint_group:
        raise ValueError("Dataset checkpoint group differs")
    config = load_config(meta["entry"], device)
    vision = meta.get("policy_modality") == "vision"
    if vision:
        validate_render_device()
    setup_env = ProbeEnv(config.env, meta["seed"])
    try:
        if vision:
            setup_env.observation_space.spaces["image"] = gym.spaces.Box(
                0, 255, tuple(config.env.size) + (3,), dtype=np.uint8
            )
        agent = load_agent(meta["entry"], setup_env, device)
    finally:
        setup_env.close()
    if agent.imag_horizon != 15:
        raise ValueError("This campaign requires trained H=15")
    ema_before = {k: v.clone() for k, v in agent.state_dict().items() if k.startswith(("trunc_kappa.", "adv_scale."))}
    output = new_directory(output)
    episodes = saved["split"]["test"] if test_limit is None else saved["split"]["test"][:test_limit]
    rows, cases, failures, traces, example = [], [], [], [], None
    with (output / "metrics.jsonl").open("x") as stream:
        for episode_index, episode in enumerate(episodes):
            shard = load_shard(path, meta["episodes"][episode])
            if shard["cores"].dtype != torch.float64 or max(points) + 15 > len(shard["actions"]):
                raise ValueError("Origin lacks 15 actions before the recorded episode boundary")
            env = ProbeEnv(config.env, shard["episode_seed"], include_image=vision)
            branch_env = ProbeEnv(config.env, shard["episode_seed"], include_image=vision)
            try:
                for start in points:
                    identity = {"episode_id": episode, "episode_seed": shard["episode_seed"], "start": start}
                    raw_seed = f"{seed}:{checkpoint_group}:{entry_id}:{episode}:{start}".encode()
                    origin_seed = int.from_bytes(hashlib.sha256(raw_seed).digest()[:4], "little")
                    torch.manual_seed(origin_seed)
                    latent = tuple(shard[key][start : start + 1].to(device) for key in ("stoch", "deter"))
                    torch.testing.assert_close(
                        agent.rssm.get_feat(*latent).cpu(), shard["features"][start : start + 1], rtol=0, atol=0
                    )
                    features, imagined_actions = agent._imagine(latent, 16)
                    torch.testing.assert_close(
                        features[:, 0].cpu(), shard["features"][start : start + 1], rtol=0, atol=0
                    )
                    diagnostics = frozen_truncation_diagnostics(agent, features)
                    actions = imagined_actions[0, :15].cpu().numpy()
                    decoded = decoder(features[0]).cpu().numpy()
                    traces.append(
                        {
                            **identity,
                            "rng_seed": origin_seed,
                            "actions": torch.from_numpy(actions.copy()),
                            "features": features[0].cpu(),
                            "decoded": torch.from_numpy(decoded.copy()),
                            "truncation": diagnostics,
                        }
                    )
                    try:
                        records, snapshots = reference_rollout(env, shard["snapshots"][start], actions, depths)
                    except (ValueError, RuntimeError, ArithmeticError) as exc:
                        failures.append({**identity, "phase": "reference", "error": str(exc)})
                        continue
                    illustration = episode_index == 0 and start == example_start
                    if illustration:
                        example = {"entry": meta["entry"], **identity, "depths": {}, "truncation": diagnostics}
                    for depth in depths:
                        if depth + 1 >= len(records):
                            failures.append(
                                {
                                    **identity,
                                    "imag_depth": depth,
                                    "phase": "reference",
                                    "error": "Reference ended early",
                                }
                            )
                            continue
                        if illustration:
                            example["depths"][depth] = {
                                "action": actions[depth].tolist(),
                                "branches": {},
                                "reference_before": snapshots[depth],
                            }
                        for source in ("features", "core64_control"):
                            core = decoded[depth] if source == "features" else records[depth]["core"]
                            branch_rows, snapshot, failure = one_step_branch(
                                branch_env,
                                snapshots[depth],
                                core,
                                records[depth],
                                records[depth + 1],
                                actions[depth],
                                meta["schema"],
                            )
                            key = {
                                **identity,
                                "imag_depth": depth,
                                "source": source,
                                "retained_steps": diagnostics["retained_steps"],
                                "action_retained": depth < diagnostics["retained_steps"],
                                "at_truncation_boundary": depth == diagnostics["retained_steps"] < 15,
                            }
                            if source == "features" and snapshot is not None:
                                consistency = metric_vectors(
                                    snapshot["core"].numpy()[None], decoded[depth + 1 : depth + 2], meta["schema"]
                                )
                                branch_rows[-1]["metrics"].update(
                                    {f"latent_next_{k}": float(v[0]) for k, v in consistency.items()}
                                )
                            for row in branch_rows:
                                record = {**key, **row}
                                rows.append(record)
                                stream.write(json.dumps(record, allow_nan=False) + "\n")
                            cases.append(
                                {
                                    **key,
                                    "failure": failure,
                                    "completed_phase": branch_rows[-1]["phase"] if branch_rows else -1,
                                }
                            )
                            if illustration:
                                example["depths"][depth]["branches"][source] = {
                                    "snapshot": snapshot,
                                    "failure": failure,
                                    "metrics": branch_rows[-1]["metrics"] if snapshot is not None else None,
                                }
                    stream.flush()
                print(f"{checkpoint_group}/{entry_id}: episodes {episode_index + 1}/{len(episodes)}", flush=True)
            finally:
                env.close()
                branch_env.close()
    for key, value in ema_before.items():
        torch.testing.assert_close(agent.state_dict()[key], value, rtol=0, atol=0)
    torch.save(traces, output / "traces.pt")
    torch.save(example, output / "example.pt")
    report = {
        "kind": "imagined_depth_one_step_restore",
        "checkpoint_group": checkpoint_group,
        "entry": meta["entry"],
        "policy_modality": "vision" if vision else "proprio",
        "test_episodes": episodes,
        "points": list(points),
        "imagined_depths": list(depths),
        "horizon": 15,
        "post_restore_steps": 1,
        "planned_origins": len(episodes) * len(points),
        "decoder_dtype": "float64",
        "decoder_sha256": decoder_digest,
        "dataset_sha256": sha256(path / "dataset.json"),
        "seed": seed,
        "reference_failures": failures,
        "cases": cases,
        "summary": summarize(rows, cases, depths, len(episodes) * len(points)),
        "ema_unchanged": True,
        "provenance": provenance(),
        "scientific_acceptance": None,
        "notes": [
            "Depth0 is the stored posterior. Depth d follows d imagined actions; restored +1 executes action[d].",
            "True reference starts at the real snapshot and executes the same imagined actions; "
            "recorded future actions are not used.",
            "Fixed depths can be beyond the diagnostic truncation boundary; all four requested depths are evaluated.",
            "No posterior feedback, decoder training, or replay insertion. "
            "Observations are not fed back into the policy.",
            "observation_rmse is proprio; image_rmse_01 is RGB/255. "
            "Initial sensor cache errors are reported separately.",
            "latent_next metrics compare simulator successor to decoder(imagined feature[d+1]); "
            "they are not ground-truth accuracy.",
            "Example is first test episode, preselected start; no selection by error.",
        ],
    }
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--entry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-group", required=True)
    parser.add_argument("--points", nargs="+", type=int, default=[0, 100, 200, 300, 400, 450])
    parser.add_argument("--depths", nargs="+", type=int, default=[8, 10, 12, 14])
    parser.add_argument("--seed", type=int, default=71317)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--example-start", type=int, default=100)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(2 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    evaluate(
        args.source_root,
        args.entry,
        args.output,
        args.checkpoint_group,
        args.points,
        args.depths,
        seed=args.seed,
        test_limit=args.test_limit,
        example_start=args.example_start,
    )


if __name__ == "__main__":
    main()
