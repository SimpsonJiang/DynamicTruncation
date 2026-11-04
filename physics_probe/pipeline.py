"""Explicit-budget data collection, supervised fitting, and paired simulation checks."""

import json
from pathlib import Path

import numpy as np
import torch

from truncation import build_truncation_mask, cal_ens_uncertainty

from .common import (
    load_agent,
    load_config,
    new_directory,
    posterior,
    provenance,
    sha256,
    statistics,
    validate_device,
    write_json,
)
from .decoder import StateDecoder
from .environment import ProbeEnv, observation_vector, state_errors, tensor


def compare_step(a, b, action):
    oa, ra, da = a.step(action)
    ob, rb, db = b.step(action)
    errors = state_errors(b.core_state(), a.core_state(), a.schema())
    errors.update(
        observation_rmse=float(np.sqrt(np.mean((observation_vector(oa) - observation_vector(ob)) ** 2))),
        reward_abs=abs(ra - rb),
        done_mismatch=float(da != db),
    )
    return errors, da or db


def restore_check(entry, output, seed, warmup, steps, atol):
    """Verify exact continuation and measure the reset-default core approximation."""
    config = load_config(entry)
    a, b = ProbeEnv(config.env, seed), ProbeEnv(config.env, seed)
    rng = np.random.default_rng(seed)
    rows = {"full": [], "core": []}
    try:
        for _ in range(warmup):
            _, _, done = a.step(rng.uniform(-1, 1, a.action_space.shape))
            if done:
                raise ValueError("Warmup crossed episode end; reduce --warmup")
        snapshot = a.snapshot()
        original_obs = observation_vector(a.observation())
        actions = rng.uniform(-1, 1, (steps, *a.action_space.shape))
        initial = {}
        for mode in rows:
            a.restore(snapshot)
            b.restore(snapshot, None if mode == "full" else snapshot["core"].numpy())
            initial[mode] = float(np.max(np.abs(observation_vector(b.observation()) - original_obs)))
            for action in actions:
                errors, done = compare_step(a, b, action)
                errors["integration_max_abs"] = float(np.max(np.abs(a.integration_state() - b.integration_state())))
                rows[mode].append(errors)
                if done:
                    break
        passed = initial["full"] <= atol and all(max(row.values()) <= atol for row in rows["full"])
        report = {
            "entry": entry,
            "kind": "restore_check",
            "seed": seed,
            "warmup": warmup,
            "provenance": provenance(),
            "requested_steps": steps,
            "full_atol": atol,
            "full_passed": passed,
            "initial_observation_max_abs": initial,
            "metrics": {k: statistics(v) for k, v in rows.items()},
            "schema": a.schema(),
            "model_hash": a.model_hash,
            "core_policy": "predict qpos/qvel/act; known time/ctrl; reset-default user fields; zero warmstart",
            "core_accepted": None,
            "note": "Core approximation has no automatic scientific acceptance threshold.",
        }
        write_json(output, report)
        if not passed:
            raise RuntimeError(f"Full restoration check failed; see {output}")
        return report
    finally:
        a.close()
        b.close()


@torch.no_grad()
def collect(entry, output, seed, episodes, max_steps, device):
    """Episode shards: state[t], posterior[t], action[t], transition reward[t]."""
    validate_device(device)
    output = new_directory(output)
    torch.manual_seed(seed)
    config = load_config(entry, device)
    env = ProbeEnv(config.env, seed)
    try:
        agent = load_agent(entry, env, device)
        records = []
        for episode in range(episodes):
            # Independent reset seeds; no overlap with earlier episode seeds.
            if episode:
                env.close()
                env = ProbeEnv(config.env, seed + episode)
            obs = env.observation(first=True)
            state = agent.get_initial_state(1)
            features, observations, snapshots, actions, rewards, dones, zs, hs = [], [], [], [], [], [], [], []
            for _ in range(max_steps):
                snapshot = env.snapshot()
                feature, action, state = posterior(agent, obs, state)
                features.append(feature[0].cpu())
                observations.append(tensor(observation_vector(obs)))
                snapshots.append(snapshot)
                zs.append(state["stoch"][0].cpu())
                hs.append(state["deter"][0].cpu())
                actions.append(action[0].cpu())
                obs, reward, done = env.step(action[0].cpu().numpy())
                rewards.append(reward)
                dones.append(done)
                if done:
                    break
            shard = {
                "features": torch.stack(features),
                "observations": torch.stack(observations),
                "cores": torch.stack([s["core"] for s in snapshots]),
                "snapshots": snapshots,
                "stoch": torch.stack(zs),
                "deter": torch.stack(hs),
                "actions": torch.stack(actions),
                "rewards": torch.tensor(rewards),
                "done": torch.tensor(dones),
                "episode_seed": seed + episode,
                "episode_id": episode,
                "complete_episode": bool(dones[-1]),
                "schema": env.schema(),
            }
            path = output / f"episode_{episode:05d}.pt"
            torch.save(shard, path)
            records.append({
                "path": path.name,
                "sha256": sha256(path),
                "episode_id": episode,
                "episode_seed": seed + episode,
                "samples": len(features),
                "complete_episode": shard["complete_episode"],
            })
            print(f"Collected {entry['id']} episode {episode}: {len(features)} states", flush=True)
        meta = {
            "version": 1,
            "entry": entry,
            "schema": env.schema(),
            "episodes": records,
            "seed": seed,
            "max_steps": max_steps,
            "device": device,
            "obs_keys": sorted(env.observation_space.spaces),
            "policy": "stochastic frozen actor",
            "alignment": "feature/state at t, action at t produces next observation; no physics labels in encoder",
            "provenance": provenance(),
        }
        write_json(output / "dataset.json", meta)
        return meta
    finally:
        env.close()


def read_dataset(path):
    path = Path(path)
    meta = json.loads((path / "dataset.json").read_text())
    return path, meta


def load_shard(path, record):
    filename = path / record["path"]
    if sha256(filename) != record["sha256"]:
        raise ValueError(f"Dataset shard changed: {filename}")
    return torch.load(filename, map_location="cpu", weights_only=True)


def episode_split(meta, seed):
    count = len(meta["episodes"])
    if count < 3:
        raise ValueError("At least 3 independently seeded episodes required for train/validation/test")
    if len({e["episode_seed"] for e in meta["episodes"]}) != count:
        raise ValueError("Episode seeds must be unique before splitting")
    ids = np.random.default_rng(seed).permutation(count).tolist()
    heldout = max(1, count // 5)
    return {"train": ids[2 * heldout :], "validation": ids[:heldout], "test": ids[heldout : 2 * heldout]}


def arrays(path, meta, ids, source):
    shards = [load_shard(path, meta["episodes"][i]) for i in ids]
    return torch.cat([s[source] for s in shards]).float(), torch.cat([s["cores"] for s in shards]).float()


def fit(dataset, output, source, seed, epochs, batch_size, hidden, lr, device):
    validate_device(device)
    output = new_directory(output)
    path, meta = read_dataset(dataset)
    split = episode_split(meta, seed)
    torch.manual_seed(seed)
    x, y = arrays(path, meta, split["train"], source)
    vx, vy = arrays(path, meta, split["validation"], source)
    model = StateDecoder(x.shape[-1], meta["schema"], hidden).to(device)
    model.normalize_from_training(x.to(device), y.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    history, best, best_export = [], float("inf"), None
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for indices in torch.randperm(len(x), generator=generator).split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(x[indices].to(device), y[indices].to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite decoder training loss")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(indices)
        model.eval()
        with torch.no_grad():
            validation = sum(
                model.loss(a.to(device), b.to(device)).item() * len(a)
                for a, b in zip(vx.split(batch_size), vy.split(batch_size))
            ) / len(vx)
        history.append({"epoch": epoch + 1, "train_loss": total / len(x), "validation_loss": validation})
        if validation < best:
            best, best_export = validation, model.export()
        print(history[-1], flush=True)
    if best_export is None:
        raise RuntimeError("No finite validation checkpoint")
    result = {
        "version": 1,
        "model": best_export,
        "entry": meta["entry"],
        "source": source,
        "provenance": provenance(),
        "dataset_sha256": sha256(path / "dataset.json"),
        "split": split,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "history": history,
        "test_used_for_selection": False,
        "target": "qpos/qvel/act, periodic hinges, sign-invariant quaternion loss",
    }
    torch.save(result, output / "decoder.pt")
    write_json(output / "training.json", {k: v for k, v in result.items() if k != "model"})
    return result


def load_probe(dataset, checkpoint, device):
    validate_device(device)
    path, meta = read_dataset(dataset)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["dataset_sha256"] != sha256(path / "dataset.json") or saved["entry"] != meta["entry"]:
        raise ValueError("Decoder belongs to a different dataset/checkpoint")
    model = StateDecoder.from_export(saved["model"], device)
    return path, meta, saved, model


def depth_statistics(rows):
    identifiers = {"depth", "episode", "start", "retained_steps", "at_boundary", "truncated"}
    return {
        str(depth): statistics(
            [{k: v for k, v in row.items() if k not in identifiers} for row in rows if row["depth"] == depth]
        )
        for depth in sorted({r["depth"] for r in rows})
    }


@torch.no_grad()
def evaluate(dataset, checkpoint, output, device, stride, horizon, max_starts):
    path, meta, saved, model = load_probe(dataset, checkpoint, device)
    config = load_config(meta["entry"])
    rows, rollout_rows, failures = [], [], []
    starts = 0
    for episode in saved["split"]["test"]:
        shard = load_shard(path, meta["episodes"][episode])
        inputs = shard[saved["source"]].float()
        predictions = torch.cat([model(x.to(device)).cpu() for x in inputs.split(512)]).numpy()
        for prediction, target in zip(predictions, shard["cores"].numpy()):
            rows.append(state_errors(prediction, target, meta["schema"]))
        a, b = ProbeEnv(config.env, shard["episode_seed"]), ProbeEnv(config.env, shard["episode_seed"])
        try:
            for t in range(0, len(inputs), stride):
                if starts >= max_starts:
                    break
                starts += 1
                try:
                    a.restore(shard["snapshots"][t])
                    ob = b.restore(shard["snapshots"][t], predictions[t])
                    rollout_rows.append({
                        "depth": 0,
                        "episode": episode,
                        "start": t,
                        "observation_rmse": float(
                            np.sqrt(np.mean((observation_vector(ob) - shard["observations"][t].numpy()) ** 2))
                        ),
                    })
                    for depth, action in enumerate(shard["actions"][t : t + horizon].numpy(), 1):
                        errors, done = compare_step(a, b, action)
                        rollout_rows.append({"depth": depth, "episode": episode, "start": t, **errors})
                        if done:
                            break
                except (ValueError, RuntimeError, ArithmeticError) as exc:
                    failures.append({"episode": episode, "start": t, "error": str(exc)})
        finally:
            a.close()
            b.close()
    report = {
        "entry": meta["entry"],
        "kind": "posterior_restore",
        "decoder_sha256": sha256(checkpoint),
        "provenance": provenance(),
        "source": saved["source"],
        "test_episodes": saved["split"]["test"],
        "state_metrics": statistics(rows),
        "restore_attempts": starts,
        "failures": failures,
        "failure_rate": len(failures) / max(starts, 1),
        "horizon": horizon,
        "stride": stride,
        "rollouts": rollout_rows,
        "by_depth": depth_statistics(rollout_rows),
        "accepted": None,
        "note": "No acceptance threshold assumed. Initial contact sensors may differ due to DMC sensor timing.",
    }
    write_json(output, report)
    return report


@torch.no_grad()
def imagine_check(dataset, checkpoint, output, device, stride, horizon, max_starts):
    """Diagnostic only: same imagined actions in true simulator, no new replay samples."""
    path, meta, saved, decoder = load_probe(dataset, checkpoint, device)
    if saved["source"] != "features":
        raise ValueError("imagine-check requires a feature decoder")
    config = load_config(meta["entry"], device)
    env = ProbeEnv(config.env, meta["seed"])
    try:
        agent = load_agent(meta["entry"], env, device)
    finally:
        env.close()
    if not bool(agent.trunc_kappa.initialized):
        raise ValueError("Checkpoint truncation threshold is uninitialized")
    rows, failures, starts = [], [], 0
    torch.manual_seed(saved["seed"])
    for episode in saved["split"]["test"]:
        shard = load_shard(path, meta["episodes"][episode])
        env = ProbeEnv(config.env, shard["episode_seed"])
        restored = ProbeEnv(config.env, shard["episode_seed"])
        try:
            for t in range(0, len(shard["features"]), stride):
                if starts >= max_starts:
                    break
                starts += 1
                try:
                    env.restore(shard["snapshots"][t])
                    start = (shard["stoch"][t : t + 1].to(device), shard["deter"][t : t + 1].to(device))
                    features, actions = agent._imagine(start, horizon + 1)
                    distributions = [head(features) for head in agent.value]
                    values = torch.stack([d.mode() for d in distributions])
                    uncertainty = cal_ens_uncertainty(distributions, values, agent.uncer_metric, beta=agent.ucb_beta)
                    mask = build_truncation_mask(uncertainty, agent.trunc_kappa.value, agent.dynamic_trunc_min_horizon)
                    boundary = int(mask[0].sum().item())
                    decoded = decoder(features[0]).cpu().numpy()
                    for depth in range(horizon + 1):
                        errors = state_errors(decoded[depth], env.core_state(), meta["schema"])
                        # This branch is a diagnostic, not a new training trajectory.
                        # restore(core=...) takes only known time/control/context from
                        # the reference; predicted core and reset-default solver inputs.
                        restored_ok = False
                        try:
                            ob = restored.restore(env.snapshot(), decoded[depth])
                            errors["restored_observation_rmse"] = float(
                                np.sqrt(np.mean((observation_vector(ob) - observation_vector(env.observation())) ** 2))
                            )
                            restored_ok = True
                        except (ValueError, RuntimeError, ArithmeticError) as exc:
                            failures.append({
                                "episode": episode,
                                "start": t,
                                "depth": depth,
                                "stage": "decoded_restore",
                                "error": str(exc),
                            })
                        errors["restore_failed"] = float(not restored_ok)
                        row = {
                            "episode": episode,
                            "start": t,
                            "depth": depth,
                            "retained_steps": boundary,
                            "at_boundary": depth == boundary,
                            "truncated": boundary < horizon,
                            **errors,
                        }
                        if depth < horizon:
                            action = actions[0, depth].cpu().numpy()
                            _, true_reward, done = env.step(action)
                            if restored_ok:
                                try:
                                    _, predicted_reward, predicted_done = restored.step(action)
                                    row.update({
                                        f"next_{k}": v
                                        for k, v in state_errors(
                                            restored.core_state(), env.core_state(), meta["schema"]
                                        ).items()
                                    })
                                    row["next_reward_abs"] = abs(predicted_reward - true_reward)
                                    row["next_done_mismatch"] = float(done != predicted_done)
                                except (ValueError, RuntimeError, ArithmeticError) as exc:
                                    failures.append({
                                        "episode": episode,
                                        "start": t,
                                        "depth": depth,
                                        "stage": "decoded_step",
                                        "error": str(exc),
                                    })
                                    row["restore_failed"] = 1.0
                        rows.append(row)
                        if depth == horizon:
                            break
                        if done:
                            break
                except (ValueError, RuntimeError, ArithmeticError) as exc:
                    failures.append({"episode": episode, "start": t, "error": str(exc)})
        finally:
            env.close()
            restored.close()
    report = {
        "entry": meta["entry"],
        "kind": "imagination_decode",
        "decoder_sha256": sha256(checkpoint),
        "provenance": provenance(),
        "horizon": horizon,
        "attempts": starts,
        "failures": failures,
        "rows": rows,
        "by_depth": depth_statistics(rows),
        "truncation_boundary_metrics": depth_statistics([r for r in rows if r["truncated"] and r["at_boundary"]]),
        "threshold": float(agent.trunc_kappa.value.cpu()),
        "threshold_policy": "fixed saved EMA; not updated",
        "accepted": None,
        "note": "Combined model/decoder error and one-step restoration diagnostics; no replay integration.",
    }
    write_json(output, report)
    return report
