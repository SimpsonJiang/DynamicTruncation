"""Stage 3 held-out prediction metrics; no physics restoration or rollout."""

import numpy as np
import torch

from .common import provenance, sha256, statistics, write_json
from .pipeline import load_probe, load_shard


def metric_vectors(prediction, target, schema):
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("Expected matching [samples, core_dim] arrays")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite prediction or target")
    nq, nv = schema["nq"], schema["nv"]
    delta = prediction - target
    position = delta[:, :nq].copy()
    position_mask = np.ones(nq, dtype=bool)
    orientations = []
    for kind, adr, limited in zip(schema["joint_types"], schema["joint_addresses"], schema["joint_limited"]):
        if kind == 3 and not limited:
            position[:, adr] = np.arctan2(np.sin(position[:, adr]), np.cos(position[:, adr]))
        if kind in (0, 1):
            start = adr + 3 if kind == 0 else adr
            p, t = prediction[:, start : start + 4], target[:, start : start + 4]
            norm = np.linalg.norm(p, axis=-1) * np.linalg.norm(t, axis=-1)
            angle = 2 * np.arccos(np.clip(np.abs((p * t).sum(-1)) / np.maximum(norm, 1e-12), 0, 1))
            orientations.append(angle)
            position_mask[start : start + 4] = False
    result = {
        "qpos_raw_rmse": np.sqrt(np.mean(delta[:, :nq] ** 2, axis=-1)),
        "qvel_rmse": np.sqrt(np.mean(delta[:, nq : nq + nv] ** 2, axis=-1)),
        "qvel_max_abs": np.max(np.abs(delta[:, nq : nq + nv]), axis=-1),
    }
    if position_mask.any():
        result["qpos_nonquat_wrapped_rmse"] = np.sqrt(np.mean(position[:, position_mask] ** 2, axis=-1))
        result["qpos_nonquat_wrapped_max_abs"] = np.max(np.abs(position[:, position_mask]), axis=-1)
    if orientations:
        result["orientation_radians"] = np.max(np.stack(orientations), axis=0)
    if schema["na"]:
        result["act_rmse"] = np.sqrt(np.mean(delta[:, nq + nv :] ** 2, axis=-1))
    return result


def summarize_vectors(vectors):
    return {
        key: {
            "mean": float(a.mean()),
            "p50": float(np.quantile(a, 0.5)),
            "p95": float(np.quantile(a, 0.95)),
            "max": float(a.max()),
            "n": len(a),
        }
        for key, a in vectors.items()
    }


@torch.no_grad()
def prediction_check(dataset, checkpoint, output, device, batch_size=256):
    path, meta, saved, model = load_probe(dataset, checkpoint, device)
    predictions, targets, episodes = [], [], []
    weighted_loss, count = 0.0, 0
    for episode in saved["split"]["test"]:
        shard = load_shard(path, meta["episodes"][episode])
        if shard["cores"].dtype != torch.float64:
            raise ValueError("Prediction evaluation requires original float64 targets")
        x = shard[saved["source"]].to(dtype=model.dtype)
        y = shard["cores"]
        chunks, total = [], 0.0
        for a, b in zip(x.split(batch_size), y.split(batch_size)):
            a, b = a.to(device), b.to(device)
            chunks.append(model(a).cpu().double())
            total += model.loss(a, b).item() * len(a)
        prediction = torch.cat(chunks).numpy()
        target = y.numpy()
        vectors = metric_vectors(prediction, target, meta["schema"])
        episodes.append({
            "episode_id": episode,
            "episode_seed": shard["episode_seed"],
            "samples": len(x),
            "loss": total / len(x),
            "metrics": summarize_vectors(vectors),
        })
        predictions.append(prediction)
        targets.append(target)
        weighted_loss += total
        count += len(x)
    prediction, target = np.concatenate(predictions), np.concatenate(targets)
    delta = prediction - target
    report = {
        "kind": "heldout_prediction_only",
        "entry": meta["entry"],
        "source": saved["source"],
        "decoder_dtype": str(model.dtype).removeprefix("torch."),
        "samples": count,
        "test_loss": weighted_loss / count,
        "best_epoch": min(saved["history"], key=lambda r: r["validation_loss"])["epoch"],
        "best_validation_loss": min(r["validation_loss"] for r in saved["history"]),
        "state_metrics": summarize_vectors(metric_vectors(prediction, target, meta["schema"])),
        "per_coordinate_raw": {
            "rmse": np.sqrt(np.mean(delta**2, axis=0)).tolist(),
            "mae": np.abs(delta).mean(0).tolist(),
            "abs_p95": np.quantile(np.abs(delta), 0.95, axis=0).tolist(),
            "abs_max": np.abs(delta).max(0).tolist(),
            "target_min": target.min(0).tolist(),
            "target_max": target.max(0).tolist(),
        },
        "schema": meta["schema"],
        "episodes": episodes,
        "episode_mean_metrics": statistics([{k: v["mean"] for k, v in e["metrics"].items()} for e in episodes]),
        "split": saved["split"],
        "checkpoint_sha256": sha256(checkpoint),
        "dataset_sha256": saved["dataset_sha256"],
        "provenance": provenance(),
        "accepted": None,
        "notes": [
            "Test episodes never select the checkpoint.",
            "Per-coordinate raw errors include angle winding and quaternion sign; use geometry-aware metrics too.",
            "Temporal samples within an episode are correlated; quantiles are descriptive, not confidence intervals.",
            "This report does not assess restored simulation or imagined features (stages 4/5).",
        ],
    }
    write_json(output, report)
    return report
