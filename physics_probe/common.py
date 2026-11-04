"""Manifest, checkpoint identity, and frozen posterior inference."""

import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from dreamer import Dreamer

from .environment import TASKS


def validate_device(device):
    """The user reserves physical GPUs 6/7 for this experiment; never select others."""
    parsed = torch.device(device)
    if parsed.type == "cpu":
        return
    if parsed.type != "cuda":
        raise ValueError("Probe supports cpu or CUDA on physical GPU 6/7")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if not visible or any(v not in ("6", "7") for v in visible) or len(set(visible)) != len(visible):
        raise ValueError("Set CUDA_VISIBLE_DEVICES=6 or 7 (or 6,7); only physical GPUs 6/7 are authorized")
    index = parsed.index if parsed.index is not None else 0
    if index >= len(visible):
        raise ValueError(f"{device} is outside CUDA_VISIBLE_DEVICES={','.join(visible)}; use logical cuda:0")


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def new_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def make_manifest(root, output):
    root = Path(root)
    entries = []
    for task in TASKS:
        for seed in (200, 300):
            matches = []
            for path in sorted(root.glob(f"dmc_{task}_proprio_s{seed}_*/.hydra/config.yaml")):
                cfg = OmegaConf.load(path)
                if (
                    cfg.env.task == f"dmc_{task}"
                    and cfg.seed == seed
                    and cfg.model.value_ensemble_size == 5
                    and cfg.model.dynamic_trunc.enabled
                    and cfg.model.uncer_metric == "wasser"
                    and cfg.model.imag_horizon == 15
                    and cfg.model.rep_loss == "dreamer"
                    and cfg.env.encoder.cnn_keys == "$^"
                ):
                    run = path.parent.parent
                    if (run / "latest.pt").is_file():
                        matches.append(run.resolve())
            if len(matches) != 1:
                raise ValueError(f"Expected one completed run for {task} seed {seed}, found: {matches}")
            run = matches[0]
            entries.append({
                "id": f"{task}_s{seed}",
                "task": task,
                "model_seed": seed,
                "run": str(run),
                "checkpoint_sha256": sha256(run / "latest.pt"),
                "config_sha256": sha256(run / ".hydra/config.yaml"),
            })
    write_json(output, {"version": 1, "group": root.name, "entries": entries})
    return entries


def select_entry(manifest, entry_id):
    manifest = json.loads(Path(manifest).read_text())
    matches = [e for e in manifest["entries"] if e["id"] == entry_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicate entry: {entry_id}")
    entry = matches[0]
    verify_entry(entry)
    return entry


def verify_entry(entry):
    run = Path(entry["run"])
    if sha256(run / "latest.pt") != entry["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed since manifest was created")
    if sha256(run / ".hydra/config.yaml") != entry["config_sha256"]:
        raise ValueError("Saved config changed since manifest was created")


def load_config(entry, device="cpu"):
    if sha256(Path(entry["run"]) / ".hydra/config.yaml") != entry["config_sha256"]:
        raise ValueError("Saved config changed since manifest was created")
    config = OmegaConf.load(Path(entry["run"]) / ".hydra/config.yaml")
    # Resolve interpolation after setting every device consumed by the constructor.
    config.device = device
    config.model.device = device
    config.model.rssm.device = device
    config.model.compile = False
    for name in ("encoder", "decoder"):
        config.model[name].mlp.device = device
    for name in ("actor", "critic", "reward", "cont"):
        config.model[name].device = device
    OmegaConf.resolve(config)
    return config


def load_agent(entry, env, device):
    validate_device(device)
    verify_entry(entry)
    config = load_config(entry, device)
    agent = Dreamer(config.model, env.observation_space, env.action_space).to(device)
    checkpoint = torch.load(Path(entry["run"]) / "latest.pt", map_location="cpu", weights_only=True)
    agent.load_state_dict(checkpoint["agent_state_dict"], strict=True)
    agent.requires_grad_(False)
    agent.eval()
    return agent


def provenance():
    root = Path(__file__).resolve().parent.parent
    files = [
        *sorted((root / "physics_probe").glob("*.py")),
        *(root / name for name in ("dreamer.py", "rssm.py", "networks.py", "distributions.py", "truncation.py")),
    ]
    return {
        "versions": {name: version(name) for name in ("torch", "dm_control", "mujoco", "numpy")},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "source_sha256": {str(p.relative_to(root)): sha256(p) for p in files},
    }


@torch.no_grad()
def posterior(agent, obs, state):
    inputs = {k: torch.as_tensor(np.asarray(v).copy(), device=agent.device).reshape(1, -1) for k, v in obs.items()}
    embed = agent.encoder(inputs)
    z, h, _ = agent.rssm.obs_step(state["stoch"], state["deter"], state["prev_action"], embed, inputs["is_first"])
    feature = agent.rssm.get_feat(z, h)
    action = agent.actor(feature).rsample()
    return feature, action, {"stoch": z, "deter": h, "prev_action": action}


def statistics(rows):
    keys = sorted({k for row in rows for k in row})
    result = {}
    for key in keys:
        values = np.asarray([r[key] for r in rows if key in r], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite metric: {key}")
        result[key] = {
            "mean": float(values.mean()),
            "p95": float(np.quantile(values, 0.95)),
            "max": float(values.max()),
            "n": len(values),
        }
    return result
