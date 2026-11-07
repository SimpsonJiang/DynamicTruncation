"""Validate the CQ/legacy checkpoints and unchanged historical frozen-model inference."""

import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from .common import (
    load_agent,
    load_config,
    make_manifest,
    new_directory,
    posterior,
    provenance,
    write_json,
)
from .environment import ProbeEnv
from .pipeline import load_probe, load_shard
from .stage5 import frozen_truncation_diagnostics

CQ_GROUP = "CQ_wasser_symlog_advpositive_only_f64_H15"


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "6":
        raise ValueError("This campaign uses GPU6 only")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(2 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    root = new_directory(args.output).resolve()
    repo = Path(__file__).resolve().parents[1]
    cq = make_manifest(
        repo / "logdir" / CQ_GROUP,
        root / "cq_manifest.json",
        ["finger_spin", "hopper_hop", "quadruped_run"],
        [1, 42, 100],
        "symlog",
        9,
        "proprio",
    )
    legacy = [
        (
            "VE_wasser_symlog_f64_H15",
            "physics_probe_stage3_symlog_f64_20261105",
            "physics_probe_symlog_f64_20261105_preflight",
        ),
        ("VE_wasser_H15", "physics_probe_stage3_vision_20261105", "physics_probe_vision_20261105"),
    ]
    fixtures = {}
    for group, source_name, snapshot_name in legacy:
        source = repo / "logdir/dyn_recon" / source_name
        snapshot = repo / "logdir/dyn_recon" / snapshot_name / "source_snapshot"
        target = root / f"{group}_historical.pt"
        with (root / f"{group}_historical.log").open("x") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    str(Path(__file__).parent / "inference_fixture.py"),
                    "--snapshot",
                    str(snapshot),
                    "--source",
                    str(source),
                    "--output",
                    str(target),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        fixtures[group] = torch.load(target, map_location="cpu", weights_only=True)
    jobs = [(CQ_GROUP, entry, None) for entry in cq]
    for group, source_name, _ in legacy:
        source = repo / "logdir/dyn_recon" / source_name
        protocol = json.loads((source / "protocol.json").read_text())
        for entry_id in protocol["entries"]:
            base = source / entry_id
            _, meta, _, _ = load_probe(base / "data", base / "features/decoder.pt", "cpu")
            jobs.append((group, meta["entry"], base))
    checks = []
    for group, entry, base in jobs:
        config = load_config(entry, "cuda:0")
        vision = config.env.encoder.cnn_keys == "image"
        env = ProbeEnv(config.env, 50000)
        if vision:
            env.observation_space.spaces["image"] = gym.spaces.Box(
                0, 255, tuple(config.env.size) + (3,), dtype=np.uint8
            )
        try:
            agent = load_agent(entry, env, "cuda:0")
            if base is None:
                if not agent.mixed_value_ensemble or agent.adv_trunc_mode != "positive_only":
                    raise ValueError("CQ group is not the requested mixed positive-only model")
                state = agent.get_initial_state(1)
                obs = env.observation(first=True)
                torch.manual_seed(71000)
                _, action, next_state = posterior(agent, obs, state)
                torch.manual_seed(71000)
                native_obs = {k: torch.as_tensor(v.copy(), device="cuda:0").unsqueeze(0) for k, v in obs.items()}
                native_action, native_state = agent.act(native_obs, state)
                torch.testing.assert_close(action, native_action, rtol=0, atol=0)
                for key in next_state:
                    torch.testing.assert_close(next_state[key], native_state[key], rtol=0, atol=0)
                latent = next_state["stoch"], next_state["deter"]
                torch.manual_seed(71100)
                features, actions = agent._imagine(latent, 16)
            else:
                fixture = fixtures[group][entry["id"]]
                meta = json.loads((base / "data/dataset.json").read_text())
                shard = load_shard(base / "data", meta["episodes"][fixture["episode"]])
                latent = tuple(shard[key][100:101].cuda() for key in ("stoch", "deter"))
                torch.manual_seed(fixture["seed"])
                features, actions = agent._imagine(latent, 16)
                for value, expected in [
                    (features.cpu(), fixture["features"]),
                    (actions.cpu(), fixture["actions"]),
                    (agent.reward(features).mode().cpu(), fixture["reward"]),
                    (agent.cont(features).mean.cpu(), fixture["cont"]),
                    (torch.stack([head(features).mode() for head in agent.value]).cpu(), fixture["critic_modes"]),
                ]:
                    torch.testing.assert_close(value, expected, rtol=0, atol=0)
                torch.testing.assert_close(features[:, 0].cpu(), shard["features"][100:101], rtol=0, atol=0)
            before = {
                k: v.clone() for k, v in agent.state_dict().items() if k.startswith(("adv_scale.", "trunc_kappa."))
            }
            diagnostics = frozen_truncation_diagnostics(agent, features)
            for key, value in before.items():
                torch.testing.assert_close(agent.state_dict()[key], value, rtol=0, atol=0)
            result = {
                "group": group,
                "entry": entry,
                "strict_load": True,
                "historical_inference_exact": base is not None,
                "native_policy_checked": base is None,
                "ema_unchanged": True,
                "actual_critic_count": len(agent.value),
                "distance_space": agent.distance_space,
                "advantage_mode": agent.adv_trunc_mode,
                "feature_dtype": str(features.dtype),
                "trainable_parameters": sum(p.numel() for p in agent.parameters() if p.requires_grad),
                "truncation": diagnostics,
            }
            write_json(root / f"{group}__{entry['id']}.json", result)
            checks.append(result)
            print(f"PASS {group}/{entry['id']}", flush=True)
            del agent, features, actions, latent
        finally:
            env.close()
            gc.collect()
            torch.cuda.empty_cache()
    write_json(root / "checks.json", {"models": len(checks), "checks": checks, "provenance": provenance()})


if __name__ == "__main__":
    main()
