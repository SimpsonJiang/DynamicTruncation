"""Run under a selected historical source snapshot to verify frozen inference equivalence."""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.snapshot).resolve()))
    import gc

    import gymnasium as gym
    import numpy as np
    import torch

    from physics_probe.common import load_agent, load_config
    from physics_probe.environment import ProbeEnv
    from physics_probe.pipeline import load_shard

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(2 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    source = Path(args.source)
    protocol = json.loads((source / "protocol.json").read_text())
    result = {}
    for entry in protocol["entries"]:
        base = source / entry
        meta = json.loads((base / "data/dataset.json").read_text())
        saved = torch.load(base / "features/decoder.pt", map_location="cpu", weights_only=True)
        episode = saved["split"]["test"][0]
        shard = load_shard(base / "data", meta["episodes"][episode])
        config = load_config(meta["entry"], "cuda:0")
        env = ProbeEnv(config.env, shard["episode_seed"])
        if config.env.encoder.cnn_keys == "image":
            env.observation_space.spaces["image"] = gym.spaces.Box(
                0, 255, tuple(config.env.size) + (3,), dtype=np.uint8
            )
        try:
            agent = load_agent(meta["entry"], env, "cuda:0")
            start = tuple(shard[key][100:101].cuda() for key in ("stoch", "deter"))
            torch.manual_seed(91000 + meta["entry"]["model_seed"])
            features, actions = agent._imagine(start, 16)
            result[entry] = {
                "features": features.cpu(),
                "actions": actions.cpu(),
                "episode": episode,
                "start": 100,
                "seed": 91000 + meta["entry"]["model_seed"],
                "reward": agent.reward(features).mode().cpu(),
                "cont": agent.cont(features).mean.cpu(),
                "critic_modes": torch.stack([head(features).mode() for head in agent.value]).cpu(),
            }
            print(f"Fixture {entry}", flush=True)
            del agent, features, actions, start, saved, shard
        finally:
            env.close()
            gc.collect()
            torch.cuda.empty_cache()
    torch.save(result, args.output)


if __name__ == "__main__":
    main()
