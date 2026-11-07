"""Decode stored imagined features with the checkpoint's native observation decoder.

Run as a standalone file to load the exact implementation archived by stage 5.
No imagined trajectory is regenerated and no environment action is executed.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage5", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    stage5, snapshot, output = [Path(p).resolve() for p in (args.stage5, args.snapshot, args.output)]
    protocol = json.loads((stage5 / "protocol.json").read_text())
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "6":
        raise ValueError("Use GPU6 only")
    if output.exists():
        raise FileExistsError(output)
    if "physics_probe" in sys.modules:
        raise RuntimeError("Run this script directly, not with -m")
    for name, expected in protocol["source_sha256"].items():
        if hashlib.sha256((snapshot / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Archived source changed: {name}")
    sys.path.insert(0, str(snapshot))
    import gc
    from types import SimpleNamespace
    from unittest.mock import patch

    import gymnasium as gym
    import numpy as np
    import torch

    from physics_probe.common import load_agent, load_config, sha256, write_json
    from physics_probe.environment import ProbeEnv

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(2 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    output.mkdir(parents=True)
    all_outputs, audit = {}, []
    for job in protocol["jobs"]:
        directory = stage5 / job["key"]
        report = json.loads((directory / "report.json").read_text())
        example = torch.load(directory / "example.pt", map_location="cpu", weights_only=True)
        traces = torch.load(directory / "traces.pt", map_location="cpu", weights_only=True)
        matched = [t for t in traces if t["episode_id"] == example["episode_id"] and t["start"] == example["start"]]
        if len(matched) != 1:
            raise ValueError("Could not identify the exact saved imagined trajectory")
        trace = matched[0]
        depths = report["imagined_depths"]
        if depths != [8, 10, 12, 14]:
            raise ValueError("Expected the four requested imagined depths")
        features = trace["features"][depths].unsqueeze(0).cuda()
        config = load_config(report["entry"], "cuda:0")
        vision = report["policy_modality"] == "vision"
        if vision:
            env = SimpleNamespace(
                observation_space=gym.spaces.Dict(
                    {"image": gym.spaces.Box(0, 255, tuple(config.env.size) + (3,), dtype=np.uint8)}
                ),
                action_space=gym.spaces.Box(-1, 1, (trace["actions"].shape[-1],), dtype=np.float32),
            )
        else:
            # Shapes only: this reset observation is never fed into the decoder.
            env = ProbeEnv(config.env, example["episode_seed"])
        try:
            agent = load_agent(report["entry"], env, "cuda:0")
            with torch.no_grad(), patch.object(
                ProbeEnv, "step", side_effect=RuntimeError("No environment steps allowed")
            ), patch.object(agent, "_imagine", side_effect=RuntimeError("Use saved features only")):
                flat = agent.rssm.flat_stoch
                stoch = features[..., :flat].reshape(1, len(depths), agent.rssm._stoch, agent.rssm._discrete)
                deter = features[..., flat:]
                torch.testing.assert_close(agent.rssm.get_feat(stoch, deter), features, rtol=0, atol=0)
                distributions = agent.decoder(stoch, deter)
                decoded = {key: dist.mode()[0].detach().cpu() for key, dist in distributions.items()}
                if vision and (set(decoded) != {"image"} or tuple(decoded["image"].shape) != (4, 64, 64, 3)):
                    raise ValueError("Unexpected native image decoder output")
                if not all(torch.isfinite(value).all() for value in decoded.values()):
                    raise ValueError("Native decoder produced nonfinite observations")
            record = {
                "job": job,
                "entry": report["entry"],
                "policy_modality": report["policy_modality"],
                "episode_id": example["episode_id"],
                "episode_seed": example["episode_seed"],
                "start": example["start"],
                "depths": depths,
                "features": features[0].cpu(),
                "observations": decoded,
                "actions": trace["actions"],
                "feature_trace_sha256": sha256(directory / "traces.pt"),
                "world_model_sha256": report["entry"]["checkpoint_sha256"],
                "post_decode_actions": 0,
            }
            all_outputs[job["key"]] = record
            audit.append(
                {
                    "key": job["key"],
                    "group": job["group"],
                    "modality": report["policy_modality"],
                    "episode_id": example["episode_id"],
                    "start": example["start"],
                    "depths": depths,
                    "shapes": {key: list(value.shape) for key, value in decoded.items()},
                    "dtypes": {key: str(value.dtype) for key, value in decoded.items()},
                    "trace_sha256": record["feature_trace_sha256"],
                    "native_checkpoint_sha256": record["world_model_sha256"],
                    "post_decode_actions": 0,
                    "native_decoder": True,
                    "image_min": float(decoded["image"].min()) if vision else None,
                    "image_max": float(decoded["image"].max()) if vision else None,
                }
            )
            print(f"Native decoded {job['key']}: imagined depths {depths}; no actions", flush=True)
            del agent, distributions, decoded, features, stoch, deter, traces, trace, matched
        finally:
            if not vision:
                env.close()
            gc.collect()
            torch.cuda.empty_cache()
    torch.save(all_outputs, output / "native_observations.pt")
    write_json(
        output / "native_manifest.json",
        {
            "kind": "native_observation_decoder_at_imagined_depth",
            "stage5": str(stage5),
            "snapshot": str(snapshot),
            "models": audit,
            "environment_steps": 0,
            "imagination_regenerated": False,
            "script_sha256": sha256(__file__),
            "data_sha256": sha256(output / "native_observations.pt"),
        },
    )


if __name__ == "__main__":
    main()
