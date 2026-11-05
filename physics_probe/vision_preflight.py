"""Check vision checkpoints, native policy equivalence and one-step restoration before collection."""

import argparse
import gc
import os
from pathlib import Path

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
from .environment import TASKS, ProbeEnv, observation_vector


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "6" or os.environ.get("MUJOCO_EGL_DEVICE_ID") != "6":
        raise ValueError("Preflight uses CUDA and EGL on physical GPU6 only")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = "cuda:0"
    torch.cuda.set_per_process_memory_fraction(2 * 1024**3 / torch.cuda.get_device_properties(device).total_memory)
    root = new_directory(Path(args.output).resolve())
    entries = make_manifest(args.root, root / "manifest.json", TASKS, [0, 100, 200], "raw", 12, "vision")
    results = []
    for entry in entries:
        config = load_config(entry, device)
        env, oracle = ProbeEnv(config.env, 50000, include_image=True), ProbeEnv(config.env, 50000)
        try:
            agent = load_agent(entry, env, device)
            if agent.encoder.mlp_shapes or set(agent.encoder.cnn_shapes) != {"image"}:
                raise ValueError("Policy encoder must use images only")
            state, obs = agent.get_initial_state(1), env.observation(first=True)
            snapshot = env.snapshot()
            torch.manual_seed(50000)
            feature, action, probe_state = posterior(agent, obs, state)
            native_obs = {k: torch.as_tensor(v.copy(), device=device).unsqueeze(0) for k, v in obs.items()}
            torch.manual_seed(50000)
            native_action, native_state = agent.act(native_obs, state)
            torch.testing.assert_close(action, native_action, rtol=0, atol=0)
            for key in probe_state:
                torch.testing.assert_close(probe_state[key], native_state[key], rtol=0, atol=0)
            next_obs, reward, done = env.step(action[0].cpu().numpy())
            oracle.restore(snapshot)
            restored_obs, restored_reward, restored_done = oracle.step(action[0].cpu().numpy())
            core_error = float(np.max(np.abs(env.core_state() - oracle.core_state())))
            proprio = {k: v for k, v in next_obs.items() if k != "image"}
            np.testing.assert_allclose(observation_vector(proprio), observation_vector(restored_obs), rtol=0, atol=1e-6)
            np.testing.assert_allclose(env.core_state(), oracle.core_state(), rtol=0, atol=1e-9)
            if abs(reward - restored_reward) > 1e-9 or done != restored_done:
                raise ValueError("One-step reference transition mismatch")
            pixels = oracle.physics.render(*env.image_size, camera_id=env.camera_id)
            pixel_difference = pixels.astype(np.int16) - next_obs["image"].astype(np.int16)
            pixel_max = int(np.abs(pixel_difference).max())
            if pixel_max > 1:
                raise ValueError(f"Restored image differs by more than one raster gray level: {pixel_max}")
            result = {
                "entry": entry,
                "strict_load": True,
                "native_policy_exact_match": True,
                "proprio_used_by_policy": False,
                "image_shape": list(obs["image"].shape),
                "image_dtype": str(obs["image"].dtype),
                "feature_shape": list(feature.shape),
                "feature_dtype": str(feature.dtype),
                "core_dtype": str(env.core_state().dtype),
                "native_parameter_dtypes": sorted({str(p.dtype) for p in agent.parameters()}),
                "trainable_parameters": sum(p.numel() for p in agent.parameters() if p.requires_grad),
                "restored_core_max_abs": core_error,
                "restored_image_exact_match": pixel_max == 0,
                "restored_image_max_abs_gray": pixel_max,
                "restored_image_changed_fraction": float(np.mean(pixel_difference != 0)),
            }
            results.append(result)
            write_json(root / f"{entry['id']}.json", result)
            print(f"PASS {entry['id']}: strict checkpoint, native actor, one-step state and pixels", flush=True)
            del agent, feature, action, native_action, native_state, probe_state, native_obs, state
        finally:
            env.close()
            oracle.close()
            del env, oracle
            gc.collect()
            torch.cuda.empty_cache()
    write_json(root / "checks.json", {"results": results, "provenance": provenance(), "models": len(results)})


if __name__ == "__main__":
    main()
