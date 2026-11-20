"""Manual, bounded EGL rendering check with an untrained CPU model; no training."""

import argparse
import json
import os
import time
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from dreamer import Dreamer
from envs import make_env
from envs.parallel import ParallelEnv
from reconstruction.visualization import ReconstructionVisualizer


class LocalRenderWorker:
    def __init__(self, env):
        self.env = env

    def render_reconstructions(self, *args):
        return lambda: self.env.render_reconstructions(*args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(23)
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(version_base=None, config_dir=str(root / "configs")):
        config = compose(
            config_name="configs",
            overrides=[
                "env=dmc_vision",
                "env.task=dmc_finger_spin",
                "env.env_num=1",
                "env.eval_episode_num=0",
                "device=cpu",
                "model.compile=false",
                "model.rep_loss=dreamer",
                "model.deter=32",
                "model.hidden=16",
                "model.units=16",
                "model.discrete=4",
                "model.rssm.stoch=4",
                "model.rssm.blocks=4",
                "model.encoder.cnn.depth=4",
                "model.decoder.cnn.depth=4",
                "model.physics_decoder.enabled=true",
                "model.physics_decoder.hidden=16",
                "trainer.reconstruction_visualization.every=2",
                "seed=23",
            ],
        )
    env = make_env(config.env, 0, physics=True)
    try:
        row = env.online_step(np.zeros(env.action_space.shape), reset=True)
        with open_dict(config.model.physics_decoder):
            config.model.physics_decoder.schema = env.physics_schema()
        agent = Dreamer(config.model, env.observation_space, env.action_space).to("cpu")
        batch = ParallelEnv.__new__(ParallelEnv)
        batch.device, batch.physics_enabled = "cpu", True
        batch.envs = [LocalRenderWorker(env)]
        trans, _ = batch._pack_steps([row])
        _, state = agent.act(trans.clone(), agent.get_initial_state(1))
        config.trainer.reconstruction_visualization.task = "dmc_finger_spin [UNTRAINED RENDER SMOKE]"
        writer = ReconstructionVisualizer.create(config.trainer.reconstruction_visualization, agent, batch, args.output)
        rng = torch.random.get_rng_state()
        before = env.physics_labels().copy()
        started = time.perf_counter()
        with mock.patch.object(
            env, "step", side_effect=AssertionError("No environment action allowed")
        ), mock.patch.object(env.physics, "step", side_effect=AssertionError("No physics step allowed")):
            assert writer.maybe_write(2, trans, state), "Gallery generation failed"
        elapsed = time.perf_counter() - started
        assert torch.equal(rng, torch.random.get_rng_state())
        np.testing.assert_array_equal(env.physics_labels(), before)
        folder = args.output / "reconstruction_visualization" / "step_000000002"
        metadata = json.loads((folder / "manifest.json").read_text())
        assert all(cell["failure"] is None for row in metadata["rows"] for cell in row["details"]), metadata
        assert len(metadata["rows"]) == 5, "Expected exactly five experiments/rows"
        report = dict(
            fixture="untrained CPU model; scheduler Step 2 is simulated, no training was run",
            training_updates=0,
            actions_executed=0,
            experiments=len(metadata["rows"]),
            paired_cells=len(metadata["rows"]) * len(metadata["depths"]),
            egl_device=os.environ.get("MUJOCO_EGL_DEVICE_ID"),
            elapsed_seconds=elapsed,
            image=str(folder / "comparison.png"),
            rng_preserved=True,
            live_state_unchanged=True,
        )
        (args.output / "smoke.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        env.close()
        env.physics.free()


if __name__ == "__main__":
    main()
