"""DMC vision-only scheduling, shared features, RNG and scratch-render contracts."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gymnasium as gym
import numpy as np
import test_value_ensemble as fixtures
import torch
from omegaconf import OmegaConf
from PIL import Image
from tensordict import TensorDict

from dreamer import Dreamer
from envs import make_env
from envs.parallel import ParallelEnv
from reconstruction.visualization import ReconstructionVisualizer
from trainer import OnlineTrainer


class ReconstructionVisualizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.ValueEnsembleTest.setUpClass()

    def setUp(self):
        torch.manual_seed(23)
        torch.set_num_threads(1)

    @staticmethod
    def vision_agent():
        config = OmegaConf.create(fixtures.ValueEnsembleTest.config)
        config.dynamic_trunc.enabled = False
        config.encoder.cnn_keys = config.decoder.cnn_keys = "image"
        config.encoder.mlp_keys = config.decoder.mlp_keys = "$^"
        config.encoder.cnn.depth = config.decoder.cnn.depth = 4
        config.imag_horizon = 15
        config.physics_decoder.enabled = True
        config.physics_decoder.hidden = 16
        config.physics_decoder.schema = dict(
            nq=2, nv=2, na=0, joint_types=[3, 2], joint_addresses=[0, 1], joint_limited=[0, 1], task="test"
        )
        obs_space = gym.spaces.Dict({"image": gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8)})
        with contextlib.redirect_stdout(io.StringIO()):
            return Dreamer(config, obs_space, fixtures.ValueEnsembleTest.act_space).to("cpu")

    @staticmethod
    def settings(**changes):
        values = dict(
            enabled=True,
            task="dmc_finger_spin",
            every=100000,
            origins=5,
            depths=[8, 10, 12, 14],
            size=[64, 64],
            seed=23,
        )
        return OmegaConf.create({**values, **changes})

    @staticmethod
    def render_rows(slots, cores, actions, depths, size):
        return [
            dict(
                images=np.full((len(depths), *size, 3), 30 + slot, np.uint8),
                details=[dict(depth=d, failure=None, actions_executed=0) for d in depths],
                source=dict(policy_step=100, time=4.0),
                camera_id=0,
                live_state_unchanged=True,
                live_model_unchanged=True,
            )
            for slot in slots
        ]

    def test_same_features_rng_parameters_and_source_state_preserved(self):
        agent = self.vision_agent()
        state = agent.get_initial_state(2)
        before_state = state.clone()
        parameters = copy.deepcopy(agent.state_dict())
        rng = torch.random.get_rng_state()
        with mock.patch.object(
            agent.physics_head, "forward", wraps=agent.physics_head.forward
        ) as physics, mock.patch.object(
            agent.decoder, "forward", wraps=agent.decoder.forward
        ) as native, mock.patch.object(
            agent._frozen_rssm, "obs_step", side_effect=AssertionError("No extra obs_step")
        ):
            packet = agent.reconstruction_visualization(state, [8, 10, 12, 14], 23)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        torch.testing.assert_close(physics.call_args.args[0].cpu(), packet["features"], rtol=0, atol=0)
        torch.testing.assert_close(
            agent.rssm.get_feat(*native.call_args.args).cpu(), packet["features"], rtol=0, atol=0
        )
        self.assertEqual(packet["predicted_core"].dtype, torch.float64)
        self.assertEqual(packet["native_images"].shape, (2, 4, 64, 64, 3))
        self.assertEqual(packet["native_images"].dtype, torch.uint8)
        for key, value in agent.state_dict().items():
            torch.testing.assert_close(value, parameters[key], rtol=0, atol=0)
        for key in state.keys():
            torch.testing.assert_close(state[key], before_state[key], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in agent.parameters()))
        torch.rand(7)
        repeated = agent.reconstruction_visualization(state, [8, 10, 12, 14], 23)
        for key in packet:
            torch.testing.assert_close(packet[key], repeated[key], rtol=0, atol=0)

    def test_proprio_and_other_suites_do_not_enable_gallery(self):
        agent = fixtures.ValueEnsembleTest().make_agent()
        envs = mock.Mock(physics_enabled=True)
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(ReconstructionVisualizer.create(self.settings(), agent, envs, directory))
            other = self.settings()
            other.task = "metaworld_reach"
            self.assertIsNone(ReconstructionVisualizer.create(other, object(), object(), directory))
            self.assertEqual(list(Path(directory).iterdir()), [])
        envs.render_reconstructions.assert_not_called()

    def test_explicit_vision_request_requires_physics_decoder(self):
        agent = self.vision_agent()
        agent.physics_head = None
        with self.assertRaisesRegex(ValueError, "online physics decoder"):
            ReconstructionVisualizer.create(self.settings(), agent, mock.Mock(physics_enabled=True), "unused")

    def test_default_runs_five_experiments_using_distinct_available_origins(self):
        agent = self.vision_agent()
        envs = mock.Mock(physics_enabled=True)
        envs.render_reconstructions.side_effect = self.render_rows
        settings = self.settings()
        del settings["origins"]
        trans = TensorDict(dict(is_last=torch.zeros(6, 1, dtype=torch.bool)), [6])
        with tempfile.TemporaryDirectory() as directory:
            writer = ReconstructionVisualizer.create(settings, agent, envs, directory)
            self.assertTrue(writer.maybe_write(100000, trans, agent.get_initial_state(6)))
            target = Path(directory) / "reconstruction_visualization" / "step_000100000"
            info = json.loads((target / "manifest.json").read_text())
            self.assertEqual(info["experiments"], 5)
            self.assertEqual(info["source_slots"], [0, 1, 2, 3, 4])
            with Image.open(target / "comparison.png") as sheet:
                self.assertEqual(sheet.size, (576, 880))

    def test_milestones_shared_artifacts_and_native_pixels(self):
        agent = self.vision_agent()
        envs = mock.Mock(physics_enabled=True)
        envs.render_reconstructions.side_effect = self.render_rows
        state = agent.get_initial_state(4)
        trans = TensorDict(
            dict(
                is_last=torch.tensor([[False], [True], [False], [False]]),
                is_recon=torch.tensor([[False], [False], [True], [False]]),
            ),
            [4],
        )
        with tempfile.TemporaryDirectory() as directory:
            writer = ReconstructionVisualizer.create(self.settings(size=[320, 320]), agent, envs, directory)
            self.assertFalse(writer.maybe_write(99999, trans, state))
            envs.render_reconstructions.assert_not_called()
            self.assertTrue(writer.maybe_write(100016, trans, state))
            self.assertFalse(writer.maybe_write(100048, trans, state))
            target = Path(directory) / "reconstruction_visualization" / "step_000100000"
            info = json.loads((target / "manifest.json").read_text())
            self.assertEqual(info["source_slots"], [0, 3, 0, 3, 0])
            self.assertEqual(info["experiments"], 5)
            self.assertEqual(len(info["rows"]), 5)
            self.assertEqual(info["actual_env_step"], 100016)
            self.assertEqual(info["actions_executed"], 0)
            packet = torch.load(target / "samples.pt", weights_only=True)
            self.assertEqual(packet["features"].shape[:2], (5, 4))
            self.assertEqual(packet["predicted_core"].dtype, torch.float64)
            with Image.open(target / "comparison.png") as sheet:
                self.assertEqual(sheet.size, (2624, 2160))
                cell = np.asarray(sheet.crop((328, 163, 648, 483)))
            expected = packet["native_images"][0, 0].numpy().repeat(5, axis=0).repeat(5, axis=1)
            np.testing.assert_array_equal(cell, expected)
            self.assertTrue(writer.maybe_write(200000, trans, state))
            self.assertEqual(envs.render_reconstructions.call_count, 2)
            index = (target.parent / "README.md").read_text()
            self.assertIn("step_000100000/comparison.png", index)
            self.assertIn("step_000200000/comparison.png", index)
            self.assertTrue((target.parent / "index.html").exists())

    def test_missing_origins_defers_and_failed_render_is_explicit(self):
        agent = self.vision_agent()
        envs = mock.Mock(physics_enabled=True)
        trans = TensorDict(
            dict(is_last=torch.zeros(1, 1, dtype=torch.bool), is_recon=torch.ones(1, 1, dtype=torch.bool)), [1]
        )
        state = agent.get_initial_state(1)
        with tempfile.TemporaryDirectory() as directory:
            writer = ReconstructionVisualizer.create(self.settings(), agent, envs, directory)
            self.assertFalse(writer.maybe_write(100000, trans, state))
            self.assertEqual(writer.next_step, 100000)
            trans["is_recon"].zero_()
            rows = self.render_rows([0] * 5, None, None, writer.depths, writer.size)
            rows[0]["details"][0]["failure"] = "ValueError: nonfinite_core"
            envs.render_reconstructions.return_value = rows
            self.assertTrue(writer.maybe_write(100032, trans, state))
            target = Path(directory) / "reconstruction_visualization" / "step_000100000"
            info = json.loads((target / "manifest.json").read_text())
            self.assertIn("nonfinite_core", info["rows"][0]["details"][0]["failure"])
            self.assertEqual(info["source_slots"], [0] * 5)
            self.assertEqual(len(info["rows"]), 5)
            packet = torch.load(target / "samples.pt", weights_only=True)
            self.assertEqual(packet["features"].shape[0], 5)
            self.assertFalse(torch.equal(packet["features"][0], packet["features"][1]))
            with Image.open(target / "comparison.png") as sheet:
                native = np.asarray(sheet.crop((72, 163, 136, 227)))
            np.testing.assert_array_equal(native, packet["native_images"][0, 0].numpy())

    def test_whole_gallery_error_is_recorded_without_repeated_attempts(self):
        agent = self.vision_agent()
        envs = mock.Mock(physics_enabled=True)
        envs.render_reconstructions.side_effect = RuntimeError("renderer unavailable")
        trans = TensorDict(dict(is_last=torch.zeros(1, 1, dtype=torch.bool)), [1])
        with tempfile.TemporaryDirectory() as directory:
            writer = ReconstructionVisualizer.create(self.settings(), agent, envs, directory)
            self.assertFalse(writer.maybe_write(100000, trans, agent.get_initial_state(1)))
            self.assertFalse(writer.maybe_write(100032, trans, agent.get_initial_state(1)))
            error = Path(directory) / "reconstruction_visualization" / "step_000100000" / "error.json"
            self.assertIn("renderer unavailable", error.read_text())
            self.assertEqual(envs.render_reconstructions.call_count, 1)

    def test_trainer_collect_loop_generates_only_interval_crossings(self):
        cfg = OmegaConf.create(
            dict(
                steps=5,
                pretrain=0,
                eval_every=0,
                eval_episode_num=0,
                video_pred_log=False,
                params_hist_log=False,
                batch_length=1,
                batch_size=1,
                train_ratio=1,
                update_log_every=100,
                action_repeat=1,
            )
        )
        settings = self.settings()
        settings.every, settings.origins = 2, 1
        cfg.reconstruction_visualization = settings
        envs = mock.Mock(env_num=1, physics_enabled=True)
        trans = TensorDict(dict(reward=torch.zeros(1, 1), is_last=torch.zeros(1, 1, dtype=torch.bool)), [1])
        envs.step.side_effect = lambda act, done: (trans.clone(), torch.zeros(1, dtype=torch.bool))
        envs.render_reconstructions.side_effect = self.render_rows
        state = TensorDict(
            dict(prev_action=torch.zeros(1, 2), stoch=torch.zeros(1, 1, 2), deter=torch.zeros(1, 4)), [1]
        )
        agent = mock.Mock(
            device=torch.device("cpu"),
            rep_loss="dreamer",
            physics_head=object(),
            imag_horizon=15,
            reconstruction_updates=0,
        )
        agent.encoder.cnn_shapes = agent.decoder.cnn_shapes = {"image": (64, 64, 3)}
        agent.get_initial_state.return_value = state
        agent.act.return_value = state["prev_action"], state
        agent.update.return_value = {"loss": 1.0}
        agent.reconstruction_visualization.side_effect = lambda state, depths, seed: dict(
            features=torch.zeros(1, len(depths), 6),
            actions=torch.zeros(1, max(depths) + 1, 2),
            predicted_core=torch.zeros(1, len(depths), 4, dtype=torch.float64),
            native_images=torch.zeros(1, len(depths), 64, 64, 3, dtype=torch.uint8),
        )
        replay = mock.Mock()
        replay.count.return_value = 0
        with tempfile.TemporaryDirectory() as directory:
            OnlineTrainer(cfg, replay, mock.Mock(), directory, envs, None).begin(agent)
            root = Path(directory) / "reconstruction_visualization"
            self.assertEqual(sorted(p.name for p in root.glob("step_*")), ["step_000000002", "step_000000004"])
            self.assertTrue(all((p / "comparison.png").exists() for p in root.glob("step_*")))

    def test_worker_rpc_dispatch_uses_selected_slots_only(self):
        batch = ParallelEnv.__new__(ParallelEnv)
        batch.envs = [mock.Mock(), mock.Mock(), mock.Mock()]
        for i, env in enumerate(batch.envs):
            env.render_reconstructions.return_value = lambda i=i: {"slot": i}
        result = batch.render_reconstructions([2, 0, 2], np.zeros((3, 1, 4)), np.zeros((3, 2, 1)), [1], [32, 32])
        self.assertEqual(result, [{"slot": 2}, {"slot": 0}, {"slot": 2}])
        self.assertEqual(batch.envs[2].render_reconstructions.call_count, 2)
        batch.envs[1].render_reconstructions.assert_not_called()

    def test_scratch_render_preserves_live_physics_context_rng_and_guard(self):
        cfg = OmegaConf.create(dict(task="dmc_finger_spin", action_repeat=2, time_limit=200, size=[32, 32], seed=23))
        env = make_env(cfg, 0, physics=True)
        self.addCleanup(env.close)
        with mock.patch.object(env.base, "render", return_value=np.zeros((32, 32, 3), np.uint8)):
            env.setup_guard()
            env.reconstruction_step(np.zeros(2), True, None, "off")
            env.reconstruction_step(np.zeros(2), False, None, "off")
        core, vector = env.physics_labels(), env.context_vector()
        rng = copy.deepcopy(env.base._env.task.random.get_state())
        mode, steps = env.guard.mode, env.time_limit._step
        rendered = []

        def render(physics, height, width, camera_id):
            self.assertIsNot(physics.model.ptr, env.physics.model.ptr)
            rendered.append(
                dict(time=physics.data.time, ctrl=physics.data.ctrl.copy(), core=physics.get_state().copy())
            )
            return np.full((height, width, 3), 70, np.uint8)

        cores = np.stack([core, core.copy(), core])
        cores[1, 0] = np.nan
        cores[2, 0] += 0.1
        actions = np.full((15, 2), 0.25, np.float32)
        with mock.patch.object(type(env.physics), "render", autospec=True, side_effect=render), mock.patch.object(
            type(env.physics), "step", side_effect=AssertionError("No action")
        ), mock.patch.object(env.guard, "check", side_effect=AssertionError("No guard filtering")):
            result = env.render_reconstructions(cores, actions, [8, 10, 14], [32, 32])
        self.assertEqual(len(rendered), 2)
        np.testing.assert_array_equal(rendered[1]["core"], cores[2])
        self.assertIn("nonfinite_core", result["details"][1]["failure"])
        self.assertIsNone(result["details"][2]["failure"])
        self.assertAlmostEqual(rendered[0]["time"], vector[0] + 8 * env.repeat * env.base._env.control_timestep())
        low, high = env.base.action_space.low, env.base.action_space.high
        np.testing.assert_allclose(rendered[0]["ctrl"], (0.25 + 1) / 2 * (high - low) + low)
        np.testing.assert_array_equal(env.physics_labels(), core)
        np.testing.assert_array_equal(env.context_vector(), vector)
        for a, b in zip(env.base._env.task.random.get_state(), rng):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(env.guard.mode, mode)
        self.assertEqual(env.time_limit._step, steps)
        self.assertTrue(result["live_state_unchanged"])
