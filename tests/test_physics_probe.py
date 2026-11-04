"""CPU checks for the offline probe; integration checks need the optional DMC extra."""

import importlib.util
import unittest
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from physics_probe.decoder import StateCodec, StateDecoder


class DecoderTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.schema = {
            "nq": 8,
            "nv": 7,
            "na": 0,
            "joint_types": [0, 3],
            "joint_addresses": [0, 7],
            "joint_limited": [0, 0],
            "task": "synthetic",
        }

    def test_periodic_angle_and_quaternion_sign(self):
        codec = StateCodec(self.schema)
        state = torch.zeros(3, 15)
        state[:, 3] = 1
        state[:, 7] = torch.tensor([0.3, 0.3 + 2 * np.pi, 0.3 - 2 * np.pi])
        encoded = codec.encode(state)
        torch.testing.assert_close(encoded[0], encoded[1])
        torch.testing.assert_close(encoded[0], encoded[2])
        decoded = codec.decode(encoded)
        torch.testing.assert_close(decoded[:, 7], torch.full((3,), 0.3))
        model = StateDecoder(4, self.schema, hidden=8)
        x = torch.randn(3, 4)
        model.normalize_from_training(x, state)
        negated = state.clone()
        negated[:, 3:7] *= -1
        torch.testing.assert_close(model.loss(x, state), model.loss(x, negated))

    def test_train_only_normalization_and_roundtrip(self):
        model = StateDecoder(4, self.schema, hidden=8)
        x, y = torch.randn(10, 4), torch.randn(10, 15)
        y[:, 3:7] = torch.nn.functional.normalize(y[:, 3:7], dim=-1)
        model.normalize_from_training(x, y)
        expected = model.input_mean.clone()
        loss = model.loss(x + 1000, y)
        loss.backward()
        torch.testing.assert_close(model.input_mean, expected)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        restored = StateDecoder.from_export(model.export())
        torch.testing.assert_close(model(x), restored(x))


HAS_DMC = importlib.util.find_spec("dm_control") is not None and importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(HAS_DMC, "Install the optional dmc extra for real state restoration tests")
class PhysicsRestoreTest(unittest.TestCase):
    def test_core_restore_does_not_borrow_target_core_or_solver_cache(self):
        from physics_probe.environment import ProbeEnv, observation_vector

        config = OmegaConf.create({"task": "dmc_finger_spin", "action_repeat": 2, "time_limit": 1000})
        env = ProbeEnv(config, 9)
        try:
            for _ in range(4):
                env.step(np.array([0.2, -0.1]))
            first = env.snapshot()
            predicted = first["core"].numpy().copy()
            predicted[0] += 0.05
            env.physics.data.qpos[:] += 0.1
            env.physics.data.qvel[:] += 0.5
            env.physics.data.qacc_warmstart[:] = 123
            env.physics.data.sensordata[:] = 456
            second = env.snapshot()
            env.restore(first, predicted)
            state = env.integration_state()
            observation = observation_vector(env.observation())
            env.restore(second, predicted)
            np.testing.assert_array_equal(env.integration_state(), state)
            np.testing.assert_array_equal(observation_vector(env.observation()), observation)
        finally:
            env.close()

    def test_device_restriction_and_episode_split(self):
        from physics_probe.common import validate_device
        from physics_probe.pipeline import episode_split

        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "6"}):
            validate_device("cuda:0")
            with self.assertRaises(ValueError):
                validate_device("cuda:6")
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "7,6"}):
            validate_device("cuda:1")
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0,6"}):
            with self.assertRaises(ValueError):
                validate_device("cuda:1")
        meta = {"episodes": [{"episode_seed": i} for i in range(10)]}
        split = episode_split(meta, 3)
        self.assertEqual(split, episode_split(meta, 3))
        self.assertEqual(sorted(sum(split.values(), [])), list(range(10)))
        meta["episodes"][1]["episode_seed"] = 0
        with self.assertRaises(ValueError):
            episode_split(meta, 3)

    def test_all_four_tasks_roundtrip_and_core_restore(self):
        from physics_probe.environment import TASKS, ProbeEnv, observation_vector

        for task in TASKS:
            with self.subTest(task=task):
                config = OmegaConf.create({"task": f"dmc_{task}", "action_repeat": 2, "time_limit": 1000})
                a, b = ProbeEnv(config, 123), ProbeEnv(config, 456)
                try:
                    rng = np.random.default_rng(123)
                    for _ in range(12):
                        a.step(rng.uniform(-0.5, 0.5, a.action_space.shape))
                    saved = a.snapshot()
                    initial = observation_vector(a.observation())
                    # Snapshot owns its arrays, regardless of later stepping.
                    frozen_core = saved["core"].clone()
                    b.restore(saved)
                    np.testing.assert_array_equal(initial, observation_vector(b.observation()))
                    for _ in range(5):
                        action = rng.uniform(-0.5, 0.5, a.action_space.shape)
                        oa, ra, da = a.step(action)
                        ob, rb, db = b.step(action)
                        np.testing.assert_allclose(a.integration_state(), b.integration_state(), atol=1e-9, rtol=0)
                        np.testing.assert_allclose(observation_vector(oa), observation_vector(ob), atol=1e-6, rtol=0)
                        self.assertAlmostEqual(ra, rb, places=9)
                        self.assertEqual(da, db)
                    torch.testing.assert_close(saved["core"], frozen_core)
                    b.restore(saved, saved["core"].numpy())
                    np.testing.assert_allclose(b.core_state(), saved["core"].numpy(), atol=1e-12)
                    self.assertEqual(b.steps, saved["steps"])
                    self.assertEqual(b.env._step_count, saved["inner_steps"])
                    self.assertNotIn("core", b.observation_space.spaces)
                    self.assertNotIn("state", b.observation_space.spaces)
                finally:
                    a.close()
                    b.close()

    def test_explicit_episode_end_and_context_rejection(self):
        from physics_probe.environment import ProbeEnv

        config = OmegaConf.create({"task": "dmc_finger_spin", "action_repeat": 2, "time_limit": 4})
        env = ProbeEnv(config, 4)
        try:
            snap = env.snapshot()
            env.step(np.zeros(env.action_space.shape))
            _, _, done = env.step(np.zeros(env.action_space.shape))
            self.assertTrue(done)
            with self.assertRaises(RuntimeError):
                env.step(np.zeros(env.action_space.shape))
            env.restore(snap)
            self.assertFalse(env.finished)
            bad = dict(snap, model_hash="wrong")
            with self.assertRaises(ValueError):
                env.restore(bad)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
