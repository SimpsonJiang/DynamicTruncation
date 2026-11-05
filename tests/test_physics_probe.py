"""CPU checks for the offline probe; integration checks need the optional DMC extra."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
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

    def test_double_training_preserves_sub_float32_labels_and_input_dtype(self):
        model = StateDecoder(4, self.schema, hidden=8)
        x = torch.randn(10, 4, dtype=torch.float32)
        original_x = x.clone()
        y = torch.randn(10, 15, dtype=torch.float64)
        y[:, 0] = 1 + torch.arange(10, dtype=torch.float64) * 1e-9
        y[:, 3:7] = torch.nn.functional.normalize(y[:, 3:7], dim=-1)
        model.normalize_from_training(x, y)
        # The two reductions may sum in different orders; tolerate float64
        # roundoff, still millions of times below the float32 rounding here.
        torch.testing.assert_close(model.target_mean[0], y[:, 0].mean(), rtol=0, atol=1e-15)
        self.assertNotEqual(model.target_mean[0].item(), y[:, 0].float().double().mean().item())
        optimizer = torch.optim.Adam(model.parameters())
        loss = model.loss(x, y)
        self.assertEqual(loss.dtype, torch.float64)
        loss.backward()
        optimizer.step()
        self.assertEqual(model(x).dtype, torch.float64)
        self.assertTrue(all(p.dtype == torch.float64 and p.grad.dtype == torch.float64 for p in model.parameters()))
        for state in optimizer.state.values():
            self.assertEqual(state["exp_avg"].dtype, torch.float64)
            self.assertEqual(state["exp_avg_sq"].dtype, torch.float64)
        self.assertEqual(x.dtype, torch.float32)
        torch.testing.assert_close(x, original_x, rtol=0, atol=0)

    def test_checkpoint_dtype_roundtrip_and_legacy_detection(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                model = StateDecoder(4, self.schema, hidden=8, dtype=dtype)
                if dtype == torch.float64:
                    with torch.no_grad():
                        model.net[0].weight[0, 0] = 1 + 1e-10
                exported = model.export()
                if dtype == torch.float32:
                    exported.pop("dtype")
                    exported.pop("version")  # Old decoder exports did not declare a dtype.
                restored = StateDecoder.from_export(exported)
                self.assertEqual(restored.dtype, dtype)
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
                x = torch.randn(3, 4, dtype=torch.float32)
                torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)
        bad = StateDecoder(4, self.schema).export()
        bad["dtype"] = "float32"
        with self.assertRaises(ValueError):
            StateDecoder.from_export(bad)


HAS_DMC = importlib.util.find_spec("dm_control") is not None and importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(HAS_DMC, "Install the optional dmc extra for real state restoration tests")
class PhysicsRestoreTest(unittest.TestCase):
    def test_manifest_explicit_tasks_seeds_and_distance_space(self):
        from physics_probe.common import make_manifest

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "models"
            for task in ("finger_spin", "hopper_hop", "quadruped_run"):
                for seed in (1, 42, 100):
                    run = root / f"dmc_{task}_proprio_s{seed}_test"
                    (run / ".hydra").mkdir(parents=True)
                    (run / "latest.pt").write_bytes(b"manifest test only")
                    config = OmegaConf.create({
                        "seed": seed,
                        "env": {"task": f"dmc_{task}", "encoder": {"cnn_keys": "$^"}},
                        "model": {
                            "value_ensemble_size": 5,
                            "dynamic_trunc": {"enabled": True},
                            "uncer_metric": "wasser",
                            "imag_horizon": 15,
                            "rep_loss": "dreamer",
                            "distance_space": "symlog",
                        },
                    })
                    OmegaConf.save(config, run / ".hydra/config.yaml")
            tasks = ["finger_spin", "hopper_hop", "quadruped_run"]
            entries = make_manifest(root, Path(temporary) / "manifest.json", tasks, [1, 42, 100], "symlog", 9)
            self.assertEqual(len(entries), 9)
            self.assertEqual(entries[0]["id"], "finger_spin_s1")
            self.assertEqual(entries[-1]["id"], "quadruped_run_s100")
            with self.assertRaises(ValueError):
                make_manifest(root, Path(temporary) / "wrong.json", tasks, [1, 42, 100], "raw", 9)
            with self.assertRaises(ValueError):
                make_manifest(root, Path(temporary) / "duplicate.json", tasks, [1, 1], "symlog")

    def test_heldout_geometry_metrics_separate_rotation_and_winding(self):
        from physics_probe.prediction import metric_vectors

        schema = {"nq": 8, "nv": 7, "na": 0, "joint_types": [0, 3], "joint_addresses": [0, 7], "joint_limited": [0, 0]}
        target = np.zeros((2, 15))
        target[:, 3] = 1
        prediction = target.copy()
        prediction[:, 3] = -1
        prediction[:, 7] = 2 * np.pi
        metrics = metric_vectors(prediction, target, schema)
        np.testing.assert_allclose(metrics["qpos_nonquat_wrapped_rmse"], 0, atol=1e-15)
        np.testing.assert_array_equal(metrics["orientation_radians"], 0)
        self.assertTrue((metrics["qpos_raw_rmse"] > 1).all())
        prediction[:, 0] = 0.1
        metrics = metric_vectors(prediction, target, schema)
        np.testing.assert_allclose(metrics["qpos_nonquat_wrapped_rmse"], 0.05)
        prediction[0, 0] = np.nan
        with self.assertRaises(ValueError):
            metric_vectors(prediction, target, schema)

    def test_dataset_labels_are_not_rounded_before_double_training(self):
        from physics_probe.pipeline import arrays

        x = torch.ones(2, 4, dtype=torch.float32)
        y = torch.tensor([[1 + 1e-10], [1 - 1e-10]], dtype=torch.float64)
        meta = {"episodes": [{}]}
        with patch("physics_probe.pipeline.load_shard", return_value={"features": x, "cores": y}):
            loaded_x, loaded_y = arrays(Path("unused"), meta, [0], "features")
        self.assertEqual(loaded_x.dtype, torch.float64)
        torch.testing.assert_close(loaded_y, y, rtol=0, atol=0)
        with patch("physics_probe.pipeline.load_shard", return_value={"features": x, "cores": y.float()}):
            with self.assertRaises(ValueError):
                arrays(Path("unused"), meta, [0], "features")

    def test_restore_experiment_keeps_reference_and_stops_at_episode_end(self):
        from physics_probe.restore_experiment import run_task

        config = OmegaConf.create({"task": "dmc_finger_spin", "action_repeat": 2, "time_limit": 6})
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "experiment"
            report = run_task(config, [123], [0, 2], [1], 4, output, 1e-9)
            self.assertTrue(report["full_passed"])
            self.assertEqual(report["failures"], [])
            self.assertEqual(len(report["cases"]), 6)
            self.assertTrue(all(c["completed_depth"] == c["expected_depth"] for c in report["cases"]))
            rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 18)
            self.assertTrue(all(r["depth"] + r["start"] <= 3 for r in rows))
            self.assertEqual(report["by_mode"]["core64"]["initial"]["core_max_abs"]["max"], 0)
            self.assertGreater(report["by_mode"]["core32"]["initial"]["core_max_abs"]["max"], 0)
            reference = torch.load(output / "reference_s123_hold1.pt", weights_only=True)
            self.assertEqual(reference["actions"].shape[0], 3)
            self.assertEqual(reference["cores"].shape[0], 4)
            self.assertTrue(bool(reference["done"][-1]))

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
