"""Vision observation plumbing without requiring a GPU renderer."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

HAS_DMC = importlib.util.find_spec("dm_control") is not None and importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(HAS_DMC, "Install DMC dependencies")
class VisionProbeTest(unittest.TestCase):
    def test_image_decoder_double_training_and_export(self):
        from physics_probe.decoder import ImageStateDecoder, StateDecoder
        from physics_probe.pipeline import arrays

        torch.manual_seed(8)
        schema = {"nq": 2, "nv": 2, "na": 0, "joint_types": [3, 3], "joint_addresses": [0, 1], "joint_limited": [1, 1]}
        pixels = torch.randint(0, 256, (3, 64, 64, 3), dtype=torch.uint8)
        target = torch.randn(3, 4, dtype=torch.float64)
        target[:, 0] = 1 + torch.arange(3, dtype=torch.float64) * 1e-9
        model = ImageStateDecoder((64, 64, 3), schema, hidden=8)
        model.normalize_from_training(pixels, target)
        optimizer = torch.optim.Adam(model.parameters())
        loss = model.loss(pixels, target)
        loss.backward()
        self.assertEqual(loss.dtype, torch.float64)
        self.assertTrue(
            all(
                p.dtype == torch.float64 and p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
        )
        optimizer.step()
        for state in optimizer.state.values():
            self.assertEqual(state["exp_avg"].dtype, torch.float64)
            self.assertEqual(state["exp_avg_sq"].dtype, torch.float64)
        restored = StateDecoder.from_export(model.export())
        self.assertIsInstance(restored, ImageStateDecoder)
        torch.testing.assert_close(restored(pixels), model(pixels), rtol=0, atol=0)
        with patch("physics_probe.pipeline.load_shard", return_value={"images": pixels, "cores": target}):
            inputs, labels = arrays(Path("unused"), {"episodes": [{}]}, [0], "images")
        self.assertEqual(inputs.dtype, torch.uint8)
        torch.testing.assert_close(labels, target, rtol=0, atol=0)
        with patch("physics_probe.pipeline.load_shard", return_value={"images": pixels.float() / 255, "cores": target}):
            with self.assertRaises(ValueError):
                arrays(Path("unused"), {"episodes": [{}]}, [0], "images")

    def test_image_and_proprio_error_scales_are_separate(self):
        from physics_probe.posterior_restore import step_metrics

        schema = {"nq": 1, "nv": 1, "na": 0, "joint_types": [3], "joint_addresses": [0], "joint_limited": [1]}
        reference = {
            "core": np.zeros(2),
            "obs": np.zeros(1),
            "fields": {"position": np.zeros(1)},
            "reward": 0,
            "done": False,
            "time": 0,
            "contacts": 0,
            "image": np.zeros((2, 2, 3), dtype=np.uint8),
        }
        current = {**reference, "image": np.full((2, 2, 3), 255, dtype=np.uint8)}
        metrics = step_metrics(current, reference, schema)
        self.assertEqual(metrics["image_rmse_01"], 1)
        self.assertEqual(metrics["observation_rmse"], 0)

    def test_vision_manifest_legacy_metric_and_modality_filter(self):
        from physics_probe.common import make_manifest

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "dmc_finger_spin_vision_s0_test"
            (run / ".hydra").mkdir(parents=True)
            (run / "latest.pt").write_bytes(b"fixture")
            cfg = OmegaConf.create(
                {
                    "seed": 0,
                    "env": {"task": "dmc_finger_spin", "encoder": {"cnn_keys": "image", "mlp_keys": "$^"}},
                    "model": {
                        "value_ensemble_size": 5,
                        "dynamic_trunc": {"enabled": True},
                        "VE_metric": "wasser",
                        "imag_horizon": 15,
                        "rep_loss": "dreamer",
                    },
                }
            )
            OmegaConf.save(cfg, run / ".hydra/config.yaml")
            path = root / "manifest.json"
            entries = make_manifest(root, path, ["finger_spin"], [0], "raw", 1, "vision")
            self.assertEqual(len(entries), 1)
            self.assertEqual(json.loads(path.read_text())["modality"], "vision")
            with self.assertRaises(ValueError):
                make_manifest(root, root / "bad.json", ["finger_spin"], [0], "raw", 1, "proprio")

    def test_image_shape_and_preprocessing_before_frozen_encoder(self):
        from dreamer import Dreamer
        from physics_probe.common import posterior

        observed = {}

        def encoder(inputs):
            observed.update(inputs)
            return torch.zeros(1, 4)

        agent = SimpleNamespace(
            device="cpu",
            preprocess=lambda inputs: Dreamer.preprocess(None, inputs),
            encoder=encoder,
            rssm=SimpleNamespace(
                obs_step=lambda *args: (torch.zeros(1, 2), torch.zeros(1, 3), None),
                get_feat=lambda z, h: torch.cat([z, h], -1),
            ),
            actor=lambda feat: SimpleNamespace(rsample=lambda: torch.zeros(1, 2)),
        )
        image = np.full((64, 64, 3), 255, dtype=np.uint8)
        obs = {"image": image, "position": np.zeros(3, dtype=np.float32), "is_first": np.array([True])}
        state = {"stoch": None, "deter": None, "prev_action": None}
        feature, _, _ = posterior(agent, obs, state)
        self.assertEqual(observed["image"].shape, (1, 64, 64, 3))
        self.assertEqual(observed["image"].dtype, torch.float32)
        self.assertTrue(torch.all(observed["image"] == 1))
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(observed["position"].shape, (1, 3))
        self.assertEqual(feature.shape, (1, 5))

    def test_probe_vision_observations_match_repo_adapter(self):
        from envs.dmc import DeepMindControl
        from physics_probe.environment import ProbeEnv

        config = OmegaConf.create({"task": "dmc_finger_spin", "action_repeat": 2, "time_limit": 1000, "size": [64, 64]})
        frame = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
        with patch("dm_control.mujoco.engine.Physics.render", return_value=frame):
            probe, repo = ProbeEnv(config, 123, include_image=True), DeepMindControl("finger_spin", 2, seed=123)
            try:
                repo.reset()
                self.assertEqual(probe.observation_space["image"].shape, (64, 64, 3))
                for action in (np.array([0.2, -0.3]), np.array([-0.1, 0.7])):
                    po, pr, pd = probe.step(action)
                    ro, rr, rd, _ = repo.step(action)
                    for key in po:
                        np.testing.assert_allclose(po[key], np.atleast_1d(ro[key]), atol=1e-6, rtol=1e-6)
                    self.assertAlmostEqual(pr, rr)
                    self.assertEqual(pd, rd)
                    np.testing.assert_allclose(probe.core_state()[:3], repo._env.physics.data.qpos, atol=1e-12)
            finally:
                probe.close()
                repo._env.close()


if __name__ == "__main__":
    unittest.main()
