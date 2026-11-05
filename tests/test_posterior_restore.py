"""Functional checks for held-out state restoration and failure accounting."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

HAS_DMC = importlib.util.find_spec("dm_control") is not None and importlib.util.find_spec("mujoco") is not None
if HAS_DMC:
    from physics_probe.environment import ProbeEnv, observation_vector
    from physics_probe.posterior_restore import (
        evaluate_model,
        summarize_cases,
        trace_candidate,
        trace_reference,
    )
    from physics_probe.stage4 import summarize


@unittest.skipUnless(HAS_DMC, "Install the optional DMC dependencies for restoration checks")
class PosteriorRestoreTest(unittest.TestCase):
    def test_recorded_reference_core_control_and_failed_prediction(self):
        config = OmegaConf.create({"task": "dmc_finger_spin", "action_repeat": 2, "time_limit": 12})
        source, reference, candidate = [ProbeEnv(config, 123) for _ in range(3)]
        try:
            snapshots, cores, observations, actions, rewards, done = [], [], [], [], [], []
            rng = np.random.default_rng(14)
            for _ in range(6):
                snapshots.append(source.snapshot())
                cores.append(source.core_state())
                observations.append(observation_vector(source.observation()))
                action = rng.uniform(-0.7, 0.7, source.action_space.shape).astype(np.float32)
                _, reward, terminal = source.step(action)
                actions.append(action)
                rewards.append(reward)
                done.append(terminal)
            shard = {
                "snapshots": snapshots,
                "cores": torch.from_numpy(np.stack(cores)),
                "observations": torch.from_numpy(np.stack(observations)),
                "actions": torch.from_numpy(np.stack(actions)),
                "rewards": torch.tensor(rewards),
                "done": torch.tensor(done),
            }
            records, controls, check = trace_reference(reference, shard, 2, 10)
            self.assertTrue(check["passed"])
            self.assertEqual(len(controls), 4)
            self.assertTrue(records[-1]["done"])
            self.assertEqual(check["checked_states"], 4)  # final post-action state was not stored
            rows, failure = trace_candidate(candidate, snapshots[2], cores[2], records, controls, source.schema())
            self.assertIsNone(failure)
            self.assertEqual(len(rows), 5)
            for row in rows:
                self.assertLess(row["metrics"]["qvel_rmse"], 1e-9)
                self.assertLess(row["metrics"]["reward_abs"], 1e-9)
                self.assertEqual(row["metrics"]["done_mismatch"], 0)
            invalid = cores[2].copy()
            invalid[0] = np.nan
            rows, failure = trace_candidate(candidate, snapshots[2], invalid, records, controls, source.schema())
            self.assertEqual(rows, [])
            self.assertEqual(failure["phase"], "restore")
            shard["cores"][3, 0] += 0.01
            _, _, check = trace_reference(reference, shard, 2, 2)
            self.assertFalse(check["passed"])
        finally:
            for env in (source, reference, candidate):
                env.close()

    def test_failure_and_missing_origin_denominators(self):
        cases = [
            {"source": "features", "failure": None, "completed_depth": 5, "expected_depth": 5},
            {"source": "features", "failure": {"phase": "restore"}, "completed_depth": -1, "expected_depth": 5},
        ]
        rows = [{"source": "features", "depth": 5, "metrics": {"qvel_rmse": 0.2}}]
        summary = summarize_cases(cases, rows, ["features"], 3)["features"]
        self.assertEqual(summary["attempted"], 2)
        self.assertEqual(summary["planned"], 3)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["full_horizon_completed"], 1)
        self.assertEqual(summary["by_depth"]["5"]["available_origins"], 1)
        self.assertEqual(summary["by_depth"]["5"]["planned_origins"], 3)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "example").mkdir()
            (root / "example/report.json").write_text(
                json.dumps({"reference_passed": True, "summary": {"features": summary}})
            )
            summarize(root, ["example"])
            saved = json.loads((root / "summary.json").read_text())
            self.assertIsNone(saved["scientific_acceptance"])
            self.assertEqual(saved["outcomes"][0]["failed"], 1)
            self.assertEqual(saved["rows"][0]["qvel_rmse_mean"], 0.2)
            self.assertTrue((root / "summary.csv").is_file())
            self.assertTrue((root / "REPORT.md").is_file())

    def test_reject_other_checkpoint_groups_before_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "protocol.json").write_text(json.dumps({"checkpoint_group": "VE_wasser_symlog_w50k_f64_H15"}))
            with self.assertRaisesRegex(ValueError, "restricted"):
                evaluate_model(root, "unused", root / "output", [0], 1, ["features"])
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
