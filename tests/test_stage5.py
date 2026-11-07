"""Time alignment, frozen EMA semantics and failure denominators for imagined-depth restores."""

import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

HAS_DMC = importlib.util.find_spec("dm_control") is not None and importlib.util.find_spec("mujoco") is not None
if HAS_DMC:
    from physics_probe.stage5 import (
        frozen_truncation_diagnostics,
        one_step_branch,
        reference_rollout,
        summarize,
    )


class LinearEnv:
    def __init__(self):
        self.physics = SimpleNamespace(data=SimpleNamespace(time=0.0, ncon=0))
        self.finished = False
        self.core = np.zeros(2)
        self.calls = []

    def core_state(self):
        return self.core.copy()

    def snapshot(self):
        return {"core": torch.from_numpy(self.core.copy()), "time": self.physics.data.time}

    def observation(self):
        return {"position": self.core[:1].copy(), "velocity": self.core[1:].copy()}

    def restore(self, snapshot, core=None):
        self.core = snapshot["core"].numpy().copy() if core is None else np.array(core, copy=True)
        if not np.isfinite(self.core).all():
            raise ValueError("Invalid core")
        self.physics.data.time = snapshot["time"]
        return self.observation()

    def step(self, action):
        self.calls.append(float(action[0]))
        self.core[0] += action[0]
        self.core[1] = action[0]
        self.physics.data.time += 1
        return self.observation(), float(self.core[0]), False


@unittest.skipUnless(HAS_DMC, "Install the optional DMC dependencies")
class Stage5Test(unittest.TestCase):
    def test_depth_d_is_after_d_actions_and_successor_uses_action_d(self):
        env, branch = LinearEnv(), LinearEnv()
        actions = np.arange(1, 16, dtype=np.float64).reshape(-1, 1)
        depths = [8, 10, 12, 14]
        records, snapshots = reference_rollout(env, env.snapshot(), actions, depths)
        self.assertEqual(len(records), 16)
        schema = {"nq": 1, "nv": 1, "na": 0, "joint_types": [2], "joint_addresses": [0], "joint_limited": [0]}
        for depth in depths:
            self.assertEqual(records[depth]["core"][0], actions[:depth].sum())
            rows, restored, failure = one_step_branch(
                branch,
                snapshots[depth],
                records[depth]["core"],
                records[depth],
                records[depth + 1],
                actions[depth],
                schema,
            )
            self.assertIsNone(failure)
            self.assertEqual(rows[0]["metrics"]["reward_abs"], 0)  # don't compare incoming reward to restore-time zero
            self.assertEqual(rows[1]["metrics"]["qvel_rmse"], 0)
            self.assertEqual(rows[1]["metrics"]["reward_abs"], 0)
            self.assertEqual(float(restored["core"][0]), actions[: depth + 1].sum())
            self.assertEqual(restored["time"], depth + 1)
        self.assertEqual(branch.calls, [9, 11, 13, 15])

    def test_failures_and_reference_missing_origins_are_not_hidden(self):
        rows = [{"imag_depth": 8, "source": "features", "phase": 1, "metrics": {"qvel_rmse": 0.5}}]
        cases = [
            {"imag_depth": 8, "source": "features", "failure": None},
            {"imag_depth": 8, "source": "features", "failure": {"phase": "restore"}},
        ]
        report = summarize(rows, cases, [8], 3)
        result = next(r for r in report if r["source"] == "features" and r["phase"] == 1)
        self.assertEqual(result["available_origins"], 1)
        self.assertEqual(result["attempted"], 2)
        self.assertEqual(result["planned_origins"], 3)
        self.assertEqual(result["failed"], 1)

    def test_advantage_and_threshold_ema_are_read_without_forward(self):
        value = torch.ones(1, 16, 1)
        scale = SimpleNamespace(value=torch.tensor(2.0), eps=1e-6, initialized=torch.tensor(True))
        threshold = SimpleNamespace(value=torch.tensor(1.0), initialized=torch.tensor(True))
        agent = SimpleNamespace(
            value=[lambda f: SimpleNamespace(mode=lambda: value)] * 2,
            uncer_metric="wasser",
            ucb_beta=0.1,
            distance_space="symlog",
            trunc_kappa=threshold,
            adv_scale=scale,
            mixed_value_ensemble=True,
            adv_trunc_mode="positive_only",
            adv_trunc_max_factor=2.0,
            dynamic_trunc_min_horizon=8,
            horizon=333,
            lamb=0.95,
            _frozen_reward=lambda f: SimpleNamespace(mode=lambda: value),
            _frozen_cont=lambda f: SimpleNamespace(mean=value),
            _lambda_return=lambda *args: torch.ones(1, 15, 1) * 3,
        )
        with patch("physics_probe.stage5.cal_ens_uncertainty", return_value=torch.ones(1, 15, 1)) as uncertainty:
            result = frozen_truncation_diagnostics(agent, torch.zeros(1, 16, 4))
        self.assertEqual(scale.value.item(), 2)
        self.assertEqual(threshold.value.item(), 1)
        self.assertEqual(result["retained_steps"], 15)
        self.assertEqual(result["bootstrap_heads"], [0, 1])
        self.assertEqual(uncertainty.call_args.kwargs["distance_space"], "symlog")


if __name__ == "__main__":
    unittest.main()
