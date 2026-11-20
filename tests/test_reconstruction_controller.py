"""CPU controller cooperation tests with real MuJoCo restoration and TorchRL replay.

The decoder returns a known source core as an oracle fixture; these tests check
scheduling and transaction behavior, not learned reconstruction accuracy.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from buffer import Buffer
from envs import make_env
from envs.parallel import ParallelEnv
from reconstruction.controller import Candidate, ReconstructionController
from reconstruction.guard import UnsafeState
from trainer import OnlineTrainer


class _LocalWorker:
    """Run the actual environment RPC method synchronously without a child process."""

    def __init__(self, env):
        self.env = env

    def setup_guard(self):
        return self.env.setup_guard

    def reconstruction_step(self, *args):
        return lambda: self.env.reconstruction_step(*args)


class ReconstructionControllerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(1)
        self.physical_envs = []
        cfg = OmegaConf.create(dict(task="dmc_finger_spin", action_repeat=2, time_limit=200, size=[16, 16], seed=19))
        for slot in range(2):
            env = make_env(cfg, slot, physics=True)
            self.physical_envs.append(env)
            self.addCleanup(env.close)
            render = mock.patch.object(env.base, "render", return_value=np.zeros((16, 16, 3), np.uint8))
            render.start()
            self.addCleanup(render.stop)
        self.envs = ParallelEnv.__new__(ParallelEnv)
        self.envs.envs = [_LocalWorker(env) for env in self.physical_envs]
        self.envs.device, self.envs.physics_enabled = "cpu", True
        self.replay = Buffer(
            OmegaConf.create(dict(device="cpu", storage_device="cpu", batch_size=1, batch_length=1, max_size=64))
        )
        self.agent = mock.Mock(reconstruction_updates=0)
        self.agent.act_with_seeds.side_effect = self.act
        self.agent.decode_physics.side_effect = lambda feature: torch.from_numpy(self.source_core.copy())
        config = OmegaConf.create(
            dict(
                ratio=0.1,
                max_active=2,
                branch_steps=3,
                max_consecutive=2,
                credit_cap_steps=6,
                warmup_env_steps=0,
                calibration_env_steps=2,
                validation_env_steps=2,
                max_false_reject=0.01,
                seed=19,
                candidates=dict(
                    capacity=8,
                    top_k=8,
                    ttl_updates=128,
                    refresh_every_updates=8,
                    max_insert=8,
                    deter_cosine_threshold=0.98,
                    action_rms_threshold=0.1,
                ),
            )
        )
        self.controller = ReconstructionController(config, self.envs, self.replay, self.agent, 2)
        self.addCleanup(self.controller.close)
        # Isolate lifecycle tests from preflight policy; actual guard checks still
        # run during root restoration and each of the first five policy steps.
        guard = self.controller.guard
        guard.thresholds = [float("inf")] * len(guard.profile["pairs"])
        guard.calibration_count = guard.validation_count = 2
        self.state = TensorDict(
            dict(stoch=torch.zeros(2, 1, 2), deter=torch.zeros(2, 4), prev_action=torch.zeros(2, 2)), [2]
        )
        self.tick = 0
        initial, _ = self.step(torch.ones(2, dtype=torch.bool))
        self.source_core = initial["physics", "core"][0].numpy().copy()
        self.source_vector = initial["physics", "context"][0].numpy().copy()
        self.source_origin = tuple(initial["origin"][0].tolist())
        self.source_context = self.controller.contexts[self.source_origin[0]][1]

    @staticmethod
    def act(trans, state, features, actions):
        state = state.clone()
        roots = trans["is_branch"].reshape(-1)
        if features is not None:
            seed = torch.from_numpy(features).float()
            state["stoch"][roots] = seed[:, :2].reshape(-1, 1, 2)
            state["deter"][roots] = seed[:, 2:]
            state["prev_action"][roots] = torch.from_numpy(actions).float()
        return state["prev_action"], state

    def step(self, done):
        self.tick += 1
        trans, done = self.controller.step(self.state["prev_action"], done, self.tick * 4)
        action, self.state = self.controller.act(trans, self.state)
        trans.update(
            dict(
                action=action * ~done[:, None],
                stoch=self.state["stoch"],
                deter=self.state["deter"],
                episode=torch.arange(2, dtype=torch.int32),
            )
        )
        self.controller.record(trans, done)
        return trans, done

    def queue(self, count=1):
        self.controller.pool.items = [
            Candidate(
                float(i + 1),
                0,
                self.source_origin,
                np.arange(6, dtype=np.float32) + i,
                np.zeros((1, 2), np.float32),
                self.source_context,
                self.source_vector.copy(),
                0,
            )
            for i in range(count)
        ]
        self.controller.budget.credit = count * 3

    def assert_released(self):
        self.assertFalse(self.controller.budget.active)
        self.assertFalse(self.controller.active_candidates)
        self.assertFalse(self.replay.has_pending())

    def test_restore_rejection_refunds_full_budget_and_stores_normal_reset(self):
        self.queue()
        self.agent.decode_physics.side_effect = lambda feature: torch.full_like(
            torch.from_numpy(self.source_core), torch.nan
        )
        trans, done = self.step(torch.ones(2, dtype=torch.bool))
        self.assert_released()
        self.assertEqual(self.controller.budget.credit, 3)
        self.assertEqual(self.controller.stats["rejected_restores"], 1)
        self.assertEqual(self.controller.stats["started_branches"], 0)
        self.assertTrue(trans["is_first"].all())
        self.assertFalse(trans["is_recon"].any())
        self.assertTrue(self.replay._buffer.storage[:]["_ready"].all())
        (event,) = self.controller.take_rejections()
        self.assertEqual(event["reason"], "invalid_core")
        self.assertEqual(event["phase"], "restore")
        self.assertEqual(event["branch_step"], 0)
        self.assertEqual(event["origin"], list(self.source_origin))

    def test_mid_branch_failure_discards_all_staged_rows_and_refunds_only_unused_steps(self):
        self.queue()
        root, done = self.step(torch.ones(2, dtype=torch.bool))
        self.assertTrue(root["is_branch"][0])
        self.assertTrue(self.replay.has_pending(0))
        _, done = self.step(done)
        details = dict(geom_pair=[0, 1], depth=0.3, threshold=0.01)
        with mock.patch.object(self.physical_envs[0].guard, "check", side_effect=UnsafeState("penetration", **details)):
            trans, _ = self.step(done)
        self.assert_released()
        self.assertEqual(self.controller.stats["discarded_branches"], 1)
        self.assertEqual(self.controller.budget.stats["reconstruction_steps"], 2)
        self.assertEqual(self.controller.budget.active, {})
        # Slot 1 earned two ordinary steps of credit after the reservation.
        self.assertAlmostEqual(self.controller.budget.credit, 1 + 2 / 9)
        stored = self.replay._buffer.storage[:]
        staged = stored["is_recon"].squeeze(-1)
        self.assertTrue(staged.any())
        self.assertFalse(stored["_ready"][staged].any())
        self.assertTrue(trans["is_first"][0])
        (event,) = self.controller.take_rejections()
        self.assertEqual(event["branch_step"], 2)
        for key, value in details.items():
            self.assertEqual(event[key], value)

    def test_same_batch_commits_early_termination_and_discards_other_slot(self):
        self.queue(2)
        _, done = self.step(torch.ones(2, dtype=torch.bool))
        original_step = self.physical_envs[0].step

        def terminate(action):
            obs, reward, _, info = original_step(action)
            obs["is_last"], obs["is_terminal"] = True, True
            return obs, reward, True, info

        with (
            mock.patch.object(self.physical_envs[0], "step", side_effect=terminate),
            mock.patch.object(self.physical_envs[1].guard, "check", side_effect=UnsafeState("penetration")),
        ):
            trans, _ = self.step(done)
        self.assert_released()
        self.assertEqual(self.controller.stats["committed_branches"], 1)
        self.assertEqual(self.controller.stats["discarded_branches"], 1)
        self.assertEqual(self.controller.budget.credit, 4)
        stored = self.replay._buffer.storage[:]
        self.assertTrue(stored["_ready"][:, 0].all())
        self.assertFalse(stored["_ready"][1, 1])
        self.assertTrue(trans["is_terminal"][0])
        self.assertFalse(trans["is_recon"][1])

    def test_normal_completion_commits_and_keeps_root_seed_immutable(self):
        self.queue()
        root, done = self.step(torch.ones(2, dtype=torch.bool))
        seed = root["deter"][0].clone()
        for _ in range(3):
            _, done = self.step(done)
        self.assert_released()
        self.assertEqual(self.controller.stats["committed_branches"], 1)
        stored = self.replay._buffer.storage[:]
        self.assertTrue(stored["_ready"].all())
        roots = stored["is_branch"].squeeze(-1)
        torch.testing.assert_close(stored["deter"][roots][0], seed)
        # Force a window whose target row is the root, rather than relying on
        # the random sampler to happen to include it.
        window = stored[:2, 0].clone()
        info = {"index": (torch.tensor([0, 1]), torch.tensor([0, 0]))}
        with mock.patch.object(self.replay._buffer, "sample", return_value=(window, info)):
            data, index, _ = self.replay.sample()
        self.assertTrue(data["is_branch"].all())
        self.replay.update(index, torch.zeros_like(data["stoch"]), torch.zeros_like(data["deter"]))
        torch.testing.assert_close(self.replay._buffer.storage[:]["deter"][roots][0], seed)

    def test_close_discards_pending_branch_cleans_candidate_and_is_idempotent(self):
        self.queue()
        self.step(torch.ones(2, dtype=torch.bool))
        self.controller.close()
        self.assert_released()
        self.assertEqual(self.controller.budget.credit, 3)
        self.assertEqual(self.controller.stats["discarded_branches"], 1)
        self.controller.close()
        self.assertEqual(self.controller.stats["discarded_branches"], 1)

    def test_overwritten_transaction_is_discarded_and_releases_reservation(self):
        # Two vector frames cannot retain a root plus three actions.
        self.replay = Buffer(
            OmegaConf.create(dict(device="cpu", storage_device="cpu", batch_size=1, batch_length=1, max_size=4))
        )
        self.replay.enable_transactions()
        self.controller.replay = self.replay
        self.queue()
        _, done = self.step(torch.ones(2, dtype=torch.bool))
        for _ in range(3):
            _, done = self.step(done)
        self.assert_released()
        self.assertEqual(self.controller.stats["committed_branches"], 0)
        self.assertEqual(self.controller.stats["discarded_branches"], 1)
        self.assertEqual(self.controller.stats["reject/ring_overwrite"], 1)

    def test_rejection_details_are_persisted_once_and_drained(self):
        self.queue()
        details = dict(geom_pair=[0, 1], geom_names=["a", "b"], depth=0.3, threshold=0.01)
        with mock.patch.object(self.physical_envs[0].guard, "check", side_effect=UnsafeState("penetration", **details)):
            self.step(torch.ones(2, dtype=torch.bool))
        trainer = OnlineTrainer.__new__(OnlineTrainer)
        trainer._reconstruction_status = None
        with tempfile.TemporaryDirectory() as directory:
            trainer.logdir = directory
            trainer.write_reconstruction_diagnostics(self.controller)
            trainer.write_reconstruction_diagnostics(self.controller, final=True)
            events = [
                json.loads(line)
                for line in (Path(directory) / "reconstruction_rejections.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["reason"], "penetration")
            self.assertEqual(events[0]["branch_step"], 0)
            for key, value in details.items():
                self.assertEqual(events[0][key], value)
            self.assertTrue((Path(directory) / "reconstruction_summary.json").exists())
        self.assertEqual(self.controller.take_rejections(), [])
