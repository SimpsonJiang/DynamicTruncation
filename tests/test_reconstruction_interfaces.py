"""Regression contracts for the online reconstruction interfaces (CPU only)."""

import copy
import unittest
from unittest import mock

import numpy as np
import test_value_ensemble as fixtures
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from buffer import ReplayNotReady
from envs.parallel import ParallelEnv
from envs.physics_protocol import EnvStepResult, PhysicsContext
from physics_decoder import OnlinePhysicsDecoder
from physics_probe.decoder import StateDecoder
from trainer import OnlineTrainer


class ReconstructionInterfaceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)

    def test_online_export_roundtrip_and_resume_statistics(self):
        schema = dict(nq=2, nv=2, na=0, joint_types=[3, 2], joint_addresses=[0, 1], joint_limited=[0, 1], task="test")
        config = OmegaConf.create(dict(schema=schema, hidden=16, norm_decay=0.99, norm_eps=1e-3))
        head = OnlinePhysicsDecoder(8, config)
        x, y = torch.randn(5, 8), torch.randn(5, 4, dtype=torch.float64)
        head.training_loss(x, y)
        exported = head.export()
        rng = torch.random.get_rng_state()
        restored = OnlinePhysicsDecoder.from_export(exported)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        torch.testing.assert_close(head(x), restored(x), rtol=0, atol=0)
        restored.train()
        torch.testing.assert_close(head.training_loss(x, y), restored.training_loss(x, y), rtol=0, atol=0)
        for key in head.state_dict():
            torch.testing.assert_close(head.state_dict()[key], restored.state_dict()[key], rtol=0, atol=0)
        bad = copy.deepcopy(exported)
        bad["state_dict"]["net.0.weight"] = bad["state_dict"]["net.0.weight"].float()
        with self.assertRaisesRegex(RuntimeError, "float64"):
            OnlinePhysicsDecoder.from_export(bad)
        with self.assertRaisesRegex(ValueError, "architecture"):
            StateDecoder.from_export(exported)
        offline = StateDecoder(8, schema, hidden=16)
        offline.normalize_from_training(x, y)
        torch.testing.assert_close(offline(x), StateDecoder.from_export(offline.export())(x), rtol=0, atol=0)

    def test_all_collection_modes_share_dtype_and_shape_packing(self):
        # Exercise RPC dispatch too, without spawning processes or using a renderer.
        rows = [
            EnvStepResult({"value": np.array([i], np.float32), "is_first": bool(i == 0)}, float(i), False)
            for i in range(2)
        ]
        batch = ParallelEnv.__new__(ParallelEnv)
        batch.device, batch.physics_enabled = "cpu", False
        batch.envs = [mock.Mock(), mock.Mock()]
        batch.envs[0].reset.return_value = lambda: rows[0].observation
        batch.envs[1].step.return_value = lambda: (rows[1].observation, rows[1].reward, False, {})
        actions, done = torch.zeros(2, 1), torch.tensor([True, False])
        plain, _ = batch.step(actions, done)
        core = np.array([1 + 1e-12, 2.0], dtype=np.float64)
        labelled = [row._replace(core=core.copy()) for row in rows]
        for env, row in zip(batch.envs, labelled):
            env.online_step.return_value = lambda row=row: row
        batch.physics_enabled = True
        online, _ = batch.step(actions, done)
        branch = [
            row._replace(context_vector=np.zeros(4, np.float64), report={"root": False}, context={"task": "x"})
            for row in labelled
        ]
        for env, row in zip(batch.envs, branch):
            env.reconstruction_step.return_value = lambda row=row: row
        reconstructed, _, reports, contexts = batch.reconstruction_step(actions, done, [None, None], "off")
        for td in (plain, online, reconstructed):
            torch.testing.assert_close(td["value"], plain["value"])
            torch.testing.assert_close(td["reward"], plain["reward"])
            self.assertEqual(td["is_first"].shape, (2, 1))
        self.assertNotIn("physics", plain)
        for td in (online, reconstructed):
            self.assertEqual(td["physics", "core"].dtype, torch.float64)
            self.assertEqual(float(td["physics", "core"][0, 0]), core[0])
        self.assertEqual(reports, [{"root": False}] * 2)
        self.assertEqual(contexts, [{"task": "x"}] * 2)

    def test_context_advances_only_from_source_and_imagined_actions(self):
        source = np.array([0.5, 3, 6, 0.12, 0.34], np.float64)
        timeline = PhysicsContext.from_vector(source)
        context = dict(
            repeat=2,
            policy_dt=0.04,
            limit=10,
            inner_limit=20,
            action_low=np.array([-2.0, -4.0], np.float32),
            action_high=np.array([2.0, 4.0], np.float32),
        )
        actions = np.array([[0, 0], [1, -1], [0.25, 0.25]], np.float32)
        advanced = timeline.advance(2, actions, context)
        np.testing.assert_array_equal(source, timeline.to_vector())
        np.testing.assert_array_equal(advanced.control, [2.0, -4.0])
        self.assertAlmostEqual(advanced.time, 0.58)
        self.assertEqual((advanced.policy_steps, advanced.inner_steps), (5, 10))
        self.assertEqual(timeline.remaining_steps(context, 2), 5)
        np.testing.assert_array_equal(timeline.advance(0, actions, context).to_vector(), source)
        with self.assertRaises(ValueError):
            PhysicsContext.from_vector([0, 1.5, 2, 0])

    def test_geometry_hook_installation_has_one_owner_and_restores_original(self):
        from dm_control import mujoco as dm_mujoco

        from reconstruction_guard import GeometryMonitor

        physics = dm_mujoco.Physics.from_xml_string("""<mujoco><worldbody>
        <geom type="plane" size="1 1 .1"/><body pos="0 0 1"><freejoint/>
        <geom type="sphere" size=".1"/></body></worldbody></mujoco>""")
        original = physics.step
        first, second = GeometryMonitor(physics, "test"), GeometryMonitor(physics, "test")
        self.assertEqual(physics.step, original)
        first.install_hook()
        first.install_hook()
        self.assertEqual(first._original_step, original)
        with self.assertRaisesRegex(RuntimeError, "already installed"):
            second.install_hook()
        first.uninstall_hook()
        first.uninstall_hook()
        self.assertEqual(physics.step, original)
        second.install_hook()
        second.uninstall_hook()
        self.assertEqual(physics.step, original)

    def test_seed_action_interface_preserves_rng_and_observes_only_normal_slots(self):
        fixtures.ValueEnsembleTest.setUpClass()
        agent = fixtures.ValueEnsembleTest().make_agent()
        data, _ = fixtures.ValueEnsembleTest().make_batch(agent)
        obs = data[:, 0].clone()
        obs["is_branch"] = torch.tensor([[False], [True]])
        state = agent.get_initial_state(2)
        features = torch.randn(1, agent.rssm.feat_size, dtype=torch.float64)
        actions = torch.tensor([[0.2, -0.3]], dtype=torch.float64)
        rng = torch.random.get_rng_state()
        normal_action, normal_state = agent.act(obs[:1].clone(), state[:1], eval=False)
        after_normal = torch.random.get_rng_state()
        torch.random.set_rng_state(rng)
        with mock.patch.object(agent, "act", wraps=agent.act) as act:
            result_action, result_state = agent.act_with_seeds(obs, state, features, actions)
        self.assertEqual(act.call_count, 1)
        self.assertEqual(act.call_args.args[0].batch_size, torch.Size([1]))
        self.assertTrue(torch.equal(after_normal, torch.random.get_rng_state()))
        torch.testing.assert_close(result_action[:1], normal_action, rtol=0, atol=0)
        torch.testing.assert_close(result_state["deter"][:1], normal_state["deter"], rtol=0, atol=0)
        torch.testing.assert_close(
            agent.rssm.get_feat(result_state["stoch"][1:], result_state["deter"][1:]), features.float()
        )
        torch.testing.assert_close(result_action[1:], actions.float())
        agent._reconstruction_candidates = {"score": torch.tensor([1.0])}
        self.assertIsNotNone(agent.take_reconstruction_candidates())
        self.assertIsNone(agent.take_reconstruction_candidates())

    def test_video_logging_skips_pending_replay_and_updates_resume(self):
        config = OmegaConf.create(
            dict(
                steps=6,
                pretrain=1,
                eval_every=0,
                eval_episode_num=0,
                video_pred_log=True,
                params_hist_log=False,
                batch_length=1,
                batch_size=1,
                train_ratio=1,
                update_log_every=1,
                action_repeat=1,
            )
        )
        envs = mock.Mock(env_num=1)
        transition = TensorDict({"reward": torch.zeros(1, 1)}, [1])
        envs.step.side_effect = lambda action, done: (transition.clone(), torch.zeros(1, dtype=torch.bool))
        state = TensorDict(
            {"prev_action": torch.zeros(1, 1), "stoch": torch.zeros(1, 2, 2), "deter": torch.zeros(1, 4)}, [1]
        )
        agent = mock.Mock(device=torch.device("cpu"))
        agent.get_initial_state.return_value = state
        agent.act.return_value = (state["prev_action"], state)
        agent.update.side_effect = [ReplayNotReady("pending")] + [{"loss": 1.0}] * 4
        agent.video_pred.return_value = torch.zeros(1, 1, 1, 1, 3)
        replay = mock.Mock()
        replay.count.return_value = 0
        replay.sample.side_effect = [ReplayNotReady("pending")] + [(transition, None, None)] * 4
        logger = mock.Mock()
        OnlineTrainer(config, replay, logger, None, envs, None).begin(agent)
        self.assertGreater(agent.video_pred.call_count, 0)
        self.assertEqual([call.kwargs["env_step"] for call in agent.update.call_args_list], [3, 4, 4, 5, 6])
        self.assertTrue(any(c.args == ("train/opt/updates", 4) for c in logger.scalar.call_args_list))
