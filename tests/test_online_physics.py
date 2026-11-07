"""Small CPU contracts for online physics training and reconstruction."""

import copy
import unittest
from unittest import mock

import test_value_ensemble as fixtures
import torch
from omegaconf import OmegaConf

from physics_decoder import Float64Adam, OnlinePhysicsDecoder

SCHEMA = dict(nq=2, nv=2, na=0, joint_types=[3, 2], joint_addresses=[0, 1], joint_limited=[0, 1], task="test")


class CPUContractTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)


class DecoderTest(CPUContractTest):
    def config(self):
        return OmegaConf.create(dict(schema=SCHEMA, hidden=16, norm_decay=0.99, norm_eps=1e-3))

    def test_precision_statistics_checkpoint_rng(self):
        rng = torch.random.get_rng_state()
        with mock.patch("torch.cuda.manual_seed_all") as cuda_seed:
            head = OnlinePhysicsDecoder(8, self.config())
        cuda_seed.assert_not_called()
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        x, y = torch.randn(10, 8, requires_grad=True), torch.randn(10, 4, dtype=torch.float64)
        opt = Float64Adam(head.parameters())
        loss = head.training_loss(x, y)
        self.assertEqual(loss.dtype, torch.float64)
        from physics_probe.decoder import StateDecoder

        torch.testing.assert_close(loss, StateDecoder.loss(head, x.double(), y))
        loss.backward()
        self.assertGreater(x.grad.abs().sum(), 0)
        opt.step()
        self.assertTrue(all(s["exp_avg"].dtype == torch.float64 for s in opt.state.values()))
        optimizer_state = copy.deepcopy(opt.state_dict())
        restored_optimizer = Float64Adam(head.parameters())
        restored_optimizer.load_state_dict(optimizer_state)
        self.assertEqual(len(restored_optimizer.state), len(opt.state))
        bad_optimizer = copy.deepcopy(optimizer_state)
        next(iter(bad_optimizer["state"].values()))["exp_avg"] = next(iter(bad_optimizer["state"].values()))[
            "exp_avg"
        ].float()
        with self.assertRaisesRegex(ValueError, "float64"):
            restored_optimizer.load_state_dict(bad_optimizer)
        state = copy.deepcopy(head.state_dict())
        head.eval()
        head.training_loss(x, y * 3)
        self.assertTrue(torch.equal(head.input_mean, state["input_mean"]))
        other = OnlinePhysicsDecoder(8, self.config())
        other.load_state_dict(state)
        wrong_schema = self.config()
        wrong_schema.schema.task = "different_task"
        with self.assertRaisesRegex(RuntimeError, "schema"):
            OnlinePhysicsDecoder(8, wrong_schema).load_state_dict(state)
        torch.testing.assert_close(head(x), other(x))
        bad = {k: v.float() if v.is_floating_point() else v for k, v in state.items()}
        with self.assertRaisesRegex(RuntimeError, "float64"):
            other.load_state_dict(bad)
        with self.assertRaises(TypeError):
            head.training_loss(x, y.float())

    def test_auxiliary_detach_and_joint_gradients(self):
        fixtures.ValueEnsembleTest.setUpClass()
        fixture = fixtures.ValueEnsembleTest()
        for detach in (True, False):
            agent = fixture.make_agent()
            agent.physics_head = OnlinePhysicsDecoder(agent.rssm.feat_size, self.config())
            agent.physics_grad_to_wm = not detach
            agent.physics_loss_scale, agent.physics_grad_clip = 1.0, 100.0
            agent.physics_optimizer = torch.optim.Adam(agent.physics_head.parameters(), lr=1e-3)
            data, initial = fixture.make_batch(agent)
            data["physics", "core"] = torch.randn(2, 4, 4, dtype=torch.float64)
            before = [p.clone() for p in agent.physics_head.parameters()]
            replay = mock.Mock()
            replay.sample.return_value = (data, None, initial)
            metrics = agent.update(replay)
            self.assertIn("loss/physics", metrics)
            self.assertTrue(any(not torch.equal(p, q) for p, q in zip(before, agent.physics_head.parameters())))
            self.assertTrue(
                all(id(p) not in {id(q) for q in agent._named_params.values()} for p in agent.physics_head.parameters())
            )


class SafetyAndReplayTest(CPUContractTest):
    def test_atomic_replay_and_ring_generation(self):
        from tensordict import TensorDict

        from buffer import Buffer, ReplayNotReady

        b = Buffer(OmegaConf.create(dict(device="cpu", storage_device="cpu", batch_size=1, batch_length=2, max_size=8)))
        b.enable_transactions()

        def row(t, branch=False):
            return TensorDict(
                dict(
                    episode=torch.zeros(1, dtype=torch.int32),
                    stoch=torch.full((1, 2, 2), float(t)),
                    deter=torch.full((1, 3), float(t)),
                    branch_stoch=torch.full((1, 2, 2), 99.0),
                    branch_deter=torch.full((1, 3), 99.0),
                    is_branch=torch.tensor([[branch]]),
                    action=torch.tensor([[float(t)]]),
                ),
                [1],
            )

        b.reserve(0)
        for t in range(4):
            b.add_transition(row(t, t == 0))
        with self.assertRaises(ReplayNotReady):
            b.sample()
        self.assertTrue(b.commit(0))
        for _ in range(4):
            data, index, initial = b.sample()
            torch.testing.assert_close(data["action"].squeeze(-1) + 1, data["deter"][..., 0])
            b.update(index, data["stoch"], data["deter"])
        b.reserve(0)
        for t in range(10):
            b.add_transition(row(t))
        self.assertFalse(b.commit(0))
        with self.assertRaises(ReplayNotReady):
            b.sample()

    def test_branch_seed_overrides_observation_and_has_no_incoming_gradient(self):
        fixtures.ValueEnsembleTest.setUpClass()
        agent = fixtures.ValueEnsembleTest().make_agent()
        embed = torch.randn(2, 4, agent.embed_size, requires_grad=True)
        action = torch.randn(2, 4, 2)
        roots = torch.zeros(2, 4, 1, dtype=torch.bool)
        roots[0, 0] = roots[0, 3] = roots[1, 2] = True
        seed_s = torch.randn(2, 4, 4, 4)
        seed_d = torch.randn(2, 4, 32)
        s, d, _ = agent.rssm.observe(
            embed, action, agent.rssm.initial(2), torch.zeros_like(roots), roots, (seed_s, seed_d)
        )
        torch.testing.assert_close(s[roots[..., 0]], seed_s[roots[..., 0]])
        torch.testing.assert_close(d[roots[..., 0]], seed_d[roots[..., 0]])
        d.sum().backward()
        self.assertTrue(torch.equal(embed.grad[roots[..., 0]], torch.zeros_like(embed.grad[roots[..., 0]])))

    def test_budget_refund_failed_actions_and_ratio(self):
        from reconstruction import ReconstructionBudget

        cfg = OmegaConf.create(dict(ratio=0.1, max_active=2, branch_steps=64, max_consecutive=2, credit_cap_steps=128))
        b = ReconstructionBudget(cfg)
        b.account(90)
        self.assertEqual(b.credit, 0)
        b.enabled = True
        b.account(90)
        self.assertAlmostEqual(b.credit, 10)
        b.reserve(0, 10)
        for _ in range(3):
            b.executed(0)
        b.account(reconstructed=3)
        b.finish(0)
        self.assertAlmostEqual(b.credit, 7)
        self.assertEqual(b.stats["reconstruction_steps"], 3)
        b.account(ordinary=1800)
        b.reserve(0, 64)
        b.reserve(1, 64)
        self.assertEqual(b.allowance(2, 64), (0, "concurrency"))
        b.finish(0)
        b.finish(1)
        self.assertEqual(b.allowance(0, 64), (0, "consecutive"))
        b.ordinary_reset(0, "consecutive")
        self.assertEqual(b.allowance(0, 64), (64, None))

    def test_candidate_topk_diversity_expiry(self):
        import numpy as np

        from reconstruction import Candidate, CandidatePool

        cfg = OmegaConf.create(
            dict(
                capacity=20,
                top_k=3,
                refresh_every_updates=8,
                ttl_updates=8,
                max_insert=20,
                max_per_origin=2,
                deter_cosine_threshold=0.98,
                action_rms_threshold=0.1,
            )
        )

        def candidate(i):
            return Candidate(i + 1.0, 0, (i, 0), np.array([i + 1.0, 1.0]), np.full((2, 1), i), {}, np.zeros(3), 1)

        p, q = CandidatePool(cfg, 3), CandidatePool(cfg, 3)
        for pool in (p, q):
            pool.insert([candidate(i) for i in range(8)], 0, np.zeros(2), np.ones(2), 0)
        first = p.choose(0)
        self.assertEqual(first.score, q.choose(0).score)
        self.assertIn(first.score, [6, 7, 8])
        self.assertNotIn(id(first), [id(c) for c in p.items])
        shortlist = q.items[:3]
        eligible = {id(c) for c in shortlist}
        selected = [q.choose(0, shortlist) for _ in range(3)]
        self.assertEqual({id(c) for c in selected}, eligible)
        self.assertIsNone(q.choose(0, shortlist))
        p.expire(8)
        self.assertFalse(p.items)

    def test_guard_calibration_validation_freeze(self):
        from reconstruction_guard import GuardCalibration

        c = GuardCalibration(dict(floors=[0.001, 0.002], signature="s"), 4, 4, 0.0)
        c.observe(dict(samples=[[0.01, 0], [0.02, 0]], violation=False), 4, "calibrate")
        self.assertAlmostEqual(c.thresholds[0], 1.25 * 0.01999)
        self.assertEqual(c.thresholds[1], 0.002)
        c.observe(dict(samples=[], violation=True), 4, "validate")
        self.assertFalse(c.ready)
        self.assertIn("false_rejection_rate", c.failure)

    def test_mujoco_substep_monitor_preserves_dynamics_and_detects_intrusion(self):
        import numpy as np
        from dm_control import suite

        from reconstruction_guard import GeometryMonitor, UnsafeState

        a = suite.load("walker", "walk", task_kwargs={"random": 3})
        b = suite.load("walker", "walk", task_kwargs={"random": 3})
        a.reset()
        b.reset()
        guard = GeometryMonitor(b.physics, "walker_walk")
        guard.install_hook()
        guard.begin_policy_step("calibrate")
        for _ in range(3):
            aa, bb = a.step(np.zeros(a.action_spec().shape)), b.step(np.zeros(b.action_spec().shape))
            np.testing.assert_array_equal(a.physics.get_state(), b.physics.get_state())
            self.assertEqual(aa.reward, bb.reward)
            self.assertEqual(aa.step_type, bb.step_type)
        self.assertEqual(len(guard.samples), 3 * b._n_sub_steps)
        guard.install_thresholds(guard.signature, np.full(len(guard.pairs), 0.01))
        b.physics.named.data.qpos["rootz"] = -1.0
        guard.begin_policy_step("branch")
        with self.assertRaises(UnsafeState):
            guard.check()
        b.physics.data.qpos[0] = np.nan
        with self.assertRaisesRegex(UnsafeState, "nonfinite"):
            guard.check()

    def test_filtered_pairs_and_transient_substep(self):
        import numpy as np
        from dm_control import mujoco as dm_mujoco

        from reconstruction_guard import GeometryMonitor, UnsafeState

        xml = """<mujoco><worldbody>
        <body pos="0 0 1"><freejoint/><geom type="sphere" size=".2" contype="1" conaffinity="1"/></body>
        <body pos=".1 0 1"><freejoint/><geom type="sphere" size=".2" contype="2" conaffinity="2"/></body>
        </worldbody></mujoco>"""
        physics = dm_mujoco.Physics.from_xml_string(xml)
        guard = GeometryMonitor(physics, "filtered")
        guard.install_hook()
        self.assertEqual(guard.pairs, [(0, 1)])
        self.assertEqual(physics.data.ncon, 0)
        self.assertGreater(guard.depths()[0], 0.2)
        guard.install_thresholds(guard.signature, [0.01])
        guard.begin_policy_step("branch")
        with self.assertRaisesRegex(UnsafeState, "penetration"):
            guard.check()
        # Validate every substep: a violation that vanishes before policy observation still counts.
        guard.begin_policy_step("validate")
        calls = []

        def depths():
            calls.append(1)
            return np.array([0.1 if len(calls) == 2 else 0.0])

        with mock.patch.object(guard, "depths", side_effect=depths):
            physics.step(4)
        self.assertEqual(len(calls), 4)
        self.assertTrue(guard.violation)
        self.assertEqual(guard.max_depth, 0.1)

    def test_root_labels_and_incoming_rewards_do_not_train(self):
        fixtures.ValueEnsembleTest.setUpClass()
        f = fixtures.ValueEnsembleTest()
        agent = f.make_agent()
        data, initial = f.make_batch(agent)
        data["is_branch"] = torch.zeros(2, 4, 1, dtype=torch.bool)
        data["is_branch"][:, 2] = True
        data["is_recon"] = data["is_branch"].clone()
        data["branch_stoch"] = torch.randn(2, 4, 4, 4)
        data["branch_deter"] = torch.randn(2, 4, 32)
        other = data.clone()
        other["state"][:, 2] = 1000
        other["reward"][:, 2] = 1000
        other["is_terminal"][:, 2] = True
        saved = copy.deepcopy(agent.state_dict())

        def grads(batch):
            agent.load_state_dict(saved)
            agent.zero_grad(set_to_none=True)
            torch.manual_seed(63)
            agent._cal_grad(batch, initial)
            return {n: p.grad.clone() for n, p in agent._named_params.items() if p.grad is not None}

        before, after = grads(data), grads(other)
        for k in before:
            torch.testing.assert_close(before[k], after[k], msg=k)

    def test_four_tasks_labels_and_substep_state_equivalence(self):
        import numpy as np
        from dm_control import suite

        from envs import make_env
        from reconstruction_guard import GeometryMonitor

        for name in ("finger_spin", "hopper_hop", "quadruped_run", "walker_walk"):
            domain, task = name.split("_", 1)
            a = suite.load(domain, task, task_kwargs={"random": 7})
            b = suite.load(domain, task, task_kwargs={"random": 7})
            a.reset()
            b.reset()
            guard = GeometryMonitor(b.physics, name)
            guard.install_hook()
            guard.begin_policy_step("calibrate")
            for _ in range(2):
                aa = a.step(np.zeros(a.action_spec().shape))
                bb = b.step(np.zeros(b.action_spec().shape))
                np.testing.assert_array_equal(a.physics.get_state(), b.physics.get_state())
                np.testing.assert_array_equal(a.physics.data.qacc_warmstart, b.physics.data.qacc_warmstart)
                self.assertEqual(aa.reward, bb.reward)
            cfg = OmegaConf.create(dict(task="dmc_" + name, action_repeat=2, time_limit=1000, size=[64, 64], seed=7))
            env = make_env(cfg, 0, physics=True)
            with mock.patch.object(env.base, "render", return_value=np.zeros((64, 64, 3), np.uint8)):
                result = env.online_step(np.zeros(env.action_space.shape), reset=True)
                obs, core = result.observation, result.core
            self.assertNotIn("physics", obs)
            self.assertNotIn("physics", env.observation_space.spaces)
            self.assertEqual(core.dtype, np.float64)
            np.testing.assert_array_equal(core, env.physics_labels())
            self.assertTrue(np.any(core != core.astype(np.float32).astype(np.float64)))

    def test_auxiliary_gradient_switch_is_exact(self):
        from physics_decoder import OnlinePhysicsDecoder

        fixtures.ValueEnsembleTest.setUpClass()
        fixture = fixtures.ValueEnsembleTest()
        for grad_to_wm in (False, True):
            agent = fixture.make_agent()
            head_config = OmegaConf.create(dict(schema=SCHEMA, hidden=16, norm_decay=0.99, norm_eps=1e-3))
            agent.physics_head = OnlinePhysicsDecoder(agent.rssm.feat_size, head_config)
            agent.physics_grad_to_wm, agent.physics_loss_scale = grad_to_wm, 1.0
            agent._loss_scales = {k: 0.0 for k in agent._loss_scales}
            data, initial = fixture.make_batch(agent)
            data["physics", "core"] = torch.randn(2, 4, 4, dtype=torch.float64)
            agent._cal_grad(data, initial)
            norm = fixture.grad_norm(agent.encoder) + fixture.grad_norm(agent.rssm)
            if grad_to_wm:
                self.assertGreater(norm, 0)
            else:
                self.assertEqual(norm, 0)
            parameters = list(agent.encoder.parameters()) + list(agent.rssm.parameters())
            before = [p.detach().clone() for p in parameters]
            agent._agc(agent._named_params.values())
            agent._optimizer.step()
            changed = any(not torch.equal(a, b) for a, b in zip(before, parameters))
            self.assertEqual(changed, grad_to_wm)

    def test_candidate_packet_excludes_reconstruction_roots(self):
        fixtures.ValueEnsembleTest.setUpClass()
        fixture = fixtures.ValueEnsembleTest()
        agent = fixture.make_agent(trunc_enabled=True, min_horizon=1, warmup_steps=0)
        agent.reconstruction_config = OmegaConf.create(dict(candidates=dict(capacity=8)))
        data, initial = fixture.make_batch(agent)
        data["is_recon"] = torch.ones(2, 4, 1, dtype=torch.bool)
        data["origin"] = torch.zeros(2, 4, 2, dtype=torch.int64)
        data["physics", "context"] = torch.zeros(2, 4, 5, dtype=torch.float64)
        mask = torch.tensor([1, 0, 0], dtype=torch.bool).view(1, 3, 1).expand(8, -1, -1)

        def positive_returns(last, term, reward, value, boot, disc, lamb):
            return torch.ones_like(value[:, :-1]) * 100

        with mock.patch("dreamer.build_truncation_mask", return_value=mask), mock.patch.object(
            agent, "_lambda_return", side_effect=positive_returns
        ):
            ordinary = data.clone()
            ordinary["is_recon"].zero_()
            _, allowed = agent._cal_grad(ordinary, initial, truncation_active=True, collect_candidates=True)
            self.assertTrue(torch.isfinite(allowed["_candidates"]["score"]).any())
            _, rejected = agent._cal_grad(data, initial, truncation_active=True, collect_candidates=True)
            self.assertFalse(torch.isfinite(rejected["_candidates"]["score"]).any())

    def test_guard_uses_first_reset_without_extra_rng_draw(self):
        import numpy as np

        from envs import make_env

        cfg = OmegaConf.create(dict(task="dmc_finger_spin", action_repeat=2, time_limit=100, size=[64, 64], seed=11))
        guarded, plain = make_env(cfg, 0, physics=True), make_env(cfg, 0, physics=True)
        image = np.zeros((64, 64, 3), np.uint8)
        with mock.patch.object(guarded.base, "render", return_value=image), mock.patch.object(
            plain.base, "render", return_value=image
        ):
            profile = guarded.setup_guard()
            self.assertIs(profile, guarded.setup_guard())
            row = guarded.reconstruction_step(np.zeros(2), True, None, "calibrate")
            plain.reset()
        self.assertNotIn("failure", profile)
        self.assertNotIn("guard_failure", row.report)
        np.testing.assert_array_equal(guarded.physics_labels(), plain.physics_labels())
        a, b = guarded.base._env.task.random.get_state(), plain.base._env.task.random.get_state()
        np.testing.assert_array_equal(a[1], b[1])
        self.assertEqual(a[2:], b[2:])

    def test_r2_short_branch_has_finite_masked_representation_loss(self):
        fixtures.ValueEnsembleTest.setUpClass()
        fixture = fixtures.ValueEnsembleTest()
        with mock.patch.dict(fixture.config, {"rep_loss": "r2dreamer"}):
            agent = fixture.make_agent()
        data, _ = fixture.make_batch(agent)
        data = data[:1, :2].clone()
        data["is_branch"] = torch.tensor([[[False], [True]]])
        data["is_recon"] = data["is_branch"].clone()
        data["branch_stoch"] = torch.randn(1, 2, 4, 4)
        data["branch_deter"] = torch.randn(1, 2, 32)
        _, metrics = agent._cal_grad(data, agent.rssm.initial(1))
        self.assertEqual(float(metrics["loss/barlow"].detach()), 0.0)
        self.assertTrue(torch.isfinite(metrics["opt/loss"]))

    def test_welded_parent_groups_are_excluded(self):
        from dm_control import mujoco as dm_mujoco

        from reconstruction_guard import GeometryMonitor

        xml = """<mujoco><worldbody><geom type="plane" size="2 2 .1"/>
          <body pos="0 0 1"><freejoint/><geom type="sphere" size=".2"/>
            <body><geom type="sphere" size=".2"/>
              <body><joint type="hinge"/><geom type="sphere" size=".2"/></body>
            </body>
          </body></worldbody></mujoco>"""
        guard = GeometryMonitor(dm_mujoco.Physics.from_xml_string(xml), "welded_parent")
        self.assertNotIn((1, 2), guard.pairs)
        self.assertNotIn((1, 3), guard.pairs)
        self.assertNotIn((2, 3), guard.pairs)
        self.assertIn((0, 3), guard.pairs)
