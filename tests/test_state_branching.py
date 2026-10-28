"""CPU checks for detached reconstruction, state restoration and branch replay."""

import contextlib
import copy
import io
import unittest
from pathlib import Path
from unittest import mock

import gymnasium as gym
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tensordict import TensorDict

from branch_collection import BranchCollector, BranchEnvironment, IsolatedTorchRNG, add_source_metadata, validate_branch_config
from buffer import Buffer, MixedReplay
from dreamer import Dreamer
from state_decoder import FingerStateDecoder, StateDecoder, STATE_DECODERS, make_state_decoder


def small_config(*overrides):
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[1] / "configs")):
        return compose(config_name="configs", overrides=[
            "env=dmc_proprio", "env.task=dmc_finger_spin", "env.env_num=1", "env.eval_episode_num=0",
            "model.rep_loss=dreamer", "device=cpu", "model.compile=false", "model.deter=32",
            "model.hidden=16", "model.units=16", "model.discrete=4", "model.rssm.stoch=4",
            "model.rssm.blocks=4", "model.imag_horizon=3", "model.state_decoder.units=16",
            "batch_size=4", "batch_length=4", "buffer.max_size=100",
            "branch_collection.replay_capacity=100", *overrides,
        ])


SPEC = {"task": "dmc_finger_spin", "dimensions": [3, 3, 0], "joint_ranges": [[-1.919862177, 1.919862177]] * 2}
OBS = gym.spaces.Dict({"position": gym.spaces.Box(-10, 10, (4,)),
                       "velocity": gym.spaces.Box(-100, 100, (3,)), "touch": gym.spaces.Box(0, 100, (2,))})


def make_agent(config):
    with contextlib.redirect_stdout(io.StringIO()):
        agent = Dreamer(config.model, OBS, gym.spaces.Box(-1, 1, (2,)), state_spec=SPEC,
                        collect_branch_candidates=config.branch_collection.mode == "imagined_state").to("cpu")
    agent._scaler = torch.amp.GradScaler("cpu", enabled=False)
    return agent


def make_batch():
    data = TensorDict({
        "position": torch.randn(2, 4, 4), "velocity": torch.randn(2, 4, 3), "touch": torch.rand(2, 4, 2),
        "action": torch.randn(2, 4, 2), "reward": torch.randn(2, 4, 1),
        "is_first": torch.zeros(2, 4, 1, dtype=torch.bool),
        "is_last": torch.zeros(2, 4, 1, dtype=torch.bool),
        "is_terminal": torch.zeros(2, 4, 1, dtype=torch.bool),
        "_physics_state": torch.randn(2, 4, 6),
        "_source": torch.zeros(2, 4, 1, dtype=torch.int64),
        "_origin_step": torch.arange(8).reshape(2, 4, 1),
    }, batch_size=(2, 4))
    data["is_first"][:, 0] = True
    return data


class StateDecoderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_config_switches_and_scope(self):
        cfg = small_config()
        self.assertEqual(validate_branch_config(cfg), (False, False))
        cfg.model.state_decoder.enabled = True
        self.assertEqual(validate_branch_config(cfg), (True, False))
        cfg.branch_collection.mode = "imagined_state"
        self.assertEqual(validate_branch_config(cfg), (True, False))
        cfg.model.state_decoder.enabled = False
        with self.assertRaisesRegex(ValueError, "requires"):
            validate_branch_config(cfg)
        cfg.branch_collection.mode = "replay_state"
        self.assertEqual(validate_branch_config(cfg), (True, True))
        cfg.env.task = "dmc_hopper_hop"
        with self.assertRaisesRegex(ValueError, "finger_spin"):
            validate_branch_config(cfg)

    def test_base_class_supports_other_codecs_and_factory_dispatch(self):
        class VectorDecoder(StateDecoder):
            encoded_size = 2

            def setup_state_spec(self, spec):
                self.register_buffer("offset", torch.tensor(spec["offset"]).float())

            def encode(self, state):
                return state - self.offset

            def decode(self, prediction):
                return prediction + self.offset

        config = small_config().model.state_decoder
        with self.assertRaises(TypeError):
            StateDecoder(config, 8, SPEC)
        with mock.patch.dict(STATE_DECODERS, {"test_vector": VectorDecoder}):
            decoder = make_state_decoder(config, 8, {"task": "test_vector", "offset": [1, 2]})
        self.assertIsInstance(decoder, StateDecoder)
        self.assertIsNone(config.shape)  # task resolution must not mutate shared config
        features = torch.randn(2, 3, 8, requires_grad=True)
        targets = torch.randn(2, 3, 2, requires_grad=True)
        self.assertEqual(decoder(features).shape, (2, 3, 2))
        torch.testing.assert_close(decoder.decode(decoder.encode(targets)), targets)
        loss, metrics = decoder.loss(features, targets)
        loss.backward()
        self.assertIsNone(features.grad)
        self.assertIsNone(targets.grad)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in decoder.parameters()))
        self.assertFalse(metrics["state_decoder/mse"].requires_grad)
        self.assertIsNone(decoder.prediction_rejection_reason(torch.zeros(1, 2)))
        self.assertEqual(decoder.prediction_rejection_reason(torch.tensor([[float("nan"), 0]])), "nonfinite")

    def test_task_selection_output_size_and_finger_specific_validation(self):
        config = small_config().model.state_decoder
        decoder = make_state_decoder(config, 8, SPEC)
        self.assertIsInstance(decoder, FingerStateDecoder)
        self.assertEqual(decoder(torch.zeros(1, 8)).shape, (1, 7))
        self.assertEqual(decoder.prediction_rejection_reason(torch.zeros(1, 7)), "angle_norm")
        prediction = torch.zeros(1, 7)
        prediction[:, 3] = 1
        self.assertIsNone(decoder.prediction_rejection_reason(prediction))
        with self.assertRaisesRegex(ValueError, "No StateDecoder registered"):
            make_state_decoder(config, 8, {**SPEC, "task": "dmc_hopper_hop"})
        with self.assertRaisesRegex(ValueError, "No StateDecoder registered"):
            make_state_decoder(config, 8, None)
        config.shape = [6]
        with self.assertRaisesRegex(ValueError, "shape"):
            make_state_decoder(config, 8, SPEC)

    def test_codec_periodicity_and_detached_input(self):
        decoder = FingerStateDecoder(small_config().model.state_decoder, 8, SPEC)
        states = torch.tensor([[0.1, -0.3, np.pi - 0.01, 1.2, -4.0, 15.0],
                               [0.1, -0.3, -np.pi + 0.01, 0.0, 0.0, 0.0]])
        encoded = decoder.encode(states)
        torch.testing.assert_close(decoder.decode(encoded), states)
        wrapped = states.clone()
        wrapped[:, 2] += 2 * np.pi
        torch.testing.assert_close(decoder.encode(wrapped), encoded, atol=1e-6, rtol=1e-5)
        features = torch.randn(2, 8, requires_grad=True)
        loss, _ = decoder.loss(features, states)
        loss.backward()
        self.assertIsNone(features.grad)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in decoder.parameters()))

    def test_probe_does_not_change_base_rng_initialization_or_gradients(self):
        cfg = small_config()
        torch.manual_seed(22)
        baseline = make_agent(copy.deepcopy(cfg))
        rng_baseline = torch.get_rng_state().clone()
        cfg.model.state_decoder.enabled = True
        torch.manual_seed(22)
        probe = make_agent(cfg)
        self.assertTrue(torch.equal(rng_baseline, torch.get_rng_state()))
        self.assertFalse(hasattr(baseline, "state_decoder"))
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(value, probe.state_dict()[key], rtol=0, atol=0)
        data = make_batch()
        for agent in (baseline, probe):
            torch.manual_seed(91)
            agent._cal_grad(data.clone(), agent.rssm.initial(2))
        for name, p in baseline._named_params.items():
            torch.testing.assert_close(p.grad, probe._named_params[name].grad, rtol=0, atol=0)
        self.assertGreater(sum(p.grad.abs().sum() for p in probe.state_decoder.parameters()), 0)
        probe.zero_grad(set_to_none=True)
        embed = probe.encoder(data)
        stoch, deter, _ = probe.rssm.observe(embed, data["action"], probe.rssm.initial(2), data["is_first"])
        loss, _ = probe.state_decoder.loss(probe.rssm.get_feat(stoch, deter), data["_physics_state"])
        loss.backward()
        for module in (probe.encoder, probe.rssm, probe.actor, probe.value):
            self.assertTrue(all(p.grad is None for p in module.parameters()))

    def test_metadata_is_excluded_even_when_in_observation_space(self):
        space = gym.spaces.Dict({**OBS.spaces, "_physics_state": gym.spaces.Box(-1, 1, (6,))})
        cfg = small_config("model.state_decoder.enabled=true")
        with contextlib.redirect_stdout(io.StringIO()):
            agent = Dreamer(cfg.model, space, gym.spaces.Box(-1, 1, (2,)), SPEC)
        self.assertNotIn("_physics_state", agent.encoder.mlp_shapes)
        self.assertNotIn("_physics_state", agent.decoder.all_keys)

    def test_shared_head_config_and_loss_scale(self):
        from networks import MLPHead

        cfg = small_config("model.state_decoder.enabled=true", "model.state_decoder.layers=3",
                           "model.loss_scales.state_recon=0.5")
        torch.manual_seed(12)
        first = make_agent(copy.deepcopy(cfg))
        cfg.model.loss_scales.state_recon = 1.5
        torch.manual_seed(12)
        second = make_agent(cfg)
        self.assertIsInstance(first.state_decoder.net, MLPHead)
        self.assertEqual(sum(isinstance(m, torch.nn.RMSNorm) for m in first.state_decoder.modules()), 3)
        self.assertNotIn("loss_scale", cfg.model.state_decoder)
        data = make_batch()
        for agent in (first, second):
            torch.manual_seed(34)
            agent._cal_grad(data.clone(), agent.rssm.initial(2))
        for name, param in first._named_params.items():
            multiplier = 3 if name.startswith("state_decoder.") else 1
            torch.testing.assert_close(second._named_params[name].grad, param.grad * multiplier)

    def test_checkpoint_contains_probe_and_optimizer_state(self):
        cfg = small_config("model.state_decoder.enabled=true")
        agent = make_agent(copy.deepcopy(cfg))
        data = make_batch()
        replay = mock.Mock()
        replay.sample.return_value = data, [], agent.rssm.initial(2)
        agent.update(replay)
        stream = io.BytesIO()
        torch.save({"agent": agent.state_dict(), "optimizer": agent._optimizer.state_dict()}, stream)
        stream.seek(0)
        state = torch.load(stream, weights_only=True)
        restored = make_agent(cfg)
        restored.load_state_dict(state["agent"])
        restored._optimizer.load_state_dict(state["optimizer"])
        for name, value in agent.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[name])
        self.assertTrue(all(restored._optimizer.state[p]["step"] == 1 for p in restored.state_decoder.parameters()))

    def test_isolated_rng_restores_and_advances_private_stream(self):
        torch.manual_seed(5)
        before = torch.get_rng_state().clone()
        rng = IsolatedTorchRNG("cpu", 71)
        with rng.use():
            first = torch.rand(5)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        with self.assertRaisesRegex(RuntimeError, "test"):
            with rng.use():
                second = torch.rand(5)
                raise RuntimeError("test")
        self.assertFalse(torch.equal(first, second))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))


def replay_row(t, slots, source):
    row = TensorDict({
        "clock": torch.full((slots, 1), float(t)), "action": torch.full((slots, 2), float(t)),
        "stoch": torch.zeros(slots, 2, 2), "deter": torch.zeros(slots, 3),
        "is_first": torch.full((slots, 1), t % 5 == 0, dtype=torch.bool),
        "is_last": torch.full((slots, 1), t % 5 == 4, dtype=torch.bool),
        "is_terminal": torch.zeros(slots, 1, dtype=torch.bool),
        "episode": torch.arange(slots, dtype=torch.int32),
    }, batch_size=(slots,), device="cpu")
    return add_source_metadata(row, source, source, t, 0)


class MixedReplayTest(unittest.TestCase):
    def test_alignment_mix_fallback_boundaries_and_routed_writes(self):
        cfg = small_config()
        normal, branch = Buffer(cfg.buffer), Buffer(cfg.buffer)
        mixed = MixedReplay(normal, branch, 0.25)
        self.assertEqual(normal.storage_device, branch.storage_device)
        self.assertFalse(mixed.can_sample())
        for t in range(12):
            normal.add_transition(replay_row(t, 2, 0))
        data, index, initial = mixed.sample()
        self.assertEqual(mixed.last_branch_fraction, 0)
        self.assertTrue((data["_source"] == 0).all())
        for t in range(12):
            branch.add_transition(replay_row(t, 1, 1))
        data, index, initial = mixed.sample()
        self.assertEqual(tuple(data.batch_size), (4, 4))
        self.assertEqual(mixed.last_branch_fraction, 0.25)
        self.assertTrue((data["_source"][:3] == 0).all())
        self.assertTrue((data["_source"][3:] == 1).all())
        torch.testing.assert_close(data["action"][..., :1], data["clock"] - 1)
        self.assertTrue(torch.equal(data["is_first"], data["clock"].remainder(5) == 0))
        self.assertTrue(torch.equal(data["is_last"], data["clock"].remainder(5) == 4))
        stoch = torch.ones(4, 4, 2, 2)
        deter = torch.ones(4, 4, 3)
        stoch[3:] *= 7
        deter[3:] *= 9
        mixed.update(index, stoch, deter)
        for replay, addresses, s, d in ((normal, index.normal, 1, 1), (branch, index.branch, 7, 9)):
            rows = replay._buffer[addresses[1].reshape(-1), addresses[0].reshape(-1)]
            self.assertTrue((rows["stoch"] == s).all())
            self.assertTrue((rows["deter"] == d).all())


class LocalWorker:
    """Same future-style interface as Parallel, without process/render overhead."""
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return lambda *args: lambda: getattr(self.env, name)(*args)

    def close(self):
        self.env.env.close()


class PhysicsBranchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        try:
            import dm_control  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("DMC extra not installed")

    def setUp(self):
        from envs.dmc import DeepMindControl
        self.render = mock.patch.object(DeepMindControl, "render", lambda env: np.zeros((*env._size, 3), np.uint8))
        self.render.start()
        self.addCleanup(self.render.stop)

    def test_restore_precision_counters_replay_and_rejection(self):
        from envs import make_env
        cfg = small_config()
        env = make_env(cfg.env, 0, True, True)
        self.addCleanup(env.close)
        first = env.reset()
        self.assertEqual(first["_integration_state"].dtype, np.float64)
        self.assertNotIn("_physics_state", env.observation_space.spaces)
        for _ in range(5):
            obs, _, _, _ = env.step(np.array([0.1, -0.1]))
        snapshot = obs["_integration_state"].copy()
        reference = [env.step(np.array([0.05, 0.03])) for _ in range(10)]
        restored = env.restore_state(snapshot, kind="integration", max_penetration=1.0)
        np.testing.assert_array_equal(restored["_integration_state"], snapshot)
        self.assertTrue(restored["is_first"])
        self.assertEqual(env.env._step, 0)
        self.assertEqual(env.unwrapped._env._step_count, 0)
        for expected in reference:
            actual = env.step(np.array([0.05, 0.03]))
            np.testing.assert_array_equal(actual[0]["_integration_state"], expected[0]["_integration_state"])
            self.assertEqual(actual[1], expected[1])
        bad = first["_physics_state"].copy()
        bad[0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            env.restore_state(bad)
        bad[0] = 10
        with self.assertRaisesRegex(ValueError, "joint_limit"):
            env.restore_state(bad)
        bad = first["_physics_state"].copy()
        bad[3] = 201
        with self.assertRaisesRegex(ValueError, "velocity"):
            env.restore_state(bad)

    def _prepare(self, mode):
        from envs import make_env
        cfg = small_config("model.state_decoder.enabled=true", f"branch_collection.mode={mode}",
                           "branch_collection.horizon=4", "branch_collection.branches_per_round=1",
                           "branch_collection.max_attempts=2", "model.dynamic_trunc.min_horizon=1")
        agent = make_agent(cfg)
        # Keep this plumbing test away from deliberately aggressive random controls;
        # the production collector still uses the unmodified stochastic policy.
        original_act = agent.act

        def zero_control(obs, memory, eval=False, metrics=None):
            action, memory = original_act(obs, memory, eval=eval, metrics=metrics)
            action = torch.zeros_like(action)
            memory["prev_action"] = action
            return action, memory

        agent.act = zero_control
        env = make_env(cfg.env, 0, True, mode == "replay_state")
        self.addCleanup(env.close)
        normal, branch = Buffer(cfg.buffer), Buffer(cfg.buffer)
        collector = BranchCollector(cfg.branch_collection, cfg.env, "cpu", 1)
        self.addCleanup(collector.close)
        worker_env = BranchEnvironment(cfg.env, cfg.branch_collection, mode == "replay_state")
        collector.worker = LocalWorker(worker_env)
        obs = env.reset()
        state = obs["_physics_state"].copy()
        memory = agent.get_initial_state(1)
        for t in range(10):
            if t:
                obs, reward, done, _ = env.step(np.zeros(2))
            else:
                reward = 0.0
            row = collector._transition(obs, reward, t * 2, 0)
            add_source_metadata(row, 0, -1, t * 2, 0)
            action, memory = agent.act(row.clone(), memory)
            row["action"] = torch.zeros_like(action)
            row["stoch"], row["deter"] = memory["stoch"], memory["deter"]
            normal.add_transition(row.detach())
        if mode == "imagined_state":
            # Controlled VE scores produce an actual training cutoff at depth 1.
            agent.trunc_kappa.value.fill_(1.0)
            agent.trunc_kappa.initialized.fill_(True)
            scores = torch.tensor([0.0, 2.0, 0.0]).reshape(1, 3, 1).expand(16, -1, -1)
            with mock.patch.object(agent, "_imag_wasserstein", return_value=scores):
                agent.update(normal)
            collector.offer_candidates(agent.take_branch_candidates())
            # Known valid output verifies the real simulator collection plumbing.
            encoded = agent.state_decoder.encode(torch.tensor(state))
            with torch.no_grad():
                agent.state_decoder.net.last.weight.zero_()
                agent.state_decoder.net.last.bias.copy_(encoded)
        return cfg, agent, normal, branch, collector

    def test_real_and_imagined_collection_then_mixed_update(self):
        for mode in ("replay_state", "imagined_state"):
            with self.subTest(mode=mode):
                cfg, agent, normal, branch, collector = self._prepare(mode)
                before = torch.get_rng_state().clone()
                used, metrics = collector.collect(agent, normal, branch, 20000, budget=8)
                self.assertTrue(torch.equal(before, torch.get_rng_state()))
                self.assertEqual(used, 8)
                self.assertEqual(metrics["branch/successes"], 1, metrics)
                self.assertEqual(branch.count(), 5)
                self.assertEqual(collector.next_normal_step, 22000)
                rows = branch._buffer.storage[:]
                self.assertTrue(rows["is_first"][0].all())
                self.assertTrue(rows["is_last"][-1].all())
                self.assertFalse(rows["is_terminal"][-1].any())
                self.assertEqual(rows["reward"][0].item(), 0.0)
                self.assertTrue((rows["action"][-1] == 0).all())
                self.assertTrue((rows["_imag_depth"] == (1 if mode == "imagined_state" else 0)).all())
                mixed = MixedReplay(normal, branch, 0.25)
                metrics = agent.update(mixed)
                self.assertEqual(mixed.last_branch_fraction, 0.25)
                self.assertTrue(all(torch.isfinite(torch.as_tensor(v)).all() for v in metrics.values()))

    def test_failed_step_is_charged_and_not_stored(self):
        _, agent, normal, branch, collector = self._prepare("replay_state")
        collector.config.max_attempts = 1
        with mock.patch.object(collector.worker.env, "step", return_value=dict(ok=False, reason="physics_error", steps=2)):
            used, metrics = collector.collect(agent, normal, branch, 20000, budget=8)
        self.assertEqual(used, 2)
        self.assertEqual(metrics["branch/discarded_steps"], 2)
        self.assertEqual(branch.count(), 0)
        self.assertIsNone(collector.worker)

    def test_rejections_exhaust_attempts_without_spending_steps(self):
        _, agent, normal, branch, collector = self._prepare("imagined_state")
        with mock.patch.object(collector, "_candidate", return_value=(None, "angle_norm")):
            used, metrics = collector.collect(agent, normal, branch, 20000, budget=8)
        self.assertEqual(used, 0)
        self.assertEqual(metrics["branch/attempts"], 2)
        self.assertEqual(branch.count(), 0)

    def test_budget_shortens_branch_without_terminal_and_too_small_skips(self):
        _, agent, normal, branch, collector = self._prepare("replay_state")
        used, metrics = collector.collect(agent, normal, branch, 20000, budget=5)
        self.assertEqual(used, 4)
        self.assertEqual(branch.count(), 3)
        rows = branch._buffer.storage[:]
        self.assertTrue(rows["is_last"][-1].all())
        self.assertFalse(rows["is_terminal"][-1].any())
        used, metrics = collector.collect(agent, normal, branch, 22000, budget=1)
        self.assertEqual(used, 0)
        self.assertEqual(metrics["branch/attempts"], 0)

    def test_environment_time_limit_ends_branch_and_invalid_action_costs_zero(self):
        _, agent, normal, branch, collector = self._prepare("replay_state")
        collector.worker.env.env.env._duration = 1
        invalid = collector.worker.env.step(np.array([np.nan, 0]))
        self.assertEqual(invalid["steps"], 0)
        self.assertFalse(invalid["ok"])
        used, metrics = collector.collect(agent, normal, branch, 20000, budget=8)
        self.assertEqual(used, 2)
        self.assertEqual(metrics["branch/length"], 1)
        self.assertTrue(branch._buffer.storage[:]["is_last"][-1].all())
        self.assertFalse(branch._buffer.storage[:]["is_terminal"][-1].any())

    def test_trainer_counts_total_budget_and_trains_mixed_sequences(self):
        self._check_trainer("replay_state")

    def test_trainer_collects_from_training_cutoffs(self):
        self._check_trainer("imagined_state")

    def _check_trainer(self, mode):
        from envs import make_env
        from trainer import OnlineTrainer

        cfg, agent, _, _, collector = self._prepare(mode)
        cfg.branch_collection.warmup_steps = 12
        cfg.branch_collection.every_steps = 4
        collector.next_normal_step = 12
        cfg.env.train_ratio = 16
        cfg.env.steps = 40
        cfg.trainer.update_log_every = 4
        real_env = make_env(cfg.env, 1, True, mode == "replay_state")
        self.addCleanup(real_env.close)

        class NormalEnv:
            env_num = 1

            def step(self, action, done):
                if bool(done[0]):
                    obs, reward, ended = real_env.reset(), 0.0, False
                else:
                    obs, reward, ended, _ = real_env.step(action[0].numpy())
                row = collector._transition(obs, reward, 0, 0)
                return row, torch.tensor([ended])

        mixed = MixedReplay(Buffer(cfg.buffer), Buffer(cfg.buffer), 0.25)
        trainer = OnlineTrainer(cfg.trainer, mixed, mock.Mock(), None, NormalEnv(), None, collector)
        scores = torch.tensor([0.0, 2.0, 0.0]).reshape(1, 3, 1).expand(16, -1, -1)
        score_context = (mock.patch.object(agent, "_imag_wasserstein", return_value=scores)
                         if mode == "imagined_state" else contextlib.nullcontext())
        with score_context:
            trainer.begin(agent)
        self.assertEqual(trainer.total_steps, 40)
        self.assertEqual(trainer.total_steps, trainer.normal_steps + trainer.branch_steps)
        self.assertGreater(trainer.branch_steps, 0)
        self.assertEqual(mixed.last_branch_fraction, 0.25)
        if mode == "imagined_state":
            self.assertTrue((mixed.branch._buffer.storage[:]["_imag_depth"] == 1).all())


if __name__ == "__main__":
    unittest.main()
