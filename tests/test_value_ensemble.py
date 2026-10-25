"""Run with: python -m unittest discover -s tests -v."""

import contextlib
import io
import unittest
from pathlib import Path
from unittest import mock

import gymnasium as gym
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tensordict import TensorDict

from distributions import TwoHot
from dreamer import Dreamer
from networks import MLPHead


class ValueEnsembleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[1] / "configs")):
            cfg = compose(
                config_name="configs",
                overrides=[
                    "env=dmc_proprio",
                    "model.rep_loss=dreamer",
                    "device=cpu",
                    "model.compile=False",
                    "model.deter=32",
                    "model.hidden=16",
                    "model.units=16",
                    "model.discrete=4",
                    "model.rssm.stoch=4",
                    "model.rssm.blocks=4",
                    "model.imag_horizon=3",
                ],
            )
        cls.config = OmegaConf.to_container(cfg.model, resolve=True)
        cls.obs_space = gym.spaces.Dict({"state": gym.spaces.Box(-1, 1, (4,))})
        cls.act_space = gym.spaces.Box(-1, 1, (2,))

    def setUp(self):
        torch.manual_seed(17)

    def make_agent(self, size=5, outscale=0.0, trunc_enabled=False, min_horizon=8):
        config = OmegaConf.create(self.config)
        config.value_ensemble_size = size
        config.critic.outscale = outscale
        config.dynamic_trunc.enabled = trunc_enabled
        config.dynamic_trunc.min_horizon = min_horizon
        with contextlib.redirect_stdout(io.StringIO()):
            agent = Dreamer(config, self.obs_space, self.act_space).to("cpu")
        # CPU unit tests must not depend on whether this host exposes CUDA.
        agent._scaler = torch.amp.GradScaler("cpu", enabled=False)
        return agent

    def make_batch(self, agent):
        batch, length = 2, 4
        data = TensorDict(
            {
                "state": torch.randn(batch, length, 4),
                "action": torch.randn(batch, length, 2),
                "reward": torch.randn(batch, length, 1),
                "is_first": torch.zeros(batch, length, 1, dtype=torch.bool),
                "is_last": torch.zeros(batch, length, 1, dtype=torch.bool),
                "is_terminal": torch.zeros(batch, length, 1, dtype=torch.bool),
            },
            batch_size=(batch, length),
        )
        data["is_first"][:, 0] = True
        data["is_last"][0, 2] = True
        data["is_first"][0, 3] = True
        data["is_last"][1, -1] = True
        data["is_terminal"][1, -1] = True
        return data, agent.rssm.initial(batch)

    @staticmethod
    def grad_norm(module):
        return sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None)

    def assert_frozen_sharing(self, agent):
        for online, frozen in [(agent.value, agent._frozen_value), (agent._slow_value, agent._frozen_slow_value)]:
            for param, frozen_param in zip(online.parameters(), frozen.parameters()):
                self.assertEqual(param.data_ptr(), frozen_param.data_ptr())
                self.assertFalse(frozen_param.requires_grad)
                self.assertIsNone(frozen_param.grad)

    def test_independent_members_preserve_architecture_and_initialization(self):
        agent = self.make_agent()
        self.assertEqual(self.config["value_ensemble_size"], 5)
        self.assertTrue(self.config["dynamic_trunc"]["enabled"])
        self.assertEqual(len(agent.value), 5)
        reference = MLPHead(OmegaConf.create(self.config["critic"]), agent.rssm.feat_size)
        shapes = {name: param.shape for name, param in reference.named_parameters()}
        pointers = set()
        optimizer_ids = {id(p) for group in agent._optimizer.param_groups for p in group["params"]}
        for i, head in enumerate(agent.value):
            self.assertEqual({name: p.shape for name, p in head.named_parameters()}, shapes)
            self.assertEqual(torch.count_nonzero(head.last.weight).item(), 0)
            for param, slow in zip(head.parameters(), agent._slow_value[i].parameters()):
                self.assertNotIn(param.data_ptr(), pointers)
                pointers.add(param.data_ptr())
                self.assertIn(id(param), optimizer_ids)
                self.assertNotIn(id(slow), optimizer_ids)
                self.assertNotEqual(param.data_ptr(), slow.data_ptr())
                torch.testing.assert_close(param, slow)
            if i:
                self.assertFalse(torch.equal(head.mlp.layers[0].weight, agent.value[0].mlp.layers[0].weight))
        self.assert_frozen_sharing(agent)

    def test_invalid_member_counts(self):
        for size in [0, -2, 2.5, "5", True]:
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "value_ensemble_size"):
                self.make_agent(size=size)

    def test_single_head_requires_disabled_truncation(self):
        with self.assertRaisesRegex(ValueError, "Dynamic truncation requires at least 2"):
            self.make_agent(size=1, trunc_enabled=True)

    def test_single_head_bootstrap_and_finite_zero_disagreement(self):
        agent = self.make_agent(size=1, outscale=0.1)
        data, initial = self.make_batch(agent)
        predictions = []
        hook = agent._frozen_value[0].register_forward_hook(
            lambda module, args, dist: predictions.append(dist.mode().detach())
        )
        try:
            with (
                mock.patch("dreamer.torch.randperm", side_effect=AssertionError("Single head must not sample a pair")),
                mock.patch.object(agent, "_lambda_return", wraps=agent._lambda_return) as returns,
            ):
                _, metrics = agent._cal_grad(data, initial)
        finally:
            hook.remove()
        imag_args = returns.call_args_list[0].args
        torch.testing.assert_close(imag_args[3], predictions[0])
        torch.testing.assert_close(imag_args[4], predictions[0])
        torch.testing.assert_close(metrics["imag_wasser"], torch.zeros(3, 1))
        self.assertEqual(metrics["imag_wasser"].dtype, torch.float32)
        self.assertFalse(metrics["imag_wasser"].requires_grad)
        self.assertEqual(metrics["value_ensemble_std"].item(), 0)
        self.assertEqual(metrics["value_ensemble_std_replay"].item(), 0)
        self.assertTrue(all(torch.isfinite(torch.as_tensor(value)).all() for value in metrics.values()))
        self.assertFalse(any(key.startswith("dynamic_trunc/") for key in metrics))

    def test_loss_matches_categorical_targets_and_averages_member_gradients(self):
        agent = self.make_agent(outscale=0.1)
        feat = torch.randn(2, 4, agent.rssm.feat_size, requires_grad=True)
        target = torch.zeros(2, 4, 1, requires_grad=True)
        # Exact bin targets make the two-hot NLL an ordinary categorical cross entropy.
        bins = agent.value[0](feat).bins
        slow = torch.stack([bins[128 + i].expand_as(target) for i in range(len(agent.value))]).requires_grad_()
        weight = torch.tensor([[[1.0], [0.0], [0.3], [1.0]], [[0.5], [1.0], [1.0], [0.0]]])
        expected = []
        for i, head in enumerate(agent.value):
            logp = head.last(head.mlp(feat)).log_softmax(-1)
            nll = -logp[:, :-1, 127] - logp[:, :-1, 128 + i]
            expected.append((nll * weight[:, :-1, 0]).mean())
        expected = torch.stack(expected).mean()
        parameters = list(agent.value.parameters()) + [feat]
        expected_grads = torch.autograd.grad(expected, parameters)
        value_dists = [head(feat) for head in agent.value]
        loss = agent._value_loss(value_dists, target, slow, weight)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        for param, expected_grad in zip(parameters, expected_grads):
            torch.testing.assert_close(param.grad, expected_grad)
        self.assertIsNone(target.grad)
        self.assertIsNone(slow.grad)
        self.assertEqual(feat.grad[:, -1].abs().sum().item(), 0)
        self.assertEqual(feat.grad[0, 1].abs().sum().item(), 0)

    def test_wasserstein_identical_and_point_mass_distributions(self):
        agent = self.make_agent()
        bins = torch.tensor([-3.0, -1.0, 4.0])
        left = TwoHot(torch.tensor([1.0, 0.0, 0.0]).log().expand(2, 4, -1), bins)
        right = TwoHot(torch.tensor([0.0, 0.0, 1.0]).log().expand(2, 4, -1), bins)
        torch.testing.assert_close(agent._imag_wasserstein([left] * 5), torch.zeros(2, 3, 1))
        torch.testing.assert_close(agent._imag_wasserstein([left, right]), torch.full((2, 3, 1), 7.0))

    def test_wasserstein_uses_full_distribution_and_nonuniform_spacing(self):
        agent = self.make_agent()
        # Moving half the mass from -3 to 4 costs 0.5 * 7, regardless of the middle bin.
        bins = torch.tensor([-3.0, -1.0, 4.0])
        left = TwoHot(torch.tensor([0.5, 0.5, 0.0]).log().expand(2, 4, -1), bins)
        right = TwoHot(torch.tensor([0.0, 0.5, 0.5]).log().expand(2, 4, -1), bins)
        torch.testing.assert_close(agent._imag_wasserstein([left, right]), torch.full((2, 3, 1), 3.5))
        # Equal means do not imply equal distributions: each half travels a distance of 2.
        bins = torch.tensor([-2.0, 0.0, 2.0])
        spread = TwoHot(torch.tensor([0.5, 0.0, 0.5]).log().expand(2, 4, -1), bins)
        center = TwoHot(torch.tensor([0.0, 1.0, 0.0]).log().expand(2, 4, -1), bins)
        torch.testing.assert_close(spread.mode(), center.mode())
        torch.testing.assert_close(agent._imag_wasserstein([spread, center]), torch.full((2, 3, 1), 2.0))

    def test_wasserstein_averages_all_unordered_pairs(self):
        agent = self.make_agent()
        bins = torch.tensor([-3.0, -1.0, 0.0, 4.0, 10.0])
        dists = [TwoHot(prob.log().expand(2, 4, -1), bins) for prob in torch.eye(5)]
        # The ten distances are 2, 3, 7, 13, 1, 5, 11, 4, 10, 6; their sum is 62.
        torch.testing.assert_close(agent._imag_wasserstein(dists), torch.full((2, 3, 1), 6.2))
        permuted = [dists[i] for i in [3, 0, 4, 1, 2]]
        torch.testing.assert_close(agent._imag_wasserstein(permuted), torch.full((2, 3, 1), 6.2))

    def test_wasserstein_averages_states_and_excludes_bootstrap(self):
        agent = self.make_agent()
        bins = torch.tensor([-1.0, 3.0])
        left = TwoHot(torch.tensor([1.0, 0.0]).log().expand(2, 3, -1), bins)
        probabilities = torch.tensor([[[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0]] * 3])
        right = TwoHot(probabilities.log(), bins)
        # Distances retain both the trajectory and time axes; only head pairs are averaged.
        expected = torch.tensor([[[4.0], [0.0]], [[4.0], [4.0]]])
        torch.testing.assert_close(agent._imag_wasserstein([left, right]), expected)
        torch.testing.assert_close(expected.mean(dim=0), torch.tensor([[4.0], [2.0]]))
        probabilities[:, -1] = torch.tensor([1.0, 0.0])
        right = TwoHot(probabilities.log(), bins)
        torch.testing.assert_close(agent._imag_wasserstein([left, right]), expected)

    def test_wasserstein_is_detached_float32_under_autocast(self):
        agent = self.make_agent()
        # This spacing overflows float16 unless bins are promoted before taking their difference.
        bins = torch.tensor([-60000.0, 60000.0], dtype=torch.float16)
        left_logits = torch.tensor([0.0, -torch.inf], dtype=torch.float16, requires_grad=True)
        right_logits = torch.tensor([-torch.inf, 0.0], dtype=torch.float16, requires_grad=True)
        with torch.autocast(device_type="cpu", dtype=torch.float16):
            dists = [TwoHot(logits.expand(2, 4, -1), bins) for logits in [left_logits, right_logits]]
            metric = agent._imag_wasserstein(dists)
        self.assertEqual(metric.dtype, torch.float32)
        self.assertEqual(metric.shape, (2, 3, 1))
        self.assertFalse(metric.requires_grad)
        torch.testing.assert_close(metric, torch.full((2, 3, 1), 120000.0))
        self.assertIsNone(left_logits.grad)
        self.assertIsNone(right_logits.grad)

    def test_wasserstein_logging_preserves_losses_and_gradients(self):
        agent = self.make_agent(outscale=0.1)
        data, initial = self.make_batch(agent)
        state = {name: value.clone() for name, value in agent.state_dict().items()}
        torch.manual_seed(29)
        with mock.patch.object(agent, "_imag_wasserstein", return_value=torch.zeros(8, 3, 1)):
            _, reference = agent._cal_grad(data, initial)
        losses = {key: value.detach().clone() for key, value in reference.items() if "loss" in key}
        grads = {name: None if p.grad is None else p.grad.clone() for name, p in agent._named_params.items()}
        agent.load_state_dict(state)
        agent.zero_grad(set_to_none=True)
        torch.manual_seed(29)
        with mock.patch.object(agent, "_value_loss", wraps=agent._value_loss) as value_loss:
            _, metrics = agent._cal_grad(data, initial)
        self.assertEqual(value_loss.call_count, 2)
        imag_dists = value_loss.call_args_list[0].args[0]
        self.assertEqual(len(imag_dists), 5)
        torch.testing.assert_close(metrics["imag_wasser"], agent._imag_wasserstein(imag_dists).mean(dim=0))
        self.assertEqual(metrics["imag_wasser"].shape, (3, 1))
        self.assertFalse(metrics["imag_wasser"].requires_grad)
        self.assertTrue(torch.isfinite(metrics["imag_wasser"]).all())
        self.assertGreater(metrics["imag_wasser"].mean().item(), 0)
        self.assertNotIn("img_wasser", metrics)
        for key, value in losses.items():
            torch.testing.assert_close(metrics[key], value, rtol=0, atol=0)
        for name, param in agent._named_params.items():
            if grads[name] is None:
                self.assertIsNone(param.grad)
            else:
                torch.testing.assert_close(param.grad, grads[name], rtol=0, atol=0)

    def test_loss_gradient_boundaries(self):
        for size, active in [(size, loss) for size in (1, 5) for loss in ("policy", "value", "repval")]:
            with self.subTest(size=size, loss=active):
                # Nonzero output weights exercise upstream gradients beyond the initial zero-output step.
                agent = self.make_agent(size=size, outscale=0.1)
                agent._loss_scales = {key: float(key == active) for key in agent._loss_scales}
                data, initial = self.make_batch(agent)
                agent._cal_grad(data, initial)
                self.assertEqual(self.grad_norm(agent.actor) > 0, active == "policy")
                for head in agent.value:
                    self.assertEqual(self.grad_norm(head) > 0, active in ["value", "repval"])
                self.assertEqual(self.grad_norm(agent.encoder) > 0, active == "repval")
                self.assertEqual(self.grad_norm(agent.rssm._obs_net) > 0, active == "repval")
                self.assertEqual(self.grad_norm(agent.rssm._deter_net) > 0, active == "repval")
                for module in [agent.rssm._img_net, agent.reward, agent.cont, agent.decoder, agent._slow_value]:
                    self.assertEqual(self.grad_norm(module), 0)
                self.assert_frozen_sharing(agent)

    def test_dynamic_threshold_curve_masks_and_return_normalization(self):
        agent = self.make_agent(outscale=0.1, trunc_enabled=True, min_horizon=1)
        agent.dynamic_trunc_quantile = 0.5
        agent.dynamic_trunc_threshold_scale = 2.0
        data, initial = self.make_batch(agent)
        scores = torch.arange(1, 9, dtype=torch.float32)[:, None, None] + torch.tensor([0.0, 10.0, 20.0])[None, :, None]
        scores[0, 1:] = 0
        expected_mask = torch.zeros(8, 3, 1, dtype=torch.bool)
        expected_mask[:, 0] = True
        expected_mask[0] = True
        with (
            mock.patch.object(agent, "_imag_wasserstein", return_value=scores),
            mock.patch.object(agent.trunc_kappa, "forward", wraps=agent.trunc_kappa.forward) as threshold_call,
            mock.patch.object(agent, "_value_loss", wraps=agent._value_loss) as value_loss,
            mock.patch.object(agent, "_lambda_return", wraps=agent._lambda_return) as returns,
        ):
            _, metrics = agent._cal_grad(data, initial)
        threshold_call.assert_called_once()
        torch.testing.assert_close(metrics["dynamic_trunc/threshold_raw"], torch.tensor(9.0))
        torch.testing.assert_close(metrics["dynamic_trunc/threshold"], torch.tensor(9.0))
        torch.testing.assert_close(metrics["imag_wasser"], torch.tensor([[4.5], [13.125], [21.875]]))
        torch.testing.assert_close(metrics["dynamic_trunc/keep_rate"], expected_mask.float().mean())
        torch.testing.assert_close(metrics["dynamic_trunc/effective_horizon"], torch.tensor(1.25))
        imag_loss_args = value_loss.call_args_list[0].args
        loss_mask = imag_loss_args[4]
        torch.testing.assert_close(loss_mask, expected_mask.float() * 2.4)
        self.assertEqual(len(value_loss.call_args_list[1].args), 4)  # Replay does not receive the imagination mask.
        imag_ret = imag_loss_args[1][:, :-1]
        imag_value = returns.call_args_list[0].args[3]
        stats = torch.where(expected_mask, imag_ret, imag_value[:, :-1])
        expected_ema = agent.return_ema.alpha * torch.quantile(stats.flatten(), agent.return_ema.range)
        torch.testing.assert_close(agent.return_ema.ema_vals, expected_ema)
        agent.zero_grad(set_to_none=True)
        with mock.patch.object(agent, "_imag_wasserstein", return_value=scores * 2):
            _, next_metrics = agent._cal_grad(data, initial)
        torch.testing.assert_close(next_metrics["dynamic_trunc/threshold_raw"], torch.tensor(18.0))
        torch.testing.assert_close(next_metrics["dynamic_trunc/threshold"], torch.tensor(9.0))
        torch.testing.assert_close(agent.trunc_kappa.value, torch.tensor(9.009))
        self.assertFalse(agent.trunc_kappa.value.requires_grad)

    def test_dynamic_gradient_boundaries_on_a_partial_prefix(self):
        for active in ["policy", "value", "repval"]:
            with self.subTest(loss=active):
                agent = self.make_agent(outscale=0.1, trunc_enabled=True, min_horizon=1)
                agent._loss_scales = {key: float(key == active) for key in agent._loss_scales}
                data, initial = self.make_batch(agent)
                scores = torch.tensor([0.0, 2.0, 0.0]).reshape(1, 3, 1).expand(8, -1, -1)
                with mock.patch.object(agent, "_imag_wasserstein", return_value=scores):
                    agent._cal_grad(data, initial)
                self.assertEqual(self.grad_norm(agent.actor) > 0, active == "policy")
                for head in agent.value:
                    self.assertEqual(self.grad_norm(head) > 0, active in ["value", "repval"])
                for module in [agent.encoder, agent.rssm._obs_net, agent.rssm._deter_net]:
                    self.assertEqual(self.grad_norm(module) > 0, active == "repval")
                for module in [agent.rssm._img_net, agent.reward, agent.cont, agent.decoder, agent._slow_value]:
                    self.assertEqual(self.grad_norm(module), 0)
                self.assert_frozen_sharing(agent)

    def test_empty_mask_removes_imagined_losses_but_keeps_replay_learning(self):
        agent = self.make_agent(outscale=0.1, trunc_enabled=True, min_horizon=0)
        agent.dynamic_trunc_threshold_scale = 0.5
        agent._loss_scales = {key: float(key == "repval") for key in agent._loss_scales}
        data, initial = self.make_batch(agent)
        with mock.patch.object(agent, "_imag_wasserstein", return_value=torch.ones(8, 3, 1)):
            _, metrics = agent._cal_grad(data, initial)
        self.assertEqual(metrics["dynamic_trunc/keep_rate"].item(), 0)
        self.assertEqual(metrics["loss/policy"].item(), 0)
        self.assertEqual(metrics["loss/value"].item(), 0)
        self.assertGreater(metrics["loss/repval"].item(), 0)
        self.assertGreater(self.grad_norm(agent.encoder), 0)
        for head in agent.value:
            self.assertGreater(self.grad_norm(head), 0)
        self.assertEqual(self.grad_norm(agent.actor), 0)

    def test_value_mask_blocks_each_members_feature_gradients(self):
        agent = self.make_agent(outscale=0.1)
        feat = torch.randn(2, 4, agent.rssm.feat_size, requires_grad=True)
        target = torch.zeros(2, 4, 1)
        slow = torch.zeros(5, 2, 4, 1)
        mask = torch.tensor([[[1.0], [0.0], [0.0]], [[1.0], [1.0], [0.0]]])
        dists = [head(feat) for head in agent.value]
        agent._value_loss(dists, target, slow, torch.ones_like(target), mask).backward()
        self.assertEqual(feat.grad[0, 1:].abs().sum().item(), 0)
        self.assertEqual(feat.grad[1, 2:].abs().sum().item(), 0)
        self.assertGreater(feat.grad[0, 0].abs().sum().item(), 0)
        for head in agent.value:
            self.assertGreater(self.grad_norm(head), 0)

    def test_zero_initialized_heads_keep_full_horizon(self):
        agent = self.make_agent(trunc_enabled=True, min_horizon=1)
        data, initial = self.make_batch(agent)
        _, metrics = agent._cal_grad(data, initial)
        torch.testing.assert_close(metrics["imag_wasser"], torch.zeros(3, 1))
        self.assertEqual(metrics["dynamic_trunc/threshold"].item(), 0)
        self.assertEqual(metrics["dynamic_trunc/keep_rate"].item(), 1)
        self.assertEqual(metrics["dynamic_trunc/effective_horizon"].item(), 3)

    def test_one_random_pair_is_shared_by_bootstrap_and_advantage(self):
        agent = self.make_agent(outscale=0.1)
        data, initial = self.make_batch(agent)
        predictions = [[] for _ in agent.value]
        hooks = [
            head.register_forward_hook(lambda module, args, dist, i=i: predictions[i].append(dist.mode().detach()))
            for i, head in enumerate(agent._frozen_value)
        ]
        returns = []
        original_return = agent._lambda_return

        def record_return(*args):
            result = original_return(*args)
            returns.append((args, result))
            return result

        try:
            with (
                mock.patch("dreamer.torch.randperm", return_value=torch.tensor([4, 1, 3, 0, 2])) as sample_pair,
                mock.patch.object(agent, "_lambda_return", side_effect=record_return),
            ):
                _, metrics = agent._cal_grad(data, initial)
            sample_pair.assert_called_once_with(5, device=torch.device("cpu"))
        finally:
            for hook in hooks:
                hook.remove()
        expected_value = (predictions[4][0] + predictions[1][0]) / 2
        imag_args, imag_return = returns[0]
        torch.testing.assert_close(imag_args[3], expected_value)
        torch.testing.assert_close(imag_args[4], expected_value)
        scale = (agent.return_ema.ema_vals[1] - agent.return_ema.ema_vals[0]).clamp(min=1)
        torch.testing.assert_close(metrics["adv"], ((imag_return - expected_value[:, :-1]) / scale).mean())
        # Every replay sequence position bootstraps from its corresponding imagination return.
        torch.testing.assert_close(returns[1][0][4], imag_return[:, 0].reshape(2, 4, 1))

    def test_slow_targets_update_matching_members_and_keep_frozen_sharing(self):
        agent = self.make_agent()
        agent.slow_target_update = 2
        before = [[p.detach().clone() for p in head.parameters()] for head in agent._slow_value]
        with torch.no_grad():
            for i, head in enumerate(agent.value):
                for param in head.parameters():
                    param.add_(i + 1)
        agent._update_slow_target()
        for i, slow in enumerate(agent._slow_value):
            for old, param in zip(before[i], slow.parameters()):
                torch.testing.assert_close(param, old + agent.slow_target_fraction * (i + 1))
        before_skip = [p.detach().clone() for p in agent._slow_value.parameters()]
        agent._update_slow_target()
        for old, param in zip(before_skip, agent._slow_value.parameters()):
            torch.testing.assert_close(param, old)
        agent.train()
        self.assertTrue(all(head.training for head in agent.value))
        self.assertTrue(all(not head.training for head in agent._slow_value))
        self.assert_frozen_sharing(agent.to("cpu"))

    def test_update_and_checkpoint_roundtrip(self):
        for size in (1, 2):
            with self.subTest(size=size):
                self._check_update_and_checkpoint_roundtrip(size)

    def _check_update_and_checkpoint_roundtrip(self, size):
        agent = self.make_agent(size=size, trunc_enabled=size > 1)
        data, initial = self.make_batch(agent)
        replay = mock.Mock()
        replay.sample.side_effect = lambda: (data.clone(), "indices", initial)
        for _ in range(2):
            metrics = agent.update(replay)
            self.assertTrue(all(torch.isfinite(torch.as_tensor(value)).all() for value in metrics.values()))
        self.assertEqual(replay.update.call_count, 2)
        for head in agent.value:
            self.assertGreater(head.last.weight.abs().sum().item(), 0)
            for param in head.parameters():
                self.assertIsNone(param.grad)
                self.assertEqual(agent._optimizer.state[param]["step"], 2)
        for tensor in replay.update.call_args.args[1:]:
            self.assertFalse(tensor.requires_grad)
        checkpoint = io.BytesIO()
        torch.save({"agent": agent.state_dict(), "optimizer": agent._optimizer.state_dict()}, checkpoint)
        checkpoint.seek(0)
        state = torch.load(checkpoint, weights_only=True)
        restored = self.make_agent(size=size, trunc_enabled=size > 1)
        restored.load_state_dict(state["agent"])
        restored._optimizer.load_state_dict(state["optimizer"])
        for name, value in restored.state_dict().items():
            torch.testing.assert_close(value, agent.state_dict()[name])
        for param in restored.value.parameters():
            self.assertEqual(restored._optimizer.state[param]["step"], 2)
        self.assert_frozen_sharing(restored)
        if size > 1:
            self.assertTrue(restored.trunc_kappa.initialized)
        else:
            self.assertIsNone(restored.trunc_kappa)


if __name__ == "__main__":
    unittest.main()
