"""CPU contracts for categorical + quantile value ensembles."""

import contextlib
import io
import math
import unittest
from unittest import mock

import test_value_ensemble as legacy
import torch
from omegaconf import OmegaConf

from distributions import Quantile, TwoHot
from dreamer import Dreamer
from truncation import cal_ens_uncertainty


def reference_w1(cat_atoms, cat_probs, quantiles, space):
    """Independent scalar CDF integration, including repeated support locations."""
    transform = (lambda x: math.copysign(math.log1p(abs(x)), x)) if space == "symlog" else float
    a = [transform(x) for x in cat_atoms]
    b = [transform(x) for x in quantiles]
    points = sorted(set(a + b))
    total = 0.0
    for left, right in zip(points, points[1:]):
        midpoint = (left + right) / 2
        cdf_a = math.fsum(p for x, p in zip(a, cat_probs) if x <= midpoint)
        cdf_b = sum(x <= midpoint for x in b) / len(b)
        total += abs(cdf_a - cdf_b) * (right - left)
    return total


class QuantileDistributionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_huber_matches_scalar_reference_and_gradients(self):
        for kappa in (0.25, 1.0, 3.0):
            atoms = torch.tensor([[2.0, -3.0, 0.2], [-0.2, 1.0, 4.0]], requires_grad=True)
            target = torch.tensor([[0.5], [-1.0]], requires_grad=True)
            dist = Quantile(atoms, kappa)
            expected, gradients = [], []
            for row, y in zip(atoms.detach().tolist(), target.detach().flatten().tolist()):
                losses, grads = [], []
                for i, value in enumerate(row):
                    delta = y - value
                    weight = abs((i + 0.5) / len(row) - float(delta < 0))
                    huber = 0.5 * delta**2 if abs(delta) <= kappa else kappa * (abs(delta) - 0.5 * kappa)
                    losses.append(weight * huber / kappa)
                    grads.append(-weight * max(-kappa, min(kappa, delta)) / (kappa * len(row)))
                expected.append(sum(losses) / len(row))
                gradients.append(grads)
            actual = dist.quantile_huber_loss(target)
            torch.testing.assert_close(actual, torch.tensor(expected))
            torch.testing.assert_close(dist.log_prob(target), -actual)
            actual.sum().backward()
            torch.testing.assert_close(atoms.grad, torch.tensor(gradients))
            self.assertIsNone(target.grad)
            torch.testing.assert_close(dist.mode(), atoms.mean(-1, keepdim=True))
            torch.testing.assert_close(dist.probs.sum(-1), torch.ones(2))

    def test_zero_residual_and_float32_under_autocast(self):
        atoms = torch.tensor([[60000.0, -60000.0]], dtype=torch.float16, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.float16):
            dist = Quantile(atoms)
            loss = dist.quantile_huber_loss(torch.tensor([[0.0]]))
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss).all())
        loss.sum().backward()
        self.assertTrue(torch.isfinite(atoms.grad).all())
        same = torch.ones(2, 3, 7, requires_grad=True)
        zero = Quantile(same).quantile_huber_loss(torch.ones(2, 3, 1))
        zero.sum().backward()
        torch.testing.assert_close(zero, torch.zeros(2, 3))
        torch.testing.assert_close(same.grad, torch.zeros_like(same))
        large = Quantile(torch.zeros(1, 2)).log_prob(torch.tensor([[1e20]]))
        self.assertTrue(torch.isfinite(large).all())

    def test_invalid_distribution_inputs(self):
        for kappa in (0, -1, float("inf"), float("nan")):
            with self.subTest(kappa=kappa), self.assertRaisesRegex(ValueError, "huber_kappa"):
                Quantile(torch.zeros(2, 3), kappa)
        for value in (torch.zeros(()), torch.zeros(2, 0)):
            with self.assertRaisesRegex(ValueError, "atom"):
                Quantile(value)
        with self.assertRaisesRegex(ValueError, "target"):
            Quantile(torch.zeros(2, 3)).log_prob(torch.zeros(2))

    def test_huber_keeps_nominal_quantile_order(self):
        target = torch.tensor([[4.0]])
        ordered = Quantile(torch.tensor([[-1.0, 2.0]]))
        crossed = Quantile(torch.tensor([[2.0, -1.0]]))
        self.assertNotEqual(ordered.log_prob(target).item(), crossed.log_prob(target).item())
        torch.testing.assert_close(ordered.mode(), crossed.mode())

    def test_mixed_w1_matches_reference_both_spaces(self):
        torch.manual_seed(8)
        bins = torch.tensor([-100.0, -0.5, 0.0, 2.0, 90.0])
        logits = torch.randn(3, 4, 5, requires_grad=True)
        atoms = (torch.randn(3, 4, 7) * 100).requires_grad_()
        cat, q = TwoHot(logits, bins), Quantile(atoms)
        before = atoms.detach().clone()
        for space in ("raw", "symlog"):
            expected = torch.empty(3, 3, 1)
            for b in range(3):
                for t in range(3):
                    expected[b, t, 0] = reference_w1(
                        bins.tolist(), logits[b, t].double().softmax(-1).tolist(), atoms[b, t].tolist(), space
                    )
            actual = cal_ens_uncertainty([cat, q], None, "wasser", distance_space=space)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(actual, cal_ens_uncertainty([q, cat], None, "wasser", distance_space=space))
            self.assertFalse(actual.requires_grad)
            self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(atoms, before)
        self.assertIsNone(logits.grad)
        self.assertIsNone(atoms.grad)

    def test_equal_means_different_laws_and_bootstrap_exclusion(self):
        logits = torch.zeros(2, 3, 2)
        cat = TwoHot(logits, torch.tensor([-1.0, 1.0]))
        q = Quantile(torch.zeros(2, 3, 3))
        torch.testing.assert_close(cat.mode(), q.mode())
        torch.testing.assert_close(cal_ens_uncertainty([cat, q], None, "wasser"), torch.ones(2, 2, 1))
        q.quantiles[:, -1] = 1e6
        torch.testing.assert_close(cal_ens_uncertainty([cat, q], None, "wasser"), torch.ones(2, 2, 1))
        identical = Quantile(torch.tensor([1.0, -1.0, 1.0, -1.0]).expand(2, 3, -1))
        torch.testing.assert_close(cal_ens_uncertainty([cat, identical], None, "wasser"), torch.zeros(2, 2, 1))

    def test_empty_tails_and_chunk_boundary(self):
        bins = torch.tensor([-485165184.0, -1.0, 0.0, 1.0, 485165184.0], requires_grad=True)
        logits = torch.tensor([-torch.inf, 1.9940087795, -8.35092735, 3.40969348, -torch.inf], requires_grad=True)
        # 1025*2 rows crosses the internal 2048-row working-memory boundary.
        cat = TwoHot(logits.expand(1025, 3, -1), bins)
        q = Quantile(torch.zeros(1025, 3, 7, dtype=torch.float16, requires_grad=True))
        probabilities = logits.detach().double().softmax(-1)
        for space in ("raw", "symlog"):
            with torch.autocast("cpu", dtype=torch.float16):
                actual = cal_ens_uncertainty([cat, q], None, "wasser", distance_space=space)
            expected = float(probabilities[1] + probabilities[3]) * (math.log(2) if space == "symlog" else 1)
            torch.testing.assert_close(actual, torch.full((1025, 2, 1), expected), atol=1e-7, rtol=1e-7)
            self.assertFalse(actual.requires_grad)
        self.assertIsNone(logits.grad)
        self.assertIsNone(bins.grad)

    def test_mixed_w1_rejects_invalid_inputs(self):
        c = TwoHot(torch.zeros(1, 3, 2), torch.tensor([-1.0, 1.0]))
        q = Quantile(torch.zeros(1, 3, 3))
        for metric in ("cramer", "sqrt_cramer", "sigma", "ucb"):
            with self.assertRaisesRegex(ValueError, "wasser"):
                cal_ens_uncertainty([c, q], None, metric)
        with self.assertRaisesRegex(ValueError, "exactly"):
            cal_ens_uncertainty([q, q], None, "wasser")
        with self.assertRaisesRegex(ValueError, "distance_space"):
            cal_ens_uncertainty([c, q], None, "wasser", distance_space="bad")
        with self.assertRaisesRegex(ValueError, "batch shapes"):
            cal_ens_uncertainty([c, Quantile(torch.zeros(2, 3, 3))], None, "wasser")

    def test_compile_fullgraph_distance_and_huber(self):
        def compute(logits, atoms, target):
            cat = TwoHot(logits, torch.tensor([-2.0, 0.0, 3.0], device=logits.device))
            q = Quantile(atoms, 0.5)
            return cal_ens_uncertainty([cat, q], None, "wasser", distance_space="symlog"), q.log_prob(target)

        torch.manual_seed(19)
        args = (torch.randn(2, 4, 3), torch.randn(2, 4, 5, requires_grad=True), torch.randn(2, 4, 1))
        compiled = torch.compile(compute, backend="eager", fullgraph=True)
        actual, expected = compiled(*args), compute(*args)
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e)
        torch.testing.assert_close(
            torch.autograd.grad(actual[1].sum(), args[1])[0], torch.autograd.grad(expected[1].sum(), args[1])[0]
        )


class MixedValueEnsembleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        legacy.ValueEnsembleTest.setUpClass()
        cls.config = legacy.ValueEnsembleTest.config
        cls.obs_space = legacy.ValueEnsembleTest.obs_space
        cls.act_space = legacy.ValueEnsembleTest.act_space

    def setUp(self):
        torch.manual_seed(17)

    make_batch = legacy.ValueEnsembleTest.make_batch
    grad_norm = staticmethod(legacy.ValueEnsembleTest.grad_norm)
    assert_frozen_sharing = legacy.ValueEnsembleTest.assert_frozen_sharing

    def make_agent(self, mode="categorical_quantile", *, overrides=None):
        config = OmegaConf.create(self.config)
        config.value_ensemble_type = mode
        config.critic.outscale = 0.1
        config.quantile.num_quantiles = 7
        config.dynamic_trunc.enabled = False
        config.dynamic_trunc.warmup_steps = 0
        for key, value in (overrides or {}).items():
            OmegaConf.update(config, key, value)
        with contextlib.redirect_stdout(io.StringIO()):
            agent = Dreamer(config, self.obs_space, self.act_space).to("cpu")
        agent._scaler = torch.amp.GradScaler("cpu", enabled=False)
        return agent

    def test_topology_and_initialization(self):
        agent = self.make_agent(
            overrides={"value_ensemble_size": 99, "critic.outscale": 0.0, "quantile.num_quantiles": 255}
        )
        self.assertEqual(agent.value_ensemble_size, 2)
        self.assertTrue(agent.quantile_repval_grad)
        self.assertEqual(len(agent.value), 2)
        self.assertEqual(agent.value[0].last.out_features, 255)
        self.assertEqual(agent.value[1].last.out_features, 255)
        x = torch.randn(2, 4, agent.rssm.feat_size)
        c, q = [head(x) for head in agent.value]
        self.assertIsInstance(c, TwoHot)
        self.assertIsInstance(q, Quantile)
        torch.testing.assert_close(c.mode(), torch.zeros(2, 4, 1))
        torch.testing.assert_close(q.mode(), torch.zeros(2, 4, 1))
        self.assertTrue((cal_ens_uncertainty([c, q], None, "wasser") > 0).all())
        self.assert_frozen_sharing(agent)
        optimizer_ids = {id(p) for group in agent._optimizer.param_groups for p in group["params"]}
        for online, slow in zip(agent.value.parameters(), agent._slow_value.parameters()):
            self.assertIn(id(online), optimizer_ids)
            self.assertNotIn(id(slow), optimizer_ids)
            self.assertNotEqual(online.data_ptr(), slow.data_ptr())
            torch.testing.assert_close(online, slow)

    def test_invalid_configuration(self):
        for overrides in (
            {"uncer_metric": "cramer"},
            {"quantile.num_quantiles": 0},
            {"quantile.num_quantiles": True},
            {"quantile.num_quantiles": 1.5},
            {"quantile.huber_kappa": -1},
            {"quantile.huber_kappa": float("nan")},
            {"quantile.repval_grad": "false"},
            {"critic.dist.name": "identity"},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.make_agent(overrides=overrides)
        with self.assertRaisesRegex(ValueError, "value_ensemble_type"):
            self.make_agent(mode="bad")
        # N is categorical-only: a mixed ensemble remains valid when N=1.
        self.assertEqual(self.make_agent(overrides={"value_ensemble_size": 1}).value_ensemble_size, 2)

    def test_hydra_interpolations_and_categorical_config_are_preserved(self):
        root = OmegaConf.create({"device": "cpu", "model": self.config})
        root.model.value_ensemble_type = "categorical_quantile"
        root.model.quantile.num_quantiles = 9
        root.model.critic.units = "${model.units}"
        root.model.critic.device = "${device}"
        OmegaConf.set_struct(root, True)
        with contextlib.redirect_stdout(io.StringIO()):
            agent = Dreamer(root.model, self.obs_space, self.act_space).to("cpu")
        self.assertEqual(agent.value[1].last.out_features, 9)
        self.assertEqual(agent.value[1].mlp.out_dim, root.model.units)
        self.assertEqual(root.model.critic.dist.name, "symexp_twohot")
        self.assertEqual(root.model.critic.shape[0], 255)

    def test_mixture_means_used_for_both_return_paths(self):
        agent = self.make_agent()
        data, initial = self.make_batch(agent)
        predictions = [[], []]
        hooks = [
            head.register_forward_hook(lambda module, args, dist, i=i: predictions[i].append(dist.mode().detach()))
            for i, head in enumerate(agent._frozen_value)
        ]
        try:
            with mock.patch(
                "dreamer.torch.randperm", side_effect=AssertionError("Mixed mode must not sample heads")
            ), mock.patch.object(agent, "_lambda_return", wraps=agent._lambda_return) as returns:
                _, metrics = agent._cal_grad(data, initial)
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(returns.call_count, 2)
        for call_index in (0, 1):
            expected = (predictions[0][call_index] + predictions[1][call_index]) / 2
            torch.testing.assert_close(returns.call_args_list[call_index].args[3], expected)
        torch.testing.assert_close(returns.call_args_list[0].args[4], (predictions[0][0] + predictions[1][0]) / 2)
        for loss in ("value", "repval"):
            torch.testing.assert_close(
                metrics[f"loss/{loss}"], (metrics[f"loss/{loss}_categorical"] + metrics[f"loss/{loss}_quantile"]) / 2
            )
        self.assertEqual(metrics["value_ensemble_size"].item(), 2)

    def test_mixed_loss_targets_masks_and_gradient_boundaries(self):
        agent = self.make_agent()
        feat = torch.randn(2, 4, agent.rssm.feat_size, requires_grad=True)
        target = torch.randn(2, 4, 1, requires_grad=True)
        slow = torch.randn(2, 2, 4, 1, requires_grad=True)
        weight = torch.ones(2, 4, 1, requires_grad=True)
        mask = torch.tensor([1.0, 0.0, 1.0]).reshape(1, 3, 1)
        dists = agent._replay_value_dist(feat)
        expected = (
            sum(((-d.log_prob(target) - d.log_prob(slow[i]))[:, :-1, None] * mask).mean() for i, d in enumerate(dists))
            / 2
        )
        loss = agent._value_loss(dists, target, slow, weight, mask)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertIsNone(target.grad)
        self.assertIsNone(slow.grad)
        self.assertIsNone(weight.grad)
        torch.testing.assert_close(feat.grad[:, -1], torch.zeros_like(feat.grad[:, -1]))
        torch.testing.assert_close(feat.grad[:, 1], torch.zeros_like(feat.grad[:, 1]))
        self.assertGreater(feat.grad[:, 0].abs().sum().item(), 0)

    def test_quantile_repval_input_detach_preserves_head_learning(self):
        for enabled in (False, True):
            agent = self.make_agent(overrides={"quantile.repval_grad": enabled})
            for index in (0, 1):
                agent.zero_grad(set_to_none=True)
                feat = torch.randn(2, 4, agent.rssm.feat_size, requires_grad=True)
                dist = agent._replay_value_dist(feat)[index]
                (-dist.log_prob(torch.ones(2, 4, 1))).mean().backward()
                self.assertGreater(self.grad_norm(agent.value[index]), 0)
                should_flow = index == 0 or enabled
                self.assertEqual(feat.grad is not None and bool(feat.grad.abs().sum() > 0), should_flow)

    def test_quantile_only_repval_world_model_gradients(self):
        class ZeroLoss:
            def __init__(self, dist):
                self.dist = dist

            def log_prob(self, target):
                return self.dist.log_prob(target) * 0

        for enabled in (False, True):
            agent = self.make_agent(overrides={"quantile.repval_grad": enabled})
            agent._loss_scales = {key: float(key == "repval") for key in agent._loss_scales}
            original = agent._value_loss

            def quantile_only(dists, *args, **kwargs):
                if kwargs.get("prefix") == "repval":
                    dists = [ZeroLoss(dists[0]), dists[1]]
                return original(dists, *args, **kwargs)

            data, initial = self.make_batch(agent)
            with mock.patch.object(agent, "_value_loss", side_effect=quantile_only):
                agent._cal_grad(data, initial)
            self.assertGreater(self.grad_norm(agent.value[1]), 0)
            self.assertEqual(self.grad_norm(agent.value[0]), 0)
            self.assertEqual(self.grad_norm(agent.encoder) > 0, enabled)
            self.assertEqual(self.grad_norm(agent.rssm._obs_net) > 0, enabled)
            self.assertEqual(self.grad_norm(agent.actor), 0)
            self.assertEqual(self.grad_norm(agent._slow_value), 0)

    def test_imagined_value_does_not_train_world_model(self):
        agent = self.make_agent()
        agent._loss_scales = {key: float(key == "value") for key in agent._loss_scales}
        agent._cal_grad(*self.make_batch(agent))
        for head in agent.value:
            self.assertGreater(self.grad_norm(head), 0)
        for module in (agent.encoder, agent.rssm, agent.actor, agent.reward, agent.cont, agent._slow_value):
            self.assertEqual(self.grad_norm(module), 0)

    def test_slow_target_update_and_device_sharing(self):
        agent = self.make_agent()
        before = [p.detach().clone() for p in agent._slow_value.parameters()]
        with torch.no_grad():
            for param in agent.value.parameters():
                param.add_(0.2)
        agent._update_slow_target()
        for old, online, slow in zip(before, agent.value.parameters(), agent._slow_value.parameters()):
            torch.testing.assert_close(
                slow, old * (1 - agent.slow_target_fraction) + online * agent.slow_target_fraction
            )
        agent.train(True)
        self.assertFalse(agent._slow_value.training)
        self.assert_frozen_sharing(agent.to("cpu"))

    def test_update_warmup_gate_and_checkpoint_roundtrip(self):
        for space in ("raw", "symlog"):
            agent = self.make_agent(
                overrides={
                    "critic.outscale": 0.0,
                    "distance_space": space,
                    "dynamic_trunc.enabled": True,
                    "dynamic_trunc.warmup_steps": 10,
                    "dynamic_trunc.min_horizon": 1,
                    "dynamic_trunc.advantage.mode": "positive_only",
                }
            )
            data, initial = self.make_batch(agent)
            replay = mock.Mock()
            replay.sample.side_effect = lambda: (data.clone(), "indices", initial)
            forced = torch.tensor([0.0, 10.0, 10.0]).reshape(1, 3, 1).expand(8, -1, -1)
            for step in (0, 20):
                with mock.patch("dreamer.cal_ens_uncertainty", return_value=forced):
                    metrics = agent.update(replay, env_step=step)
                self.assertTrue(all(torch.isfinite(torch.as_tensor(v)).all() for v in metrics.values()))
                self.assertEqual(metrics["dynamic_trunc/effective_horizon"].item(), 3 if step == 0 else 1)
                self.assertEqual(metrics["imag_wasser"].shape, (3, 1))
            # Also exercise the real mixed W1 inside update's autocast context.
            metrics = agent.update(replay, env_step=21)
            self.assertTrue(all(torch.isfinite(torch.as_tensor(v)).all() for v in metrics.values()))
            for head in agent.value:
                self.assertGreater(head.last.weight.abs().sum().item(), 0)
                for param in head.parameters():
                    self.assertEqual(agent._optimizer.state[param]["step"], 3)
            blob = io.BytesIO()
            torch.save({"agent": agent.state_dict(), "optimizer": agent._optimizer.state_dict()}, blob)
            blob.seek(0)
            saved = torch.load(blob, weights_only=True)
            restored = self.make_agent(
                overrides={
                    "distance_space": space,
                    "dynamic_trunc.enabled": True,
                    "dynamic_trunc.advantage.mode": "positive_only",
                }
            )
            restored.load_state_dict(saved["agent"], strict=True)
            restored._optimizer.load_state_dict(saved["optimizer"])
            for name, value in restored.state_dict().items():
                torch.testing.assert_close(value, agent.state_dict()[name])
            for param in restored.value.parameters():
                self.assertEqual(restored._optimizer.state[param]["step"], 3)
            self.assert_frozen_sharing(restored)

    def test_compiled_mixed_cal_grad_matches_eager(self):
        eager, compiled = self.make_agent(), self.make_agent()
        compiled.load_state_dict(eager.state_dict())
        compiled._cal_grad = torch.compile(compiled._cal_grad, backend="eager")
        data, initial = self.make_batch(eager)
        rng = torch.get_rng_state()
        _, expected = eager._cal_grad(data, initial)
        torch.set_rng_state(rng)
        _, actual = compiled._cal_grad(data, initial)
        for name, value in expected.items():
            torch.testing.assert_close(actual[name], value)
        for name, param in eager._named_params.items():
            if param.grad is not None:
                torch.testing.assert_close(compiled._named_params[name].grad, param.grad)

    def test_checkpoint_mode_guard_before_parameter_copy(self):
        mixed = self.make_agent(overrides={"quantile.num_quantiles": 255})
        categorical = self.make_agent(mode="categorical", overrides={"value_ensemble_size": 2})
        self.assertNotIn("_categorical_quantile_marker", categorical.state_dict())
        for target, source in ((mixed, categorical), (categorical, mixed)):
            for strict in (True, False):
                before = {k: v.clone() for k, v in target.state_dict().items()}
                with self.assertRaisesRegex(RuntimeError, "mode mismatch"):
                    target.load_state_dict(source.state_dict(), strict=strict)
                for key, value in target.state_dict().items():
                    torch.testing.assert_close(value, before[key])
        bad = mixed.state_dict()
        bad["_categorical_quantile_marker"] = torch.tensor(2)
        with self.assertRaisesRegex(RuntimeError, "marker"):
            mixed.load_state_dict(bad)

    def test_old_config_and_default_mode_are_identical(self):
        full = OmegaConf.create(self.config)
        full.dynamic_trunc.enabled = False
        old = OmegaConf.create(OmegaConf.to_container(full, resolve=True))
        del old.value_ensemble_type
        del old.quantile
        agents = []
        for cfg in (full, old):
            torch.manual_seed(42)
            with contextlib.redirect_stdout(io.StringIO()):
                agent = Dreamer(cfg, self.obs_space, self.act_space).to("cpu")
            agent._scaler = torch.amp.GradScaler("cpu", enabled=False)
            agents.append(agent)
        self.assertEqual(agents[0].state_dict().keys(), agents[1].state_dict().keys())
        agents[1].load_state_dict(agents[0].state_dict(), strict=True)
        data, initial = self.make_batch(agents[0])
        rng = torch.get_rng_state()
        results = []
        for agent in agents:
            torch.set_rng_state(rng)
            results.append(agent._cal_grad(data, initial)[1])
        for key in results[0]:
            torch.testing.assert_close(results[0][key], results[1][key], rtol=0, atol=0)
        for key, param in agents[0]._named_params.items():
            other = agents[1]._named_params[key]
            if param.grad is not None:
                torch.testing.assert_close(param.grad, other.grad, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
