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

    def make_agent(self, size=5, outscale=0.0):
        config = OmegaConf.create(self.config)
        config.value_ensemble_size = size
        config.critic.outscale = outscale
        with contextlib.redirect_stdout(io.StringIO()):
            agent = Dreamer(config, self.obs_space, self.act_space).to("cpu")
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
        for size in [0, 1, -2, 2.5, "5", True]:
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "value_ensemble_size"):
                self.make_agent(size=size)

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
        loss = agent._value_loss(feat, target, slow, weight)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        for param, expected_grad in zip(parameters, expected_grads):
            torch.testing.assert_close(param.grad, expected_grad)
        self.assertIsNone(target.grad)
        self.assertIsNone(slow.grad)
        self.assertEqual(feat.grad[:, -1].abs().sum().item(), 0)
        self.assertEqual(feat.grad[0, 1].abs().sum().item(), 0)

    def test_loss_gradient_boundaries(self):
        for active in ["policy", "value", "repval"]:
            with self.subTest(loss=active):
                # Nonzero output weights exercise upstream gradients beyond the initial zero-output step.
                agent = self.make_agent(outscale=0.1)
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
        agent = self.make_agent(size=2)
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
        restored = self.make_agent(size=2)
        restored.load_state_dict(state["agent"])
        restored._optimizer.load_state_dict(state["optimizer"])
        for name, value in restored.state_dict().items():
            torch.testing.assert_close(value, agent.state_dict()[name])
        for param in restored.value.parameters():
            self.assertEqual(restored._optimizer.state[param]["step"], 2)
        self.assert_frozen_sharing(restored)


if __name__ == "__main__":
    unittest.main()
