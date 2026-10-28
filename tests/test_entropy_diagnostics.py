import unittest
from unittest import mock

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

import test_value_ensemble as value_ensemble_tests
from diagnostics import imagined_entropy_metrics
from trainer import OnlineTrainer


class EntropyDiagnosticsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.factory = value_ensemble_tests.ValueEnsembleTest()
        cls.factory.setUpClass()

    def setUp(self):
        self.factory.setUp()

    def test_valid_mean_weights_positions_and_preserves_depth(self):
        entropy = torch.tensor([[[1.0], [2.0], [30.0]], [[4.0], [50.0], [60.0]]], requires_grad=True)
        mask = torch.tensor([[[True], [True], [False]], [[True], [False], [False]]])
        metrics = imagined_entropy_metrics(entropy, mask)
        torch.testing.assert_close(metrics["action_entropy_valid"], torch.tensor(7.0 / 3))
        torch.testing.assert_close(metrics["imag_action_entropy"], torch.tensor([[2.5], [26.0], [45.0]]))
        for value in metrics.values():
            self.assertEqual(value.dtype, torch.float32)
            self.assertFalse(value.requires_grad)
        self.assertIsNone(entropy.grad)
        empty = imagined_entropy_metrics(entropy, torch.zeros_like(mask))
        self.assertEqual(empty["action_entropy_valid"].item(), 0)
        full = imagined_entropy_metrics(entropy)
        torch.testing.assert_close(full["action_entropy_valid"], entropy.mean())
        self.assertEqual(imagined_entropy_metrics(entropy[:, :1])["imag_action_entropy"].shape, (1, 1))

    def test_act_entropy_reporting_preserves_actions_state_and_rng(self):
        agent = self.factory.make_agent(outscale=0.1)
        data, _ = self.factory.make_batch(agent)
        obs = data[:, 0].clone()
        state = agent.get_initial_state(2)
        for evaluation in (False, True):
            torch.manual_seed(71)
            action, next_state = agent.act(obs.clone(), state.clone(), eval=evaluation)
            rng = torch.get_rng_state()
            torch.manual_seed(71)
            metrics = {}
            measured_action, measured_state = agent.act(obs.clone(), state.clone(), eval=evaluation, metrics=metrics)
            torch.testing.assert_close(measured_action, action, rtol=0, atol=0)
            for key in next_state.keys():
                torch.testing.assert_close(measured_state[key], next_state[key], rtol=0, atol=0)
            torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
            feat = agent.rssm.get_feat(next_state["stoch"], next_state["deter"])
            torch.testing.assert_close(metrics["action_entropy"], agent.actor(feat).entropy())
            self.assertEqual(metrics["action_entropy"].shape, (2,))
            self.assertFalse(metrics["action_entropy"].requires_grad)

    def test_update_entropy_excludes_bootstrap_and_does_not_change_gradients(self):
        for truncation in (False, True):
            with self.subTest(truncation=truncation):
                reference = self.factory.make_agent(outscale=0.1, trunc_enabled=truncation, min_horizon=1)
                measured = self.factory.make_agent(outscale=0.1, trunc_enabled=truncation, min_horizon=1)
                measured.load_state_dict(reference.state_dict())
                data, initial = self.factory.make_batch(reference)
                torch.manual_seed(89)
                with mock.patch("dreamer.imagined_entropy_metrics", return_value={}):
                    _, previous = reference._cal_grad(data, initial)
                previous_rng = torch.get_rng_state()
                torch.manual_seed(89)
                with mock.patch("dreamer.imagined_entropy_metrics", wraps=imagined_entropy_metrics) as summary:
                    _, current = measured._cal_grad(data, initial)
                entropy, mask = summary.call_args.args
                self.assertEqual(entropy.shape, (8, 3, 1))
                self.assertEqual(mask is not None, truncation)
                self.assertEqual(current["imag_action_entropy"].shape, (3, 1))
                if not truncation:
                    torch.testing.assert_close(current["action_entropy_valid"], current["action_entropy"])
                for key in ("action_entropy_valid", "imag_action_entropy"):
                    self.assertFalse(current[key].requires_grad)
                for key, value in previous.items():
                    torch.testing.assert_close(current[key], value, rtol=0, atol=0)
                torch.testing.assert_close(torch.get_rng_state(), previous_rng, rtol=0, atol=0)
                for name, param in reference._named_params.items():
                    other = measured._named_params[name]
                    if param.grad is None:
                        self.assertIsNone(other.grad)
                    else:
                        torch.testing.assert_close(param.grad, other.grad, rtol=0, atol=0)

    def test_collection_mean_excludes_terminal_and_clears_each_log_window(self):
        config = OmegaConf.create(dict(
            steps=4, pretrain=1, eval_every=0, eval_episode_num=0, video_pred_log=False,
            params_hist_log=False, batch_length=1, batch_size=1, train_ratio=1,
            update_log_every=1, action_repeat=1,
        ))
        calls = 0
        trans = TensorDict({"reward": torch.zeros(1, 1)}, batch_size=(1,))
        state = TensorDict({"prev_action": torch.zeros(1, 1), "stoch": torch.zeros(1, 2, 2),
                            "deter": torch.zeros(1, 4)}, batch_size=(1,))
        envs = mock.Mock(env_num=1)

        def step(action, done):
            return trans.clone(), torch.tensor([calls == 0])

        def act(obs, memory, eval=False, metrics=None):
            nonlocal calls
            calls += 1
            metrics["action_entropy"] = torch.tensor([100.0 if calls == 1 else float(calls)])
            return torch.zeros(1, 1), state

        envs.step.side_effect = step
        agent = mock.Mock(device=torch.device("cpu"))
        agent.get_initial_state.return_value = state
        agent.act.side_effect = act
        agent.update.return_value = {"loss": torch.tensor(3.0)}
        replay = mock.Mock()
        replay.count.return_value = 0
        logger = mock.Mock()
        OnlineTrainer(config, replay, logger, None, envs, None).begin(agent)
        means = [call.args[1].item() for call in logger.scalar.call_args_list
                 if call.args[0] == "train/action_entropy_real"]
        self.assertEqual(means, [3.5, 6.0])


if __name__ == "__main__":
    unittest.main()
