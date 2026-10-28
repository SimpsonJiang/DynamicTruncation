"""Branch candidates must be the exact states excluded by training's VE mask."""

import copy
import unittest
from unittest import mock

import torch

from branch_collection import BranchCollector, validate_branch_config
from dynamic_truncation import build_truncation_mask, truncation_candidates
from test_state_branching import make_agent, make_batch, small_config


class TruncationBranchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def packet(self, scores, minimum=0, eligible=None):
        scores = torch.tensor(scores).float().unsqueeze(-1)
        n, h, _ = scores.shape
        features = torch.arange(n * (h + 1) * 2).float().reshape(n, h + 1, 2).requires_grad_()
        threshold = torch.tensor(1.0)
        mask = build_truncation_mask(scores, threshold, minimum)
        packet = truncation_candidates(features, scores, mask, threshold,
                                       torch.ones(n, dtype=torch.bool) if eligible is None else eligible,
                                       torch.arange(n) * 100)
        return packet, features

    def test_exact_first_excluded_feature_and_full_horizon_skip(self):
        packet, features = self.packet([[0, 1, 2, 0], [2, 0, 0, 0], [0, 0, 0, 2], [1, 1, 1, 1]])
        self.assertEqual(packet["depth"].tolist(), [2, 0, 3, 4])
        self.assertEqual(packet["eligible"].tolist(), [True, True, True, False])
        torch.testing.assert_close(packet["features"], features[torch.arange(4), torch.tensor([2, 0, 3, 4])])
        self.assertEqual(packet["origin_step"].tolist(), [0, 100, 200, 300])
        self.assertTrue(all(not tensor.requires_grad for tensor in packet.values()))

    def test_forced_prefix_is_the_actual_boundary_even_if_score_recovers(self):
        packet, features = self.packet([[2, 0, 0, 0]], minimum=2)
        self.assertTrue(packet["eligible"].item())
        self.assertEqual(packet["depth"].item(), 2)
        self.assertEqual(packet["score"].item(), 0)
        torch.testing.assert_close(packet["features"], features[:, 2])
        packet, _ = self.packet([[2, 0, 0, 0]], minimum=4)
        self.assertFalse(packet["eligible"].any())

    def test_nonfinite_causes_and_ineligible_origins_are_excluded(self):
        packet, _ = self.packet([[0, float("nan"), 0, 0], [float("inf"), 0, 0, 0],
                                 [0, 2, 0, 0], [0, 0, 2, float("nan")]], minimum=2,
                                eligible=torch.tensor([True, True, False, True]))
        self.assertEqual(packet["eligible"].tolist(), [False, False, False, True])

    def test_collector_copies_bounded_latest_pool_without_advancing_training_rng(self):
        cfg = small_config("branch_collection.mode=imagined_state", "branch_collection.max_attempts=2")
        collector = BranchCollector(cfg.branch_collection, cfg.env, "cpu", 5)
        packet, _ = self.packet([[0, 2, 0]] * 6)
        before = torch.get_rng_state().clone()
        collector.offer_candidates(packet)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(len(collector.candidates), 2)
        expected = [c["features"].clone() for c in collector.candidates]
        packet["features"].zero_()
        for candidate, feature in zip(collector.candidates, expected):
            torch.testing.assert_close(candidate["features"], feature)
        packet["eligible"].zero_()
        collector.offer_candidates(packet)
        self.assertEqual(collector.candidates, [])
        agent, normal, branch = mock.Mock(), mock.Mock(), mock.Mock()
        used, metrics = collector.collect(agent, normal, branch, 20000, 100)
        self.assertEqual((used, metrics["branch/attempts"], metrics["branch/successes"]), (0, 0, 0))
        self.assertEqual(metrics["branch/rejected/no_truncation"], 1)
        self.assertIsNone(collector.worker)
        normal.sample.assert_not_called()
        agent.state_decoder.assert_not_called()
        branch.add_transition.assert_not_called()

    def test_both_metrics_export_actual_training_cutoffs_and_previous_threshold(self):
        for metric in ("wasser", "ucb"):
            with self.subTest(metric=metric):
                cfg = small_config("model.state_decoder.enabled=true", "branch_collection.mode=imagined_state",
                                   "model.dynamic_trunc.min_horizon=1", f"model.VE_metric={metric}")
                agent = make_agent(cfg)
                agent.trunc_kappa.value.fill_(1)
                agent.trunc_kappa.initialized.fill_(True)
                data = make_batch()
                data["_source"][0, 1] = 1
                data["is_last"][0, 2] = True
                data["is_terminal"][0, 3] = True
                scores = torch.tensor([[0, 2, 0]] * 4 + [[0, 0, 0], [0, 1, 2], [2, 0, 0], [0, 0, 2]]).float().unsqueeze(-1)
                replay = mock.Mock()
                replay.sample.return_value = data, [], agent.rssm.initial(2)
                captured = {}
                imagine = agent._imagine

                def record_imagination(*args):
                    result = imagine(*args)
                    captured["features"] = result[0].detach().clone()
                    return result

                selected_method = "_imag_wasserstein" if metric == "wasser" else "_imag_ucb"
                unused_method = "_imag_ucb" if metric == "wasser" else "_imag_wasserstein"
                with mock.patch.object(agent, selected_method, return_value=scores) as score_fn, \
                        mock.patch.object(agent, unused_method, side_effect=AssertionError("wrong VE_metric")), \
                        mock.patch.object(agent, "_imagine", side_effect=record_imagination), \
                        mock.patch.object(agent.trunc_kappa, "forward", wraps=agent.trunc_kappa.forward) as ema:
                    metrics = agent.update(replay)
                score_fn.assert_called_once()
                ema.assert_called_once()
                self.assertFalse(any(k.startswith("_branch/") for k in metrics))
                packet = agent.take_branch_candidates()
                self.assertIsNone(agent.take_branch_candidates())
                self.assertEqual(packet["depth"].tolist(), [1, 1, 1, 1, 3, 2, 1, 2])
                self.assertEqual(packet["eligible"].tolist(), [True, False, False, False, False, True, True, True])
                torch.testing.assert_close(packet["features"], captured["features"][torch.arange(8), packet["depth"]])
                torch.testing.assert_close(packet["threshold"], torch.ones(8))
                self.assertNotEqual(agent.trunc_kappa.value.item(), 1.0)
                self.assertTrue(all(not v.requires_grad for v in packet.values()))

    def test_candidate_export_does_not_change_training_gradients_or_rng(self):
        cfg = small_config("model.state_decoder.enabled=true", "model.dynamic_trunc.min_horizon=1")
        torch.manual_seed(123)
        baseline = make_agent(copy.deepcopy(cfg))
        cfg.branch_collection.mode = "imagined_state"
        torch.manual_seed(123)
        branching = make_agent(cfg)
        data = make_batch()
        for agent in (baseline, branching):
            torch.manual_seed(456)
            agent._cal_grad(data.clone(), agent.rssm.initial(2))
            if agent is baseline:
                expected_rng = torch.get_rng_state().clone()
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        for name, parameter in baseline._named_params.items():
            torch.testing.assert_close(parameter.grad, branching._named_params[name].grad, atol=0, rtol=0)

    def test_imagined_mode_requires_dynamic_truncation(self):
        cfg = small_config("model.state_decoder.enabled=true", "branch_collection.mode=imagined_state",
                           "model.dynamic_trunc.enabled=false")
        with self.assertRaisesRegex(ValueError, "dynamic_trunc"):
            validate_branch_config(cfg)
        with self.assertRaisesRegex(ValueError, "dynamic truncation"):
            make_agent(cfg)


if __name__ == "__main__":
    unittest.main()
