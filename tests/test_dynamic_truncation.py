import io
import unittest

import torch

from dreamer import Dreamer
from truncation import ThresholdEMA, build_truncation_mask, mask_to_last, normalized_mask


class DynamicTruncationTest(unittest.TestCase):
    def test_first_crossing_is_per_trajectory_and_does_not_recover(self):
        scores = torch.tensor([[0.0, 1.0, 2.0, 0.0], [0.0, 2.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
        mask = build_truncation_mask(scores.unsqueeze(-1), torch.tensor(1.0), min_horizon=0)
        expected = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
        self.assertTrue(torch.equal(mask[..., 0], expected))
        self.assertFalse(mask.requires_grad)

    def test_forced_prefix_keeps_early_crossing_history_and_clamps_to_horizon(self):
        scores = torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]]).unsqueeze(-1)
        mask = build_truncation_mask(scores, torch.tensor(1.0), min_horizon=2)
        self.assertTrue(torch.equal(mask[..., 0], torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0]]).bool()))
        self.assertTrue(build_truncation_mask(scores, torch.tensor(1.0), min_horizon=8).all())
        single = build_truncation_mask(scores[:, :1], torch.tensor(1.0), min_horizon=1)
        self.assertEqual(single.shape, (2, 1, 1))
        self.assertTrue(single.all())

    def test_nonfinite_scores_cross_threshold(self):
        scores = torch.tensor([[0.0, torch.nan, 0.0], [0.0, torch.inf, 0.0], [0.0, -torch.inf, 0.0]])
        mask = build_truncation_mask(scores.unsqueeze(-1), torch.tensor(1.0), min_horizon=0)
        self.assertTrue(torch.equal(mask[..., 0], torch.tensor([[1, 0, 0]] * 3).bool()))

    def test_mask_normalization_including_empty_masks(self):
        mask = torch.tensor([[[True], [False]], [[True], [True]]])
        normalized = normalized_mask(mask)
        torch.testing.assert_close(normalized.mean(), torch.tensor(1.0))
        torch.testing.assert_close(normalized[mask], torch.full((3,), 4.0 / 3.0))
        self.assertEqual(normalized[~mask].sum().item(), 0)
        empty = normalized_mask(torch.zeros_like(mask))
        self.assertTrue(torch.isfinite(empty).all())
        self.assertEqual(empty.sum().item(), 0)

    def test_threshold_uses_previous_ema_and_checkpoint_roundtrip(self):
        ema = ThresholdEMA(decay=0.5)
        first = ema(torch.tensor(10.0, requires_grad=True))
        second = ema(torch.tensor(0.0, requires_grad=True))
        third = ema(torch.tensor(0.0))
        torch.testing.assert_close(first, torch.tensor(10.0))
        torch.testing.assert_close(second, torch.tensor(10.0))
        torch.testing.assert_close(third, torch.tensor(5.0))
        self.assertFalse(first.requires_grad)
        checkpoint = io.BytesIO()
        torch.save(ema.state_dict(), checkpoint)
        checkpoint.seek(0)
        restored = ThresholdEMA(decay=0.5)
        restored.load_state_dict(torch.load(checkpoint, weights_only=True))
        torch.testing.assert_close(restored(torch.tensor(100.0)), torch.tensor(2.5))

    def test_truncated_return_keeps_boundary_bootstrap_and_ignores_suffix(self):
        mask = torch.tensor([[[1], [1], [1], [0], [0], [0]]], dtype=torch.bool)
        last = mask_to_last(mask)
        torch.testing.assert_close(last, torch.tensor([[[0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0]]]))
        reward = torch.arange(7, dtype=torch.float32).reshape(1, 7, 1)
        value = reward * 10
        term = torch.zeros_like(reward)

        def returns(r, v, terminal):
            return Dreamer._lambda_return(None, last, terminal, r, v, v, 0.9, 0.95)

        original = returns(reward, value, term)
        torch.testing.assert_close(original[:, 2], torch.tensor([[30.0]]))  # r_3 + gamma * V_3
        changed_reward, changed_value = reward.clone(), value.clone()
        changed_reward[:, 4:] += 10000
        changed_value[:, 4:] -= 20000
        torch.testing.assert_close(returns(changed_reward, changed_value, term)[:, :3], original[:, :3])
        changed_value[:, 3] += 2
        torch.testing.assert_close(returns(reward, changed_value, term)[:, 2], torch.tensor([[31.8]]))
        term[:, 3] = 1
        torch.testing.assert_close(returns(reward, value, term)[:, 2], torch.tensor([[3.0]]))


if __name__ == "__main__":
    unittest.main()
