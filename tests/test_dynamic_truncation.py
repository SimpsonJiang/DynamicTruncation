import io
import unittest

import torch

from dreamer import Dreamer
from truncation import (
    AdvantageScaleEMA,
    ThresholdEMA,
    advantage_drop_fractions,
    advantage_truncation_score,
    build_truncation_mask,
    mask_to_last,
    normalized_mask,
)


class DynamicTruncationTest(unittest.TestCase):
    def test_advantage_score_sign_monotonicity_bounds_and_detachment(self):
        advantage = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0], requires_grad=True).reshape(1, 5, 1)
        uncertainty = torch.full_like(advantage, 2.0, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            for mode in ("positive_only", "signed"):
                score = advantage_truncation_score(uncertainty, advantage, mode=mode)
                self.assertFalse(score.requires_grad)
                self.assertEqual(score.dtype, torch.float32)
                self.assertTrue((score[:, 1:] <= score[:, :-1]).all())
                self.assertTrue(((score >= 1) & (score <= 4)).all())
                self.assertEqual(score[0, 2, 0].item(), 2)
                self.assertEqual(score[0, -1, 0].item(), 1)
                if mode == "positive_only":
                    torch.testing.assert_close(score[:, :3], uncertainty[:, :3], rtol=0, atol=0)
                else:
                    self.assertEqual(score[0, 0, 0].item(), 4)
                    self.assertAlmostEqual(score[0, 1, 0].item(), 3.39075, places=4)
        zero = advantage_truncation_score(torch.zeros_like(advantage), advantage)
        torch.testing.assert_close(zero, torch.zeros_like(zero))

    def test_advantage_identity_and_invalid_scores_respect_forced_prefix(self):
        uncertainty = torch.tensor([0.1, 0.2, 0.3, 0.4]).reshape(1, 4, 1)
        advantage = torch.tensor([torch.nan, torch.inf, -torch.inf, 1.0]).reshape(1, 4, 1)
        torch.testing.assert_close(
            advantage_truncation_score(uncertainty, None, mode="off"), uncertainty, rtol=0, atol=0
        )
        for mode in ("positive_only", "signed"):
            torch.testing.assert_close(
                advantage_truncation_score(uncertainty, advantage, mode=mode, max_factor=1),
                uncertainty,
                rtol=0,
                atol=0,
            )
            score = advantage_truncation_score(uncertainty, advantage, mode=mode)
            self.assertTrue(torch.isinf(score[:, :3]).all())
            self.assertFalse(torch.isnan(score).any())
            mask = build_truncation_mask(score, torch.tensor(1.0), min_horizon=2)
            self.assertEqual(mask.flatten().tolist(), [True, True, False, False])
        invalid_u = torch.tensor([torch.nan, torch.inf, -torch.inf, -1.0])
        self.assertTrue(torch.isinf(advantage_truncation_score(invalid_u, torch.ones(4))).all())

    def test_advantage_scale_lag_zero_invalid_and_checkpoint(self):
        ema = AdvantageScaleEMA(quantile=0.5, decay=0.5)
        invalid = torch.tensor([torch.nan, torch.inf, -torch.inf])
        _, scale = ema(invalid)
        self.assertAlmostEqual(scale.item(), 1e-6)
        self.assertFalse(ema.initialized)
        self.assertEqual(ema.value.item(), 0)
        normalized, scale = ema(torch.tensor([-2.0, 4.0, torch.nan], requires_grad=True))
        self.assertEqual(scale.item(), 3)
        torch.testing.assert_close(normalized[:2], torch.tensor([-2 / 3, 4 / 3]))
        self.assertFalse(normalized.requires_grad)
        _, scale = ema(torch.tensor([10.0]))
        self.assertEqual(scale.item(), 3)
        self.assertEqual(ema.value.item(), 6.5)
        ema(invalid)
        self.assertEqual(ema.value.item(), 6.5)
        checkpoint = io.BytesIO()
        torch.save(ema.state_dict(), checkpoint)
        checkpoint.seek(0)
        restored = AdvantageScaleEMA(quantile=0.5, decay=0.5)
        restored.load_state_dict(torch.load(checkpoint, weights_only=True))
        normalized, scale = restored(torch.zeros(5))
        self.assertEqual(scale.item(), 6.5)
        self.assertEqual(restored.value.item(), 3.25)
        torch.testing.assert_close(normalized, torch.zeros(5))
        normalized, scale = AdvantageScaleEMA()(torch.zeros(5))
        self.assertTrue(torch.isfinite(normalized).all())
        self.assertAlmostEqual(scale.item(), 1e-6)

    def test_advantage_drop_fractions_use_weight_sign_and_actual_mask(self):
        advantage = torch.tensor([2.0, 4.0, -3.0, -6.0, torch.nan], requires_grad=True)
        weight = torch.tensor([1.0, 2.0, 3.0, 1.0, 5.0], requires_grad=True)
        mask = torch.tensor([True, False, True, False, False])
        positive, negative = advantage_drop_fractions(advantage, weight, mask)
        torch.testing.assert_close(positive, torch.tensor(0.8))
        torch.testing.assert_close(negative, torch.tensor(0.4))
        self.assertFalse(positive.requires_grad or negative.requires_grad)
        for a, w, m in (
            (advantage, weight, torch.ones_like(mask)),
            (torch.zeros(5), weight, mask),
            (advantage, torch.zeros(5), mask),
        ):
            for fraction in advantage_drop_fractions(a, w, m):
                self.assertEqual(fraction.item(), 0)

    def test_advantage_gate_cpu_compile_matches_eager_across_warmup(self):
        torch.compiler.reset()
        eager = AdvantageScaleEMA(quantile=0.5, decay=0.5)
        compiled_ema = AdvantageScaleEMA(quantile=0.5, decay=0.5)

        def gate(ema, uncertainty, advantage, active):
            normalized, scale = ema(advantage)
            score = advantage_truncation_score(uncertainty, normalized, mode="signed")
            mask = build_truncation_mask(score, uncertainty.new_tensor(1.0), 1)
            if not active:
                mask = torch.ones_like(mask)
            return normalized, scale, score, mask

        compiled = torch.compile(gate, backend="eager")
        uncertainty = torch.tensor([0.5, 1.4, 0.5]).reshape(1, 3, 1)
        for active, values in ((False, [0.0, 2.0, -1.0]), (True, [1.0, -3.0, 2.0]), (True, [0.0] * 3)):
            advantage = torch.tensor(values).reshape(1, 3, 1)
            expected = gate(eager, uncertainty, advantage, active)
            actual = compiled(compiled_ema, uncertainty, advantage, active)
            for left, right in zip(actual, expected):
                torch.testing.assert_close(left, right)
            torch.testing.assert_close(compiled_ema.value, eager.value)
        torch.compiler.reset()

    def test_invalid_advantage_configuration(self):
        for mode in ("unknown", None):
            with self.assertRaisesRegex(ValueError, "advantage.mode"):
                advantage_truncation_score(torch.ones(1), torch.ones(1), mode=mode)
        for factor in (0, 0.9, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "max_factor"):
                advantage_truncation_score(torch.ones(1), torch.ones(1), max_factor=factor)
        for key, values in {
            "quantile": [-0.1, 1.1, float("nan")],
            "decay": [-0.1, 1.0, float("nan")],
            "eps": [0, -1, float("inf"), float("nan")],
        }.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    AdvantageScaleEMA(**{key: value})

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
