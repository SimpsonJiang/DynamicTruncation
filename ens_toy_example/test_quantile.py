"""Check unequal-support W1 and quantile regression before comparing ensembles."""

import unittest

import numpy as np
import torch
from scipy.stats import wasserstein_distance

from ens_toy_example.compare_quantile import (
    distances,
    pinball,
    uniform_error,
    weighted_w1,
)
from ens_toy_example.run import truth


class TestQuantile(unittest.TestCase):
    def test_unequal_supports_against_scipy(self):
        rng = np.random.default_rng(12)
        a, b = rng.normal(size=(8, 7)), rng.normal(size=(8, 11))
        pa, pb = rng.uniform(size=a.shape), rng.uniform(size=b.shape)
        actual = weighted_w1(a, pa, b, pb)
        expected = [wasserstein_distance(a[i], b[i], pa[i], pb[i]) for i in range(8)]
        np.testing.assert_allclose(actual, expected, atol=1e-14)
        np.testing.assert_allclose(actual, weighted_w1(b, pb, a, pa), atol=1e-14)

    def test_wide_support_and_duplicate_atoms(self):
        a, pa = np.array([-4.8e8, 0, 4.8e8]), np.array([1e-5, 0.99997, 2e-5])
        b, pb = np.array([-0.1, 0, 0, 0.1]), np.ones(4)
        self.assertAlmostEqual(weighted_w1(a, pa, b, pb), wasserstein_distance(a, b, pa, pb), delta=1e-6)
        np.testing.assert_allclose(weighted_w1(a, pa, a, pa), 0, atol=1e-10)

    def test_pair_normalization_identical_groups(self):
        bins = np.linspace(-2, 2, 7)
        q = np.broadcast_to(bins, (5, 3, 7)).copy()
        p = np.full((5, 3, 7), 1 / 7)
        scores, pairs = distances(p, bins, q)
        for score in scores.values():
            np.testing.assert_allclose(score, 0, atol=1e-14)
        self.assertEqual(pairs["cc_pairs"].shape, (10, 3))
        self.assertEqual(pairs["qq_pairs"].shape, (10, 3))
        self.assertEqual(pairs["cq_pairs"].shape, (25, 3))
        shifted, _ = distances(p, bins, q + 3)
        np.testing.assert_allclose(shifted["between"], 3)
        np.testing.assert_allclose(shifted["quantile_within"], 0)

    def test_uniform_error_known_optimal_quantiles(self):
        x = np.linspace(-2, 2, 9)
        mean, scale = truth(x)
        for k in (1, 17, 255):
            tau = (np.arange(k) + 0.5) / k
            atoms = mean[:, None] + scale[:, None] * (2 * tau - 1)
            # Midpoint quantization error for a uniform interval of width 2s.
            np.testing.assert_allclose(uniform_error(atoms, np.ones_like(atoms), x), scale / (2 * k), atol=1e-12)

    def test_pinball_learns_uniform_quantiles(self):
        # Direct parameters isolate loss correctness from network generalization.
        target = torch.linspace(-1, 1, 2001)
        q = torch.nn.Parameter(torch.zeros(1, 1, 5))
        opt = torch.optim.Adam([q], lr=0.03)
        for _ in range(250):
            opt.zero_grad()
            loss = pinball(q.expand(1, len(target), 5), target).sum()
            loss.backward()
            opt.step()
        torch.testing.assert_close(q.detach().flatten(), torch.linspace(-0.8, 0.8, 5), atol=0.015, rtol=0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
