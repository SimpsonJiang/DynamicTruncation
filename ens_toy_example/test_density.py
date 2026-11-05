"""Validate gradual density, genuine empty intervals and target preservation."""

import unittest

import numpy as np
from scipy.integrate import quad

from ens_toy_example.density_sweep import (
    NORMALIZER,
    REGIONS,
    coverage,
    masks,
    sample_data,
    sampling_density,
)
from ens_toy_example.run import truth


class TestDensity(unittest.TestCase):
    def test_density_mass_and_order(self):
        areas = [quad(lambda x: float(sampling_density(np.array([x]))[0]), a, b)[0] for _, a, b in REGIONS]
        np.testing.assert_allclose(areas, [0.675, 0.1075, 0.024, 0, 0.028, 0], atol=1e-12)
        self.assertAlmostEqual(sum(areas), NORMALIZER)
        x = np.linspace(0, 2.1, 10001)
        self.assertTrue(np.all(np.diff(sampling_density(x)) <= 1e-14))
        self.assertLessEqual(sampling_density(np.linspace(2.5, 3.2, 1001)).max(), 0.08)

    def test_internal_boundary_continuity_and_slope(self):
        eps = 1e-5
        for _, _, b in REGIONS[:-1]:
            left, at, right = sampling_density(np.array([b - eps, b, b + eps]))
            self.assertLess(abs(left - right), 1e-8)
            self.assertLess(abs((at - left) / eps - (right - at) / eps), 1e-3)

    def test_sampler_and_conditional_truth(self):
        x, y = sample_data(20000, np.random.default_rng(123))
        mean, scale = truth(x)
        self.assertTrue(np.all(np.abs(y - mean) <= scale + 1e-6))
        self.assertTrue(np.all(sampling_density(x) > 0))
        expected = np.array([0.675, 0.1075, 0.024, 0, 0.028, 0]) / NORMALIZER
        observed = np.array([m.sum() for m in masks(x).values()])
        tolerance = 6 * np.sqrt(len(x) * expected * (1 - expected)) + 1
        self.assertTrue(np.all(np.abs(observed - len(x) * expected) < tolerance))
        self.assertEqual(observed[3], 0)
        self.assertEqual(observed[5], 0)
        c = coverage(x, x)
        self.assertGreater(c[0]["train_count_per_x_unit"], c[1]["train_count_per_x_unit"])
        self.assertGreater(c[1]["train_count_per_x_unit"], c[2]["train_count_per_x_unit"])
        self.assertGreater(c[1]["train_count_per_x_unit"], c[4]["train_count_per_x_unit"])

    def test_regions_partition_unequal_domain(self):
        x = np.unique(np.r_[np.linspace(0, 4.2, 841), [a for _, a, _ in REGIONS]])
        np.testing.assert_array_equal(np.stack(list(masks(x).values())).sum(0), 1)
        self.assertGreater(len(set(round(b - a, 6) for _, a, b in REGIONS)), 1)


if __name__ == "__main__":
    unittest.main()
