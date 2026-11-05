"""Numerical checks needed before interpreting uncertainty experiments."""

import unittest

import numpy as np
import torch
from scipy.stats import wasserstein_distance

from distributions import TwoHot, symexp_twohot
from ens_toy_example.run import (
    discrete_w1,
    features,
    logits_of,
    make_ensemble,
    projected_truth,
    risk_curve,
    truth,
    twohot,
    uniform_w1,
)
from truncation import cal_ens_uncertainty


class TestToy(unittest.TestCase):
    def test_labels_match_repository_loss_and_gradient(self):
        torch.manual_seed(42)
        bins = symexp_twohot(torch.zeros(1, 255), 255).bins
        y = torch.tensor([-1e10, -2.1, 0, 0.05, 2.3, 1e10])
        logits = torch.randn(len(y), 255, requires_grad=True)
        ours = -(twohot(y, bins) * logits.log_softmax(-1)).sum()
        reference = -TwoHot(logits, bins).log_prob(y[:, None]).sum()
        torch.testing.assert_close(ours, reference)
        torch.testing.assert_close(torch.autograd.grad(ours, logits)[0], torch.autograd.grad(reference, logits)[0])

    def test_continuous_w1_against_point_mass(self):
        # W1(delta_mean, Uniform(mean-s, mean+s)) = s/2.
        x = np.array([0.0])
        bins = np.array([-2.0, 0, 2.0])
        self.assertAlmostEqual(uniform_w1(np.array([[0, 1, 0]]), bins, x)[0], truth(x)[1][0] / 2)

    def test_continuous_w1_against_scipy_quadrature(self):
        x = np.array([0.2])
        bins = np.array([-2.0, -0.5, 1.0, 4.0])
        p = np.array([[0.1, 0.2, 0.4, 0.3]])
        mean, scale = truth(x)
        reference_samples = mean[0] + scale[0] * (2 * (np.arange(20000) + 0.5) / 20000 - 1)
        reference = wasserstein_distance(bins, reference_samples, p[0], None)
        self.assertAlmostEqual(uniform_w1(p, bins, x)[0], reference, delta=1e-5)

    def test_exact_uniform_projection(self):
        x = np.linspace(-2, 2, 9)
        bins = torch.linspace(-5, 5, 255)
        exact = projected_truth(x, bins.double().numpy())
        mean, scale = truth(x)
        q = (np.arange(20000) + 0.5) / 20000
        for i in range(len(x)):
            y = torch.tensor(mean[i] + scale[i] * (2 * q - 1), dtype=torch.float32)
            numerical = twohot(y, bins).mean(0).numpy()
            np.testing.assert_allclose(exact[i], numerical, atol=2e-6)
        np.testing.assert_allclose(exact.sum(-1), 1)
        self.assertGreaterEqual(exact.min(), -1e-14)

    def test_discrete_w1_matches_scipy_and_repo(self):
        torch.manual_seed(8)
        logits = torch.randn(5, 1, 2, 255)
        distributions = [symexp_twohot(member_logits, 255) for member_logits in logits]
        bins = distributions[0].bins.double().numpy()
        p = logits.double().softmax(-1).numpy()[:, 0, 0]
        distances = [discrete_w1(p[i], p[j], bins) for i in range(5) for j in range(i + 1, 5)]
        self.assertAlmostEqual(distances[0], wasserstein_distance(bins, bins, p[0], p[1]), delta=1e-6)
        score = cal_ens_uncertainty(distributions, None, "wasser").item()
        self.assertAlmostEqual(score / np.mean(distances), 1, places=6)

    def test_initialization_and_identical_member_control(self):
        heads = make_ensemble(11, 16)
        x = features(np.linspace(-2, 2, 20))
        logits = logits_of(heads, x)
        # Repository zero output scale means equal initial logits, despite different hidden weights.
        torch.testing.assert_close(logits, torch.zeros_like(logits))
        self.assertFalse(torch.equal(heads[0].mlp.layers[0].weight, heads[1].mlp.layers[0].weight))
        heads[1].load_state_dict(heads[0].state_dict())
        optimizer = torch.optim.Adam(heads.parameters(), lr=1e-3)
        for _ in range(2):
            logits = logits_of(heads, x)
            optimizer.zero_grad()
            (-logits.log_softmax(-1)[..., 127].mean()).backward()
            optimizer.step()
        torch.testing.assert_close(logits_of(heads, x)[0], logits_of(heads, x)[1], rtol=0, atol=0)

    def test_tied_scores_have_random_expected_risk(self):
        error = np.array([1, 10, 2, 4.0])
        np.testing.assert_allclose(risk_curve(error, np.zeros(4)), error.mean())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
