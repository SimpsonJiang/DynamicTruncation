"""Numerical and compilation contracts for fixed-shape reconstruction losses."""

import copy
import unittest

import torch
from omegaconf import OmegaConf

from networks import barlow_loss
from reconstruction.decoder import OnlinePhysicsDecoder


class ReconstructionLossTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(1)

    @staticmethod
    def decoder(quaternion=False):
        schema = (
            dict(nq=7, nv=6, na=0, joint_types=[0], joint_addresses=[0], joint_limited=[0], task="free")
            if quaternion
            else dict(nq=2, nv=2, na=1, joint_types=[3, 2], joint_addresses=[0, 1], joint_limited=[0, 1], task="test")
        )
        return OnlinePhysicsDecoder(8, OmegaConf.create(dict(schema=schema, hidden=16, norm_decay=0.99, norm_eps=1e-3)))

    def test_barlow_values_and_gradients_match_compacted_sample_variance(self):
        for count in (2, 5, 8):
            with self.subTest(count=count):
                x = torch.randn(8, 4, dtype=torch.float64, requires_grad=True)
                y = torch.randn(8, 4, dtype=torch.float64, requires_grad=True)
                mask = torch.randperm(8) < count
                actual = barlow_loss(x, y, 0.05, None if count == 8 else mask)
                a, b = x[mask], y[mask]
                a = (a - a.mean(0)) / (a.std(0) + 1e-8)
                b = (b - b.mean(0)) / (b.std(0) + 1e-8)
                c = a.T @ b / count
                expected = (c.diagonal() - 1).square().sum()
                expected = expected + 0.05 * c[~torch.eye(4, dtype=torch.bool)].square().sum()
                torch.testing.assert_close(actual, expected)
                for got, want in zip(torch.autograd.grad(actual, (x, y)), torch.autograd.grad(expected, (x, y))):
                    torch.testing.assert_close(got, want)

    def test_barlow_short_or_constant_batches_have_finite_gradients(self):
        for count in (0, 1, 5):
            with self.subTest(count=count):
                x = torch.ones(5, 4, requires_grad=True)
                mask = torch.arange(5) < count
                loss = barlow_loss(x, x, 0.05, mask)
                loss.backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                if count < 2:
                    self.assertEqual(loss.item(), 0)
                    self.assertEqual(x.grad.abs().sum().item(), 0)

    def test_physics_mask_matches_compacted_loss_gradients_and_statistics(self):
        for quaternion in (False, True):
            with self.subTest(quaternion=quaternion):
                head = self.decoder(quaternion)
                reference = copy.deepcopy(head)
                x = torch.randn(6, 8, requires_grad=True)
                y = torch.randn(6, head.codec.core_dim, dtype=torch.float64)
                mask = torch.tensor([False, True, True, False, True, False])
                # Exercise first-update replacement followed by EMA updates.
                for _ in range(2):
                    actual = head.training_loss(x, y, mask)
                    expected = reference.training_loss(x[mask], y[mask])
                    torch.testing.assert_close(actual, expected)
                    params = tuple(head.parameters())
                    ref_params = tuple(reference.parameters())
                    for got, want in zip(
                        torch.autograd.grad(actual, (x, *params)), torch.autograd.grad(expected, (x, *ref_params))
                    ):
                        torch.testing.assert_close(got, want)
                    for key, value in head.state_dict().items():
                        torch.testing.assert_close(value, reference.state_dict()[key])
                torch.testing.assert_close(head.input_mean, x[mask].detach().double().mean(0))

    def test_empty_physics_mask_keeps_statistics_and_returns_zero_gradients(self):
        head = self.decoder(True)
        head.training_loss(torch.randn(3, 8), torch.randn(3, 13, dtype=torch.float64))
        before = copy.deepcopy(head.state_dict())
        x = torch.randn(4, 8, requires_grad=True)
        loss = head.training_loss(x, torch.randn(4, 13, dtype=torch.float64), torch.zeros(4, dtype=torch.bool))
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertEqual(x.grad.abs().sum().item(), 0)
        for p in head.parameters():
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertEqual(p.grad.abs().sum().item(), 0)
        for key, value in head.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_physics_label_validation_remains_available_outside_compile(self):
        head = self.decoder()
        x, y = torch.randn(4, 8), torch.randn(4, 5, dtype=torch.float64)
        with self.assertRaises(TypeError):
            head.training_loss(x, y.float())
        y[0, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            head.training_loss(x, y, torch.zeros(4, dtype=torch.bool))

    def test_losses_compile_once_across_empty_short_and_full_masks(self):
        head = self.decoder(True)
        x, y = torch.randn(6, 8), torch.randn(6, 13, dtype=torch.float64)
        functions = (
            lambda a, b, mask: barlow_loss(a, b[:, :8].float(), 0.05, mask),
            lambda a, b, mask: head.training_loss(a, b, mask, validate=False),
        )
        for function in functions:
            graphs = []

            def backend(graph, example_inputs):
                graphs.append(graph)
                return graph.forward

            compiled = torch.compile(function, backend=backend, fullgraph=True)
            for count in (0, 1, 3, 6, 2, 0):
                mask = torch.arange(6) < count
                self.assertTrue(torch.isfinite(compiled(x, y, mask)))
            self.assertEqual(len(graphs), 1)
