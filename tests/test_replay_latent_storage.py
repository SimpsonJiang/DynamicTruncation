"""Branch seeds share the regular replay cache without changing sampled states."""

import unittest
from unittest import mock

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from buffer import Buffer, ReplayNotReady


class ReplayLatentStorageTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)

    def make_buffer(self, frames=8, length=3):
        buffer = Buffer(
            OmegaConf.create(
                dict(device="cpu", storage_device="cpu", batch_size=2, batch_length=length, max_size=2 * frames)
            )
        )
        buffer.enable_transactions()
        return buffer

    def row(self, step, root=False):
        values = torch.tensor([float(step), 100.0 + step])
        return TensorDict(
            dict(
                episode=torch.arange(2, dtype=torch.int32),
                stoch=values[:, None, None].expand(2, 2, 2).clone(),
                deter=values[:, None].expand(2, 3).clone(),
                action=values[:, None].clone(),
                is_branch=torch.tensor([[root], [False]]),
                is_recon=torch.tensor([[step >= 3], [False]]),
            ),
            [2],
        )

    def fixed_sample(self, buffer, start):
        # Force boundary placements while exercising the real storage, readiness
        # filtering, sample assembly, action shift and cache writeback.
        slots = torch.arange(2)[:, None].expand(2, buffer.batch_length + 1)
        times = torch.arange(start, start + buffer.batch_length + 1)[None].expand_as(slots)

        def sample(return_info):
            self.assertTrue(return_info)
            return buffer._buffer.storage[times, slots].reshape(-1), {"index": (times.reshape(-1), slots.reshape(-1))}

        with mock.patch.object(buffer._buffer, "sample", side_effect=sample):
            return buffer.sample()

    def test_all_root_positions_and_cache_writeback(self):
        # Root at context row, training first/middle/last row, or before window.
        for start in (3, 2, 1, 0, 4):
            with self.subTest(start=start):
                buffer = self.make_buffer()
                for step in range(8):
                    if step == 3:
                        buffer.reserve(0)
                    buffer.add_transition(self.row(step, root=step == 3))
                    if step == 6:
                        self.assertTrue(buffer.commit(0))
                storage = buffer._buffer.storage
                before = storage[:].transpose(0, 1).clone()
                data, index, initial = self.fixed_sample(buffer, start)
                for key, state in zip(("stoch", "deter"), initial):
                    torch.testing.assert_close(state, before[key][:, start])
                    latent = before[key][:, start + 1 : start + 4]
                    roots = data["is_branch"].reshape(2, 3, *([1] * (latent.ndim - 2)))
                    torch.testing.assert_close(data["branch_" + key], torch.where(roots, latent, 0))
                    self.assertNotIn("branch_" + key, storage[:].keys())
                torch.testing.assert_close(data["action"], before["action"][:, start : start + 3])
                torch.testing.assert_close(storage[:].transpose(0, 1)["action"], before["action"])

                buffer.update(index, torch.full_like(data["stoch"], -7), torch.full_like(data["deter"], -8))
                for key, fill in (("stoch", -7), ("deter", -8)):
                    expected = before[key].clone()
                    for slot in range(2):
                        for step in range(start + 1, start + 4):
                            if not (slot == 0 and step == 3):
                                expected[slot, step] = fill
                    torch.testing.assert_close(storage[:].transpose(0, 1)[key], expected)
                # A second sample must still see the original branch seed.
                again, _, seed = self.fixed_sample(buffer, start)
                if start == 3:
                    torch.testing.assert_close(seed[1][0], before["deter"][0, 3])
                elif start < 3:
                    torch.testing.assert_close(again["branch_deter"][0, 2 - start], before["deter"][0, 3])

    def test_storage_bytes_and_ring_reuse(self):
        buffer = self.make_buffer(frames=4)
        for step in range(4):
            buffer.add_transition(self.row(step, root=step == 0))
        storage = buffer._buffer.storage
        latent_bytes = sum(storage[:][key].numel() * storage[:][key].element_size() for key in ("stoch", "deter"))
        self.assertEqual(latent_bytes, 8 * (2 * 2 + 3) * 4)
        self.fixed_sample(buffer, 0)
        self.assertNotIn("branch_stoch", storage[:].keys())
        self.assertNotIn("branch_deter", storage[:].keys())
        # Ordinary data replacing a root must be writable again.
        for step in range(4, 8):
            buffer.add_transition(self.row(step))
        data, index, _ = self.fixed_sample(buffer, 0)
        buffer.update(index, torch.ones_like(data["stoch"]), torch.ones_like(data["deter"]))
        torch.testing.assert_close(storage[:].transpose(0, 1)["deter"][:, 1:], torch.ones(2, 3, 3))
        self.assertFalse(storage[:]["is_branch"].any())
        buffer.update([torch.tensor([[0]]), torch.tensor([[0]])], torch.ones(1, 1, 2, 2), torch.ones(1, 1, 3))
        torch.testing.assert_close(storage[0, 0]["deter"], torch.ones(3))

    def test_short_branch_pending_discard_and_commit(self):
        buffer = self.make_buffer(frames=4, length=1)
        buffer.reserve(0)
        buffer.add_transition(self.row(0, root=True))
        buffer.add_transition(self.row(1))
        # Two copies of the same forced root window are used to exclude the
        # ordinary second slot from satisfying the batch on its own.
        slots, times = torch.zeros(4, dtype=torch.long), torch.tensor([0, 1, 0, 1])

        def sample(return_info):
            return buffer._buffer.storage[times, slots], {"index": (times, slots)}

        with mock.patch.object(buffer._buffer, "sample", side_effect=sample):
            with self.assertRaises(ReplayNotReady):
                buffer.sample()
            self.assertTrue(buffer.commit(0))
            _, _, initial = buffer.sample()
            torch.testing.assert_close(initial[1], torch.zeros(2, 3))
        buffer.reserve(0)
        buffer.add_transition(self.row(2, root=True))
        buffer.discard(0)
        self.assertFalse(buffer._buffer.storage[2, 0]["_ready"])


if __name__ == "__main__":
    unittest.main()
