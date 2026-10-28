from dataclasses import dataclass

import torch
from tensordict import TensorDict
from torchrl.data.replay_buffers import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler


class Buffer:
    def __init__(self, config):
        self.device = torch.device(config.device)
        self.storage_device = torch.device(config.storage_device)
        self.batch_size = int(config.batch_size)
        self.batch_length = int(config.batch_length)
        self.num_eps = 0
        self._buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=config.max_size, device=self.storage_device, ndim=2),
            sampler=SliceSampler(
                slice_len=self.batch_length + 1, end_key=None, traj_key="episode", truncated_key=None, strict_length=True
            ),
            prefetch=0,
        )

    def add_transition(self, data):
        # This is batched data and lifted for storage.
        # (N, ...) -> (N, 1, ...), N=config.env_num
        self._buffer.extend(data.unsqueeze(1))

    def can_sample(self):
        shape = self._buffer.storage.shape
        return shape is not None and shape[0] >= self.batch_length + 1

    def sample(self, batch_size=None):
        size = self.batch_size if batch_size is None else int(batch_size)
        sample_td, info = self._buffer.sample(batch_size=size * (self.batch_length + 1), return_info=True)
        # The sampler returns a flattened batch of length B*(T+1).
        # (B*(T+1), ...) -> (B, T+1, ...)
        sample_td = sample_td.view(-1, self.batch_length + 1)
        src_dev = sample_td.device
        if src_dev.type == "cpu" and self.device.type == "cuda":
            sample_td = sample_td.pin_memory().to(self.device, non_blocking=True)
        elif src_dev != self.device:
            sample_td = sample_td.to(self.device, non_blocking=True)
        # The initial ones are used only to extract the latent vector
        initial = (sample_td["stoch"][:, 0], sample_td["deter"][:, 0])
        data = sample_td[:, 1:]
        # Source and destination overlap in sample_td. Clone before shifting;
        # otherwise an in-place copy can propagate the first action across time.
        data.set_("action", sample_td["action"][:, :-1].clone())  # action is 1 step back
        index = [ind.view(-1, self.batch_length + 1)[:, 1:] for ind in info["index"]]
        return data, index, initial

    def update(self, index, stoch, deter):
        # Flatten the data
        index = [ind.reshape(-1).to(self.storage_device) for ind in index]
        # (B, T, S, K) -> (B*T, S, K)
        stoch = stoch.reshape(-1, *stoch.shape[2:]).to(self.storage_device)
        # (B, T, D) -> (B*T, D)
        deter = deter.reshape(-1, *deter.shape[2:]).to(self.storage_device)
        # In storage, the length is the first dimension, and the batch (number of environments) is the second dimension.
        n = index[0].shape[0]
        self._buffer[index[1], index[0]] = TensorDict({"stoch": stoch, "deter": deter}, batch_size=(n,))

    def count(self):
        if self._buffer.storage.shape is None:
            return 0
        return self._buffer.storage.shape.numel()


@dataclass
class MixedIndex:
    normal: object
    branch: object
    normal_size: int


class MixedReplay:
    """Two fixed-slot storages, with sequence-level mixing and routed latent writes."""

    def __init__(self, normal, branch, fraction):
        if normal.batch_size < 2 or not 0 < fraction < 1:
            raise ValueError("Mixed replay requires batch_size >= 2 and 0 < fraction < 1")
        if normal.device != branch.device or normal.storage_device != branch.storage_device:
            raise ValueError("Branch replay must use the same devices as normal replay")
        self.normal, self.branch = normal, branch
        self.batch_size = normal.batch_size
        self.branch_size = min(self.batch_size - 1, max(1, round(self.batch_size * fraction)))
        self.last_branch_fraction = 0.0

    def add_transition(self, data):
        self.normal.add_transition(data)

    def count(self):
        return self.normal.count()

    def can_sample(self):
        return self.normal.can_sample()

    def sample(self):
        if not self.branch.can_sample():
            data, index, initial = self.normal.sample()
            self.last_branch_fraction = 0.0
            return data, MixedIndex(index, None, self.batch_size), initial
        normal_size = self.batch_size - self.branch_size
        data_n, index_n, initial_n = self.normal.sample(normal_size)
        data_b, index_b, initial_b = self.branch.sample(self.branch_size)
        self.last_branch_fraction = self.branch_size / self.batch_size
        return (
            torch.cat((data_n, data_b), dim=0),
            MixedIndex(index_n, index_b, normal_size),
            tuple(torch.cat((n, b), dim=0) for n, b in zip(initial_n, initial_b)),
        )

    def update(self, index, stoch, deter):
        n = index.normal_size
        self.normal.update(index.normal, stoch[:n], deter[:n])
        if index.branch is not None:
            self.branch.update(index.branch, stoch[n:], deter[n:])
