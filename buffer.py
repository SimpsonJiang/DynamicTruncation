import torch
from tensordict import TensorDict
from torchrl.data.replay_buffers import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler


class ReplayNotReady(RuntimeError):
    """No committed sequence is available; retain the trainer update credit."""


class Buffer:
    def __init__(self, config):
        self.device = torch.device(config.device)
        self.storage_device = torch.device(config.storage_device)
        self.batch_size = int(config.batch_size)
        self.batch_length = int(config.batch_length)
        self.num_eps = 0
        self.transactional = False
        self.generation = 0
        self.pending = {}
        self._buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=config.max_size, device=self.storage_device, ndim=2),
            sampler=SliceSampler(
                num_slices=self.batch_size, end_key=None, traj_key="episode", truncated_key=None, strict_length=True
            ),
            prefetch=0,
            batch_size=self.batch_size * (self.batch_length + 1),  # +1 for context
        )

    def enable_transactions(self):
        if not self.transactional and self.count():
            raise ValueError("Enable replay transactions before collecting data")
        self.transactional = True

    def has_pending(self, slot=None):
        return bool(self.pending) if slot is None else slot in self.pending

    def retention_frames(self, env_num):
        """Maximum stored vector frames, for source-context lifetime management."""
        return int(self._buffer.storage.max_size // env_num)

    def add_transition(self, data):
        # This is batched data and lifted for storage.
        # (N, ...) -> (N, 1, ...), N=config.env_num
        if not self.transactional:
            return self._buffer.extend(data.unsqueeze(1))
        data = data.clone()
        data["_generation"] = torch.full(data.batch_size, self.generation, dtype=torch.int64, device=data.device)
        data["_ready"] = torch.ones(data.batch_size, dtype=torch.bool, device=data.device)
        staged = {}
        for slot in self.pending:
            staged[slot] = data[slot].clone()
            data["_ready"][slot] = False
        indices = self._buffer.extend(data.unsqueeze(1))
        for slot, row in staged.items():
            location = indices[slot].cpu().tolist()
            self.pending[slot].append((tuple(location), self.generation, row.to(self.storage_device)))
        self.generation += 1
        return indices

    def reserve(self, slot):
        if slot in self.pending:
            raise RuntimeError("Slot already has a pending branch")
        self.pending[slot] = []

    def commit(self, slot):
        rows = self.pending.pop(slot)
        if not rows or any(
            int(self._buffer.storage[index]["_generation"]) != generation for index, generation, _ in rows
        ):
            return False
        # Validate the entire transaction before exposing any row. Collector and
        # sampler run on the same thread; no sampling can interleave this commit.
        for index, _, row in rows:
            row["_ready"] = torch.ones((), dtype=torch.bool, device=row.device)
            self._buffer.storage[index] = row
        return True

    def discard(self, slot):
        self.pending.pop(slot, None)

    def _committed_sample(self):
        rows, indices, count = [], [[], []], 0
        for _ in range(32):
            td, info = self._buffer.sample(return_info=True)
            td = td.view(-1, self.batch_length + 1)
            gen = td["_generation"]
            valid = td["_ready"].all(1) & ((gen[:, 1:] - gen[:, :-1]) == 1).all(1)
            take = valid.nonzero().flatten()[: self.batch_size - count]
            if take.numel():
                rows.append(td[take])
                for j in range(2):
                    indices[j].append(info["index"][j].view(-1, self.batch_length + 1)[take])
                count += len(take)
            if count == self.batch_size:
                return torch.cat(rows), {"index": tuple(torch.cat(v) for v in indices)}
        raise ReplayNotReady("No full batch of committed windows; update postponed")

    def sample(self):
        if self.transactional:
            sample_td, info = self._committed_sample()
        else:
            sample_td, info = self._buffer.sample(return_info=True)
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
        if self.transactional:
            root = sample_td["is_branch"][:, 0]
            initial = tuple(
                torch.where(root.reshape(-1, *([1] * (x.ndim - 1))), sample_td[key][:, 0], x)
                for x, key in zip(initial, ("branch_stoch", "branch_deter"))
            )
        data = sample_td[:, 1:]
        data.set_("action", sample_td["action"][:, :-1].clone())  # action is 1 step back
        index = [ind.view(-1, self.batch_length + 1)[:, 1:] for ind in info["index"]]
        return data, index, initial

    def update(self, index, stoch, deter):
        # Flatten the data
        index = [ind.reshape(-1) for ind in index]
        # (B, T, S, K) -> (B*T, S, K)
        stoch = stoch.reshape(-1, *stoch.shape[2:])
        # (B, T, D) -> (B*T, D)
        deter = deter.reshape(-1, *deter.shape[2:])
        if self.transactional:
            root = self._buffer.storage[(index[0], index[1])]["is_branch"].reshape(-1)
            keep = ~root
            index = [v[keep] for v in index]
            stoch, deter = stoch[keep], deter[keep]
        # In storage, the length is the first dimension, and the batch (number of environments) is the second dimension.
        n = index[0].shape[0]
        self._buffer[index[1], index[0]] = TensorDict({"stoch": stoch, "deter": deter}, batch_size=(n,))

    def count(self):
        if self._buffer.storage.shape is None:
            return 0
        return self._buffer.storage.shape.numel()
