"""Task-specific physical state codecs on a shared detached prediction head."""

import copy
from abc import ABC, abstractmethod

import torch
from torch import nn

from distributions import symlog, symexp
from networks import MLPHead


class StateDecoder(nn.Module, ABC):
    """Base for task-specific state decoders.

    Subclasses define encoded_size, validate their state spec in setup_state_spec,
    and implement encode/decode for [..., physical_size] <-> [..., encoded_size].
    Override error_metrics and prediction_rejection_reason when needed. Keep
    forward/loss inherited so input detachment and supervision stay consistent.
    """

    @property
    @abstractmethod
    def encoded_size(self):
        """Number of scalar outputs in the task's encoded representation."""
        raise NotImplementedError

    def __init__(self, config, feat_size, state_spec):
        super().__init__()
        if config.dist.name != "identity":
            raise ValueError("StateDecoder requires dist.name=identity")
        if config.shape is not None and tuple(config.shape) != (self.encoded_size,):
            raise ValueError(f"StateDecoder requires shape=[{self.encoded_size}] or null")
        self.setup_state_spec(state_spec)
        # Resolve task-dependent output size without mutating the Hydra config.
        head_config = copy.deepcopy(config)
        head_config.shape = [self.encoded_size]
        self.net = MLPHead(head_config, feat_size)

    @abstractmethod
    def setup_state_spec(self, state_spec):
        """Validate the task spec and register any codec constants as buffers."""
        raise NotImplementedError

    def forward(self, feat):
        # Enforce the gradient boundary even for callers outside Dreamer.
        return self.net(feat.detach())

    @abstractmethod
    def encode(self, state):
        """Convert physical states to supervised targets in encoded space."""
        raise NotImplementedError

    @abstractmethod
    def decode(self, prediction):
        """Convert encoded predictions to physical states for the simulator."""
        raise NotImplementedError

    def error_metrics(self, errors):
        return {"state_decoder/mse": errors.mean()}

    def loss(self, feat, target):
        prediction = self(feat).float()
        encoded = self.encode(target.detach()).detach().float()
        errors = (prediction - encoded).square()
        metrics = {name: value.detach() for name, value in self.error_metrics(errors).items()}
        return errors.mean(), metrics

    def prediction_rejection_reason(self, prediction, **limits):
        """Collection-only validation; return None or a rejection label."""
        if not torch.isfinite(prediction).all():
            return "nonfinite"
        return None


class FingerStateDecoder(StateDecoder):
    """Finger Spin: two bounded joints, one periodic angle and three velocities."""

    encoded_size = 7

    def setup_state_spec(self, state_spec):
        if state_spec is None or tuple(state_spec["dimensions"]) != (3, 3, 0):
            raise ValueError("StateDecoder requires the Finger state spec (nq=3, nv=3, na=0)")
        bounds = torch.as_tensor(state_spec["joint_ranges"], dtype=torch.float32)
        if bounds.shape != (2, 2) or not torch.all(bounds[:, 1] > bounds[:, 0]):
            raise ValueError("Expected two finite, nonempty Finger joint ranges")
        if not torch.isfinite(bounds).all():
            raise ValueError("Joint ranges must be finite")
        self.register_buffer("center", bounds.mean(-1))
        self.register_buffer("scale", (bounds[:, 1] - bounds[:, 0]) / 2)

    def encode(self, state):
        state = state.detach().float()
        return torch.cat((
            (state[..., :2] - self.center) / self.scale,
            state[..., 2:3].sin(), state[..., 2:3].cos(),
            symlog(state[..., 3:6]),
        ), dim=-1)

    def decode(self, prediction):
        prediction = prediction.float()
        return torch.cat((
            prediction[..., :2] * self.scale + self.center,
            torch.atan2(prediction[..., 2:3], prediction[..., 3:4]),
            symexp(prediction[..., 4:7]),
        ), dim=-1)

    def error_metrics(self, errors):
        return {
            "state_decoder/joint_mse": errors[..., :2].mean(),
            "state_decoder/angle_mse": errors[..., 2:4].mean(),
            "state_decoder/velocity_symlog_mse": errors[..., 4:].mean(),
        }

    def prediction_rejection_reason(self, prediction, **limits):
        reason = super().prediction_rejection_reason(prediction, **limits)
        if reason:
            return reason
        if (prediction[..., 2:4].norm(dim=-1) < limits.get("min_angle_norm", 1e-6)).any():
            return "angle_norm"
        return None


# Explicit task dispatch: equal state dimensions do not imply equal codecs.
STATE_DECODERS = {"dmc_finger_spin": FingerStateDecoder}


def make_state_decoder(config, feat_size, state_spec):
    task = state_spec.get("task") if state_spec is not None else None
    if task not in STATE_DECODERS:
        raise ValueError(f"No StateDecoder registered for task {task!r}")
    decoder_cls = STATE_DECODERS[task]
    if not issubclass(decoder_cls, StateDecoder):
        raise TypeError("Task decoders must inherit StateDecoder")
    return decoder_cls(config, feat_size, state_spec)
