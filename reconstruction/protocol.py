"""Named RPC results and the compact physics-context layout used in replay."""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


class EnvStepResult(NamedTuple):
    observation: dict
    reward: float
    done: bool
    core: np.ndarray | None = None
    context_vector: np.ndarray | None = None
    report: dict | None = None
    context: dict | None = None


@dataclass(frozen=True)
class PhysicsContext:
    """Keep [time, policy_steps, inner_steps, ctrl...] on disk, name it at boundaries."""

    time: float
    policy_steps: int
    inner_steps: int
    control: np.ndarray

    def to_vector(self):
        return np.r_[self.time, self.policy_steps, self.inner_steps, self.control].astype(np.float64)

    @classmethod
    def from_vector(cls, vector):
        vector = np.asarray(vector)
        if vector.ndim != 1 or len(vector) < 3 or not np.isfinite(vector).all():
            raise ValueError("Invalid physics context vector")
        time, policy_steps, inner_steps = vector[:3]
        if time < 0 or policy_steps < 0 or inner_steps < 0:
            raise ValueError("Negative physics time/counter")
        if policy_steps != int(policy_steps) or inner_steps != int(inner_steps):
            raise ValueError("Physics step counters must be integers")
        return cls(float(time), int(policy_steps), int(inner_steps), vector[3:].copy())

    def advance(self, depth, actions, context):
        control = self.control.copy()
        if depth:
            previous = actions[depth - 1].astype(np.float32)
            control[:] = (previous + 1) / 2 * (context["action_high"] - context["action_low"]) + context["action_low"]
        return PhysicsContext(
            self.time + depth * context["policy_dt"],
            self.policy_steps + depth,
            self.inner_steps + depth * context["repeat"],
            control,
        )

    def remaining_steps(self, context, depth=0):
        return int(
            min(
                context["limit"] - self.policy_steps - depth,
                (context["inner_limit"] - self.inner_steps) // context["repeat"] - depth,
            )
        )
