"""Shared physical-state codec and MLP; no collection or training policy."""

import torch
from torch import nn


class StateCodec:
    """Encode unbounded hinges by sin/cos, handle quaternion sign equivalence.

    Absolute translations remain prediction targets: no unvalidated symmetry or
    access to the target's global origin is assumed. Full raw errors remain reported.
    """

    def __init__(self, schema):
        self.schema = schema
        self.core_dim = schema["nq"] + schema["nv"] + schema["na"]
        self.periodic = []
        self.quaternions = []
        for kind, adr, limited in zip(schema["joint_types"], schema["joint_addresses"], schema["joint_limited"]):
            if kind == 3 and not limited:  # mjJNT_HINGE
                self.periodic.append(adr)
            elif kind in (0, 1):  # free, ball
                self.quaternions.append(adr + 3 if kind == 0 else adr)
        self.dim = self.core_dim + len(self.periodic)

    def encode(self, core):
        result = core.clone()
        extra = []
        for i in self.periodic:
            result[..., i] = core[..., i].sin()
            extra.append(core[..., i : i + 1].cos())
        return torch.cat([result, *extra], dim=-1)

    def decode(self, encoded):
        result = encoded[..., : self.core_dim].clone()
        for j, i in enumerate(self.periodic):
            result[..., i] = torch.atan2(encoded[..., i], encoded[..., self.core_dim + j])
        for i in self.quaternions:
            q = result[..., i : i + 4].clone()
            result[..., i : i + 4] = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return result


class PhysicsDecoderBase(nn.Module):
    def __init__(self, input_dim, schema, hidden=256, dtype=torch.float64):
        super().__init__()
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Decoder supports float64 and legacy float32 checkpoints only")
        self.codec = StateCodec(schema)
        self.input_dim, self.hidden = input_dim, hidden
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden, dtype=dtype),
            nn.SiLU(),
            nn.Linear(hidden, hidden, dtype=dtype),
            nn.SiLU(),
            nn.Linear(hidden, self.codec.dim, dtype=dtype),
        )
        self.register_buffer("input_mean", torch.zeros(input_dim, dtype=dtype))
        self.register_buffer("input_scale", torch.ones(input_dim, dtype=dtype))
        self.register_buffer("target_mean", torch.zeros(self.codec.dim, dtype=dtype))
        self.register_buffer("target_scale", torch.ones(self.codec.dim, dtype=dtype))
        self.register_buffer("active", torch.ones(self.codec.dim, dtype=torch.bool))

    @property
    def dtype(self):
        return self.input_mean.dtype

    def encoded_prediction(self, inputs):
        # Frozen Dreamer keeps its own precision. Promote only at the decoder boundary.
        inputs = inputs.to(self.input_mean)
        normalized = self.net((inputs - self.input_mean) / self.input_scale)
        normalized = normalized * self.active
        return normalized * self.target_scale + self.target_mean

    def forward(self, inputs):
        return self.codec.decode(self.encoded_prediction(inputs))
