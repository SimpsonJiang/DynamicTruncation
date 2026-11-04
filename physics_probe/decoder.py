"""Independent MLP probe with train-only normalization and geometry-aware targets."""

import copy

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


class StateDecoder(nn.Module):
    def __init__(self, input_dim, schema, hidden=256):
        super().__init__()
        self.codec = StateCodec(schema)
        self.input_dim, self.hidden = input_dim, hidden
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.codec.dim),
        )
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_scale", torch.ones(input_dim))
        self.register_buffer("target_mean", torch.zeros(self.codec.dim))
        self.register_buffer("target_scale", torch.ones(self.codec.dim))
        self.register_buffer("active", torch.ones(self.codec.dim, dtype=torch.bool))

    @torch.no_grad()
    def normalize_from_training(self, inputs, cores):
        target = self.codec.encode(cores)
        self.input_mean.copy_(inputs.mean(0))
        self.input_scale.copy_(inputs.std(0, unbiased=False).clamp_min(1e-3))
        self.target_mean.copy_(target.mean(0))
        std = target.std(0, unbiased=False)
        self.active.copy_(std > 1e-6)
        self.target_scale.copy_(std.clamp_min(1e-3))
        # Geometry coordinates use their natural scale, and are never treated as constants.
        for i in self.codec.quaternions:
            self.target_mean[i : i + 4] = 0
            self.target_scale[i : i + 4] = 1
            self.active[i : i + 4] = True
        for j, i in enumerate(self.codec.periodic):
            for index in (i, self.codec.core_dim + j):
                self.target_mean[index], self.target_scale[index], self.active[index] = 0, 1, True

    def encoded_prediction(self, inputs):
        normalized = self.net((inputs - self.input_mean) / self.input_scale)
        normalized = normalized * self.active
        return normalized * self.target_scale + self.target_mean

    def forward(self, inputs):
        return self.codec.decode(self.encoded_prediction(inputs))

    def loss(self, inputs, cores):
        prediction = self.encoded_prediction(inputs)
        target = self.codec.encode(cores)
        squared = ((prediction - target) / self.target_scale).square()
        mask = self.active.clone()
        rotation_terms = []
        for i in self.codec.quaternions:
            p, t = prediction[..., i : i + 4], target[..., i : i + 4]
            p = torch.nn.functional.normalize(p, dim=-1, eps=1e-8)
            t = torch.nn.functional.normalize(t, dim=-1, eps=1e-8)
            rotation_terms.append(1 - (p * t).sum(-1).square())
            mask[i : i + 4] = False
        # Equal weight per non-empty physical group rather than per coordinate.
        nq, nv = self.codec.schema["nq"], self.codec.schema["nv"]
        groups = [
            list(range(nq)) + list(range(self.codec.core_dim, self.codec.dim)),
            list(range(nq, nq + nv)),
            list(range(nq + nv, self.codec.core_dim)),
        ]
        terms = []
        for indices in groups:
            indices = [i for i in indices if mask[i]]
            if indices:
                terms.append(squared[..., indices].mean())
        terms.extend(t.mean() for t in rotation_terms)
        if not terms:
            return prediction.sum() * 0
        return torch.stack(terms).mean()

    def export(self):
        return {
            "input_dim": self.input_dim,
            "hidden": self.hidden,
            "schema": self.codec.schema,
            "state_dict": {k: v.detach().cpu().clone() for k, v in self.state_dict().items()},
        }

    @classmethod
    def from_export(cls, exported, device="cpu"):
        model = cls(exported["input_dim"], exported["schema"], exported["hidden"]).to(device)
        model.load_state_dict(copy.deepcopy(exported["state_dict"]), strict=True)
        return model.eval()
