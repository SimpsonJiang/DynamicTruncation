"""Opt-in online float64 physics head; the policy never sees its labels."""

import copy
import hashlib
import json

import torch
from omegaconf import OmegaConf

from physics_state import PhysicsDecoderBase


class OnlinePhysicsDecoder(PhysicsDecoderBase):
    def __init__(self, input_dim, config):
        if config.schema is None:
            raise ValueError("Enabled physics decoder requires an environment schema")
        schema = (
            OmegaConf.to_container(config.schema, resolve=True)
            if OmegaConf.is_config(config.schema)
            else dict(config.schema)
        )
        # Network initialization must not consume the policy's RNG stream.
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            # torch.manual_seed also reseeds CUDA; use only the CPU generator.
            torch.random.default_generator.manual_seed(1701)
            super().__init__(input_dim, schema, int(config.hidden), torch.float64)
        self.decay, self.eps = float(config.norm_decay), float(config.norm_eps)
        if not 0 <= self.decay < 1 or not 0 < self.eps < float("inf"):
            raise ValueError("Invalid physics normalizer settings")
        self.register_buffer("input_second", torch.ones(input_dim, dtype=torch.float64))
        self.register_buffer("target_second", torch.ones(self.codec.dim, dtype=torch.float64))
        self.register_buffer("norm_updates", torch.zeros((), dtype=torch.int64))
        signature = hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).digest()
        self.register_buffer("schema_signature", torch.tensor(list(signature), dtype=torch.uint8))
        self.register_load_state_dict_pre_hook(self._check_checkpoint)
        rotation = {j for i in self.codec.quaternions for j in range(i, i + 4)}
        nq, nv = schema["nq"], schema["nv"]
        self.loss_groups = [
            [i for i in list(range(nq)) + list(range(self.codec.core_dim, self.codec.dim)) if i not in rotation],
            list(range(nq, nq + nv)),
            list(range(nq + nv, self.codec.core_dim)),
        ]

    @staticmethod
    def _check_checkpoint(module, state, prefix, *args):
        signature = state.get(prefix + "schema_signature")
        if signature is None or not torch.equal(signature.cpu(), module.schema_signature.cpu()):
            raise RuntimeError("Physics decoder schema mismatch")
        for key, value in state.items():
            if key.startswith(prefix) and value.is_floating_point() and value.dtype != torch.float64:
                raise RuntimeError(f"Physics decoder requires float64 checkpoint: {key}")

    @staticmethod
    def validate_labels(core):
        if core.dtype != torch.float64:
            raise TypeError("Physics labels must arrive as float64; casting rounded labels is not allowed")
        if not torch.isfinite(core).all():
            raise ValueError("Nonfinite physics label")

    @torch.no_grad()
    def update_statistics(self, inputs, core):
        if not self.training:
            return
        x, y = inputs.detach().double(), self.codec.encode(core.detach())
        decay = torch.where(
            self.norm_updates > 0, self.input_mean.new_tensor(self.decay), self.input_mean.new_zeros(())
        )
        for name, values in (("input", x), ("target", y)):
            mean, second = getattr(self, name + "_mean"), getattr(self, name + "_second")
            mean.lerp_(values.mean(0), 1 - decay)
            second.lerp_(values.square().mean(0), 1 - decay)
            getattr(self, name + "_scale").copy_((second - mean.square()).clamp_min(0).sqrt().clamp_min(self.eps))
        for i in self.codec.quaternions:
            self.target_mean[i : i + 4] = 0
            self.target_scale[i : i + 4] = 1
        for j, i in enumerate(self.codec.periodic):
            self.target_mean[[i, self.codec.core_dim + j]] = 0
            self.target_scale[[i, self.codec.core_dim + j]] = 1
        self.norm_updates.add_(1)

    def loss(self, inputs, cores):
        # Online coordinates are all active. Static schema groups avoid a GPU
        # scalar read per coordinate in the offline probe's variable-active loss.
        prediction = self.encoded_prediction(inputs)
        target = self.codec.encode(cores)
        squared = ((prediction - target) / self.target_scale).square()
        terms = [squared[..., indices].mean() for indices in self.loss_groups if indices]
        for i in self.codec.quaternions:
            p = torch.nn.functional.normalize(prediction[..., i : i + 4], dim=-1, eps=1e-8)
            t = torch.nn.functional.normalize(target[..., i : i + 4], dim=-1, eps=1e-8)
            terms.append((1 - (p * t).sum(-1).square()).mean())
        return torch.stack(terms).mean() if terms else prediction.sum() * 0

    def training_loss(self, features, core, valid=None):
        self.validate_labels(core)
        features, core = features.reshape(-1, features.shape[-1]), core.reshape(-1, core.shape[-1])
        if valid is not None:
            valid = valid.reshape(-1).bool()
            features, core = features[valid], core[valid]
        if not len(core):
            return sum(p.sum() * 0 for p in self.parameters())
        with torch.autocast(device_type=features.device.type, enabled=False):
            self.update_statistics(features, core)
            return self.loss(features.double(), core)

    def export(self):
        """Export the online architecture and EMA state, independently of probe formats."""
        return {
            "version": 1,
            "architecture": "online_physics_mlp",
            "dtype": "float64",
            "input_dim": self.input_dim,
            "hidden": self.hidden,
            "schema": copy.deepcopy(self.codec.schema),
            "norm_decay": self.decay,
            "norm_eps": self.eps,
            "state_dict": {k: v.detach().cpu().clone() for k, v in self.state_dict().items()},
        }

    @classmethod
    def from_export(cls, exported, device="cpu"):
        if exported.get("architecture") != "online_physics_mlp" or exported.get("version") != 1:
            raise ValueError("Expected an online physics decoder export (version 1)")
        if exported.get("dtype") != "float64":
            raise ValueError("Online physics decoder exports must preserve float64")
        config = OmegaConf.create({key: exported[key] for key in ("schema", "hidden", "norm_decay", "norm_eps")})
        model = cls(exported["input_dim"], config).to(device)
        model.load_state_dict(copy.deepcopy(exported["state_dict"]), strict=True)
        return model.eval()


class Float64Adam(torch.optim.Adam):
    """Reject rounded optimizer moments instead of silently promoting them on load."""

    def load_state_dict(self, state_dict):
        for state in state_dict["state"].values():
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if key in state and state[key].dtype != torch.float64:
                    raise ValueError(f"Physics optimizer requires float64 {key}")
        return super().load_state_dict(state_dict)
