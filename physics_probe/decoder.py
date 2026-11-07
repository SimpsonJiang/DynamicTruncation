"""Independent MLP probe with train-only normalization and geometry-aware targets."""

import copy

import torch
from torch import nn

from physics_state import PhysicsDecoderBase, StateCodec

__all__ = ["StateCodec", "StateDecoder", "ImageStateDecoder"]


class StateDecoder(PhysicsDecoderBase):
    @torch.no_grad()
    def normalize_from_training(self, inputs, cores):
        inputs = inputs.to(self.input_mean)
        self.input_mean.copy_(inputs.mean(0))
        self.input_scale.copy_(inputs.std(0, unbiased=False).clamp_min(1e-3))
        self.normalize_targets(cores)

    @torch.no_grad()
    def normalize_targets(self, cores):
        target = self.codec.encode(cores.to(self.target_mean))
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

    def loss(self, inputs, cores):
        prediction = self.encoded_prediction(inputs)
        target = self.codec.encode(cores.to(self.target_mean))
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
            "version": 2,
            "dtype": str(self.dtype).removeprefix("torch."),
            "input_dim": self.input_dim,
            "hidden": self.hidden,
            "schema": self.codec.schema,
            "state_dict": {k: v.detach().cpu().clone() for k, v in self.state_dict().items()},
        }

    @classmethod
    def from_export(cls, exported, device="cpu"):
        # Construct in the stored dtype BEFORE load_state_dict, which otherwise
        # silently casts saved double parameters into a float32 destination.
        dtypes = {v.dtype for v in exported["state_dict"].values() if v.is_floating_point()}
        if len(dtypes) != 1 or not dtypes.issubset({torch.float32, torch.float64}):
            raise ValueError(f"Inconsistent or unsupported decoder checkpoint dtypes: {dtypes}")
        dtype = next(iter(dtypes))
        name = str(dtype).removeprefix("torch.")
        if exported.get("dtype", name) != name:
            raise ValueError("Decoder dtype metadata disagrees with saved tensors")
        architecture = exported.get("architecture", "mlp")
        if architecture == "image_cnn":
            model = ImageStateDecoder(exported["image_shape"], exported["schema"], exported["hidden"], dtype=dtype)
        elif architecture == "mlp":
            model = cls(exported["input_dim"], exported["schema"], exported["hidden"], dtype=dtype)
        else:
            raise ValueError(f"Unknown state decoder architecture: {architecture}")
        model = model.to(device)
        model.load_state_dict(copy.deepcopy(exported["state_dict"]), strict=True)
        return model.eval()


class ImageStateDecoder(StateDecoder):
    """Single-frame RGB control: four strided convolutions and the same state loss.

    Pixels stay uint8 in the dataset/host batches; only minibatches are promoted
    to float64. Input scaling is fixed at pixel / 255 - 0.5, not fitted on test data.
    This control has no temporal context and receives no privileged proprio input.
    """

    def __init__(self, image_shape, schema, hidden=256, dtype=torch.float64):
        image_shape = tuple(image_shape)
        if image_shape != (64, 64, 3):
            raise ValueError("Image state decoder currently supports saved 64x64 RGB observations")
        super().__init__(1, schema, hidden, dtype)
        self.image_shape = image_shape
        layers, channels = [], 3
        for output_channels in (16, 32, 64, 128):
            layers += [nn.Conv2d(channels, output_channels, 3, stride=2, padding=1, dtype=dtype), nn.SiLU()]
            channels = output_channels
        self.net = nn.Sequential(
            *layers,
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, hidden, dtype=dtype),
            nn.SiLU(),
            nn.Linear(hidden, hidden, dtype=dtype),
            nn.SiLU(),
            nn.Linear(hidden, self.codec.dim, dtype=dtype),
        )

    @torch.no_grad()
    def normalize_from_training(self, inputs, cores):
        if tuple(inputs.shape[1:]) != self.image_shape or inputs.dtype != torch.uint8:
            raise ValueError("Train image decoder from original uint8 NHWC pixels")
        self.normalize_targets(cores)

    def encoded_prediction(self, inputs):
        if tuple(inputs.shape[1:]) != self.image_shape:
            raise ValueError("Image state decoder expects [batch,64,64,3]")
        pixels = (inputs.to(self.input_mean) / 255.0 - 0.5).permute(0, 3, 1, 2)
        normalized = self.net(pixels) * self.active
        return normalized * self.target_scale + self.target_mean

    def export(self):
        result = super().export()
        result.update(
            version=3,
            architecture="image_cnn",
            image_shape=list(self.image_shape),
            channels=[16, 32, 64, 128],
            pixel_normalization="uint8 / 255 - 0.5",
            temporal_context=1,
        )
        return result
