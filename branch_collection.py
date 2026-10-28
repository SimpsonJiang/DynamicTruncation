"""Real simulator rollouts from replay or decoded imagination states."""

import contextlib
import math
import re
import time
from collections import Counter

import numpy as np
import torch
from tensordict import TensorDict

from envs import make_env
from envs.parallel import Parallel


def validate_branch_config(config):
    cfg = config.branch_collection
    mode = str(cfg.mode)
    enabled = bool(config.model.state_decoder.enabled)
    if mode not in ("off", "replay_state", "imagined_state"):
        raise ValueError("branch_collection.mode must be off, replay_state, or imagined_state")
    if mode == "imagined_state" and not enabled:
        raise ValueError("imagined_state requires model.state_decoder.enabled=true")
    if mode == "imagined_state" and not config.model.dynamic_trunc.enabled:
        raise ValueError("imagined_state requires model.dynamic_trunc.enabled=true")
    if "imag_depth" in cfg:
        raise ValueError("branch_collection.imag_depth was removed: branches now start at actual VE cutoffs")
    if enabled or mode != "off":
        if (config.env.task != "dmc_finger_spin" or config.model.rep_loss != "dreamer"
                or re.match(config.env.encoder.cnn_keys, "image")
                or not all(re.match(config.env.encoder.mlp_keys, k) for k in ("position", "velocity", "touch"))):
            raise ValueError("State reconstruction/branching v1 requires dmc_finger_spin + proprio + dreamer")
    if enabled:
        for key in ("units", "layers"):
            value = config.model.state_decoder[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"state_decoder.{key} must be a positive integer")
        scale = float(config.model.loss_scales.state_recon)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("loss_scales.state_recon must be finite and positive")
    if mode != "off":
        for key in ("every_steps", "branches_per_round", "max_attempts", "horizon", "replay_capacity"):
            value = cfg[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"branch_collection.{key} must be a positive integer")
        if isinstance(cfg.warmup_steps, bool) or not isinstance(cfg.warmup_steps, int) or cfg.warmup_steps < 0:
            raise ValueError("branch_collection.warmup_steps must be nonnegative")
        if cfg.replay_capacity < config.batch_length + 1:
            raise ValueError("branch replay_capacity must hold at least one T+1 sequence")
        if config.batch_size < 2 or not 0 < cfg.replay_fraction < 1:
            raise ValueError("Branch mixing requires batch_size >= 2 and 0 < replay_fraction < 1")
        for key in ("max_velocity", "max_penetration", "min_angle_norm"):
            if not math.isfinite(float(cfg[key])) or cfg[key] <= 0:
                raise ValueError(f"branch_collection.{key} must be finite and positive")
    return enabled or mode != "off", mode == "replay_state"


def add_source_metadata(data, source, branch_id, origin_step, depth):
    """Same tensor schema for both buffers; none of these keys are observations."""
    for key, value in (("_source", source), ("_branch_id", branch_id),
                       ("_origin_step", origin_step), ("_imag_depth", depth)):
        data[key] = torch.full((*data.batch_size, 1), value, dtype=torch.int64, device=data.device)
    return data


class IsolatedTorchRNG:
    """Persistent CPU/device streams without advancing the main training streams."""

    def __init__(self, device, seed):
        self.device = torch.device(device)
        self.cpu_state = torch.Generator().manual_seed(seed).get_state()
        self.cuda_index = None
        if self.device.type == "cuda":
            self.cuda_index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            self.cuda_state = torch.Generator(device=self.device).manual_seed(seed).get_state()

    @contextlib.contextmanager
    def use(self):
        devices = [] if self.cuda_index is None else [self.cuda_index]
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(self.cpu_state)
            if self.cuda_index is not None:
                torch.cuda.set_rng_state(self.cuda_state, self.cuda_index)
            try:
                yield
            finally:
                self.cpu_state = torch.get_rng_state()
                if self.cuda_index is not None:
                    self.cuda_state = torch.cuda.get_rng_state(self.cuda_index)


class BranchEnvironment:
    """Worker-side adapter. Expected invalid physics becomes a rejected candidate."""

    def __init__(self, env_config, branch_config, capture_snapshot):
        from dm_control.rl.control import PhysicsError

        self.physics_error = PhysicsError
        self.config = branch_config
        self.env = make_env(env_config, 100000, capture_state=True, capture_snapshot=capture_snapshot)

    def restore(self, state, kind):
        try:
            obs = self.env.restore_state(
                state, kind=kind, max_velocity=self.config.max_velocity,
                max_penetration=self.config.max_penetration,
            )
            return dict(ok=True, observation=obs)
        except (ValueError, self.physics_error) as error:
            return dict(ok=False, reason=str(error))

    def step(self, action):
        base = self.env.unwrapped
        if not np.isfinite(action).all():
            return dict(ok=False, reason="nonfinite", steps=0)
        try:
            obs, reward, done, _ = self.env.step(action)
            reason = base.state_rejection_reason(self.config.max_velocity, self.config.max_penetration)
            if reason:
                return dict(ok=False, reason=reason, steps=base.last_env_steps)
            return dict(ok=True, observation=obs, reward=reward, done=done, steps=base.last_env_steps)
        except (self.physics_error, FloatingPointError) as error:
            return dict(ok=False, reason="physics_error", detail=str(error), steps=base.last_env_steps)


class BranchCollector:
    def __init__(self, config, env_config, device, seed):
        self.config = config
        self.env_config = env_config
        self.device = torch.device(device)
        self.rng = IsolatedTorchRNG(device, seed + 100000)
        self.next_normal_step = int(config.warmup_steps)
        self.worker = None
        self.branch_id = 0
        self.candidates = []

    def close(self):
        if self.worker is not None:
            self.worker.close()
            self.worker = None

    def _worker(self):
        if self.worker is None:
            env_cfg, cfg = self.env_config, self.config
            self.worker = Parallel(lambda: BranchEnvironment(env_cfg, cfg, cfg.mode == "replay_state"), "process")
        return self.worker

    def due(self, normal_steps):
        return normal_steps >= self.next_normal_step

    @torch.no_grad()
    def offer_candidates(self, packet):
        """Keep a bounded sample of cutoffs from the latest training update.

        Copy before the next compiled act/update can reuse CUDA graph outputs.
        Sampling uses the branch RNG and does not advance the training stream.
        """
        self.candidates = []
        if self.config.mode != "imagined_state" or packet is None:
            return
        eligible = packet["eligible"].nonzero().flatten()
        with self.rng.use():
            selected = eligible[torch.randperm(eligible.numel(), device=eligible.device)[:self.config.max_attempts]]
        saved = {key: value[selected].detach().clone() for key, value in packet.items() if key != "eligible"}
        self.candidates = [{key: value[i] for key, value in saved.items()} for i in range(selected.numel())]

    @torch.no_grad()
    def _candidate(self, agent, replay):
        if self.config.mode == "replay_state":
            data, _, _ = replay.sample(batch_size=1)
            eligible = (~(data["is_last"] | data["is_terminal"])).reshape(-1).nonzero().flatten()
            if not eligible.numel():
                return None, "no_start"
            t = int(eligible[torch.randint(eligible.numel(), (), device=eligible.device)])
            return dict(state=data["_integration_state"][0, t].cpu().numpy().copy(),
                        origin_step=int(data["_origin_step"][0, t, 0]), depth=0), None
        if not self.candidates:
            return None, "no_truncation"
        candidate = self.candidates.pop()
        prediction = agent.state_decoder(candidate["features"].unsqueeze(0).float()).float()
        reason = agent.state_decoder.prediction_rejection_reason(
            prediction, min_angle_norm=self.config.min_angle_norm
        )
        if reason:
            return None, reason
        candidate["state"] = agent.state_decoder.decode(prediction)[0].cpu().numpy()
        return candidate, None

    def _transition(self, observation, reward, origin, depth):
        values = {k: torch.as_tensor(np.array(v, copy=True), device=self.device).reshape(1, *np.shape(v))
                  for k, v in observation.items()}
        for k, v in values.items():
            if v.ndim == 1:
                values[k] = v.unsqueeze(-1)
        values["reward"] = torch.tensor([[reward]], device=self.device, dtype=torch.float32)
        data = TensorDict(values, batch_size=(1,), device=self.device)
        data["episode"] = torch.zeros(1, dtype=torch.int32, device=self.device)
        return add_source_metadata(data, 1, self.branch_id, origin, depth)

    @torch.no_grad()
    def collect(self, agent, normal_replay, branch_replay, normal_steps, budget):
        started = time.monotonic()
        # Schedule by normal interaction only; no recursive triggering by branches.
        self.next_normal_step = normal_steps + int(self.config.every_steps)
        used = attempts = successes = accepted_steps = 0
        rewards, lengths = [], []
        depths, cutoff_scores, thresholds = [], [], []
        rejected = Counter()
        repeat = int(self.env_config.action_repeat)
        with self.rng.use():
            while (attempts < self.config.max_attempts and successes < self.config.branches_per_round
                   and budget - used >= repeat):
                candidate, reason = self._candidate(agent, normal_replay)
                if reason == "no_truncation":
                    rejected[reason] += 1
                    break
                attempts += 1
                self.branch_id += 1
                if reason:
                    rejected[reason] += 1
                    continue
                kind = "integration" if self.config.mode == "replay_state" else "physics"
                restored = self._worker().restore(candidate["state"], kind)()
                if not restored["ok"]:
                    rejected[restored["reason"]] += 1
                    continue
                origin, depth = int(candidate["origin_step"]), int(candidate["depth"])
                row = self._transition(restored["observation"], 0.0, origin, depth)
                memory = agent.get_initial_state(1)
                action, memory = agent.act(row.clone(), memory)
                row["action"] = action
                row["stoch"], row["deter"] = memory["stoch"], memory["deter"]
                rows = [row.clone().detach()]
                score = 0.0
                steps = 0
                valid = True
                horizon = min(self.config.horizon, (budget - used) // repeat)
                for decision in range(horizon):
                    result = self._worker().step(action[0].cpu().numpy())()
                    used += int(result["steps"])
                    steps += int(result["steps"])
                    if not result["ok"]:
                        rejected[result["reason"]] += 1
                        valid = False
                        self.close()
                        break
                    row = self._transition(result["observation"], result["reward"], origin, depth)
                    done = bool(result["done"] or decision + 1 == horizon)
                    if done:
                        row["is_last"].fill_(True)
                    action, memory = agent.act(row.clone(), memory)
                    row["action"] = torch.zeros_like(action) if done else action
                    row["stoch"], row["deter"] = memory["stoch"], memory["deter"]
                    rows.append(row.clone().detach())
                    score += float(result["reward"])
                    if done:
                        break
                if valid:
                    for row in rows:
                        branch_replay.add_transition(row)
                    successes += 1
                    accepted_steps += steps
                    rewards.append(score)
                    lengths.append(len(rows) - 1)
                    depths.append(depth)
                    if kind == "physics":
                        cutoff_scores.append(float(candidate["score"]))
                        thresholds.append(float(candidate["threshold"]))
        metrics = {
            "branch/attempts": attempts, "branch/successes": successes,
            "branch/accept_rate": successes / max(attempts, 1),
            "branch/round_steps": used, "branch/discarded_steps": used - accepted_steps,
            "branch/score": float(np.mean(rewards)) if rewards else 0.0,
            "branch/length": float(np.mean(lengths)) if lengths else 0.0,
            "branch/imag_depth": float(np.mean(depths)) if depths else 0.0,
            "branch/cutoff_score": float(np.mean(cutoff_scores)) if cutoff_scores else 0.0,
            "branch/cutoff_threshold": float(np.mean(thresholds)) if thresholds else 0.0,
            "branch/seconds": time.monotonic() - started,
        }
        # Stable labels for logs; detailed PhysicsError text stays out of metric names.
        for reason, count in rejected.items():
            name = reason if reason in {"no_start", "no_truncation", "nonfinite", "angle_norm", "joint_limit", "velocity",
                                       "penetration", "physics_warning", "physics_error"} else "restore_error"
            key = f"branch/rejected/{name}"
            metrics[key] = metrics.get(key, 0) + count
        return used, metrics
