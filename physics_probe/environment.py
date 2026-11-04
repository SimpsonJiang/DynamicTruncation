"""Headless proprio DMC adapter and explicit full/core state restoration contracts."""

import hashlib

import gymnasium as gym
import mujoco
import numpy as np
import torch
from dm_control import suite

TASKS = ("finger_spin", "hopper_hop", "quadruped_run", "walker_walk")
STATE_SPEC = int(mujoco.mjtState.mjSTATE_INTEGRATION)


def tensor(array):
    return torch.from_numpy(np.asarray(array).copy())


class ProbeEnv:
    """Match repository DMC action repeat/normalization/time limits without rendering.

    Restricted to four deterministic, fixed-model tasks. The model hash rejects
    cross-model restores instead of pretending integration state contains mjModel.
    No image is rendered and no physics label is added to observation_space.
    """

    def __init__(self, config, seed):
        self.task = str(config.task).removeprefix("dmc_")
        if self.task not in TASKS:
            raise ValueError(f"Unsupported probe task: {self.task}")
        self.repeat = int(config.action_repeat)
        self.limit = int(config.time_limit) // self.repeat
        if self.repeat < 1 or self.limit < 1:
            raise ValueError("action_repeat and time_limit must permit at least one step")
        self.seed = int(seed)
        domain, task = self.task.split("_", 1)
        self.env = suite.load(domain, task, task_kwargs={"random": self.seed})
        self.physics = self.env.physics
        self.steps = 0
        self.finished = True
        spec = self.env.action_spec()
        self.low, self.high = spec.minimum, spec.maximum
        self.action_space = gym.spaces.Box(-1.0, 1.0, spec.shape, dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            k: gym.spaces.Box(-np.inf, np.inf, v.shape or (1,), dtype=np.float32)
            for k, v in self.env.observation_spec().items()
        })
        self.reset()

    def observation(self, first=False, last=False, terminal=False):
        obs = {k: np.atleast_1d(v).astype(np.float32) for k, v in self.env.task.get_observation(self.physics).items()}
        obs.update(is_first=np.array([first]), is_last=np.array([last]), is_terminal=np.array([terminal]))
        return obs

    def reset(self):
        ts = self.env.reset()
        self.steps, self.finished = 0, False
        self.default_state = self.integration_state()
        self.model_hash = hashlib.sha256(self.physics.model.to_bytes()).hexdigest()
        return self.observation(first=ts.first())

    def step(self, action):
        if self.finished:
            raise RuntimeError("Explicit reset required; probe never silently auto-resets")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError("Invalid action")
        # Match NormalizeActions exactly, including its lack of clipping.
        original = (action + 1) / 2 * (self.high - self.low) + self.low
        reward = 0.0
        for _ in range(self.repeat):
            ts = self.env.step(original)
            reward += ts.reward or 0.0
            if ts.last():
                break
        self.steps += 1
        self.finished = bool(ts.last() or self.steps >= self.limit)
        terminal = bool(ts.discount == 0)
        return self.observation(last=self.finished, terminal=terminal), float(reward), self.finished

    def integration_state(self):
        result = np.empty(mujoco.mj_stateSize(self.physics.model.ptr, STATE_SPEC), dtype=np.float64)
        mujoco.mj_getState(self.physics.model.ptr, self.physics.data.ptr, result, STATE_SPEC)
        return result

    def core_state(self):
        d = self.physics.data
        return np.concatenate([d.qpos, d.qvel, d.act]).copy()

    def schema(self):
        m = self.physics.model
        return {
            "nq": int(m.nq),
            "nv": int(m.nv),
            "na": int(m.na),
            "joint_types": m.jnt_type.tolist(),
            "joint_addresses": m.jnt_qposadr.tolist(),
            "joint_limited": m.jnt_limited.tolist(),
            "task": self.task,
        }

    def snapshot(self):
        rng = self.env.task.random.get_state()
        # The supported tasks only have scalar task bookkeeping besides RandomState.
        attrs = {}
        for k, v in vars(self.env.task).items():
            if k == "_random":
                continue
            if not isinstance(v, (bool, int, float, str, type(None), np.number)):
                raise TypeError(f"Unmodelled task context: {k}: {type(v)}")
            attrs[k] = v.item() if isinstance(v, np.number) else v
        return {
            "state": tensor(self.integration_state()),
            "sensors": tensor(self.physics.data.sensordata),
            "core": tensor(self.core_state()),
            "model_hash": self.model_hash,
            "steps": self.steps,
            "inner_steps": int(self.env._step_count),
            "reset_next_step": bool(self.env._reset_next_step),
            "finished": self.finished,
            "rng": [rng[0], tensor(rng[1].astype(np.int64)), rng[2], rng[3], rng[4]],
            "task_attrs": attrs,
            "task": self.task,
            "repeat": self.repeat,
            "limit": self.limit,
        }

    def restore(self, snapshot, core=None):
        """Full oracle restore, or predicted core + known time/control + reset defaults.

        Full restore preserves cached sensors (DMC's acceleration sensors can be
        staggered). Core restore recomputes sensors and NEVER borrows true future
        warmstarts/sensors. Context is validated for this adapter's limited scope.
        """
        if (snapshot["task"], snapshot["repeat"], snapshot["limit"]) != (self.task, self.repeat, self.limit):
            raise ValueError("Incompatible environment context")
        if snapshot["model_hash"] != self.model_hash:
            raise ValueError("mjModel differs; integration state alone cannot restore this environment")
        state = snapshot["state"].numpy().copy()
        if core is not None:
            core = np.asarray(core, dtype=np.float64)
            m, d = self.physics.model, self.physics.data
            if core.shape != (m.nq + m.nv + m.na,) or not np.isfinite(core).all():
                raise ValueError("Invalid decoded core state")
            # Capture ONLY known time/control from context, not true core or warmstart.
            mujoco.mj_setState(m.ptr, d.ptr, state, STATE_SPEC)
            time, ctrl = float(d.time), d.ctrl.copy()
            for name in (
                "qfrc_applied",
                "xfrc_applied",
                "mocap_pos",
                "mocap_quat",
                "userdata",
                "eq_active",
                "plugin_state",
            ):
                current = np.asarray(getattr(d, name)).copy()
                mujoco.mj_setState(m.ptr, d.ptr, self.default_state, STATE_SPEC)
                if not np.array_equal(current, np.asarray(getattr(d, name))):
                    raise ValueError(f"Core restoration cannot assume reset-default {name}")
                mujoco.mj_setState(m.ptr, d.ptr, state, STATE_SPEC)
            mujoco.mj_setState(m.ptr, d.ptr, self.default_state, STATE_SPEC)
            d.time, d.ctrl[:] = time, ctrl
            d.qpos[:] = core[: m.nq]
            d.qvel[:] = core[m.nq : m.nq + m.nv]
            d.act[:] = core[m.nq + m.nv :]
            # Reject degenerate quaternion predictions rather than silently choosing an orientation.
            for typ, adr in zip(m.jnt_type, m.jnt_qposadr):
                start = int(adr) + 3 if typ == mujoco.mjtJoint.mjJNT_FREE else int(adr)
                if typ in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
                    q = d.qpos[start : start + 4]
                    norm = np.linalg.norm(q)
                    if norm < 1e-8:
                        raise ValueError("Degenerate predicted quaternion")
                    q[:] /= norm
            d.qacc_warmstart[:] = 0
            self.physics.forward()
        else:
            mujoco.mj_setState(self.physics.model.ptr, self.physics.data.ptr, state, STATE_SPEC)
            self.physics.forward()
            # forward computes caches but may update warmstart; restore integration inputs afterwards.
            mujoco.mj_setState(self.physics.model.ptr, self.physics.data.ptr, state, STATE_SPEC)
            self.physics.data.sensordata[:] = snapshot["sensors"].numpy()
        self.steps, self.finished = snapshot["steps"], snapshot["finished"]
        self.env._step_count = snapshot["inner_steps"]
        self.env._reset_next_step = snapshot["reset_next_step"]
        for k, v in snapshot["task_attrs"].items():
            setattr(self.env.task, k, v)
        rng = snapshot["rng"]
        self.env.task.random.set_state((rng[0], rng[1].numpy().astype(np.uint32), *rng[2:]))
        return self.observation(first=self.steps == 0, last=self.finished)

    def close(self):
        self.env.close()


def observation_vector(obs):
    return np.concatenate([np.asarray(obs[k]).reshape(-1) for k in sorted(obs) if not k.startswith("is_")])


def state_errors(prediction, target, schema):
    """Raw and geometry-aware errors, without assuming global translation invariance."""
    nq, nv = schema["nq"], schema["nv"]
    diff = np.asarray(prediction) - np.asarray(target)
    qdiff = diff[:nq].copy()
    rotations = []
    for typ, adr, limited in zip(schema["joint_types"], schema["joint_addresses"], schema["joint_limited"]):
        if typ == int(mujoco.mjtJoint.mjJNT_HINGE) and not limited:
            qdiff[adr] = np.arctan2(np.sin(qdiff[adr]), np.cos(qdiff[adr]))
        if typ in (int(mujoco.mjtJoint.mjJNT_FREE), int(mujoco.mjtJoint.mjJNT_BALL)):
            start = adr + 3 if typ == int(mujoco.mjtJoint.mjJNT_FREE) else adr
            a, b = prediction[start : start + 4], target[start : start + 4]
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            angle = 2 * np.arccos(np.clip(abs(np.dot(a, b)) / max(denom, 1e-12), 0, 1))
            rotations.append(float(angle))
            qdiff[start : start + 4] = 0
    result = {
        "qpos_raw_rmse": float(np.sqrt(np.mean(diff[:nq] ** 2))),
        "qpos_wrapped_rmse": float(np.sqrt(np.mean(qdiff**2))),
        "qvel_rmse": float(np.sqrt(np.mean(diff[nq : nq + nv] ** 2))),
    }
    if rotations:
        result["orientation_radians"] = max(rotations)
    if schema["na"]:
        result["act_rmse"] = float(np.sqrt(np.mean(diff[nq + nv :] ** 2)))
    return result
