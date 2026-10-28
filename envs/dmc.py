import gymnasium as gym
import numpy as np


class DeepMindControl(gym.Env):
    metadata = {}

    def __init__(self, name, action_repeat=1, size=(64, 64), camera=None, seed=0,
                 capture_state=False, capture_snapshot=False):
        self._name = name
        self._capture_state = capture_state
        self._capture_snapshot = capture_snapshot
        self.last_env_steps = 0
        if name.endswith("_subtle"):
            is_subtle = True
        else:
            is_subtle = False

        if "sparse" in name or "finger_turn" in name:
            _name, difficulty = name.rsplit("_", 1)
            domain, task = _name.rsplit("_", 1)
            task = task + "_" + difficulty
        else:
            domain, task = name.rsplit("_", 1)

        if is_subtle:
            from envs import dmc_subtle

            func = getattr(dmc_subtle, name)
            self._env = func(random=seed)
        elif isinstance(domain, str):
            from dm_control import suite

            self._env = suite.load(
                domain,
                task,
                task_kwargs={"random": seed},
            )
        else:
            assert task is None
            self._env = domain()

        self._action_repeat = action_repeat
        self._size = size
        if camera is None:
            camera = dict(quadruped=2, fish=3).get(domain, 0)
        self._camera = camera
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
            if len(value.shape) == 0:
                shape = (1,)
            else:
                shape = value.shape
            spaces[key] = gym.spaces.Box(-np.inf, np.inf, shape, dtype=np.float32)
        spaces["image"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0
        self.last_env_steps = 0
        for _ in range(self._action_repeat):
            # Include an attempted control step even if the simulator rejects it.
            self.last_env_steps += 1
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            if time_step.last():
                break
        obs = dict(time_step.observation)
        obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items()}
        obs["image"] = self.render()
        # There is no terminal state in DMC
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()
        obs["is_last"] = time_step.last()
        self._add_state_metadata(obs)
        done = time_step.last()
        info = {"discount": np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self, **kwargs):
        self.last_env_steps = 0
        time_step = self._env.reset()
        obs = dict(time_step.observation)
        obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items()}
        obs["image"] = self.render()
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()
        obs["is_last"] = time_step.last()
        self._add_state_metadata(obs)
        return obs

    def get_state_spec(self):
        if self._name != "finger_spin":
            raise ValueError("State reconstruction v1 supports only finger_spin")
        p = self._env.physics
        return {
            "task": f"dmc_{self._name}",
            "dimensions": (p.model.nq, p.model.nv, p.model.na),
            "joint_ranges": p.model.jnt_range[:2].copy().tolist(),
        }

    def export_snapshot(self):
        import mujoco

        p = self._env.physics
        sig = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(p.model.ptr, sig), dtype=np.float64)
        mujoco.mj_getState(p.model.ptr, p.data.ptr, state, sig)
        return state

    def _add_state_metadata(self, obs):
        # Metadata deliberately stays out of observation_space / the encoders.
        if self._capture_state:
            p = self._env.physics
            obs["_physics_state"] = np.concatenate((p.data.qpos, p.data.qvel, p.data.act)).astype(np.float32)
        if self._capture_snapshot:
            obs["_integration_state"] = self.export_snapshot()

    def state_rejection_reason(self, max_velocity, max_penetration):
        p = self._env.physics
        state = p.get_state()
        if not np.isfinite(state).all():
            return "nonfinite"
        limited = p.model.jnt_limited.astype(bool)
        addresses = p.model.jnt_qposadr[limited]
        bounds = p.model.jnt_range[limited]
        if np.any(p.data.qpos[addresses] < bounds[:, 0]) or np.any(p.data.qpos[addresses] > bounds[:, 1]):
            return "joint_limit"
        if np.any(np.abs(p.data.qvel) > max_velocity):
            return "velocity"
        if p.data.ncon and np.min(p.data.contact.dist[:p.data.ncon]) < -max_penetration:
            return "penetration"
        if np.any(p.data.warning.number):
            return "physics_warning"
        return None

    def restore_state(self, state, kind="physics", max_velocity=200.0, max_penetration=0.001):
        """Start a new episode at a physical state or an exact integration snapshot.

        Snapshots restore physics time; episode counters start at zero for branching.
        Reapplying integration inputs after forward preserves solver warmstarts.
        """
        import mujoco

        self.reset()
        p = self._env.physics
        state = np.asarray(state, dtype=np.float64)
        if not np.isfinite(state).all():
            raise ValueError("nonfinite")
        if kind == "physics":
            if state.shape != (6,) or self._name != "finger_spin":
                raise ValueError("Expected a six-dimensional finger_spin physics state")
            p.set_state(state)
            p.data.time = 0.0
            p.data.ctrl[:] = 0
            p.data.qacc_warmstart[:] = 0
        elif kind == "integration":
            sig = mujoco.mjtState.mjSTATE_INTEGRATION
            if state.shape != (mujoco.mj_stateSize(p.model.ptr, sig),):
                raise ValueError("Invalid integration snapshot size")
            mujoco.mj_setState(p.model.ptr, p.data.ptr, state, sig)
        else:
            raise ValueError(f"Unknown restore kind: {kind}")
        reason = self.state_rejection_reason(max_velocity, float("inf"))
        if reason:
            raise ValueError(reason)
        p.forward()
        if kind == "integration":
            mujoco.mj_setState(p.model.ptr, p.data.ptr, state, sig)
        reason = self.state_rejection_reason(max_velocity, max_penetration)
        if reason:
            raise ValueError(reason)
        obs = dict(self._env.task.get_observation(p))
        obs = {k: np.asarray(v).reshape(1) if np.ndim(v) == 0 else np.asarray(v).copy() for k, v in obs.items()}
        obs.update(image=self.render(), is_first=True, is_last=False, is_terminal=False)
        self._add_state_metadata(obs)
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*self._size, camera_id=self._camera)
