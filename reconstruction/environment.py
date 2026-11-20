"""Out-of-band physics labels. Never extend the agent observation space."""

import hashlib

import gymnasium as gym
import numpy as np

from .protocol import EnvStepResult, PhysicsContext

TASKS = ("finger_spin", "hopper_hop", "quadruped_run", "walker_walk")


class OnlinePhysicsEnv(gym.Wrapper):
    _BRANCH_CHECK_STEPS = 5

    def __init__(self, env, config, *, time_limit):
        super().__init__(env)
        self.task_name = str(config.task).removeprefix("dmc_")
        if self.task_name not in TASKS:
            raise ValueError(f"Online reconstruction does not support {self.task_name}")
        self.base = env.unwrapped
        self.physics = self.base._env.physics
        self.time_limit = time_limit
        self.repeat = int(config.action_repeat)
        self.guard = None
        self._guard_profile = None
        self._prepared_reset = None
        self.policy_steps, self.branch_steps, self.branch_left, self.in_branch = 0, 0, 0, False

    def reset(self):
        return self.env.reset()

    def physics_schema(self):
        m = self.physics.model
        return dict(
            nq=int(m.nq),
            nv=int(m.nv),
            na=int(m.na),
            joint_types=m.jnt_type.tolist(),
            joint_addresses=m.jnt_qposadr.tolist(),
            joint_limited=m.jnt_limited.tolist(),
            task=self.task_name,
        )

    def physics_labels(self):
        d = self.physics.data
        return np.concatenate([d.qpos, d.qvel, d.act]).astype(np.float64, copy=True)

    def online_step(self, action, reset=False):
        if reset:
            obs, reward, done = self.reset(), 0.0, False
            self.policy_steps = 0
        else:
            obs, reward, done, _ = self.step(action)
            self.policy_steps += 1
        return EnvStepResult(obs, reward, done, self.physics_labels())

    def render_reconstructions(self, cores, actions, depths, size):
        """Render predicted cores in a private model/data copy, with zero actions.

        This diagnostic deliberately bypasses the branch guard, so poor or
        rejected reconstructions remain visible. Invalid cells carry a failure.
        """
        import mujoco

        cores, actions = np.asarray(cores), np.asarray(actions)
        if cores.dtype != np.float64 or cores.shape != (len(depths), len(self.physics_labels())):
            raise ValueError("Visualization cores must be float64 [depth, nq+nv+na]")
        height, width = map(int, size)
        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION

        def integration(physics):
            state = np.empty(mujoco.mj_stateSize(physics.model.ptr, state_spec), dtype=np.float64)
            mujoco.mj_getState(physics.model.ptr, physics.data.ptr, state, state_spec)
            return state

        before = integration(self.physics)
        model_hash = hashlib.sha256(self.physics.model.to_bytes()).hexdigest()
        source = dict(
            time=float(self.physics.data.time),
            policy_step=int(self.policy_steps),
            context_vector=self.context_vector().tolist(),
            core=self.physics_labels().tolist(),
            model_hash=model_hash,
        )
        scratch = self.physics.copy(share_model=False)
        images, details = [], []
        try:
            m, d = scratch.model, scratch.data
            defaults = {
                key: np.asarray(getattr(self.physics.data, key)).copy()
                for key in (
                    "qfrc_applied",
                    "xfrc_applied",
                    "mocap_pos",
                    "mocap_quat",
                    "userdata",
                    "eq_active",
                    "plugin_state",
                )
            }
            for core, depth in zip(cores, depths):
                detail = dict(depth=int(depth), actions_executed=0, failure=None)
                pixels = np.zeros((height, width, 3), dtype=np.uint8)
                try:
                    if not np.isfinite(core).all():
                        raise ValueError("nonfinite_core")
                    for kind, address in zip(m.jnt_type, m.jnt_qposadr):
                        if kind in (0, 1):
                            start = int(address) + (3 if kind == 0 else 0)
                            if abs(np.linalg.norm(core[start : start + 4]) - 1) > 1e-5:
                                raise ValueError("invalid_quaternion")
                    mujoco.mj_resetData(m.ptr, d.ptr)
                    for key, value in defaults.items():
                        getattr(d, key)[:] = value
                    d.qpos[:], d.qvel[:], d.act[:] = core[: m.nq], core[m.nq : m.nq + m.nv], core[m.nq + m.nv :]
                    d.time = source["time"] + int(depth) * self.base._env.control_timestep() * self.repeat
                    d.ctrl[:] = self.physics.data.ctrl
                    if depth:
                        previous = actions[depth - 1].astype(np.float32)
                        low, high = self.base.action_space.low, self.base.action_space.high
                        d.ctrl[:] = (previous + 1) / 2 * (high - low) + low
                    scratch.forward()
                    restored = integration(scratch)
                    pixels = scratch.render(height=height, width=width, camera_id=self.base._camera)
                    np.testing.assert_array_equal(integration(scratch), restored)
                    detail.update(time=float(d.time), render_state_unchanged=True)
                except Exception as error:
                    # Optional rendering errors must not terminate this RPC
                    # worker and thereby take down the training collector.
                    detail["failure"] = f"{type(error).__name__}: {error}"
                images.append(pixels)
                details.append(detail)
        finally:
            scratch.free()
            np.testing.assert_array_equal(integration(self.physics), before)
            if hashlib.sha256(self.physics.model.to_bytes()).hexdigest() != model_hash:
                raise RuntimeError("Visualization changed the live physics model")
        return dict(
            images=np.stack(images),
            details=details,
            source=source,
            camera_id=self.base._camera,
            live_state_unchanged=True,
            live_model_unchanged=True,
        )

    def setup_guard(self):
        from .guard import GeometryMonitor, UnsafeState

        if self._guard_profile is not None:
            return self._guard_profile
        # DMC tasks finalize model settings in initialize_episode (e.g. finger
        # damping). This is the collector's first ordinary reset, cached for its
        # first row; do not consume an additional reset/RNG draw for diagnostics.
        self._prepared_reset = self.reset()
        try:
            self.guard = GeometryMonitor(self.physics, f"{self.task_name}:{self.repeat}:{self.base._env._n_sub_steps}")
            self.guard.install_hook()
            self._guard_profile = dict(
                signature=self.guard.signature, pairs=self.guard.pairs, floors=self.guard.floors.tolist()
            )
        except UnsafeState as exc:
            self.guard = None
            self._guard_profile = dict(failure=str(exc), floors=[], pairs=[])
        return self._guard_profile

    def close(self):
        if self.guard is not None:
            self.guard.uninstall_hook()
        return self.env.close()

    def context(self):
        task = self.base._env.task
        attrs = {}
        for k, v in vars(task).items():
            if k == "_random":
                continue
            if not isinstance(v, (bool, int, float, str, type(None), np.number)):
                raise ValueError(f"Unsupported task context: {k}")
            attrs[k] = v.item() if isinstance(v, np.number) else v
        d = self.physics.data
        defaults = {
            k: np.asarray(getattr(d, k)).copy()
            for k in (
                "qfrc_applied",
                "xfrc_applied",
                "mocap_pos",
                "mocap_quat",
                "userdata",
                "eq_active",
                "plugin_state",
            )
        }
        return dict(
            task=self.task_name,
            repeat=self.repeat,
            limit=self.time_limit.duration,
            model_hash=hashlib.sha256(self.physics.model.to_bytes()).hexdigest(),
            task_attrs=attrs,
            defaults=defaults,
            policy_dt=float(self.base._env.control_timestep()) * self.repeat,
            inner_limit=int(self.base._env._step_limit),
            action_low=self.base.action_space.low.copy(),
            action_high=self.base.action_space.high.copy(),
        )

    def context_vector(self):
        e, d = self.base._env, self.physics.data
        # TimeLimit clears _step on timeout, so retain the real counter separately.
        return PhysicsContext(float(d.time), self.policy_steps, int(e._step_count), d.ctrl).to_vector()

    def restore_branch(self, candidate):
        import mujoco

        from .guard import UnsafeState

        context, vector = candidate["context"], candidate["context_vector"]
        m, d = self.physics.model, self.physics.data
        expected = (self.task_name, self.repeat, self.time_limit.duration, hashlib.sha256(m.to_bytes()).hexdigest())
        actual = (context["task"], context["repeat"], context["limit"], context["model_hash"])
        if expected != actual:
            raise UnsafeState("incompatible_context")
        if self.guard is None or self.guard.thresholds is None:
            raise UnsafeState("unvalidated_guard")
        core = np.asarray(candidate["core"])
        if core.dtype != np.float64 or core.shape != (m.nq + m.nv + m.na,) or not np.isfinite(core).all():
            raise UnsafeState("invalid_core")
        try:
            timeline = PhysicsContext.from_vector(vector)
        except ValueError as exc:
            raise UnsafeState("invalid_context_time") from exc
        if (
            len(timeline.control) != m.nu
            or timeline.policy_steps >= self.time_limit.duration
            or timeline.inner_steps >= self.base._env._step_limit
        ):
            raise UnsafeState("invalid_context_time")
        for typ, adr in zip(m.jnt_type, m.jnt_qposadr):
            if typ in (0, 1):
                start = int(adr) + (3 if typ == 0 else 0)
                norm = np.linalg.norm(core[start : start + 4])
                if not np.isfinite(norm) or abs(norm - 1) > 1e-5:
                    raise UnsafeState("invalid_quaternion")
        # Restore only the known context and predicted core; never future truth or warmstarts.
        mujoco.mj_resetData(m.ptr, d.ptr)
        d.time, d.ctrl[:] = timeline.time, timeline.control
        for k, v in context["defaults"].items():
            getattr(d, k)[:] = v
        d.qpos[:], d.qvel[:], d.act[:] = core[: m.nq], core[m.nq : m.nq + m.nv], core[m.nq + m.nv :]
        self.physics.forward()
        self.policy_steps = timeline.policy_steps
        self.time_limit.restore_elapsed_steps(self.policy_steps)
        self.base._env._step_count = timeline.inner_steps
        self.base._env._reset_next_step = False
        for k, v in context["task_attrs"].items():
            setattr(self.base._env.task, k, v)
        self.branch_steps = 0
        self.guard.begin_policy_step("branch")
        self.guard.check()
        obs = {
            k: np.atleast_1d(v).astype(np.float32) for k, v in self.base._env.task.get_observation(self.physics).items()
        }
        obs.update(image=self.base.render(), is_first=False, is_last=False, is_terminal=False)
        return obs

    def reconstruction_step(self, action, reset, request, phase, thresholds=None):
        from dm_control.rl.control import PhysicsError

        from .guard import UnsafeState

        report = dict(
            violation=False, samples=[], failure=None, failure_details=None, max_depth=0.0, branch=False, root=False
        )
        restoring = reset and request is not None
        branch_step = 0 if restoring else self.branch_steps + 1 if self.in_branch else None
        if thresholds is not None and self.guard is not None:
            self.guard.install_thresholds(thresholds["signature"], thresholds["values"])
        context = None
        try:
            if reset and request is not None:
                obs = self.restore_branch(request)
                self.branch_left = int(request["length"])
                self.in_branch = True
                reward, done = 0.0, False
                report.update(branch=True, root=True)
                report["max_depth"] = self.guard.max_depth
            elif reset:
                if self.guard is not None:
                    self.guard.begin_policy_step("off")
                prepared = self._prepared_reset
                obs = prepared if prepared is not None else self.reset()
                self._prepared_reset = None
                reward, done = 0.0, False
                self.policy_steps, self.branch_steps, self.branch_left, self.in_branch = 0, 0, 0, False
                context = self.context()
            else:
                branch = self.in_branch
                if branch and not np.isfinite(action).all():
                    raise UnsafeState("nonfinite_action")
                if self.guard is not None:
                    mode = "branch" if self.branch_steps < self._BRANCH_CHECK_STEPS else "off"
                    self.guard.begin_policy_step(mode if branch else phase)
                obs, reward, done, _ = self.step(action)
                self.policy_steps += 1
                report["branch"] = branch
                if branch:
                    self.branch_steps += 1
                    self.branch_left -= 1
                    if self.branch_left <= 0:
                        done, obs["is_last"] = True, True
                if self.guard is not None:
                    report.update(
                        violation=self.guard.violation, samples=self.guard.samples, max_depth=self.guard.max_depth
                    )
            if (
                not np.isfinite(reward)
                or not np.isfinite(self.physics_labels()).all()
                or any(not np.isfinite(v).all() for v in obs.values())
            ):
                raise UnsafeState("nonfinite_transition")
        except (UnsafeState, FloatingPointError, PhysicsError) as exc:
            reason = (
                exc.reason
                if isinstance(exc, UnsafeState)
                else ("physics_error" if isinstance(exc, PhysicsError) else "floating_point_error")
            )
            report["failure"] = reason
            report["failure_details"] = dict(
                message=str(exc),
                phase="restore" if restoring else "rollout" if self.in_branch else "ordinary",
                branch_step=branch_step,
                **(exc.details if isinstance(exc, UnsafeState) else {}),
            )
            report["branch"] = bool(self.in_branch or request is not None)
            if self.guard is not None:
                report["max_depth"] = self.guard.max_depth
                self.guard.begin_policy_step("off")
            obs, reward, done = self.reset(), 0.0, False
            self.policy_steps, self.branch_steps, self.branch_left, self.in_branch = 0, 0, 0, False
            context = self.context()
        if context is not None and self.guard is not None and context["model_hash"] != self.guard.model_hash:
            report["guard_failure"] = "model_changed_since_guard_setup"
        return EnvStepResult(obs, reward, done, self.physics_labels(), self.context_vector(), report, context)
