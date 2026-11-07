"""Pair-specific, frozen penetration thresholds and non-mutating substep checks."""

import hashlib
from collections import defaultdict

import mujoco
import numpy as np


class UnsafeState(RuntimeError):
    pass


class GeometryMonitor:
    """Distance queries on scratch MjData, including pairs filtered by MuJoCo.

    The four supported tasks use primitive geometry only. Fail closed for a
    critical pair without an explicit distance implementation in this version.
    """

    def __init__(self, physics, task):
        self.physics, self.task = physics, task
        self.model = physics.model.ptr
        self.scratch = mujoco.MjData(self.model)
        m = self.model
        self.model_hash = hashlib.sha256(physics.model.to_bytes()).hexdigest()
        self.signature = hashlib.sha256(
            physics.model.to_bytes() + mujoco.__version__.encode() + task.encode() + str(physics.legacy_step).encode()
        ).hexdigest()
        explicit = {tuple(sorted(map(int, p))) for p in zip(m.pair_geom1, m.pair_geom2)}
        geoms = [
            i for i in range(m.ngeom) if m.geom_contype[i] or m.geom_conaffinity[i] or any(i in p for p in explicit)
        ]
        self.pairs, self.floors = [], []
        for a in geoms:
            for b in geoms:
                if a >= b:
                    continue
                ba, bb = int(m.geom_bodyid[a]), int(m.geom_bodyid[b])
                wa, wb = int(m.body_weldid[ba]), int(m.body_weldid[bb])
                parent_a = int(m.body_weldid[m.body_parentid[wa]])
                parent_b = int(m.body_weldid[m.body_parentid[wb]])
                if wa == wb or parent_a == wb or parent_b == wa:
                    # A movable world-child must still be checked against the ground.
                    if wa == wb or (wa != 0 and wb != 0):
                        continue
                if not (wa or wb):
                    continue
                ta, tb = int(m.geom_type[a]), int(m.geom_type[b])
                # Mesh/hfield/SDF distances cannot silently be treated as separated.
                if max(ta, tb) > int(mujoco.mjtGeom.mjGEOM_BOX) or 1 in (ta, tb):
                    raise UnsafeState(f"unsupported_geometry_pair:{a}:{b}:{ta}:{tb}")
                if ta == 0 and tb == 0:
                    continue
                sizes = [float(np.max(m.geom_size[g])) for g in (a, b) if int(m.geom_type[g]) != 0]
                self.pairs.append((a, b))
                self.floors.append(1e-3 * min(sizes))
        if not self.pairs:
            raise UnsafeState("no_detectable_critical_pairs")
        self.lookup = {pair: i for i, pair in enumerate(self.pairs)}
        self.floors = np.asarray(self.floors, dtype=np.float64)
        self.signature = hashlib.sha256(
            self.signature.encode()
            + b"pair_guard_v1"
            + np.asarray(self.pairs, dtype=np.int32).tobytes()
            + self.floors.tobytes()
        ).hexdigest()
        self.thresholds = None
        self.last_depths = np.zeros(len(self.pairs))
        self.samples = []
        self.violation = False
        self.max_depth = 0.0
        self.mode = "off"
        self._original_step = None

    def install_hook(self):
        """Explicit, idempotent ownership of the dm_control substep adapter."""
        current = self.physics.step
        if self._original_step is not None:
            if current != self.step:
                raise RuntimeError("Physics step changed while the geometry hook was installed")
            return
        if isinstance(getattr(current, "__self__", None), GeometryMonitor):
            raise RuntimeError("Another geometry monitor is already installed")
        self._original_step = current
        self.physics.step = self.step

    def uninstall_hook(self):
        if self._original_step is None:
            return
        if self.physics.step != self.step:
            raise RuntimeError("Refusing to overwrite another physics step hook")
        self.physics.step = self._original_step
        self._original_step = None

    def begin_policy_step(self, mode):
        self.mode, self.violation, self.max_depth, self.samples = mode, False, 0.0, []

    def depths(self):
        d, scratch, m = self.physics.data.ptr, self.scratch, self.model
        if not all(np.isfinite(getattr(d, k)).all() for k in ("qpos", "qvel", "act")):
            raise UnsafeState("nonfinite_state")
        # mj_fwdPosition needs qpos/mocap/eq_active, not the live solver state.
        scratch.qpos[:] = d.qpos
        scratch.mocap_pos[:] = d.mocap_pos
        scratch.mocap_quat[:] = d.mocap_quat
        scratch.eq_active[:] = d.eq_active
        mujoco.mj_fwdPosition(m, scratch)
        depths = np.zeros(len(self.pairs), dtype=np.float64)
        lookup = self.lookup
        for c in scratch.contact[: scratch.ncon]:
            pair = tuple(sorted(map(int, c.geom)))
            if pair in lookup:
                depths[lookup[pair]] = max(depths[lookup[pair]], -float(c.dist))
        for i, (a, b) in enumerate(self.pairs):
            distance = mujoco.mj_geomDistance(m, scratch, a, b, 1e3, None)
            if not np.isfinite(distance):
                raise UnsafeState("nonfinite_distance")
            depths[i] = max(depths[i], -distance, 0.0)
        self.last_depths = depths
        return depths

    def check(self):
        depths = self.depths()
        self.max_depth = max(self.max_depth, float(depths.max(initial=0)))
        if self.mode == "calibrate":
            self.samples.append(depths.copy())
        elif self.mode in ("validate", "branch"):
            if self.thresholds is None:
                raise UnsafeState("unvalidated_guard")
            if np.any(depths > self.thresholds):
                self.violation = True
                if self.mode == "branch":
                    raise UnsafeState("penetration")

    def step(self, nstep=1):
        if self._original_step is None:
            raise RuntimeError("Install the geometry hook before stepping")
        if self.mode == "off":
            return self._original_step(nstep)
        p, m, d = self.physics, self.model, self.physics.data.ptr
        # Exact dm_control legacy order: step2, (n-1)*step, step1.
        with p.check_invalid_state():
            if p.legacy_step and m.opt.integrator != mujoco.mjtIntegrator.mjINT_RK4:
                mujoco.mj_step2(m, d)
                self.check()
                for _ in range(1, nstep):
                    mujoco.mj_step(m, d)
                    self.check()
                mujoco.mj_step1(m, d)
            else:
                for _ in range(nstep):
                    mujoco.mj_step(m, d)
                    self.check()
                if p.legacy_step:
                    mujoco.mj_step1(m, d)

    def install_thresholds(self, signature, thresholds):
        if signature != self.signature or len(thresholds) != len(self.pairs):
            raise UnsafeState("guard_profile_mismatch")
        self.thresholds = np.asarray(thresholds, dtype=np.float64)


class GuardCalibration:
    """Aggregate ordinary policy steps from all slots, never reconstructed data."""

    def __init__(self, profile, calibration_steps=50000, validation_steps=50000, max_false_reject=0.001):
        self.profile = profile
        self.calibration_steps, self.validation_steps = int(calibration_steps), int(validation_steps)
        if self.calibration_steps <= 0 or self.validation_steps <= 0 or not 0 <= max_false_reject <= 1:
            raise ValueError("Invalid guard calibration/validation budget")
        self.max_false_reject = float(max_false_reject)
        self.calibration_count = self.validation_count = self.rejected = 0
        self.positive = defaultdict(list)
        self.thresholds = None
        self.failure = profile.get("failure")

    @property
    def phase(self):
        if self.failure:
            return "off"
        if self.calibration_count < self.calibration_steps:
            return "calibrate"
        if self.validation_count < self.validation_steps:
            return "validate"
        return "off"

    @property
    def ready(self):
        return not self.failure and self.thresholds is not None and self.validation_count >= self.validation_steps

    def observe(self, report, env_steps, phase):
        if self.failure:
            return
        if report.get("failure"):
            self.failure = report["failure"]
            return
        if phase == "calibrate":
            for row in report["samples"]:
                for i, value in enumerate(row):
                    if value > 0:
                        self.positive[i].append(float(value))
            self.calibration_count += env_steps
            if self.calibration_count >= self.calibration_steps and self.thresholds is None:
                self.thresholds = [
                    max(floor, 1.25 * float(np.quantile(self.positive[i], 0.999))) if self.positive[i] else floor
                    for i, floor in enumerate(self.profile["floors"])
                ]
                self.positive.clear()
        elif phase == "validate":
            self.validation_count += env_steps
            self.rejected += bool(report["violation"])
            # Denominator is policy steps, not integration substeps or repeated env steps.
            self.profile["validation_policy_steps"] = self.profile.get("validation_policy_steps", 0) + 1
            if self.validation_count >= self.validation_steps:
                rate = self.rejected / self.profile["validation_policy_steps"]
                if rate > self.max_false_reject:
                    self.failure = f"false_rejection_rate:{rate:.8g}"

    def export(self):
        return dict(
            profile=self.profile,
            thresholds=self.thresholds,
            failure=self.failure,
            calibration_env_steps=self.calibration_count,
            validation_env_steps=self.validation_count,
            rejected_policy_steps=self.rejected,
            ready=self.ready,
        )
