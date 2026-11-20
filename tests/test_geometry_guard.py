"""Collision-filter parity and the five-policy-step reconstruction guard window."""

import unittest
from unittest import mock

import mujoco
import numpy as np
from dm_control import mujoco as dm_mujoco
from omegaconf import OmegaConf

from envs import make_env
from reconstruction_guard import GeometryMonitor, UnsafeState


class GeometryFilterTest(unittest.TestCase):
    @staticmethod
    def physics(*, first='contype="1" conaffinity="1"', second='contype="1" conaffinity="1"', contact="", option=""):
        return dm_mujoco.Physics.from_xml_string(f"""<mujoco>{option}<worldbody>
        <body name="a" pos="0 0 1"><freejoint/><geom name="ga" type="sphere" size=".2" {first}/></body>
        <body name="b" pos=".1 0 1"><freejoint/><geom name="gb" type="sphere" size=".2" {second}/></body>
        </worldbody>{contact}</mujoco>""")

    def test_enabled_pair_uses_mujoco_contacts_without_forced_distance_queries(self):
        physics = self.physics()
        guard = GeometryMonitor(physics, "allowed")
        self.assertEqual(guard.pairs, [(0, 1)])
        with mock.patch("reconstruction_guard.mujoco.mj_geomDistance") as forced_distance:
            self.assertAlmostEqual(guard.depths()[0], 0.3)
        forced_distance.assert_not_called()
        guard.install_thresholds(guard.signature, [0.01])
        guard.begin_policy_step("branch")
        with self.assertRaisesRegex(UnsafeState, "penetration") as rejected:
            guard.check()
        self.assertEqual(rejected.exception.reason, "penetration")
        self.assertEqual(rejected.exception.details["geom_pair"], [0, 1])
        self.assertEqual(rejected.exception.details["geom_names"], ["ga", "gb"])
        self.assertAlmostEqual(rejected.exception.details["depth"], 0.3)
        self.assertEqual(rejected.exception.details["threshold"], 0.01)

    def test_rejection_identifies_threshold_excess_instead_of_overall_deepest_pair(self):
        physics = dm_mujoco.Physics.from_xml_string("""<mujoco><worldbody>
        <body><freejoint/><geom type="sphere" size=".2"/></body>
        <body><freejoint/><geom type="sphere" size=".2"/></body>
        <body><freejoint/><geom type="sphere" size=".2"/></body>
        </worldbody></mujoco>""")
        guard = GeometryMonitor(physics, "excess")
        guard.install_thresholds(guard.signature, [0.01, 0.49, 0.2])
        guard.begin_policy_step("branch")
        with mock.patch.object(guard, "depths", return_value=np.array([0.3, 0.5, 0.1])):
            with self.assertRaises(UnsafeState) as rejected:
                guard.check()
        self.assertEqual(rejected.exception.details["geom_pair"], list(guard.pairs[0]))
        self.assertEqual(rejected.exception.details["depth"], 0.3)
        self.assertEqual(guard.max_depth, 0.5)

    def test_one_direction_of_collision_bitmasks_is_enough(self):
        physics = self.physics(first='contype="1" conaffinity="0"', second='contype="0" conaffinity="1"')
        guard = GeometryMonitor(physics, "one_direction")
        self.assertEqual(guard.pairs, [(0, 1)])
        self.assertGreater(guard.depths()[0], 0)

    def test_body_exclusion_removes_overlapping_pair(self):
        physics = self.physics(contact='<contact><exclude body1="a" body2="b"/></contact>')
        guard = GeometryMonitor(physics, "excluded")
        self.assertEqual(guard.pairs, [])
        self.assertEqual(guard.depths().size, 0)
        guard.install_thresholds(guard.signature, [])
        guard.begin_policy_step("branch")
        guard.check()

    def test_explicit_pair_overrides_body_exclusion_and_incompatible_masks(self):
        physics = self.physics(
            second='contype="2" conaffinity="2"',
            contact='<contact><exclude body1="a" body2="b"/><pair geom1="ga" geom2="gb"/></contact>',
        )
        guard = GeometryMonitor(physics, "explicit")
        self.assertEqual(guard.pairs, [(0, 1)])
        self.assertAlmostEqual(guard.depths()[0], 0.3)

    def test_global_contact_or_constraint_disable_leaves_no_monitored_pairs(self):
        for flag in ("contact", "constraint"):
            with self.subTest(flag=flag):
                physics = self.physics(option=f'<option><flag {flag}="disable"/></option>')
                guard = GeometryMonitor(physics, flag)
                self.assertEqual(guard.pairs, [])
                self.assertEqual(guard.depths().size, 0)

    def test_parent_filter_disable_is_respected(self):
        for disable in (False, True):
            with self.subTest(disable=disable):
                option = '<option><flag filterparent="disable"/></option>' if disable else ""
                physics = dm_mujoco.Physics.from_xml_string(f"""<mujoco>{option}<worldbody>
                <body pos="0 0 1"><freejoint/><geom type="sphere" size=".2"/>
                  <body pos=".1 0 0"><joint type="hinge"/><geom type="sphere" size=".2"/></body>
                </body></worldbody></mujoco>""")
                guard = GeometryMonitor(physics, "parent")
                self.assertEqual(guard.pairs, [(0, 1)] if disable else [])
                if disable:
                    self.assertGreater(guard.depths()[0], 0)

    def test_runtime_contact_filter_is_respected(self):
        physics = self.physics()
        guard = GeometryMonitor(physics, "callback")
        original = mujoco.get_mjcb_contactfilter()
        try:
            mujoco.set_mjcb_contactfilter(lambda model, data, first, second: 1)
            np.testing.assert_array_equal(guard.depths(), [0.0])
        finally:
            mujoco.set_mjcb_contactfilter(original)


class BranchGuardWindowTest(unittest.TestCase):
    def environment(self):
        cfg = OmegaConf.create(dict(task="dmc_finger_spin", action_repeat=2, time_limit=200, size=[32, 32], seed=11))
        env = make_env(cfg, 0, physics=True)
        self.addCleanup(env.close)
        render = mock.patch.object(env.base, "render", return_value=np.zeros((32, 32, 3), np.uint8))
        render.start()
        self.addCleanup(render.stop)
        profile = env.setup_guard()
        self.assertNotIn("failure", profile)
        env.guard.install_thresholds(env.guard.signature, np.full(len(env.guard.pairs), np.inf))
        action = np.zeros(env.action_space.shape)
        env.reconstruction_step(action, True, None, "off")
        return env, action

    def start_branch(self, env, action, length=8):
        # The source episode is already past step five; the guard window must be
        # relative to this restore, not the source episode's policy-step counter.
        for _ in range(7):
            env.reconstruction_step(action, False, None, "off")
        request = dict(
            context=env.context(), context_vector=env.context_vector(), core=env.physics_labels(), length=length
        )
        row = env.reconstruction_step(action, True, request, "off")
        self.assertIsNone(row.report["failure"])
        self.assertTrue(row.report["root"])
        self.assertEqual(env.branch_steps, 0)
        return request

    def test_root_and_steps_one_through_five_checked_but_six_and_later_not_checked(self):
        env, action = self.environment()
        checks_per_step = env.repeat * env.base._env._n_sub_steps
        with mock.patch.object(env.guard, "check", wraps=env.guard.check) as check:
            request = self.start_branch(env, action)
            self.assertEqual(check.call_count, 1)
            self.assertGreater(env.policy_steps, 5)
            for step in range(1, 9):
                before = check.call_count
                row = env.reconstruction_step(action, False, None, "off")
                self.assertIsNone(row.report["failure"])
                self.assertTrue(row.report["branch"])
                self.assertEqual(check.call_count - before, checks_per_step if step <= 5 else 0)
                self.assertEqual(env.branch_steps, step)
            self.assertTrue(row.done)
            before = check.call_count
            env.reconstruction_step(action, True, request, "off")
            self.assertEqual(env.branch_steps, 0)
            self.assertEqual(check.call_count - before, 1)
            before = check.call_count
            env.reconstruction_step(action, False, None, "off")
            self.assertEqual(check.call_count - before, checks_per_step)

    def test_penetration_on_step_five_still_rejects_and_resets_branch_counter(self):
        env, action = self.environment()
        self.start_branch(env, action)
        original = env.guard.check

        def check():
            if env.branch_steps == 4:
                raise UnsafeState("penetration")
            original()

        with mock.patch.object(env.guard, "check", side_effect=check):
            for _ in range(4):
                self.assertIsNone(env.reconstruction_step(action, False, None, "off").report["failure"])
            row = env.reconstruction_step(action, False, None, "off")
        self.assertEqual(row.report["failure"], "penetration")
        self.assertEqual(row.report["failure_details"]["branch_step"], 5)
        self.assertEqual(row.report["failure_details"]["phase"], "rollout")
        self.assertTrue(row.report["branch"])
        self.assertFalse(env.in_branch)
        self.assertEqual(env.branch_steps, 0)

    def test_restore_rejection_preserves_pair_details_and_step_zero_after_reset(self):
        env, action = self.environment()
        request = dict(context=env.context(), context_vector=env.context_vector(), core=env.physics_labels(), length=8)
        details = dict(geom_pair=[0, 1], geom_names=["a", "b"], depth=0.3, threshold=0.01)
        with mock.patch.object(env.guard, "check", side_effect=UnsafeState("penetration", **details)):
            row = env.reconstruction_step(action, True, request, "off")
        self.assertEqual(row.report["failure"], "penetration")
        self.assertEqual(
            row.report["failure_details"], dict(message="penetration", phase="restore", branch_step=0, **details)
        )
        self.assertTrue(row.observation["is_first"])
        self.assertFalse(env.in_branch)

    def test_nonfinite_action_is_still_rejected_after_geometry_window(self):
        env, action = self.environment()
        self.start_branch(env, action)
        for _ in range(5):
            env.reconstruction_step(action, False, None, "off")
        with mock.patch.object(env.guard, "check", wraps=env.guard.check) as check:
            row = env.reconstruction_step(np.full_like(action, np.nan), False, None, "off")
        check.assert_not_called()
        self.assertEqual(row.report["failure"], "nonfinite_action")
        self.assertEqual(env.branch_steps, 0)

    def test_nonfinite_state_is_still_rejected_after_geometry_window(self):
        env, action = self.environment()
        self.start_branch(env, action)
        for _ in range(5):
            env.reconstruction_step(action, False, None, "off")
        good = env.physics_labels()
        bad = good.copy()
        bad[0] = np.nan
        with mock.patch.object(env, "physics_labels", side_effect=[bad, good]):
            row = env.reconstruction_step(action, False, None, "off")
        self.assertEqual(row.report["failure"], "nonfinite_transition")

    def test_ordinary_calibration_and_validation_remain_active_past_step_five(self):
        env, action = self.environment()
        checks_per_step = env.repeat * env.base._env._n_sub_steps
        for _ in range(7):
            row = env.reconstruction_step(action, False, None, "calibrate")
            self.assertEqual(len(row.report["samples"]), checks_per_step)
        with mock.patch.object(env.guard, "check", wraps=env.guard.check) as check:
            row = env.reconstruction_step(action, False, None, "validate")
        self.assertIsNone(row.report["failure"])
        self.assertEqual(check.call_count, checks_per_step)


if __name__ == "__main__":
    unittest.main()
