"""Scheduling tests that do not initialize Torch or CUDA."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from physics_probe.stage3 import completed_artifact, validate_manifest


class Stage3QueueTest(unittest.TestCase):
    def test_parallel_lane_runs_selected_entries_in_order(self):
        from physics_probe.parallel_entry import main

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "protocol.json").write_text(json.dumps({"entries": ["finger_spin_s1", "hopper_hop_s42"]}))
            argv = ["parallel_entry", "--output", temporary, "--entries", "finger_spin_s1", "hopper_hop_s42"]
            with patch.object(sys, "argv", argv), patch("physics_probe.parallel_entry.run_entry") as run:
                main()
            self.assertEqual([call.args[1] for call in run.call_args_list], ["finger_spin_s1", "hopper_hop_s42"])

    def test_explicit_nine_model_manifest(self):
        entries = [
            {
                "id": f"{task}_s{seed}",
                "task": task,
                "model_seed": seed,
                "run": "some/run",
                "checkpoint_sha256": "checkpoint",
                "config_sha256": "config",
            }
            for task in ("finger_spin", "hopper_hop", "quadruped_run")
            for seed in (1, 42, 100)
        ]
        self.assertEqual(validate_manifest({"entries": entries}, 9), entries)
        with self.assertRaises(ValueError):
            validate_manifest({"entries": entries}, 8)
        with self.assertRaises(ValueError):
            validate_manifest({"entries": entries + entries[:1]})
        bad = [{**entries[0], "id": "../../wrong"}]
        with self.assertRaises(ValueError):
            validate_manifest({"entries": bad})

    def test_completed_collection_must_match_identity_and_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entry = {"id": "test", "checkpoint_sha256": "abc"}
            protocol = {"episodes": 3, "data_seed": 10, "max_steps": 500}
            self.assertFalse(completed_artifact(root, entry, "collect", protocol))
            target = root / "test/data/dataset.json"
            target.parent.mkdir(parents=True)
            target.write_text(json.dumps({"entry": entry, "episodes": [{}, {}, {}], "seed": 10, "max_steps": 500}))
            self.assertTrue(completed_artifact(root, entry, "collect", protocol))
            with self.assertRaises(ValueError):
                completed_artifact(root, entry, "collect", {**protocol, "episodes": 100})
            with self.assertRaises(ValueError):
                completed_artifact(root, {**entry, "checkpoint_sha256": "changed"}, "collect", protocol)

    def test_concurrent_stage_claim_executes_once(self):
        script = """
import sys
import time
from pathlib import Path
from physics_probe.stage3 import stage_lock
root = Path(sys.argv[1])
with stage_lock(root, 'test', 'collect'):
    done = root / 'done'
    if not done.exists():
        with (root / 'executions').open('a') as f:
            f.write('executed\\n')
        time.sleep(.1)
        done.write_text('complete')
"""
        with tempfile.TemporaryDirectory() as temporary:
            processes = [subprocess.Popen([sys.executable, "-c", script, temporary]) for _ in range(2)]
            for process in processes:
                self.assertEqual(process.wait(timeout=30), 0)
            self.assertEqual((Path(temporary) / "executions").read_text().splitlines(), ["executed"])


if __name__ == "__main__":
    unittest.main()
