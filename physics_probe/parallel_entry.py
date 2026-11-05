"""Additional stage-3 lane; original queue remains the only global status writer."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .stage3 import artifact, atomic_json, fingerprint


def run_entry(root, entry_id):
    protocol = json.loads((root / "protocol.json").read_text())
    if entry_id not in protocol["entries"]:
        raise ValueError("Entry not in original campaign")
    if len(os.sched_getaffinity(0)) != 1 or os.getpriority(os.PRIO_PROCESS, 0) < 10:
        raise ValueError("Use one CPU via taskset and nice -n 19")
    directory = root / "parallel" / entry_id
    directory.mkdir(parents=True, exist_ok=False)
    status = {
        "state": "running",
        "entry": entry_id,
        "pid": os.getpid(),
        "started_unix": time.time(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "jobs": {},
    }
    atomic_json(directory / "status.json", status)
    stages = ["collect"] + [s for source in protocol["sources"] for s in (f"fit_{source}", f"test_{source}")]
    try:
        for stage in stages:
            status["current"] = stage
            status["jobs"][stage] = {"state": "running", "started_unix": time.time()}
            atomic_json(directory / "status.json", status)
            print(f"START {entry_id}/{stage}", flush=True)
            with (directory / f"{stage}.log").open("x") as log:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "physics_probe.stage3",
                        "--output",
                        str(root),
                        "--worker-entry",
                        entry_id,
                        "--worker-stage",
                        stage,
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            target = artifact(root, entry_id, stage)
            if result.returncode or not target.is_file():
                status["jobs"][stage].update(state="failed", exit_code=result.returncode)
                raise RuntimeError(f"Parallel stage failed: {entry_id}/{stage}")
            status["jobs"][stage].update(state="complete", ended_unix=time.time(), artifact_sha256=fingerprint(target))
            atomic_json(directory / "status.json", status)
            print(f"DONE {entry_id}/{stage}", flush=True)
        status.update(state="complete", current=None, ended_unix=time.time())
    except BaseException as exc:
        status.update(state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        atomic_json(directory / "status.json", status)
        raise
    atomic_json(directory / "status.json", status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--entry")
    selection.add_argument("--entries", nargs="+", help="Process these models sequentially on this lane")
    args = parser.parse_args()
    root = args.output.resolve()
    entries = args.entries if args.entries else [args.entry]
    if len(set(entries)) != len(entries):
        raise ValueError("Duplicate model entries on lane")
    protocol = json.loads((root / "protocol.json").read_text())
    if any(entry not in protocol["entries"] for entry in entries):
        raise ValueError("Entry not in original campaign")
    for entry in entries:
        run_entry(root, entry)


if __name__ == "__main__":
    main()
