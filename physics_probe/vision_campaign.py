"""Run the approved 12-model vision study through training, restoration and four first-step images."""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from .stage3 import atomic_json, fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage3-output", type=Path, required=True)
    parser.add_argument("--stage4-output", type=Path, required=True)
    parser.add_argument("--cpus", type=int, nargs=2, required=True)
    args = parser.parse_args()
    if len(set(args.cpus)) != 2 or not set(args.cpus) <= os.sched_getaffinity(0):
        raise ValueError("Choose two distinct permitted CPUs")
    preflight = args.preflight.resolve()
    checks = json.loads((preflight / "checks.json").read_text())
    smoke = json.loads((preflight / "smoke.json").read_text())
    manifest = json.loads((preflight / "manifest.json").read_text())
    if (
        checks["models"] != 12
        or not smoke["passed"]
        or manifest["modality"] != "vision"
        or manifest["group"] != "VE_wasser_H15"
    ):
        raise ValueError("Require completed vision checkpoint and integration preflight")
    root, stage3, stage4 = args.output.resolve(), args.stage3_output.resolve(), args.stage4_output.resolve()
    if any(p.exists() for p in (root, stage3, stage4)):
        raise FileExistsError("Campaign outputs must be new; previous work is never overwritten")
    root.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[1]
    snapshot = root / "source_snapshot"
    snapshot.mkdir()
    for path in repo.glob("*.py"):
        shutil.copy2(path, snapshot / path.name)
    for name in ("physics_probe", "envs", "optim"):
        shutil.copytree(repo / name, snapshot / name, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(preflight / "manifest.json", root / "manifest.json")
    protocol = {
        "checkpoint_group": "VE_wasser_H15",
        "modality": "vision",
        "entries": [entry["id"] for entry in manifest["entries"]],
        "sources": ["features", "images"],
        "episodes": 100,
        "epochs": 100,
        "max_steps": 500,
        "split": "60/20/20 episodes; seed17",
        "data_seed": 50000,
        "batch_size": 256,
        "hidden": 256,
        "lr": 1e-3,
        "frozen_dreamer_dtype": "float32",
        "decoder_dtype": "float64",
        "image_control": {
            "frames": 1,
            "channels": [16, 32, 64, 128],
            "kernel": 3,
            "stride": 2,
            "hidden": [256, 256],
            "normalization": "uint8 / 255 - 0.5",
            "proprio_input": False,
        },
        "points": [0, 100, 200, 300, 400, 450],
        "horizon": 50,
        "image_count": 4,
        "image_origin": {"episode": "first saved test episode", "start": 100, "depth": 1},
        "cpus": args.cpus,
        "nice": 19,
        "threads_per_worker": 1,
        "gpu": 6,
        "gpu_workers": 2,
        "torch_gib_per_worker": 2,
        "stage3": str(stage3),
        "stage4": str(stage4),
        "preflight": str(preflight),
        "preflight_sha256": fingerprint(preflight / "checks.json"),
        "smoke_sha256": fingerprint(preflight / "smoke.json"),
        "source_sha256": {str(p.relative_to(snapshot)): fingerprint(p) for p in snapshot.rglob("*.py")},
    }
    atomic_json(root / "protocol.json", protocol)
    state = {"state": "running", "stage": "stage3", "started_unix": time.time(), "pid": os.getpid()}
    atomic_json(root / "status.json", state)
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="6",
        MUJOCO_GL="egl",
        MUJOCO_EGL_DEVICE_ID="6",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    children = []

    def launch(name, cores, arguments):
        command = ["taskset", "-c", ",".join(map(str, cores)), "nice", "-n", "19", sys.executable, "-u", *arguments]
        with (root / f"{name}.log").open("x") as log:
            process = subprocess.Popen(
                command, cwd=snapshot, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        children.append(process)
        state[name] = {"pid": process.pid, "command": command, "started_unix": time.time()}
        atomic_json(root / "status.json", state)
        return process

    def completed(name, process):
        code = process.wait()
        state[name].update(exit_code=code, finished_unix=time.time())
        atomic_json(root / "status.json", state)
        if code:
            raise RuntimeError(f"{name} exited {code}; see {root / (name + '.log')}")

    try:
        main_queue = launch(
            "training_main",
            args.cpus[:1],
            [
                "-m",
                "physics_probe.stage3",
                "--output",
                str(stage3),
                "--manifest",
                str(root / "manifest.json"),
                "--expected-models",
                "12",
                "--episodes",
                "100",
                "--epochs",
                "100",
                "--max-steps",
                "500",
                "--images",
                "--device",
                "cuda:0",
                "--gpu-memory-gib",
                "2",
            ],
        )
        while not (stage3 / "protocol.json").exists():
            if main_queue.poll() is not None:
                raise RuntimeError("Training coordinator exited before initializing the protocol")
            time.sleep(0.5)
        helper = launch(
            "training_helper",
            args.cpus[1:],
            ["-m", "physics_probe.parallel_entry", "--output", str(stage3), "--entries", *protocol["entries"][6:]],
        )
        while main_queue.poll() is None or helper.poll() is None:
            if any(p.poll() not in (None, 0) for p in (main_queue, helper)):
                raise RuntimeError("A training lane failed; inspect training_main.log and training_helper.log")
            time.sleep(2)
        completed("training_main", main_queue)
        completed("training_helper", helper)
        state.update(stage="stage4")
        atomic_json(root / "status.json", state)
        evaluation = launch(
            "restoration",
            args.cpus,
            [
                "-m",
                "physics_probe.stage4",
                "--source-root",
                str(stage3),
                "--output",
                str(stage4),
                "--checkpoint-group",
                "VE_wasser_H15",
                "--expected-models",
                "12",
                "--points",
                "0",
                "100",
                "200",
                "300",
                "400",
                "450",
                "--horizon",
                "50",
                "--sources",
                "features",
                "images",
                "--cpus",
                *map(str, args.cpus),
            ],
        )
        completed("restoration", evaluation)
        state.update(stage="first_step_images")
        atomic_json(root / "status.json", state)
        pictures = stage4 / "first_step_images"
        renderer = launch(
            "render",
            args.cpus[:1],
            [
                "-m",
                "physics_probe.render_first_step",
                "--stage4",
                str(stage4),
                "--output",
                str(pictures),
                "--start",
                "100",
                "--frame-all",
                "--checkpoint-group",
                "VE_wasser_H15",
            ],
        )
        completed("render", renderer)
        if len(list(pictures.glob("*.png"))) != 4:
            raise ValueError("Expected exactly four task summary PNGs")
        shutil.copy2(snapshot / "physics_probe/render_first_step.py", pictures / "render_script.py")
        state.update(
            state="complete",
            stage=None,
            finished_unix=time.time(),
            report=str(stage4 / "REPORT.md"),
            image_gallery=str(pictures / "README.md"),
        )
        atomic_json(root / "status.json", state)
    except BaseException as exc:
        for process in children:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        state.update(state="failed", error=f"{type(exc).__name__}: {exc}", finished_unix=time.time())
        atomic_json(root / "status.json", state)
        raise


if __name__ == "__main__":
    main()
