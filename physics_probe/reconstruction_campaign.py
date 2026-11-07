"""Resource-bounded CQ stages 3/4 followed by stage 5 on all three requested groups."""

import argparse
import hashlib
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from .stage3 import atomic_json
from .stage3 import summarize as summarize_training
from .stage3 import validate_manifest

CQ = "CQ_wasser_symlog_advpositive_only_f64_H15"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cq-stage3", required=True)
    parser.add_argument("--cq-stage4", required=True)
    parser.add_argument("--stage5", required=True)
    parser.add_argument("--cpus", type=int, nargs="+", required=True)
    args = parser.parse_args()
    if (
        not 1 <= len(args.cpus) <= 6
        or len(set(args.cpus)) != len(args.cpus)
        or not set(args.cpus) <= os.sched_getaffinity(0)
    ):
        raise ValueError("Select one to six distinct allowed CPUs")
    preflight = Path(args.preflight).resolve()
    checks = json.loads((preflight / "checks.json").read_text())
    smoke = json.loads((preflight / "smoke.json").read_text())
    if checks["models"] != 30 or not smoke["passed"]:
        raise ValueError("Require 30-model preflight and passing CQ/stage5 integration check")
    repo = Path(__file__).resolve().parents[1]
    for relative, expected in checks["provenance"]["source_sha256"].items():
        if relative in ("dreamer.py", "rssm.py", "networks.py", "distributions.py", "truncation.py"):
            if hashlib.sha256((repo / relative).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Frozen model implementation changed after preflight: {relative}")
    root, cq3, cq4, fifth = [Path(p).resolve() for p in (args.output, args.cq_stage3, args.cq_stage4, args.stage5)]
    if any(p.exists() for p in (root, cq3, cq4, fifth)):
        raise FileExistsError("Use new campaign outputs; no automatic overwrite or restart")
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    snapshot = root / "source_snapshot"
    snapshot.mkdir()
    for file in repo.glob("*.py"):
        shutil.copy2(file, snapshot / file.name)
    for directory in ("physics_probe", "envs", "optim"):
        shutil.copytree(repo / directory, snapshot / directory, ignore=shutil.ignore_patterns("__pycache__"))
    manifest = json.loads((preflight / "cq_manifest.json").read_text())
    entries = validate_manifest(manifest, 9)
    cq3.mkdir(parents=True)
    atomic_json(cq3 / "manifest.json", manifest)
    training = {
        "entries": [e["id"] for e in entries],
        "model_count": 9,
        "checkpoint_group": CQ,
        "modality": "proprio",
        "sources": ["features", "observations"],
        "observation_control": "proprio",
        "episodes": 100,
        "epochs": 100,
        "max_steps": 500,
        "device": "cuda:0",
        "gpu_memory_gib": 2,
        "data_seed": 50000,
        "train_seed": 17,
        "batch_size": 256,
        "hidden": 256,
        "lr": 1e-3,
        "decoder_dtype": "float64",
        "split": "episode60/20/20; seed17",
        "stages4_and5_enabled": True,
    }
    atomic_json(cq3 / "protocol.json", training)
    sources = [
        {"alias": "cq", "group": CQ, "source": str(cq3)},
        {
            "alias": "ve_proprio",
            "group": "VE_wasser_symlog_f64_H15",
            "source": str(repo / "logdir/dyn_recon/physics_probe_stage3_symlog_f64_20261105"),
        },
        {
            "alias": "ve_vision",
            "group": "VE_wasser_H15",
            "source": str(repo / "logdir/dyn_recon/physics_probe_stage3_vision_20261105"),
        },
    ]
    protocol = {
        "sources": sources,
        "cpus": args.cpus,
        "gpu": 6,
        "torch_gib_per_worker": 2,
        "nice": 19,
        "threads_per_worker": 1,
        "cq_stage3": str(cq3),
        "cq_stage4": str(cq4),
        "stage5": str(fifth),
        "points": [0, 100, 200, 300, 400, 450],
        "imagined_depths": [8, 10, 12, 14],
        "post_restore_steps": 1,
        "stage4_horizon": 50,
        "stage5_seed": 71317,
        "test_episodes_per_model": 20,
        "planned_stage5_origins": 3600,
        "planned_stage5_depth_branches_per_source": 14400,
        "image_policy": (
            "CQ stage4:3 task sheets. Stage5:10 direct feature-to-physics sheets plus "
            "10 separate native observation decoder sheets; four imagined depths, no extra action."
        ),
        "stage5_visualization": {
            "primary_path": "imagined feature -> trained feature-to-core -> static simulator render",
            "native_decoder": True,
            "post_decode_actions": 0,
            "proprio_path": "native proprio -> trained observation-to-core -> static render",
            "vision_path": "native image decoder -> RGB",
        },
        "stage5_controls": ["features", "core64_control"],
        "preflight": str(preflight),
        "source_sha256": {
            str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.rglob("*.py")
        },
    }
    atomic_json(root / "protocol.json", protocol)
    state = {"state": "running", "phase": "cq_stage3", "started_unix": time.time(), "jobs": {}, "pid": os.getpid()}
    lock, stopped = threading.Lock(), threading.Event()
    live = {}
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="6",
        MUJOCO_GL="egl",
        MUJOCO_EGL_DEVICE_ID="6",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )

    def save():
        atomic_json(root / "status.json", state)
        if state["phase"] == "cq_stage3":
            atomic_json(cq3 / "status.json", {"state": state["state"], "jobs": state["jobs"], "campaign": str(root)})
        if fifth.exists():
            atomic_json(
                fifth / "status.json",
                {
                    "state": state["state"],
                    "phase": state["phase"],
                    "jobs": {k: v for k, v in state["jobs"].items() if k.startswith("stage5/")},
                    "campaign": str(root),
                },
            )

    def run(key, cores, command):
        if stopped.is_set():
            return
        log = root / "logs" / (key.replace("/", "__") + ".log")
        with lock:
            state["jobs"][key] = {"state": "running", "started_unix": time.time(), "cpus": cores, "log": str(log)}
            save()
        with log.open("x") as stream:
            process = subprocess.Popen(
                ["taskset", "-c", ",".join(map(str, cores)), "nice", "-n", "19", sys.executable, "-u", *command],
                cwd=snapshot,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        with lock:
            live[key] = process
            if stopped.is_set() and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            state["jobs"][key]["pid"] = process.pid
            save()
        code = process.wait()
        with lock:
            live.pop(key, None)
            state["jobs"][key].update(
                state="complete" if code == 0 else "failed", exit_code=code, finished_unix=time.time()
            )
            save()
        if code:
            stopped.set()
            raise RuntimeError(f"{key} exited {code}; see {log}")

    def parallel(jobs, callback):
        pending = queue.Queue()
        errors = []
        for job in jobs:
            pending.put(job)

        def lane(cpu):
            while not stopped.is_set():
                try:
                    job = pending.get_nowait()
                except queue.Empty:
                    return
                try:
                    callback(job, cpu)
                except BaseException as exc:
                    stopped.set()
                    with lock:
                        errors.append(exc)
                        for process in live.values():
                            if process.poll() is None:
                                os.killpg(process.pid, signal.SIGTERM)
                    return
                finally:
                    pending.task_done()

        workers = [threading.Thread(target=lane, args=(cpu,)) for cpu in args.cpus]
        for thread in workers:
            thread.start()
        for thread in workers:
            thread.join()
        if errors:
            raise errors[0]

    save()
    try:

        def train(entry, cpu):
            for stage in ("collect", "fit_features", "test_features", "fit_observations", "test_observations"):
                run(
                    f"cq_stage3/{entry}/{stage}",
                    [cpu],
                    [
                        "-m",
                        "physics_probe.stage3",
                        "--output",
                        str(cq3),
                        "--worker-entry",
                        entry,
                        "--worker-stage",
                        stage,
                    ],
                )

        parallel(training["entries"], train)
        summarize_training(cq3, training)
        atomic_json(cq3 / "status.json", {"state": "complete", "finished_unix": time.time(), "campaign": str(root)})
        state["phase"] = "cq_stage4"
        save()
        run(
            "cq_stage4",
            args.cpus,
            [
                "-m",
                "physics_probe.stage4",
                "--source-root",
                str(cq3),
                "--output",
                str(cq4),
                "--checkpoint-group",
                CQ,
                "--expected-models",
                "9",
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
                "observations",
                "--cpus",
                *map(str, args.cpus),
            ],
        )
        run(
            "cq_stage4_images",
            args.cpus[:1],
            [
                "-m",
                "physics_probe.render_first_step",
                "--stage4",
                str(cq4),
                "--output",
                str(cq4 / "first_step_images"),
                "--start",
                "100",
                "--frame-all",
                "--checkpoint-group",
                CQ,
            ],
        )
        if len(list((cq4 / "first_step_images").glob("*.png"))) != 3:
            raise ValueError("Expected three CQ task sheets")
        state["phase"] = "stage5"
        save()
        fifth.mkdir(parents=True)
        jobs = []
        for source in sources:
            source_protocol = json.loads((Path(source["source"]) / "protocol.json").read_text())
            jobs += [
                {**source, "entry": entry, "key": f"{source['alias']}/{entry}"} for entry in source_protocol["entries"]
            ]
        if len(jobs) != 30:
            raise ValueError("Expected 30 models for stage5")
        atomic_json(fifth / "protocol.json", {**protocol, "jobs": jobs})

        def imagined(job, cpu):
            run(
                f"stage5/{job['key']}",
                [cpu],
                [
                    "-m",
                    "physics_probe.stage5",
                    "--source-root",
                    job["source"],
                    "--entry",
                    job["entry"],
                    "--output",
                    str(fifth / job["key"]),
                    "--checkpoint-group",
                    job["group"],
                ],
            )

        parallel(jobs, imagined)
        state["phase"] = "stage5_images"
        save()
        run(
            "stage5_summary",
            args.cpus[:1],
            ["-m", "physics_probe.stage5_report", "--output", str(fifth), "--skip-images"],
        )
        native_data = fifth / "native_decoder_data"
        native_gallery = fifth / "native_decoder_images"
        run(
            "stage5_native_decode",
            args.cpus[:1],
            [
                str(snapshot / "physics_probe/export_native_observations.py"),
                "--stage5",
                str(fifth),
                "--snapshot",
                str(snapshot),
                "--output",
                str(native_data),
            ],
        )
        run(
            "stage5_native_images",
            args.cpus[:1],
            [
                str(snapshot / "physics_probe/render_native_observations.py"),
                "--decoded",
                str(native_data),
                "--output",
                str(native_gallery),
            ],
        )
        if len(list(native_gallery.glob("*.png"))) != 10:
            raise ValueError("Expected ten group/task sheets for imagined-depth images")
        physics_gallery = fifth / "physics_reconstruction_images"
        run(
            "stage5_physics_images",
            args.cpus[:1],
            [
                str(snapshot / "physics_probe/render_physics_reconstructions.py"),
                "--native-gallery",
                str(native_gallery),
                "--output",
                str(physics_gallery),
            ],
        )
        if len(list(physics_gallery.glob("*.png"))) != 10:
            raise ValueError("Expected ten direct feature-to-physics sheets")
        run(
            "stage5_comparison_images",
            args.cpus[:1],
            [str(snapshot / "physics_probe/compare_reconstruction_images.py"), "--stage5", str(fifth)],
        )
        state.update(
            state="complete",
            phase=None,
            finished_unix=time.time(),
            native_decoder_gallery=str(native_gallery / "README.md"),
            physics_reconstruction_gallery=str(physics_gallery / "README.md"),
            comparison_gallery=str(fifth / "comparison_images/README.md"),
        )
        save()
    except BaseException as exc:
        stopped.set()
        with lock:
            for process in live.values():
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
            state.update(state="failed", error=f"{type(exc).__name__}: {exc}", finished_unix=time.time())
            save()
        raise


if __name__ == "__main__":
    main()
