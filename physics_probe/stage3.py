"""Sequential, resource-bounded stage-3 queue. Run under taskset/nice and CUDA_VISIBLE_DEVICES=6 or 7."""

import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_manifest(manifest, expected_models=None):
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest must contain model entries")
    if expected_models is not None and len(entries) != expected_models:
        raise ValueError(f"Expected {expected_models} models, got {len(entries)}")
    ids = []
    for entry in entries:
        task, seed = entry.get("task"), entry.get("model_seed")
        if (
            not isinstance(task, str)
            or not re.fullmatch(r"[a-z0-9_]+", task)
            or type(seed) is not int
            or seed < 0
            or entry.get("id") != f"{task}_s{seed}"
        ):
            raise ValueError("Invalid model identity in manifest")
        if not all(isinstance(entry.get(k), str) and entry[k] for k in ("run", "checkpoint_sha256", "config_sha256")):
            raise ValueError("Manifest entry lacks run/checkpoint/config identity")
        ids.append(entry["id"])
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate model identities in manifest")
    return entries


def artifact(root, entry, stage):
    base = root / entry
    if stage == "collect":
        return base / "data/dataset.json"
    if stage.startswith("fit_"):
        return base / stage.removeprefix("fit_") / "training.json"
    return base / f"{stage}.json"


@contextmanager
def stage_lock(root, entry_id, stage):
    """Coordinate original sequential queue and additional workers without shared status writers."""
    directory = root / ".locks"
    directory.mkdir(exist_ok=True)
    with (directory / f"{entry_id}_{stage}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def completed_artifact(root, entry, stage, protocol):
    """Only reuse complete products that match this campaign; never reuse partial directories."""
    target = artifact(root, entry["id"], stage)
    if not target.exists():
        return False
    result = json.loads(target.read_text())
    if result.get("entry") != entry:
        raise ValueError(f"Artifact model identity differs: {target}")
    if stage == "collect":
        if (
            len(result["episodes"]) != protocol["episodes"]
            or result["seed"] != protocol["data_seed"]
            or result["max_steps"] != protocol["max_steps"]
        ):
            raise ValueError(f"Artifact collection budget differs: {target}")
    elif stage.startswith("fit_"):
        source = stage.removeprefix("fit_")
        if (
            result["source"] != source
            or result["epochs"] != protocol["epochs"]
            or result["seed"] != protocol["train_seed"]
            or result["batch_size"] != protocol["batch_size"]
            or result["lr"] != protocol["lr"]
            or result.get("decoder_dtype") != "float64"
            or not (target.parent / "decoder.pt").is_file()
        ):
            raise ValueError(f"Artifact training configuration differs: {target}")
        if result["dataset_sha256"] != fingerprint(root / entry["id"] / "data/dataset.json"):
            raise ValueError(f"Artifact dataset identity differs: {target}")
    else:
        source = stage.removeprefix("test_")
        if (
            result["source"] != source
            or result.get("kind") != "heldout_prediction_only"
            or result.get("decoder_dtype") != "float64"
            or result["dataset_sha256"] != fingerprint(root / entry["id"] / "data/dataset.json")
            or result["checkpoint_sha256"] != fingerprint(root / entry["id"] / source / "decoder.pt")
        ):
            raise ValueError(f"Artifact evaluation identity differs: {target}")
    return True


def worker(root, entry_id, stage):
    with stage_lock(root, entry_id, stage):
        _worker_locked(root, entry_id, stage)


def _worker_locked(root, entry_id, stage):
    import torch

    from .common import select_entry, validate_device
    from .pipeline import collect, fit
    from .prediction import prediction_check

    protocol = json.loads((root / "protocol.json").read_text())
    device = protocol["device"]
    validate_device(device)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if len(os.sched_getaffinity(0)) != 1:
        raise ValueError("Run the queue under taskset with exactly one CPU")
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        raise ValueError("Run the queue under nice -n 19 (minimum accepted niceness: 10)")
    if device.startswith("cuda"):
        cap = protocol["gpu_memory_gib"] * 1024**3
        torch.cuda.set_per_process_memory_fraction(cap / torch.cuda.get_device_properties(device).total_memory, device)
    entry = select_entry(root / "manifest.json", entry_id)
    if completed_artifact(root, entry, stage, protocol):
        print(f"REUSE verified completed artifact: {entry_id}/{stage}", flush=True)
        return
    base = root / entry_id
    data = base / "data"
    if stage == "collect":
        collect(entry, data, protocol["data_seed"], protocol["episodes"], protocol["max_steps"], device)
    elif stage.startswith("fit_"):
        source = stage.removeprefix("fit_")
        result = fit(
            data,
            base / source,
            source,
            protocol["train_seed"],
            protocol["epochs"],
            protocol["batch_size"],
            protocol["hidden"],
            protocol["lr"],
            device,
        )
        if result["decoder_dtype"] != "float64":
            raise ValueError("Stage 3 requires float64 reconstruction")
    else:
        source = stage.removeprefix("test_")
        prediction_check(data, base / source / "decoder.pt", base / f"{stage}.json", device, protocol["batch_size"])


def summarize(root, protocol):
    rows = []
    for entry in protocol["entries"]:
        for source in protocol["sources"]:
            report = json.loads((root / entry / f"test_{source}.json").read_text())
            metrics = report["state_metrics"]
            rows.append(
                {
                    "entry": entry,
                    "source": source,
                    "dtype": report["decoder_dtype"],
                    "best_epoch": report["best_epoch"],
                    "validation_loss": report["best_validation_loss"],
                    "test_loss": report["test_loss"],
                    "test_samples": report["samples"],
                    **{
                        key: metrics.get(key, {}).get("mean")
                        for key in ("qpos_nonquat_wrapped_rmse", "qvel_rmse", "act_rmse", "orientation_radians")
                    },
                }
            )
    atomic_json(root / "summary.json", {"rows": rows, "scientific_acceptance": None})
    with (root / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# 第三步：冻结表示的物理状态解码",
        "",
        "结果为独立 test episodes 上的预测误差，不包含恢复仿真或想象状态评估。尚未指定科学验收阈值。",
        "",
        "| 模型 | 输入 | 最佳轮次 | validation loss | test loss | qpos非四元数周期校正RMSE均值 | qvel RMSE均值 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['entry']} | {row['source']} | {row['best_epoch']} | {row['validation_loss']:.6g} | "
            f"{row['test_loss']:.6g} | {row['qpos_nonquat_wrapped_rmse']:.6g} | {row['qvel_rmse']:.6g} |"
        )
    lines += [
        "",
        "四元数方向误差、activation、各坐标误差和逐episode统计见各 test_*.json。",
        "所有训练与标准化只使用train；validation选checkpoint；test只用于最终评估。",
        "每个模型的数据种子与split相同，feature/observation使用同一份配对数据。",
        "随机actor采样、原始绝对位置目标、周期角度处理和常量阈值均沿用现有probe协议。",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--expected-models", type=int, help="Optional explicit campaign size check")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--observations", action="store_true")
    parser.add_argument("--images", action="store_true", help="Train the single-frame image-to-state CNN control")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-memory-gib", type=float, default=2)
    parser.add_argument("--data-seed", type=int, default=50000)
    parser.add_argument("--train-seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--resume", action="store_true", help="Skip verified completed stages; partial outputs are not overwritten"
    )
    parser.add_argument("--worker-entry")
    parser.add_argument(
        "--worker-stage",
        choices=(
            "collect",
            "fit_features",
            "test_features",
            "fit_observations",
            "test_observations",
            "fit_images",
            "test_images",
        ),
    )
    args = parser.parse_args()
    root = args.output.resolve()
    if args.worker_entry:
        if not args.worker_stage:
            parser.error("--worker-stage required")
        worker(root, args.worker_entry, args.worker_stage)
        return
    if not args.resume:
        if args.manifest is None:
            parser.error("--manifest required")
        if min(args.episodes, args.epochs, args.max_steps, args.batch_size, args.hidden) < 1 or args.episodes < 3:
            parser.error("Positive budgets and at least three episodes required")
        if not 0 < args.gpu_memory_gib <= 4 or not 0 < args.lr < 1:
            parser.error("Use a GPU cap up to 4 GiB and a positive learning rate below 1")
        from .common import (
            load_config,
            observation_mode,
            provenance,
            select_entry,
            validate_device,
        )
        from .environment import TASKS

        validate_device(args.device)
        manifest = json.loads(args.manifest.read_text())
        if args.images and (args.observations or manifest.get("modality") != "vision"):
            parser.error("Select image control only for a vision campaign, separately from proprio control")
        entries = validate_manifest(manifest, args.expected_models)
        for entry in entries:
            selected = select_entry(args.manifest, entry["id"])
            config = load_config(selected)
            if (
                entry["task"] not in TASKS
                or config.env.task != f"dmc_{entry['task']}"
                or config.seed != entry["model_seed"]
                or observation_mode(config) != manifest.get("modality", "proprio")
            ):
                raise ValueError(f"Saved environment does not match manifest: {entry['id']}")
        root.mkdir(parents=True, exist_ok=False)
        (root / "logs").mkdir()
        atomic_json(root / "manifest.json", manifest)
        protocol = {
            "entries": [e["id"] for e in manifest["entries"]],
            "model_count": len(entries),
            "checkpoint_group": manifest.get("group"),
            "modality": manifest.get("modality", "proprio"),
            "observation_control": "single RGB image" if args.images else "proprio" if args.observations else None,
            "sources": (
                ["features", "images"]
                if args.images
                else ["features", "observations"] if args.observations else ["features"]
            ),
            "episodes": args.episodes,
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "device": args.device,
            "gpu_memory_gib": args.gpu_memory_gib,
            "data_seed": args.data_seed,
            "train_seed": args.train_seed,
            "batch_size": args.batch_size,
            "hidden": args.hidden,
            "lr": args.lr,
            "decoder_dtype": "float64",
            "split": "episode 60/20/20; seed 17 unless specified",
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "nice": os.getpriority(os.PRIO_PROCESS, 0),
            "provenance": provenance(),
            "stages4_and5_enabled": False,
        }
        atomic_json(root / "protocol.json", protocol)
        status = {"state": "running", "started_unix": time.time(), "pid": os.getpid(), "jobs": {}}
    else:
        protocol = json.loads((root / "protocol.json").read_text())
        status = json.loads((root / "status.json").read_text())
        status.update(state="running", pid=os.getpid())
    try:
        for entry in protocol["entries"]:
            stages = ["collect"] + [
                stage for source in protocol["sources"] for stage in (f"fit_{source}", f"test_{source}")
            ]
            for stage in stages:
                key = f"{entry}/{stage}"
                previous = status["jobs"].get(key, {})
                target = artifact(root, entry, stage)
                if previous.get("state") == "complete":
                    if not target.is_file() or fingerprint(target) != previous["artifact_sha256"]:
                        raise ValueError(f"Completed stage artifact changed: {key}")
                    continue
                log = root / "logs" / f"{entry}_{stage}.log"
                if log.exists():
                    raise FileExistsError(f"Partial stage preserved at {log}; inspect before retrying")
                status["current"] = key
                status["jobs"][key] = {
                    "state": "running",
                    "started_unix": time.time(),
                    "log": str(log.relative_to(root)),
                }
                atomic_json(root / "status.json", status)
                print(f"START {key}", flush=True)
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "physics_probe.stage3",
                    "--output",
                    str(root),
                    "--worker-entry",
                    entry,
                    "--worker-stage",
                    stage,
                ]
                with log.open("x") as f:
                    result = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT)
                job = status["jobs"][key]
                job.update(exit_code=result.returncode, ended_unix=time.time())
                if result.returncode or not target.is_file():
                    job["state"] = "failed"
                    raise RuntimeError(f"Stage failed: {key}; inspect {log}")
                job.update(state="complete", artifact_sha256=fingerprint(target))
                atomic_json(root / "status.json", status)
                print(f"DONE {key}", flush=True)
        summarize(root, protocol)
        status.update(state="complete", ended_unix=time.time(), current=None)
    except BaseException as exc:
        status.update(state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        atomic_json(root / "status.json", status)
        raise
    atomic_json(root / "status.json", status)


if __name__ == "__main__":
    main()
