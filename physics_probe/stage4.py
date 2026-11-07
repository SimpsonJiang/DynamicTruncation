"""Bounded CPU campaign for posterior decoder restoration; no training or imagined-state evaluation."""

import argparse
import csv
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from .common import new_directory, provenance, sha256, write_json

GROUP = "VE_wasser_symlog_f64_H15"


def write_status(root, state):
    temporary = root / "status.tmp"
    temporary.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
    temporary.replace(root / "status.json")


def summarize(root, entries, checkpoint_group=GROUP):
    rows, outcomes = [], []
    for entry in entries:
        report = json.loads((root / entry / "report.json").read_text())
        for source, summary in report["summary"].items():
            outcomes.append(
                {
                    "entry": entry,
                    "source": source,
                    "reference_passed": report["reference_passed"],
                    **{k: v for k, v in summary.items() if k != "by_depth"},
                }
            )
            for depth in (0, 1, 5, 15, 50):
                if str(depth) not in summary["by_depth"]:
                    continue
                group = summary["by_depth"][str(depth)]
                row = {
                    "entry": entry,
                    "source": source,
                    "depth": depth,
                    "available_origins": group["available_origins"],
                    "planned_origins": group["planned_origins"],
                }
                for metric, stats in group["metrics"].items():
                    for statistic in ("mean", "p95", "max"):
                        row[f"{metric}_{statistic}"] = stats[statistic]
                rows.append(row)
    write_json(root / "summary.json", {"rows": rows, "outcomes": outcomes, "scientific_acceptance": None})
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (root / "summary.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Posterior 状态恢复与同动作续跑",
        "",
        f"Checkpoint group: `{checkpoint_group}`。",
        "",
        "使用第三步保留的测试 episode；不重新训练、不生成想象 feature、不接入 replay。",
        "每个起点有完整快照参考、真实核心状态恢复，以及所选 decoder 恢复；各自执行同一段已记录动作。",
        "第0步观测误差包含已知的传感器缓存差异，应与动作执行后的结果分别判断。",
        "observation_rmse为proprio诊断误差；vision另报image_rmse_01（RGB像素/255），两者不混合。",
        "完整参考通过指核心状态、proprio、奖励和终止校验；图像复现的灰度差异另见各模型reference_checks。",
        "vision的images对照只输入单帧RGB，features包含冻结RSSM的历史信息；两者不是相同时间信息量的对照。",
        "运行完成/未发生异常不等于重建可靠；没有预设科学验收阈值。",
        "",
        "## 完成与异常",
        "",
        "|模型|来源|完整参考通过|尝试/计划|异常|完成续跑|",
        "|---|---|---|---|---|---|",
    ]
    for o in outcomes:
        lines.append(
            f"|{o['entry']}|{o['source']}|{o['reference_passed']}|{o['attempted']}/{o['planned']}|"
            f"{o['failed']}|{o['full_horizon_completed']}|"
        )
    lines += [
        "",
        "## 续跑误差（起点平均）",
        "",
        "包含位置、速度、观测和奖励的完整均值/p95/最大值见 summary.csv；逐轨迹见各模型 metrics.jsonl。",
        "后续深度仅统计仍有有效指标的轨迹，必须同时查看样本覆盖与异常；同 episode 起点相关，不作为独立置信区间。",
        "",
        "|模型|来源|步数|有效/计划|qpos几何RMSE|qvel RMSE|proprio RMSE|奖励绝对差|累计奖励绝对差|",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r["depth"] not in (1, 15, 50):
            continue
        values = [
            r.get(f"{name}_mean", 0)
            for name in (
                "qpos_nonquat_wrapped_rmse",
                "qvel_rmse",
                "observation_rmse",
                "reward_abs",
                "cumulative_return_abs",
            )
        ]
        lines.append(
            f"|{r['entry']}|{r['source']}|{r['depth']}|{r['available_origins']}/{r['planned_origins']}|"
            + "|".join(f"{v:.6g}" for v in values)
            + "|"
        )
    lines += [
        "",
        "qpos几何RMSE对无限转动角做周期校正，并排除四元数；它仍混合不同关节单位，不能作为唯一指标。",
        "Quadruped四元数方向误差以弧度单独保存在CSV和各模型报告中；原始位置误差同时保留。",
        "",
    ]
    if any("image_rmse_01_mean" in row for row in rows):
        lines += [
            "## 图像误差（像素归一化到0～1）",
            "",
            "|模型|来源|步数|均值|p95|最大值|",
            "|---|---|---|---|---|---|",
        ]
        for row in rows:
            if "image_rmse_01_mean" in row:
                lines.append(
                    f"|{row['entry']}|{row['source']}|{row['depth']}|{row['image_rmse_01_mean']:.6g}|"
                    f"{row['image_rmse_01_p95']:.6g}|{row['image_rmse_01_max']:.6g}|"
                )
    (root / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--points", nargs="+", type=int, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--sources", nargs="+", choices=["features", "observations", "images"], required=True)
    parser.add_argument("--cpus", nargs="+", type=int, required=True)
    parser.add_argument("--checkpoint-group", default=GROUP)
    parser.add_argument("--expected-models", type=int, default=9)
    args = parser.parse_args()
    source = Path(args.source_root).resolve()
    protocol = json.loads((source / "protocol.json").read_text())
    if protocol.get("checkpoint_group") != args.checkpoint_group or len(protocol["entries"]) != args.expected_models:
        raise ValueError("Source checkpoint group or model count differs from the explicitly selected campaign")
    if not 1 <= len(args.cpus) <= 6 or len(set(args.cpus)) != len(args.cpus):
        raise ValueError("Use one to six distinct CPU cores for this campaign")
    if not set(args.cpus) <= os.sched_getaffinity(0):
        raise ValueError("Requested CPUs are outside current affinity")
    if args.horizon < 1 or min(args.points) < 0 or len(set(args.points)) != len(args.points):
        raise ValueError("Invalid rollout budget")
    hashes = {}
    for entry in protocol["entries"]:
        for relative in ["data/dataset.json", *[f"{s}/decoder.pt" for s in args.sources]]:
            path = source / entry / relative
            hashes[str(path)] = sha256(path)
    root = new_directory(Path(args.output).resolve())
    snapshot = root / "source_snapshot"
    snapshot.mkdir()
    repo = Path(__file__).resolve().parents[1]
    for path in repo.glob("*.py"):
        shutil.copy2(path, snapshot / path.name)
    for name in ("physics_probe", "envs", "optim"):
        shutil.copytree(repo / name, snapshot / name, ignore=shutil.ignore_patterns("__pycache__"))
    write_json(
        root / "protocol.json",
        {
            "checkpoint_group": args.checkpoint_group,
            "modality": protocol.get("modality", "proprio"),
            "observation_metrics": "proprio scalar observations; not image pixels",
            "entries": protocol["entries"],
            "source_root": str(source),
            "source_protocol_sha256": sha256(source / "protocol.json"),
            "input_sha256": hashes,
            "points": args.points,
            "horizon": args.horizon,
            "sources": args.sources,
            "cpus": args.cpus,
            "nice": 19,
            "threads_per_worker": 1,
            "device": "cpu",
            "decoder_dtype": "float64",
            "provenance": provenance(),
            "source_snapshot_sha256": {str(p.relative_to(snapshot)): sha256(p) for p in snapshot.rglob("*.py")},
        },
    )
    state = {"status": "running", "started_unix": time.time(), "completed": [], "running": {}, "failed": {}}
    write_status(root, state)
    jobs, lock, stop = queue.Queue(), threading.Lock(), threading.Event()
    for entry in protocol["entries"]:
        jobs.put(entry)

    def worker(cpu):
        while not stop.is_set():
            try:
                entry = jobs.get_nowait()
            except queue.Empty:
                return
            command = [
                "taskset",
                "-c",
                str(cpu),
                "nice",
                "-n",
                "19",
                sys.executable,
                "-u",
                "-m",
                "physics_probe.posterior_restore",
                "--source-root",
                str(source),
                "--entry",
                entry,
                "--checkpoint-group",
                args.checkpoint_group,
                "--output",
                str(root / entry),
                "--horizon",
                str(args.horizon),
                "--points",
                *map(str, args.points),
                "--sources",
                *args.sources,
            ]
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES="6" if protocol.get("modality") == "vision" else "",
                MUJOCO_EGL_DEVICE_ID="6",
                MUJOCO_GL="egl",
                OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
            )
            with lock:
                state["running"][entry] = {"cpu": cpu, "started_unix": time.time()}
                write_status(root, state)
            try:
                with (root / f"{entry}.log").open("x") as stream:
                    result = subprocess.run(command, cwd=snapshot, env=env, stdout=stream, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f"Worker exited {result.returncode}; see {entry}.log")
                report = json.loads((root / entry / "report.json").read_text())
                if not report["reference_passed"]:
                    raise RuntimeError("Full reference validation failed; inspect report.json")
                with lock:
                    state["completed"].append(entry)
            except Exception as exc:
                with lock:
                    state["failed"][entry] = str(exc)
                stop.set()
            finally:
                with lock:
                    state["running"].pop(entry)
                    write_status(root, state)
                jobs.task_done()

    threads = [threading.Thread(target=worker, args=(cpu,)) for cpu in args.cpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if state["failed"]:
        state["status"] = "failed"
    else:
        try:
            summarize(root, protocol["entries"], args.checkpoint_group)
            state["status"] = "complete"
        except Exception as exc:
            state["status"] = "failed"
            state["failed"]["summary"] = str(exc)
    state["finished_unix"] = time.time()
    write_status(root, state)
    if state["failed"]:
        raise RuntimeError(f"Campaign failed; see {root / 'status.json'}")


if __name__ == "__main__":
    main()
