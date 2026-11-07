"""Summaries and one task sheet containing every seed and requested imagined depth."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .common import load_config, new_directory, sha256, write_json
from .environment import ProbeEnv
from .render_first_step import apply_camera, capture_camera, font


def render_cell(entry_report, example, depth, width=320, height=240):
    canvas = Image.new("RGB", (2 * width, height + 112), "#f4f6f8")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 5), f"{entry_report['entry']['id']} | FEATURE: imag +{depth}", font=font(17), fill="#17212b")
    start = example["start"] if example else 100
    boundary = example["truncation"]["retained_steps"] if example else None
    draw.text(
        (10, 29),
        f"RESTORE +1 => t={start}+{depth}+1={start+depth+1} | saved-EMA boundary={boundary}",
        font=font(13),
        fill="#465565",
    )
    record = example["depths"].get(depth) if example else None
    if record is None:
        draw.text((12, 120), "Reference unavailable; see report.json", font=font(18), fill="#b12626")
        return canvas
    config = load_config(entry_report["entry"])
    env = ProbeEnv(config.env, example["episode_seed"])
    try:
        snapshots = {key: row["snapshot"] for key, row in record["branches"].items() if row["snapshot"] is not None}
        cameras, bounds = [], []
        camera_id = 2 if entry_report["entry"]["task"] == "quadruped_run" else 0
        for key, snapshot in snapshots.items():
            env.restore(snapshot)
            moving = env.physics.model.geom_bodyid != 0
            centers, radii = env.physics.data.geom_xpos[moving], env.physics.model.geom_rbound[moving, None]
            bounds.append(((centers - radii).min(0), (centers + radii).max(0)))
        reference_snapshot = snapshots.get("core64_control", next(iter(snapshots.values()), None))
        if reference_snapshot is not None:
            env.restore(reference_snapshot)
            env.physics.render(
                height=height,
                width=width,
                camera_id=camera_id,
                scene_callback=lambda p, s: capture_camera(p, s, cameras),
            )
            low, high = np.min([b[0] for b in bounds], 0), np.max([b[1] for b in bounds], 0)
            center, radius = (low + high) / 2, max(float(np.linalg.norm(high - low) / 2), 0.1)
            for camera in cameras:
                forward = np.asarray(camera["forward"])
                forward /= np.linalg.norm(forward)
                tangent = (camera["frustum_top"] - camera["frustum_bottom"]) / (2 * camera["frustum_near"])
                distance = 1.1 * radius * (1 + 1 / min(tangent, tangent * width / height))
                camera["pos"] = (center - forward * distance).tolist()
                camera["frustum_center"] = 0.0
                camera["frustum_far"] = max(camera["frustum_far"], distance + 3 * radius)
        for col, key in enumerate(("features", "core64_control")):
            draw.text(
                (col * width + 10, 51),
                f"Decode imag f(+{depth})" if key == "features" else "True core restore",
                font=font(16),
                fill="#17212b",
            )
            branch = record["branches"].get(key)
            if branch is None or branch["snapshot"] is None:
                draw.text((col * width + 12, 130), "Restore/step failed", font=font(17), fill="#b12626")
                continue
            env.restore(branch["snapshot"])
            frame = env.physics.render(
                height=height, width=width, camera_id=camera_id, scene_callback=lambda p, s: apply_camera(p, s, cameras)
            )
            canvas.paste(Image.fromarray(frame), (col * width, 75))
            metrics = branch["metrics"]
            draw.text(
                (col * width + 8, height + 80),
                f"proprio RMSE {metrics['observation_rmse']:.4g} | vel {metrics['qvel_rmse']:.4g}",
                font=font(12),
                fill="#17212b",
            )
            if "image_rmse_01" in metrics:
                draw.text(
                    (col * width + 8, height + 96),
                    f"RGB RMSE [0,1] {metrics['image_rmse_01']:.4g}",
                    font=font(12),
                    fill="#465565",
                )
    finally:
        env.close()
    return canvas


def summarize_campaign(root, render_images=True):
    root = Path(root)
    protocol = json.loads((root / "protocol.json").read_text())
    reports, rows, groups = [], [], defaultdict(list)
    for job in protocol["jobs"]:
        report_path = root / job["key"] / "report.json"
        report = json.loads(report_path.read_text())
        reports.append(report)
        groups[(report["checkpoint_group"], report["entry"]["task"])].append((report, report_path.parent))
        for summary in report["summary"]:
            row = {
                "checkpoint_group": report["checkpoint_group"],
                "entry": report["entry"]["id"],
                **{k: v for k, v in summary.items() if k != "metrics"},
            }
            for key, stats in summary["metrics"].items():
                for stat, value in stats.items():
                    row[f"{key}_{stat}"] = value
            rows.append(row)
    write_json(root / "summary.json", {"rows": rows, "scientific_acceptance": None})
    with (root / "summary.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# 步骤⑤：固定想象步的重建与一步接续",
        "",
        "f(0)是测试集posterior；f(d)经过d次prior转移。恢复f(d)解码状态后执行a(d)，图片显示t+d+1。",
        "参考环境从真实起点执行相同想象动作；不用原采集轨迹的未来动作。对照为真实核心状态恢复。",
        "选择8/10/12/14四步，即使超过固定EMA诊断的截断边界也记录，不能当作实际截断点。",
        "proprio与图像指标分开。latent_next指标只检验解码器/世界模型的内部一致性，不是对真实状态的准确度。",
        "同episode起点相关；均值/p95为描述性统计。运行完成不表示科学验收通过。",
        "",
        "|组|模型|参考失败|解码分支失败|真实核心分支失败|",
        "|---|---|---|---|---|",
    ]
    for report in reports:
        failures = {
            source: sum(c["failure"] is not None for c in report["cases"] if c["source"] == source)
            for source in ("features", "core64_control")
        }
        lines.append(
            f"|{report['checkpoint_group']}|{report['entry']['id']}|{len(report['reference_failures'])}|"
            f"{failures['features']}|{failures['core64_control']}|"
        )
    lines += [
        "",
        "## 恢复后第1步的误差均值",
        "",
        "|组|模型|想象步|来源|有效/计划|qvel RMSE|proprio RMSE|奖励绝对差|",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        if row["phase"] != 1:
            continue
        values = [row.get(f"{key}_mean") for key in ("qvel_rmse", "observation_rmse", "reward_abs")]
        lines.append(
            f"|{row['checkpoint_group']}|{row['entry']}|{row['imag_depth']}|{row['source']}|"
            f"{row['available_origins']}/{row['planned_origins']}|"
            + "|".join(f"{v:.6g}" if v is not None else "missing" for v in values)
            + "|"
        )
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    if not render_images:
        return {"models": len(reports), "images": 0}
    images = new_directory(root / "first_step_images")
    manifest = []
    for (group, task), entries in groups.items():
        entries.sort(key=lambda item: item[0]["entry"]["model_seed"])
        depths = entries[0][0]["imagined_depths"]
        sheet = Image.new("RGB", (640 * len(depths), 65 + 352 * len(entries)), "#e8edf2")
        draw = ImageDraw.Draw(sheet)
        draw.text(
            (12, 10), f"{group} | {task} | imagined features at +8 / +10 / +12 / +14", font=font(24), fill="#17212b"
        )
        draw.text(
            (12, 39),
            "Each cell: decoded imagined feature vs true core | BOTH AFTER ONE SAME ACTION | shared camera",
            font=font(17),
            fill="#465565",
        )
        for row, (report, path) in enumerate(entries):
            example = torch.load(path / "example.pt", map_location="cpu", weights_only=True)
            for col, depth in enumerate(depths):
                sheet.paste(render_cell(report, example, depth), (640 * col, 65 + 352 * row))
            manifest.append(
                {
                    "group": group,
                    "entry": report["entry"]["id"],
                    "example_sha256": sha256(path / "example.pt"),
                    "episode_id": example["episode_id"] if example else None,
                    "start": example["start"] if example else None,
                    "imagined_depths": depths,
                    "restored_steps": 1,
                }
            )
        filename = f"{group}__{task}__imag8_10_12_14_step1.png"
        sheet.save(images / filename)
    write_json(
        images / "manifest.json",
        {"examples": manifest, "selection": "first test episode, preselected start100; no error-based selection"},
    )
    image_names = sorted(p.name for p in images.glob("*.png"))
    (images / "README.md").write_text(
        "# 想象feature重建后的第一步\n\n每列依次是想象第8、10、12、14步；每行对应一个模型seed。"
        "每个单元左侧是该想象feature重建，右侧是真实核心状态恢复，均已执行同一个动作接续1步。"
        "图片没有重新生成想象轨迹，直接读取评估中保存的第一步状态。\n\n"
        + "\n\n".join(f"![{name}]({name})" for name in image_names)
        + "\n"
    )
    (images / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Stage 5</title>'
        "<style>body{font-family:sans-serif}img{width:100%;margin:16px 0}</style>"
        "<h1>Imag +8/+10/+12/+14; restored +1</h1>"
        + "".join(f'<img src="{name}" alt="{name}">' for name in image_names)
    )
    return {"models": len(reports), "images": len(image_names)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--skip-images", action="store_true", help="Write numeric reports only; native decoder images are separate"
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    print(summarize_campaign(args.output, render_images=not args.skip_images), flush=True)
