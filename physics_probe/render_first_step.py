"""Render matched stage-4 branches after exactly one recorded policy action."""

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .common import load_config, new_directory, provenance, sha256, write_json
from .environment import ProbeEnv
from .pipeline import load_probe, load_shard
from .posterior_restore import observe_record, step_metrics, trace_reference

CAMERA_FIELDS = (
    "pos",
    "forward",
    "up",
    "frustum_center",
    "frustum_width",
    "frustum_bottom",
    "frustum_top",
    "frustum_near",
    "frustum_far",
    "orthographic",
)
LABELS = {
    "features": "Feature decoder",
    "observations": "Observation decoder",
    "images": "Image CNN decoder",
    "core64_control": "True core restore",
}


def capture_camera(physics, scene, saved):
    del physics
    for camera in scene.camera:
        saved.append(
            {key: np.asarray(getattr(camera, key)).copy().tolist() for key in CAMERA_FIELDS if hasattr(camera, key)}
        )


def apply_camera(physics, scene, saved):
    del physics
    for camera, fields in zip(scene.camera, saved):
        for key, value in fields.items():
            if isinstance(value, list):
                getattr(camera, key)[:] = value
            else:
                setattr(camera, key, value)


def font(size):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


@torch.no_grad()
def render_campaign(
    stage4, output, start, width=640, height=480, frame_all=False, checkpoint_group="VE_wasser_symlog_f64_H15"
):
    if os.environ.get("MUJOCO_GL") != "egl" or os.environ.get("MUJOCO_EGL_DEVICE_ID") != "6":
        raise ValueError("Use EGL explicitly restricted to device 6 for this rendering campaign")
    stage4 = Path(stage4).resolve()
    protocol = json.loads((stage4 / "protocol.json").read_text())
    if protocol["checkpoint_group"] != checkpoint_group:
        raise ValueError("Unexpected checkpoint group")
    labels = {kind: LABELS[kind] for kind in [*protocol["sources"], "core64_control"]}
    if protocol.get("modality") == "vision":
        labels["features"] = "Vision feature decoder"
        if "observations" in labels:
            labels["observations"] = "Proprio control (privileged)"
    columns = len(labels)
    if start not in protocol["points"]:
        raise ValueError("Choose an existing stage-4 origin")
    for name in ("environment.py", "decoder.py", "prediction.py", "posterior_restore.py"):
        if sha256(Path(__file__).parent / name) != sha256(stage4 / "source_snapshot/physics_probe" / name):
            raise ValueError(f"Stage-4 implementation changed: {name}")
    root = new_directory(output)
    source = Path(protocol["source_root"])
    results, panels, common_episode = [], [], None
    for entry in protocol["entries"]:
        base = source / entry
        report = json.loads((stage4 / entry / "report.json").read_text())
        episode = report["test_episodes"][0]  # deterministic selection, never selected by error
        if common_episode is not None and episode != common_episode:
            raise ValueError("Models do not share the selected test episode")
        common_episode = episode
        predictions, meta, saved_split = {}, None, None
        for kind in protocol["sources"]:
            checkpoint = base / kind / "decoder.pt"
            for path in (checkpoint, base / "data/dataset.json"):
                if sha256(path) != protocol["input_sha256"][str(path)]:
                    raise ValueError(f"Input changed since stage 4: {path}")
            path, meta, saved, model = load_probe(base / "data", checkpoint, "cpu")
            if model.dtype != torch.float64 or episode not in saved["split"]["test"]:
                raise ValueError("Require float64 decoder and held-out episode")
            if saved_split is not None and saved_split != saved["split"]:
                raise ValueError("Decoder splits differ")
            saved_split = saved["split"]
            shard = load_shard(path, meta["episodes"][episode])
            predictions[kind] = model(shard[kind][start : start + 1]).cpu().numpy()[0]
        predictions["core64_control"] = shard["cores"][start].numpy()
        config = load_config(meta["entry"])
        include_image = meta.get("policy_modality") == "vision"
        reference, candidate = [
            ProbeEnv(config.env, shard["episode_seed"], include_image=include_image) for _ in range(2)
        ]
        try:
            records, actions, check = trace_reference(reference, shard, start, 1)
            if not check["passed"] or len(actions) != 1:
                raise ValueError("Recorded first-step reference validation failed")
            camera_id = 2 if meta["entry"]["task"] == "quadruped_run" else 0
            cameras = []
            reference.physics.render(
                height=height,
                width=width,
                camera_id=camera_id,
                scene_callback=lambda p, s: capture_camera(p, s, cameras),
            )
            canvas = Image.new("RGB", (columns * width, height + 144), "#f4f6f8")
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (16, 8),
                f"{entry} | episode {episode} (env seed {shard['episode_seed']}) | t={start} -> {start+1}",
                font=font(24),
                fill="#17212b",
            )
            draw.text(
                (16, 42),
                "One identical recorded action | fixed shared camera | float64 reconstruction",
                font=font(18),
                fill="#465565",
            )
            selected_metrics = {}
            archived = {}
            with (stage4 / entry / "metrics.jsonl").open() as stream:
                for line in stream:
                    row = json.loads(line)
                    if row["episode_id"] == episode and row["start"] == start and row["depth"] == 1:
                        archived[row["source"]] = row["metrics"]
            post_snapshots, lower_bounds, upper_bounds = {}, [], []
            for kind in labels:
                candidate.restore(shard["snapshots"][start], predictions[kind])
                obs, reward, _ = candidate.step(actions[0])
                metrics = step_metrics(observe_record(candidate, obs, reward), records[1], meta["schema"])
                for key in ("qvel_rmse", "observation_rmse", "reward_abs", "qpos_nonquat_wrapped_rmse"):
                    np.testing.assert_allclose(metrics[key], archived[kind][key], rtol=1e-9, atol=1e-10)
                post_snapshots[kind] = candidate.snapshot()
                selected_metrics[kind] = metrics
                moving = candidate.physics.model.geom_bodyid != 0
                centers = candidate.physics.data.geom_xpos[moving].copy()
                radii = candidate.physics.model.geom_rbound[moving, None]
                lower_bounds.append((centers - radii).min(axis=0))
                upper_bounds.append((centers + radii).max(axis=0))
            if frame_all:
                low, high = np.min(lower_bounds, axis=0), np.max(upper_bounds, axis=0)
                center = (low + high) / 2
                radius = max(float(np.linalg.norm(high - low) / 2), 0.1)
                for camera in cameras:
                    forward = np.asarray(camera["forward"])
                    forward /= np.linalg.norm(forward)
                    tangent = (camera["frustum_top"] - camera["frustum_bottom"]) / (2 * camera["frustum_near"])
                    distance = 1.1 * radius * (1 + 1 / min(tangent, tangent * width / height))
                    camera["pos"] = (center - forward * distance).tolist()
                    camera["frustum_center"] = 0.0
                    camera["frustum_far"] = max(camera["frustum_far"], distance + 3 * radius)
            for col, kind in enumerate(labels):
                # Restore the exact post-action snapshot only for rendering; take no more actions.
                candidate.restore(post_snapshots[kind])
                metrics = selected_metrics[kind]
                frame = candidate.physics.render(
                    height=height,
                    width=width,
                    camera_id=camera_id,
                    scene_callback=lambda p, s: apply_camera(p, s, cameras),
                )
                canvas.paste(Image.fromarray(frame), (col * width, 96))
                draw.text((col * width + 16, 70), labels[kind], font=font(22), fill="#17212b")
                if include_image:
                    draw.text(
                        (col * width + 16, height + 100),
                        f"RGB RMSE [0,1] {metrics['image_rmse_01']:.5g} | velocity {metrics['qvel_rmse']:.5g}",
                        font=font(17),
                        fill="#17212b",
                    )
                    draw.text(
                        (col * width + 16, height + 123),
                        f"Proprio diagnostic RMSE {metrics['observation_rmse']:.5g}",
                        font=font(15),
                        fill="#465565",
                    )
                else:
                    draw.text(
                        (col * width + 16, height + 105),
                        f"obs RMSE {metrics['observation_rmse']:.5g} | velocity RMSE {metrics['qvel_rmse']:.5g}",
                        font=font(18),
                        fill="#17212b",
                    )
            filename = f"{meta['entry']['task']}_all_seeds_step1.png"
            panels.append((entry, canvas.copy()))
            results.append(
                {
                    "entry": entry,
                    "episode_id": episode,
                    "episode_seed": shard["episode_seed"],
                    "start": start,
                    "depth": 1,
                    "policy_action": actions[0].tolist(),
                    "action_repeat": reference.repeat,
                    "camera_id": camera_id,
                    "shared_camera": cameras,
                    "framing": "shared bounds of all branches" if frame_all else "reference camera",
                    "metrics": selected_metrics,
                    "reference_check": check,
                    "image": filename,
                    "matches_stage4_metrics": True,
                }
            )
            print(f"Rendered {entry}: episode {episode}, {start}->{start+1}; stage-4 metrics matched", flush=True)
        finally:
            reference.close()
            candidate.close()
            del reference, candidate
            gc.collect()
    image_names = []
    tasks = list(dict.fromkeys(result["entry"].rsplit("_s", 1)[0] for result in results))
    for task in tasks:
        selected = [im for filename, im in panels if filename.startswith(task + "_")]
        overview = Image.new("RGB", (columns * width, len(selected) * (height + 144)), "white")
        for index, im in enumerate(selected):
            overview.paste(im, (0, index * (height + 144)))
        filename = f"{task}_all_seeds_step1.png"
        overview.save(root / filename)
        image_names.append(filename)
    write_json(
        root / "manifest.json",
        {
            "checkpoint_group": protocol["checkpoint_group"],
            "stage4": str(stage4),
            "selection": "First saved held-out episode, preselected time index; not selected by error",
            "depth": 1,
            "results": results,
            "provenance": provenance(),
            "render_script_sha256": sha256(__file__),
            "egl_device_id": os.environ["MUJOCO_EGL_DEVICE_ID"],
        },
    )
    lines = [
        "# 第一步接续图片",
        "",
        f"仅 {checkpoint_group}；从左到右：" + " / ".join(labels.values()) + "。",
        "图下注明的obs RMSE为proprio诊断误差；vision图像RMSE（像素/255）另存于manifest，不与proprio混合。",
        "",
        (
            f"全部模型使用测试 episode {common_episode}，恢复第 {start} 步状态后，执行同一条已保存动作，仅拍摄第"
            f" {start+1} 步。"
        ),
        "每个策略动作包含2个底层环境步。各栏使用完全相同的相机，各分支不独立跟随或重新居中。",
        (
            "统一取景模式会调整共享相机的位置和距离，覆盖三个状态的身体范围；物理状态坐标未对齐或修改。"
            if frame_all
            else "使用完整快照参考在该帧的原相机视角。"
        ),
        "数值已与第四步存档逐项核对。单帧图片不能显示速度和触觉/IMU等全部状态，也不代表整体平均误差。",
        "",
    ]
    for filename in image_names:
        lines += [f"![{filename}]({filename})", ""]
    (root / "README.md").write_text("\n".join(lines))
    (root / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Step 1'
        " restoration</title><style>body{margin:24px;background:#edf0f4;font-family:sans-serif}"
        "img{width:100%;max-width:1920px;margin:12px"
        " 0}</style><h1>Step 1: " + " / ".join(labels.values()) + "</h1><p>Same recorded action and fixed camera. No"
        " later frames.</p>" + "".join(f'<img src="{name}" alt="{name}">' for name in image_names)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage4", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--frame-all", action="store_true", help="Use one shared camera enclosing all branch bodies")
    parser.add_argument("--checkpoint-group", default="VE_wasser_symlog_f64_H15")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    render_campaign(
        args.stage4, args.output, args.start, frame_all=args.frame_all, checkpoint_group=args.checkpoint_group
    )


if __name__ == "__main__":
    main()
