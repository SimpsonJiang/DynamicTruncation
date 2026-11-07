"""Redraw saved stage-4 examples with fixed task overview and constant-scale centered detail.

Run this file directly: simulation and decoder imports come from the evaluated
stage-4 source snapshot, not from potentially edited workspace model code.
"""

import argparse
import copy
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage4", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    stage4 = Path(args.stage4).resolve()
    protocol = json.loads((stage4 / "protocol.json").read_text())
    original_path = stage4 / "first_step_images/manifest_original.json"
    if not original_path.exists():
        original_path = stage4 / "first_step_images/manifest.json"
    original = json.loads(original_path.read_text())
    if original.get("depth") != 1:
        raise ValueError("Only the original first-step gallery is supported")
    if "physics_probe" in sys.modules:
        raise RuntimeError("Run this script directly, not with python -m, to use the archived implementation")
    snapshot = stage4 / "source_snapshot"
    for relative, expected in protocol["source_snapshot_sha256"].items():
        if hashlib.sha256((snapshot / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Archived implementation changed: {relative}")
    sys.path.insert(0, str(snapshot))
    import numpy as np
    import torch
    from PIL import Image, ImageDraw

    from physics_probe.common import load_config, new_directory, sha256, write_json
    from physics_probe.environment import ProbeEnv
    from physics_probe.pipeline import load_probe, load_shard
    from physics_probe.posterior_restore import (
        observe_record,
        step_metrics,
        trace_reference,
    )
    from physics_probe.render_first_step import LABELS, apply_camera, font

    if os.environ.get("MUJOCO_GL") != "egl" or os.environ.get("MUJOCO_EGL_DEVICE_ID") != "6":
        raise ValueError("Use EGL on physical GPU6 only")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = new_directory(args.output)
    source_root = Path(protocol["source_root"])
    width, height, gap, row_height, header = 640, 360, 26, 902, 86
    kinds = [*protocol["sources"], "core64_control"]
    if kinds != ["features", "observations", "core64_control"]:
        raise ValueError("This redraw is for the requested CQ proprio comparison")
    grouped, examples = {}, []

    def corners(low, high):
        return np.asarray(list(itertools.product(*zip(low, high))), dtype=np.float64)

    def basis(camera):
        forward = np.asarray(camera["forward"], dtype=np.float64)
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, camera["up"])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        return forward, right, up

    def fit_distance(camera, offsets):
        forward, right, up = basis(camera)
        tan_y = (camera["frustum_top"] - camera["frustum_bottom"]) / (2 * camera["frustum_near"])
        tan_x = tan_y * width / height
        depth = offsets @ forward
        distance = max(
            float((np.abs(offsets @ right) * 1.18 / tan_x - depth).max()),
            float((np.abs(offsets @ up) * 1.18 / tan_y - depth).max()),
            float((camera["frustum_near"] * 3 - depth).max()),
            0.3,
        )
        # Check the actual projected AABB corners, rather than an oversized bounding sphere.
        assert np.max(np.abs(offsets @ right) / ((distance + depth) * tan_x)) < 1
        assert np.max(np.abs(offsets @ up) / ((distance + depth) * tan_y)) < 1
        return distance

    def camera_at(template, center, distance):
        cameras = copy.deepcopy(template)
        for camera in cameras:
            forward, _, up = basis(camera)
            camera.update(
                pos=(center - forward * distance).tolist(),
                forward=forward.tolist(),
                up=up.tolist(),
                frustum_center=0.0,
                frustum_far=max(camera["frustum_far"], 10 * distance),
            )
        return cameras

    with torch.no_grad():
        for old in original["results"]:
            entry_id, start, episode = old["entry"], old["start"], old["episode_id"]
            base = source_root / entry_id
            predictions = {}
            for kind in protocol["sources"]:
                checkpoint = base / kind / "decoder.pt"
                for path in (checkpoint, base / "data/dataset.json"):
                    if sha256(path) != protocol["input_sha256"][str(path)]:
                        raise ValueError(f"Evaluation input changed: {path}")
                path, meta, saved, decoder = load_probe(base / "data", checkpoint, "cpu")
                if episode not in saved["split"]["test"] or decoder.dtype != torch.float64:
                    raise ValueError("Require the same held-out episode and double decoder")
                shard = load_shard(path, meta["episodes"][episode])
                predictions[kind] = decoder(shard[kind][start : start + 1]).cpu().numpy()[0]
            predictions["core64_control"] = shard["cores"][start].numpy()
            config = load_config(meta["entry"])
            reference, candidate = [ProbeEnv(config.env, shard["episode_seed"]) for _ in range(2)]
            try:
                records, actions, check = trace_reference(reference, shard, start, 1)
                if not check["passed"]:
                    raise ValueError("Original first-step reference did not reproduce")
                example = {
                    "entry": entry_id,
                    "task": meta["entry"]["task"],
                    "model_seed": meta["entry"]["model_seed"],
                    "episode_id": episode,
                    "episode_seed": shard["episode_seed"],
                    "start": start,
                    "depth": 1,
                    "camera_id": old["camera_id"],
                    "policy_action": actions[0].tolist(),
                    "config_entry": meta["entry"],
                    "template": old["shared_camera"],
                    "branches": {},
                }
                for kind in kinds:
                    candidate.restore(shard["snapshots"][start], predictions[kind])
                    obs, reward, _ = candidate.step(actions[0])
                    metrics = step_metrics(observe_record(candidate, obs, reward), records[1], meta["schema"])
                    for key, expected in old["metrics"][kind].items():
                        np.testing.assert_allclose(metrics[key], expected, rtol=1e-9, atol=1e-10)
                    moving = candidate.physics.model.geom_bodyid != 0
                    centers = candidate.physics.data.geom_xpos[moving].copy()
                    radii = candidate.physics.model.geom_rbound[moving, None]
                    low, high = (centers - radii).min(0), (centers + radii).max(0)
                    example["branches"][kind] = {
                        "snapshot": candidate.snapshot(),
                        "metrics": metrics,
                        "corners": corners(low, high),
                        "center": (low + high) / 2,
                    }
                grouped.setdefault(example["task"], []).append(example)
                examples.append(example)
                print(f"Verified original one-step metrics: {entry_id}", flush=True)
            finally:
                reference.close()
                candidate.close()

    task_cameras, manifest_results, filenames = {}, [], []
    for task, cases in grouped.items():
        cases.sort(key=lambda e: e["model_seed"])
        template = cases[0]["template"]
        for case in cases:
            np.testing.assert_allclose(case["template"][0]["forward"], template[0]["forward"], atol=1e-7)
            np.testing.assert_allclose(case["template"][0]["up"], template[0]["up"], atol=1e-7)
        branches = [case["branches"][kind] for case in cases for kind in kinds]
        all_corners = np.concatenate([b["corners"] for b in branches])
        overview_center = (all_corners.min(0) + all_corners.max(0)) / 2
        overview_distance = fit_distance(template[0], all_corners - overview_center)
        overview_camera = camera_at(template, overview_center, overview_distance)
        detail_distance = fit_distance(template[0], np.concatenate([b["corners"] - b["center"] for b in branches]))
        task_cameras[task] = {
            "overview_camera": overview_camera,
            "overview_center": overview_center.tolist(),
            "overview_distance": overview_distance,
            "detail_distance": detail_distance,
            "detail_policy": "only camera translation varies; same forward/up/FOV/distance across all states",
        }
        canvas = Image.new("RGB", (width * 3, header + row_height * len(cases)), "#eef2f6")
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 10), f"{original['checkpoint_group']} | {task}", font=font(25), fill="#17212b")
        draw.text(
            (16, 47),
            "OVERVIEW: one fixed world camera for all seeds | DETAIL: centered camera, same scale, no pose alignment",
            font=font(21),
            fill="#465565",
        )
        for row, case in enumerate(cases):
            y = header + row * row_height
            draw.text(
                (16, y + 5),
                f"{case['entry']} | episode {case['episode_id']} | "
                f"t={case['start']} -> {case['start']+1} | same recorded action",
                font=font(24),
                fill="#17212b",
            )
            draw.text(
                (16, y + 38),
                f"OVERVIEW  (fixed across all 9 states; camera distance {overview_distance:.2f})",
                font=font(19),
                fill="#42566d",
            )
            config = load_config(case["config_entry"])
            env = ProbeEnv(config.env, case["episode_seed"])
            detail_cameras = {}
            try:
                for col, kind in enumerate(kinds):
                    branch = case["branches"][kind]
                    env.restore(branch["snapshot"])
                    draw.text((col * width + 14, y + 66), LABELS[kind], font=font(21), fill="#17212b")
                    before = env.integration_state().copy()
                    frame = env.physics.render(
                        height=height,
                        width=width,
                        camera_id=case["camera_id"],
                        scene_callback=lambda p, s: apply_camera(p, s, overview_camera),
                    )
                    canvas.paste(Image.fromarray(frame), (col * width, y + 96))
                    camera = camera_at(template, branch["center"], detail_distance)
                    detail_cameras[kind] = camera
                    frame = env.physics.render(
                        height=height,
                        width=width,
                        camera_id=case["camera_id"],
                        scene_callback=lambda p, s: apply_camera(p, s, camera),
                    )
                    canvas.paste(Image.fromarray(frame), (col * width, y + 96 + height + gap))
                    np.testing.assert_array_equal(env.integration_state(), before)
                    metrics = branch["metrics"]
                    draw.text(
                        (col * width + 14, y + 850),
                        f"obs RMSE {metrics['observation_rmse']:.5g} | velocity RMSE {metrics['qvel_rmse']:.5g}",
                        font=font(18),
                        fill="#17212b",
                    )
                    center = branch["center"]
                    draw.text(
                        (col * width + 14, y + 876),
                        f"Body bounds center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})",
                        font=font(17),
                        fill="#465565",
                    )
                draw.text(
                    (14, y + 96 + height + 3),
                    "DETAIL  (camera centers separately; "
                    f"same direction and distance {detail_distance:.2f}; states are not aligned)",
                    font=font(18),
                    fill="#245e78",
                )
            finally:
                env.close()
            manifest_results.append(
                {
                    k: case[k]
                    for k in (
                        "entry",
                        "task",
                        "model_seed",
                        "episode_id",
                        "episode_seed",
                        "start",
                        "depth",
                        "camera_id",
                        "policy_action",
                    )
                }
                | {
                    "overview_camera": overview_camera,
                    "detail_cameras": detail_cameras,
                    "metrics": {k: b["metrics"] for k, b in case["branches"].items()},
                    "metrics_match_original": True,
                    "render_preserves_integration_state": True,
                }
            )
        filename = f"{task}_all_seeds_step1.png"
        canvas.save(root / filename)
        filenames.append(filename)
        print(f"Saved {filename}", flush=True)
    if len(filenames) != 3:
        raise ValueError("Expected exactly three task sheets")
    write_json(
        root / "manifest.json",
        {
            "checkpoint_group": original["checkpoint_group"],
            "stage4": str(stage4),
            "depth": 1,
            "layout": "fixed_task_overview_and_constant_scale_centered_detail",
            "original_manifest_sha256": sha256(original_path),
            "task_cameras": task_cameras,
            "results": manifest_results,
            "render_script_sha256": sha256(__file__),
            "simulation_source": str(snapshot),
            "notes": [
                "Only cameras/layout changed; physical states and metrics unchanged.",
                "Overview uses exactly the same camera across seeds and branches within each task.",
                "Detail changes camera translation only; no physical position/orientation alignment.",
            ],
        },
    )
    (root / "README.md").write_text(
        "# 第一步接续：统一概览与居中近景\n\n仅重绘CQ步骤④的3张汇总图。\n\n"
        "每行对应一个模型seed，三栏依次为feature decoder、observation decoder、真实核心状态恢复。"
        "所有图片仍为测试episode60、第100步恢复后，执行同一动作得到的第101步。\n\n"
        "每个模型上半部是概览：同一任务的全部seed和分支共用同一台固定相机，保留绝对位置差异。"
        "下半部是近景：分别平移相机使身体居中，但保持方向、距离、视场和尺度相同；没有旋转或对齐身体姿态。"
        "近景不能用来比较绝对位置，请结合概览和图下坐标。\n\n"
        "所有数值已与原图逐项核对；渲染前后integration state保持完全一致。未重新训练或更改实验误差。\n\n"
        + "\n\n".join(f"![{name}]({name})" for name in filenames)
        + "\n"
    )
    (root / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Overview and centered detail</title>'
        "<style>body{font-family:sans-serif;background:#eef2f6}img{width:100%;margin:20px 0}</style>"
        "<h1>Fixed overview + constant-scale centered detail</h1>"
        "<p>Overview preserves world positions. Detail translates cameras only; no pose alignment.</p>"
        + "".join(f'<img src="{name}" alt="{name}">' for name in filenames)
    )


if __name__ == "__main__":
    main()
