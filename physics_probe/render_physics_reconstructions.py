"""Render saved imagined feature-to-core predictions without advancing the simulator."""

import argparse
import copy
import json
import os
import shutil
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-gallery", required=True, help="Paired origins and archived runtime provenance")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    native, output = Path(args.native_gallery).resolve(), Path(args.output).resolve()
    manifest = json.loads((native / "native_manifest.json").read_text())
    stage5 = Path(manifest["stage5"])
    if output.exists():
        raise FileExistsError(output)
    if "physics_probe" in sys.modules:
        raise RuntimeError("Run this script directly to use the archived implementation")
    sys.path.insert(0, manifest["snapshot"])

    import gc
    from collections import defaultdict
    from unittest.mock import patch

    import numpy as np
    import torch
    from PIL import Image, ImageDraw

    from physics_probe.common import load_config, sha256, write_json
    from physics_probe.environment import ProbeEnv
    from physics_probe.pipeline import load_probe, load_shard
    from physics_probe.render_first_step import capture_camera, font

    if os.environ.get("MUJOCO_GL") != "egl" or os.environ.get("MUJOCO_EGL_DEVICE_ID") != "6":
        raise ValueError("Use EGL on physical GPU6 only")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if sha256(native / "native_observations.pt") != manifest["data_sha256"]:
        raise ValueError("Paired origins changed")
    records = torch.load(native / "native_observations.pt", map_location="cpu", weights_only=True)
    output.mkdir(parents=True)
    frames, audits, states = defaultdict(list), [], {}
    with torch.no_grad(), patch.object(ProbeEnv, "step", side_effect=RuntimeError("No dynamics steps allowed")):
        for key, record in records.items():
            job, entry, depths = record["job"], record["entry"], record["depths"]
            trace_path = stage5 / key / "traces.pt"
            if sha256(trace_path) != record["feature_trace_sha256"]:
                raise ValueError(f"Trace changed: {key}")
            traces = torch.load(trace_path, map_location="cpu", weights_only=True)
            matches = [t for t in traces if t["episode_id"] == record["episode_id"] and t["start"] == record["start"]]
            if len(matches) != 1:
                raise ValueError(f"Ambiguous origin: {key}")
            trace = matches[0]
            features, cores = trace["features"][depths], trace["decoded"][depths]
            torch.testing.assert_close(features, record["features"], rtol=0, atol=0)
            torch.testing.assert_close(trace["actions"], record["actions"], rtol=0, atol=0)
            if cores.dtype != torch.float64 or not torch.isfinite(cores).all():
                raise ValueError("Expected finite float64 direct feature-to-core predictions")
            report = json.loads((stage5 / key / "report.json").read_text())
            base = Path(job["source"]) / job["entry"]
            checkpoint = base / "features/decoder.pt"
            if sha256(checkpoint) != report["decoder_sha256"]:
                raise ValueError(f"Physical-state decoder changed: {key}")
            path, meta, saved, decoder = load_probe(base / "data", checkpoint, "cpu")
            if meta["entry"] != entry or saved["source"] != "features" or decoder.dtype != torch.float64:
                raise ValueError("Expected the original direct feature-to-core decoder")
            recomputed = decoder.requires_grad_(False)(features)
            torch.testing.assert_close(recomputed, cores, rtol=1e-8, atol=1e-8)
            decoder_difference = float((recomputed - cores).abs().max())
            shard = load_shard(path, meta["episodes"][record["episode_id"]])
            initial = shard["snapshots"][record["start"]]
            config = load_config(entry)
            env = ProbeEnv(config.env, record["episode_seed"])
            cells, details, restored = [], [], []
            try:
                for index, depth in enumerate(depths):
                    env.restore(copy.deepcopy(initial), cores[index].numpy())
                    env.physics.data.time += depth * env.repeat * env.env.control_timestep()
                    previous_action = trace["actions"][depth - 1].double().numpy()
                    env.physics.data.ctrl[:] = (previous_action + 1) / 2 * (env.high - env.low) + env.low
                    env.steps = initial["steps"] + depth
                    env.env._step_count = initial["inner_steps"] + depth * env.repeat
                    env.physics.forward()
                    before = env.integration_state().copy()
                    camera = []
                    pixels = env.physics.render(
                        height=320,
                        width=320,
                        camera_id=env.camera_id,
                        scene_callback=lambda p, s: capture_camera(p, s, camera),
                    )
                    np.testing.assert_array_equal(env.integration_state(), before)
                    np.testing.assert_allclose(env.core_state(), cores[index].numpy(), atol=1e-12, rtol=1e-12)
                    cells.append(Image.fromarray(pixels))
                    restored.append(env.snapshot())
                    details.append(
                        dict(
                            imag_depth=depth,
                            absolute_step=record["start"] + depth,
                            camera_id=env.camera_id,
                            camera=camera,
                            state_unchanged_by_render=True,
                            actions_executed=0,
                            simulation_time=float(env.physics.data.time),
                        )
                    )
            finally:
                env.close()
            states[key] = dict(features=features, predicted_core=cores, restored_states=restored, depths=depths)
            audits.append(
                dict(
                    key=key,
                    entry=entry,
                    episode_id=record["episode_id"],
                    start=record["start"],
                    decoder_sha256=report["decoder_sha256"],
                    trace_sha256=record["feature_trace_sha256"],
                    cpu_decoder_recheck_max_abs=decoder_difference,
                    details=details,
                )
            )
            frames[(job["group"], entry["task"])].append((record, cells))
            print(f"Rendered direct feature -> physical state -> simulator: {key}", flush=True)
            del decoder, shard, traces, trace, matches, recomputed
            gc.collect()

    names = []
    for (group, task), entries in frames.items():
        entries.sort(key=lambda item: item[0]["entry"]["model_seed"])
        width, row_height, header = 320, 404, 90
        canvas = Image.new("RGB", (4 * width, header + len(entries) * row_height), "#eef2f6")
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 8), f"{group} | {task}", font=font(21), fill="#17212b")
        draw.text(
            (12, 38),
            "Imagined feature -> trained physical-state decoder -> simulator render",
            font=font(19),
            fill="#465565",
        )
        draw.text(
            (12, 64), "DIRECT CORE RECONSTRUCTION | RESTORE AT IMAG +d | NO EXTRA ACTION", font=font(16), fill="#245e78"
        )
        for row, (record, cells) in enumerate(entries):
            y = header + row * row_height
            draw.text(
                (12, y + 5),
                f"{record['entry']['id']} | episode {record['episode_id']} | posterior start t={record['start']}",
                font=font(18),
                fill="#17212b",
            )
            for col, (depth, cell) in enumerate(zip(record["depths"], cells)):
                draw.text(
                    (col * width + 10, y + 33),
                    f"IMAG +{depth} | restored t={record['start'] + depth}",
                    font=font(17),
                    fill="#17212b",
                )
                canvas.paste(cell, (col * width, y + 61))
                draw.text(
                    (col * width + 10, y + 384), "feature -> qpos/qvel/act -> render", font=font(14), fill="#465565"
                )
        name = f"{group}__{task}__physics_imag8_10_12_14.png"
        canvas.save(output / name)
        names.append(name)
    if len(names) != 10 or len(audits) != 30:
        raise ValueError("Expected 30 models in ten group/task sheets")
    torch.save(states, output / "physical_states.pt")
    write_json(
        output / "manifest.json",
        dict(
            kind="direct_imagined_feature_to_physics_reconstruction",
            stage5=str(stage5),
            snapshot=manifest["snapshot"],
            path="imagined feature -> trained feature-to-core decoder -> simulator render",
            script_sha256=sha256(__file__),
            states_sha256=sha256(output / "physical_states.pt"),
            core_dtype="float64",
            models=audits,
            images=names,
            actions_executed=0,
            imagination_regenerated=False,
            reference_rollout_replayed=False,
            known_context="initial snapshot plus analytical elapsed time and previous imagined control",
            paired_native_gallery=str(native),
        ),
    )
    (output / "README.md").write_text(
        "# 主线：想象 feature → 物理状态 → 仿真重建\n\n"
        "这里是直接feature→物理核心状态decoder的重建图。Vision和proprio模型均使用这条主线，"
        "解码qpos/qvel/act（float64）后恢复仿真并立即截图。\n\n"
        "每张图三行对应三个模型seed，四列对应**想象第8、10、12、14步**。"
        "同一测试episode60、posterior起点100；图示状态分别位于108/110/112/114步。"
        "复用原实验保存的feature和直接解码的核心状态，并重新核对decoder输出。"
        "没有重新想象、没有重放真实动作序列、没有执行恢复后的额外动作。辅助时间/control由已知上下文设置。\n\n"
        "[另一条展示：原生 image/proprio decoder](../native_decoder_images/README.md)。"
        "两组使用逐值相同的feature和相同起点，分开保留。"
        "原步骤⑤REPORT/CSV仍衡量恢复后接续一步的误差；这些静态图显示接续前的状态。\n\n"
        + "\n\n".join(f"![{name}]({name})" for name in sorted(names))
        + "\n"
    )
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Feature to physics reconstruction</title>'
        "<style>body{font-family:sans-serif;background:#eef2f6}img{width:100%;max-width:1280px;margin:16px 0}</style>"
        "<h1>Feature → physical state → simulator</h1><p>Imagined +8 / +10 / +12 / +14; no extra action.</p>"
        '<p><a href="../native_decoder_images/index.html">Native observation decoder gallery</a></p>'
        + "".join(f'<img src="{name}" alt="{name}">' for name in sorted(names))
    )
    shutil.copy2(__file__, output / Path(__file__).name)


if __name__ == "__main__":
    main()
