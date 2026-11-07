"""Display native decoder outputs; map proprio through its trained inverse only for static rendering."""

import argparse
import copy
import json
import os
import shutil
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoded", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    decoded_root, output = Path(args.decoded).resolve(), Path(args.output).resolve()
    decoded_manifest = json.loads((decoded_root / "native_manifest.json").read_text())
    if output.exists():
        raise FileExistsError(output)
    if "physics_probe" in sys.modules:
        raise RuntimeError("Run this script directly to use the archived implementation")
    sys.path.insert(0, decoded_manifest["snapshot"])
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
    if sha256(decoded_root / "native_observations.pt") != decoded_manifest["data_sha256"]:
        raise ValueError("Native decoder outputs changed")
    records = torch.load(decoded_root / "native_observations.pt", map_location="cpu", weights_only=True)
    output.mkdir(parents=True)
    frames, audits, mapped_states = defaultdict(list), [], {}
    with torch.no_grad(), patch.object(ProbeEnv, "step", side_effect=RuntimeError("No dynamics steps allowed")):
        for key, record in records.items():
            job, depths = record["job"], record["depths"]
            entry = record["entry"]
            vision = record["policy_modality"] == "vision"
            cell_frames, details = [], []
            if vision:
                raw = record["observations"]["image"]
                for index, depth in enumerate(depths):
                    pixels = raw[index].clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
                    cell_frames.append(Image.fromarray(pixels).resize((320, 320), Image.Resampling.NEAREST))
                    details.append(
                        {
                            "imag_depth": depth,
                            "absolute_step": record["start"] + depth,
                            "path": "saved feature -> native image decoder -> RGB",
                            "pixel_min": float(raw[index].min()),
                            "pixel_max": float(raw[index].max()),
                            "clipped_fraction": float(((raw[index] < 0) | (raw[index] > 1)).float().mean()),
                            "simulator_rendered": False,
                            "actions_executed": 0,
                        }
                    )
            else:
                base = Path(job["source"]) / job["entry"]
                checkpoint = base / "observations/decoder.pt"
                path, meta, saved, inverse = load_probe(base / "data", checkpoint, "cpu")
                if meta["entry"] != entry or saved["source"] != "observations" or inverse.dtype != torch.float64:
                    raise ValueError("Wrong observation-to-core mapping")
                expected = json.loads((base / "test_observations.json").read_text())["checkpoint_sha256"]
                if sha256(checkpoint) != expected:
                    raise ValueError("Observation-to-core mapping changed since its held-out test")
                keys = meta["obs_keys"]
                if set(keys) != set(record["observations"]):
                    raise ValueError("Native proprio fields do not match the trained inverse input")
                vector = torch.cat([record["observations"][name].reshape(len(depths), -1) for name in keys], dim=-1)
                if vector.shape[-1] != inverse.input_dim:
                    raise ValueError("Proprio input dimensions differ")
                cores = inverse.requires_grad_(False)(vector)
                if cores.dtype != torch.float64:
                    raise ValueError("Preserve float64 core reconstruction")
                shard = load_shard(path, meta["episodes"][record["episode_id"]])
                initial = shard["snapshots"][record["start"]]
                config = load_config(entry)
                env = ProbeEnv(config.env, record["episode_seed"])
                restored_states = []
                try:
                    for index, depth in enumerate(depths):
                        metadata = {
                            "imag_depth": depth,
                            "absolute_step": record["start"] + depth,
                            "path": "saved feature -> native proprio decoder -> "
                            "trained observation-to-core -> static render",
                            "mapping_sha256": expected,
                            "simulator_rendered": False,
                            "actions_executed": 0,
                            "core_dtype": "float64",
                            "known_context": "initial snapshot plus analytical elapsed time "
                            "and previous imagined control",
                        }
                        try:
                            env.restore(copy.deepcopy(initial), cores[index].numpy())
                            base_time = float(env.physics.data.time)
                            # Set known bookkeeping/control analytically. Never replay the imagined action sequence.
                            env.physics.data.time = base_time + depth * env.repeat * env.env.control_timestep()
                            previous_action = record["actions"][depth - 1].double().numpy()
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
                            cell_frames.append(Image.fromarray(pixels))
                            metadata.update(
                                simulator_rendered=True,
                                camera_id=env.camera_id,
                                camera=camera,
                                state_unchanged_by_render=True,
                                simulation_time=float(env.physics.data.time),
                                failure=None,
                            )
                            restored_states.append(env.snapshot())
                        except (ValueError, RuntimeError, ArithmeticError) as exc:
                            cell_frames.append(None)
                            metadata["failure"] = f"{type(exc).__name__}: {exc}"
                            restored_states.append(None)
                        details.append(metadata)
                finally:
                    env.close()
                mapped_states[key] = {
                    "proprio_keys": keys,
                    "native_proprio": vector,
                    "predicted_core": cores,
                    "restored_states": restored_states,
                    "mapping_sha256": expected,
                }
                del inverse, shard, cores
            frames[(job["group"], entry["task"])].append((record, cell_frames))
            audits.append(
                {
                    "key": key,
                    "entry": entry,
                    "modality": record["policy_modality"],
                    "episode_id": record["episode_id"],
                    "start": record["start"],
                    "details": details,
                }
            )
            print(f"Prepared native reconstruction {key}; no action or reference replay", flush=True)
            gc.collect()

    names = []
    for (group, task), entries in frames.items():
        entries.sort(key=lambda item: item[0]["entry"]["model_seed"])
        width, cell_height, header = 320, 404, 90
        canvas = Image.new("RGB", (4 * width, header + len(entries) * cell_height), "#eef2f6")
        draw = ImageDraw.Draw(canvas)
        vision = entries[0][0]["policy_modality"] == "vision"
        draw.text((12, 8), f"{group} | {task}", font=font(21), fill="#17212b")
        path_label = (
            "Native image decoder (64x64, nearest-neighbor display)"
            if vision
            else "Native proprio decoder -> observation-to-core -> static simulator render"
        )
        draw.text((12, 38), path_label, font=font(17), fill="#465565")
        draw.text(
            (12, 62),
            "DECODED STATE AT THE LABELLED IMAGINED STEP | NO ACTION EXECUTED | NO REAL-ROLLOUT REFERENCE",
            font=font(15),
            fill="#245e78",
        )
        for row, (record, cells) in enumerate(entries):
            y = header + row * cell_height
            draw.text(
                (12, y + 5),
                f"{record['entry']['id']} | episode {record['episode_id']} | posterior start t={record['start']}",
                font=font(18),
                fill="#17212b",
            )
            for col, (depth, cell) in enumerate(zip(record["depths"], cells)):
                draw.text(
                    (col * width + 10, y + 33),
                    f"IMAG +{depth} | decoded t={record['start']+depth}",
                    font=font(17),
                    fill="#17212b",
                )
                if cell is None:
                    draw.text(
                        (col * width + 10, y + 145), "Core mapping / render failed", font=font(17), fill="#b12626"
                    )
                else:
                    canvas.paste(cell, (col * width, y + 61))
                draw.text(
                    (col * width + 10, y + 384),
                    "native decoder output" if vision else "two-decoder chain; not a true state",
                    font=font(14),
                    fill="#465565",
                )
        filename = f"{group}__{task}__native_decoder_imag8_10_12_14.png"
        canvas.save(output / filename)
        names.append(filename)
    if len(names) != 10:
        raise ValueError("Expected ten group/task sheets")
    torch.save(mapped_states, output / "proprio_mapping_states.pt")
    write_json(
        output / "manifest.json",
        {
            "kind": "native_decoder_reconstruction",
            "stage5": decoded_manifest["stage5"],
            "native_export_sha256": sha256(decoded_root / "native_observations.pt"),
            "script_sha256": sha256(__file__),
            "models": audits,
            "images": names,
            "actions_executed": 0,
            "imagination_regenerated": False,
            "reference_rollout_replayed": False,
            "numeric_stage5_report_recomputed": False,
            "note": "Old stage5 CSV measures feature-to-core plus one action. "
            "These images use native observation decoder at d, with no action.",
        },
    )
    (output / "README.md").write_text(
        "# 原生 observation decoder 的想象重建\n\n"
        "每列依次显示想象第8、10、12、14步的解码结果，每行是一个模型seed。"
        "复用步骤⑤保存的同一批feature，没有重新想象、没有重放真实动作，也没有执行接续动作或放入真实参考列。"
        "示例仍是测试episode60、posterior起点100步，所以图示分别对应108/110/112/114步。\n\n"
        "**Vision：** feature拆回原始stoch/deter，直接经checkpoint内的Dreamer图像decoder输出RGB。"
        "原图分辨率64×64，最近邻放大，不经过物理状态decoder或仿真器渲染。\n\n"
        "**Proprio：** feature→Dreamer原生proprio decoder→已训练的observation→核心状态decoder→恢复后立即截图。"
        "原生观测与完整物理状态不同，图中包含第二个decoder的误差；不代表唯一正确的物理状态。"
        "仅按已知初始时间和上一想象动作设置辅助上下文，调用forward计算几何与相机，不做动力学步进。"
        "使用任务原生相机，未对齐身体姿态。\n\n"
        "**数值报告的范围：** 原步骤⑤REPORT.md/summary.csv仍是旧的feature→核心状态decoder及一步接续诊断，"
        "未修改，不能把其中的误差当作这组新图的误差。原生输出保存在native_observations.pt，proprio映射保存在proprio_mapping_states.pt。\n\n"
        + "\n\n".join(f"![{name}]({name})" for name in sorted(names))
        + "\n"
    )
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Native observation decoder</title>'
        "<style>body{font-family:sans-serif;background:#eef2f6}img{width:100%;max-width:1280px;margin:16px 0}</style>"
        "<h1>Native observation decoder: imag +8 / +10 / +12 / +14</h1>"
        "<p>No post-decoding action; no real-rollout reference.</p>"
        + "".join(f'<img src="{name}" alt="{name}">' for name in sorted(names))
    )
    for name in ("native_observations.pt", "native_manifest.json"):
        shutil.copy2(decoded_root / name, output / name)
    shutil.copy2(__file__, output / "render_native_observations.py")
    exporter = Path(__file__).with_name("export_native_observations.py")
    if sha256(exporter) != decoded_manifest["script_sha256"]:
        raise ValueError("Exporter source changed since native outputs were created")
    shutil.copy2(exporter, output / exporter.name)


if __name__ == "__main__":
    main()
