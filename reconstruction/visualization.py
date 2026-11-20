"""Periodic paired physics/native-image diagnostics for online DMC vision runs."""

import hashlib
import html
import json
from pathlib import Path

import numpy as np
import torch


class ReconstructionVisualizer:
    @classmethod
    def create(cls, config, agent, envs, logdir):
        if not config.get("enabled", False) or not str(config.get("task", "")).startswith("dmc_"):
            return None
        # Proprio still carries RGB observations, so inspect what the encoder
        # actually consumes, rather than merely checking for an image field.
        if "image" not in agent.encoder.cnn_shapes:
            return None
        if agent.rep_loss != "dreamer" or "image" not in agent.decoder.cnn_shapes:
            print("Reconstruction visualization skipped: no native image decoder.", flush=True)
            return None
        if agent.physics_head is None or not envs.physics_enabled:
            raise ValueError("DMC vision reconstruction visualization requires an enabled online physics decoder")
        return cls(config, agent, envs, logdir)

    def __init__(self, config, agent, envs, logdir):
        self.agent, self.envs = agent, envs
        self.root = Path(logdir) / "reconstruction_visualization"
        self.task, self.seed = str(config["task"]), int(config.get("seed", 0))
        self.every, self.origins = int(config.get("every", 100000)), int(config.get("origins", 5))
        requested = list(config.get("depths", [8, 10, 12, 14]))
        self.size = tuple(config.get("size", [320, 320]))
        if self.every < 1 or self.origins < 1 or len(self.size) != 2 or min(self.size) < 1:
            raise ValueError("Invalid reconstruction visualization interval/origins/size")
        if not requested or any(isinstance(d, bool) or not isinstance(d, int) or d < 0 for d in requested):
            raise ValueError("Visualization depths must be nonnegative integers")
        # Retain available requested depths for short imagination horizons.
        self.depths = sorted({d for d in requested if d <= agent.imag_horizon}) or [agent.imag_horizon]
        self.next_step = self.every
        print(
            f"DMC vision reconstruction visualization: every {self.every} Step, "
            f"depths={self.depths}, {self.origins} experiments per sheet, output={self.root}",
            flush=True,
        )

    def maybe_write(self, step, trans, state):
        if step < self.next_step:
            return False
        eligible = ~trans["is_last"].reshape(-1)
        if "is_recon" in trans:
            eligible = eligible & ~trans["is_recon"].reshape(-1)
        available = eligible.nonzero().flatten()[: self.origins].tolist()
        if not available:
            return False
        # Always run the requested number of imagined trials. With few ordinary
        # environments, reuse starting latents but sample each batch row anew;
        # never pad the sheet by copying an already generated image.
        slots = [available[i % len(available)] for i in range(self.origins)]
        milestone = int(step // self.every * self.every)
        self.next_step = milestone + self.every
        directory = self.root / f"step_{milestone:09d}"
        if directory.exists():
            print(f"Reconstruction visualization already exists: {directory}", flush=True)
            return False
        try:
            directory.mkdir(parents=True)
            packet = self.agent.reconstruction_visualization(state[slots], self.depths, self.seed)
            cores, actions = packet["predicted_core"].numpy(), packet["actions"].numpy()
            rows = self.envs.render_reconstructions(slots, cores, actions, self.depths, self.size)
            packet["physical_images"] = torch.from_numpy(np.stack([row["images"] for row in rows]))
            packet.update(depths=self.depths, source_slots=slots)
            torch.save(packet, directory / "samples.pt")
            metadata = dict(
                kind="paired_physics_vs_native_image",
                task=self.task,
                seed=self.seed,
                milestone=milestone,
                actual_env_step=int(step),
                model_updates=int(self.agent.reconstruction_updates),
                depths=self.depths,
                experiments=len(slots),
                source_slots=slots,
                source_selection="first ordinary nonterminal slots, cycled if fewer than experiments",
                rows=[{k: v for k, v in row.items() if k != "images"} for row in rows],
                same_features=True,
                shared_features_sha256=hashlib.sha256(packet["features"].numpy().tobytes()).hexdigest(),
                actions_executed=0,
                penetration_guard_applied=False,
                render_size=list(self.size),
                native_pixel_shape=list(packet["native_images"].shape[-3:]),
                live_state_unchanged=all(row["live_state_unchanged"] for row in rows),
                live_model_unchanged=all(row["live_model_unchanged"] for row in rows),
                training_rng_preserved=True,
            )
            self._write_sheet(directory, packet, metadata)
            (directory / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
            self._write_index()
            print(f"Reconstruction visualization: {directory / 'comparison.png'}", flush=True)
            return True
        except Exception as error:
            print(f"Reconstruction visualization failed at Step {step}: {type(error).__name__}: {error}", flush=True)
            try:
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "error.json").write_text(
                    json.dumps(dict(step=int(step), error=repr(error)), indent=2) + "\n"
                )
            except OSError:
                pass
            return False

    def _write_sheet(self, directory, packet, metadata):
        from PIL import Image, ImageDraw, ImageFont

        def font(size):
            try:
                return ImageFont.truetype("DejaVuSans.ttf", size)
            except OSError:
                return ImageFont.load_default()

        height, width = self.size
        header, row_height, pair_width = 110, height + 90, 2 * width + 16
        canvas = Image.new(
            "RGB", (len(self.depths) * pair_width, header + len(metadata["rows"]) * row_height), "#eef2f6"
        )
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (12, 8),
            f"{self.task} | seed {self.seed} | Step {metadata['actual_env_step']}",
            fill="#17212b",
            font=font(23),
        )
        draw.text(
            (12, 42),
            "LEFT: feature -> physics -> simulator | RIGHT: native image decoder",
            fill="#17212b",
            font=font(20),
        )
        draw.text(
            (12, 73),
            "Same imagined feature; no real action. Native pixels enlarged with nearest neighbor.",
            fill="#465565",
            font=font(17),
        )
        for row, (slot, details) in enumerate(zip(metadata["source_slots"], metadata["rows"])):
            y = header + row * row_height
            for col, depth in enumerate(self.depths):
                x = col * pair_width
                source_step = details["source"]["policy_step"]
                draw.text(
                    (x + 6, y + 3),
                    f"Trial {row + 1} | Slot {slot} | origin {source_step} | IMAG +{depth}",
                    fill="#17212b",
                    font=font(18),
                )
                draw.text((x + 6, y + 28), "Physical-state reconstruction", fill="#125c8a", font=font(15))
                draw.text((x + width + 14, y + 28), "Native image decoder", fill="#9a4311", font=font(15))
                physical = Image.fromarray(packet["physical_images"][row, col].numpy())
                native = Image.fromarray(packet["native_images"][row, col].numpy())
                native = native.resize((width, height), Image.Resampling.NEAREST)
                failure = details["details"][col]["failure"]
                if failure:
                    ImageDraw.Draw(physical).text((5, 5), "RENDER FAILED", fill="#ff6b6b", font=font(18))
                canvas.paste(physical, (x, y + 53))
                canvas.paste(native, (x + width + 8, y + 53))
                note = failure or f"camera {details['camera_id']} | no action | shared feature"
                draw.text((x + 6, y + height + 60), note[:100], fill="#a62929" if failure else "#465565", font=font(13))
        canvas.save(directory / "comparison.png")
        (directory / "README.md").write_text(
            f"# {self.task} · Step {metadata['actual_env_step']}\n\n"
            "每对左侧为想象 feature → 在线物理 decoder → 仿真截图，右侧为同一 feature → 原生 image decoder。\n\n"
            f"想象深度：{self.depths}；实验次数：{metadata['experiments']}；环境槽：{metadata['source_slots']}。"
            "每次实验各占一行，属于同一模型 seed；优先使用不同普通环境起点，起点不足时循环使用，"
            "每行独立采样想象动作与潜在状态，不复制已有图片。不会按重建质量挑选样本，"
            "不要求 guard 放行；失败格会明确标记。"
            "仿真副本中只设置已知起点上下文和预测状态，不执行动作。\n\n"
            "这是训练中当前来源的可视化诊断，不是固定测试集的精度评估；左右都不是未来真实画面。"
            "原始共享 feature、动作链、预测核心状态及两路像素保存在 samples.pt，元信息见 manifest.json。\n\n"
            "![comparison](comparison.png)\n"
        )

    def _write_index(self):
        entries = []
        for path in sorted(self.root.glob("step_*/manifest.json")):
            info = json.loads(path.read_text())
            entries.append((info["actual_env_step"], path.parent.name))
        text = "# DMC vision 训练重建对比\n\n左：feature → 物理状态 → 仿真；右：原生 image decoder。\n\n"
        text += "\n\n".join(
            f"## Step {step}\n\n[详情]({name}/README.md)\n\n![Step {step}]({name}/comparison.png)"
            for step, name in entries
        )
        (self.root / "README.md").write_text(text + "\n")
        body = "".join(
            f'<h2>Step {step}</h2><a href="{name}/comparison.png"><img src="{name}/comparison.png"></a>'
            for step, name in entries
        )
        style = "body{font-family:sans-serif;background:#eef2f6}img{max-width:100%;margin-bottom:20px}"
        heading = f"<h1>{html.escape(self.task)} · seed {self.seed}</h1>"
        (self.root / "index.html").write_text(
            '<!doctype html><meta charset="utf-8"><title>Training reconstruction</title>'
            f"<style>{style}</style>{heading}"
            f"<p>Left: physics reconstruction; right: native image decoder.</p>{body}\n"
        )
