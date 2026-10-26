"""Record policy rollouts from a completed training run without starting training."""

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from omegaconf import OmegaConf

import tools
from dreamer import Dreamer
from envs import make_env


def positive_float(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return value


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logdir", type=Path, required=True, help="Training run directory containing latest.pt.")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("latest.pt"), help="Absolute path or path under logdir."
    )
    parser.add_argument("--config", type=Path, help="Defaults to <logdir>/.hydra/config.yaml.")
    parser.add_argument("--output-dir", type=Path, help="Defaults to <logdir>/eval_videos.")
    parser.add_argument("--seconds", type=positive_float, default=30.0, help="Duration of EACH video (default: 30).")
    parser.add_argument("--num-videos", type=positive_int, default=3, help="Number of videos to generate (default: 3).")
    parser.add_argument(
        "--fps", type=positive_float, help="DMC defaults to its control rate / action repeat; others: 30."
    )
    parser.add_argument("--width", type=positive_int, default=512, help="DMC video width, independent of policy input.")
    parser.add_argument(
        "--height", type=positive_int, default=512, help="DMC video height, independent of policy input."
    )
    parser.add_argument("--device", help="Inference device, e.g. cpu or cuda:0 (default: CUDA if available).")
    parser.add_argument("--seed", type=int, help="Evaluation seed (default: training seed + 10000).")
    args = parser.parse_args()
    if args.width % 2 or args.height % 2:
        parser.error("--width and --height must be even for H.264 video")
    return args


def policy_observation(obs, device):
    # Match ParallelEnv: add environment batch dimension and lift scalar flags.
    return {
        key: torch.as_tensor(np.asarray(value).copy(), device=device).reshape(1, *np.atleast_1d(value).shape)
        for key, value in obs.items()
    }


def video_frame(env, obs, suite, width, height):
    if suite == "dmc":
        raw = env.unwrapped
        frame = raw._env.physics.render(height, width, camera_id=raw._camera)
    else:
        frame = obs["image"]
    frame = np.asarray(frame, dtype=np.uint8)
    if frame.ndim == 2:
        frame = frame[..., None]
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"Expected an RGB frame, got {frame.shape}")
    # H.264 yuv420p needs even dimensions. Pad native frames without changing observations.
    return np.pad(frame, ((0, frame.shape[0] % 2), (0, frame.shape[1] % 2), (0, 0)), mode="edge")


@torch.no_grad()
def record_video(agent, env, path, seconds, fps, suite, width, height):
    frame_count = max(1, math.ceil(seconds * fps))
    obs = env.reset()
    state = agent.get_initial_state(1)
    episode_return, episode_steps = 0.0, 0
    completed_episodes = []
    frame = video_frame(env, obs, suite, width, height)
    # Stream frames to FFmpeg rather than holding the entire video in memory.
    partial = path.with_suffix(".partial.mp4")
    try:
        with FFMPEG_VideoWriter(
            str(partial),
            (frame.shape[1], frame.shape[0]),
            fps,
            codec="libx264",
            preset="medium",
            threads=1,
            ffmpeg_params=["-pix_fmt", "yuv420p"],
        ) as writer:
            for step in range(frame_count):
                action, state = agent.act(policy_observation(obs, agent.device), state, eval=True)
                obs, reward, done, _ = env.step(action[0].cpu().numpy())
                episode_return += float(reward)
                episode_steps += 1
                writer.write_frame(video_frame(env, obs, suite, width, height))
                if done:
                    completed_episodes.append({"return": episode_return, "steps": episode_steps})
                    episode_return, episode_steps = 0.0, 0
                    if step + 1 < frame_count:
                        obs = env.reset()
                        state = agent.get_initial_state(1)
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return {
        "file": str(path),
        "frames": frame_count,
        "fps": fps,
        "seconds": frame_count / fps,
        "completed_episodes": completed_episodes,
        "partial_episode": {"return": episode_return, "steps": episode_steps} if episode_steps else None,
    }


def main():
    args = parse_args()
    logdir = args.logdir.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser()
    if not checkpoint.is_absolute():
        checkpoint = logdir / checkpoint
    config_path = args.config.expanduser() if args.config else logdir / ".hydra" / "config.yaml"
    for path in (checkpoint, config_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    config = OmegaConf.load(config_path)
    config.logdir = str(logdir)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    config.device = device
    config.model.compile = False
    seed = args.seed if args.seed is not None else int(config.seed) + 10000
    config.seed = seed
    # Set explicitly too: saved configs may contain resolved device/seed values.
    config.env.device = config.model.device = device
    config.env.seed = seed
    config.model.rssm.device = device
    for name in ("encoder", "decoder"):
        config.model[name].mlp.device = device
    for name in ("actor", "critic", "reward", "cont"):
        config.model[name].device = device
    OmegaConf.resolve(config)
    tools.set_seed_everywhere(seed)
    torch.set_float32_matmul_precision("high")
    if torch.device(device).type == "cpu":
        torch.set_num_threads(1)

    suite = config.env.task.split("_", 1)[0]
    if suite == "isaaclab":
        raise ValueError("This script uses make_env; IsaacLab's vector environment is not supported.")
    if suite in ("dmc", "metaworld") and not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else logdir / "eval_videos"
    print(f"Checkpoint: {checkpoint}\nTask: {config.env.task}\nDevice: {device}; seed: {seed}", flush=True)
    env = make_env(config.env, 0)
    try:
        if suite == "dmc":
            # Enlarge before the first render creates MuJoCo's offscreen context.
            # The default framebuffer is only 640x480 for several DMC tasks.
            visual = env.unwrapped._env.physics.model.vis.global_
            visual.offwidth = max(visual.offwidth, args.width)
            visual.offheight = max(visual.offheight, args.height)
        state_spec = env.get_state_spec() if config.model.get("state_decoder", {}).get("enabled", False) else None
        agent = Dreamer(config.model, env.observation_space, env.action_space, state_spec=state_spec).to(device)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        agent.load_state_dict(saved["agent_state_dict"], strict=True)
        del saved
        agent.eval()
        fps = args.fps
        if fps is None:
            fps = 1.0 / (env.unwrapped._env.control_timestep() * config.env.action_repeat) if suite == "dmc" else 30.0
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        results = []
        for index in range(args.num_videos):
            path = output_dir / f"{checkpoint.stem}_{timestamp}_{index + 1:03d}.mp4"
            result = record_video(agent, env, path, args.seconds, fps, suite, args.width, args.height)
            results.append(result)
            print(f"[{index + 1}/{args.num_videos}] Saved {path} ({result['seconds']:.3f}s, {fps:g} FPS)", flush=True)
        metadata = {
            "checkpoint": str(checkpoint),
            "config": str(config_path),
            "task": config.env.task,
            "seed": seed,
            "requested_seconds": args.seconds,
            "videos": results,
        }
        (output_dir / f"eval_{timestamp}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    finally:
        env.close()
        # The DMC/MetaWorld adapter inherits Gym's no-op close().
        if suite in ("dmc", "metaworld"):
            env.unwrapped._env.close()


if __name__ == "__main__":
    main()
