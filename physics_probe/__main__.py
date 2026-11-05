"""Run independent frozen-feature physics probes (python -m physics_probe --help)."""

import argparse
import math
from pathlib import Path

import torch

from .common import (
    load_agent,
    load_config,
    make_manifest,
    observation_mode,
    select_entry,
    validate_device,
    write_json,
)
from .environment import TASKS, ProbeEnv
from .pipeline import collect, evaluate, fit, imagine_check, restore_check
from .prediction import prediction_check


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def positive_float(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=positive_int, default=1, help="CPU PyTorch threads (default 1).")
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser(
        "manifest", help="Select an explicit task/seed grid; defaults to four tasks x seeds 200/300."
    )
    manifest.add_argument("--root", type=Path, default=Path("logdir/VE_wasser_H15"))
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--tasks", nargs="+", choices=TASKS)
    manifest.add_argument("--seeds", nargs="+", type=int)
    manifest.add_argument("--distance-space", choices=("raw", "symlog"))
    manifest.add_argument("--expected-count", type=positive_int)
    manifest.add_argument("--modality", choices=("proprio", "vision"), default="proprio")
    for name in ("checkpoint-check", "restore-check", "collect"):
        p = commands.add_parser(name)
        p.add_argument("--manifest", type=Path, required=True)
        p.add_argument("--entry", required=True, help="For example finger_spin_s200")
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--seed", type=int, required=True, help="Environment/data seed, separate from model seed.")
        if name in ("checkpoint-check", "collect"):
            p.add_argument("--device", required=True)
        if name == "restore-check":
            p.add_argument("--warmup", type=positive_int, required=True)
            p.add_argument("--steps", type=positive_int, required=True)
            p.add_argument("--atol", type=positive_float, required=True, help="Tolerance for full oracle restore only.")
        if name == "collect":
            p.add_argument("--episodes", type=positive_int, required=True)
            p.add_argument("--max-steps", type=positive_int, required=True, help="Maximum policy steps per episode.")
    p = commands.add_parser("fit")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source", choices=("features", "observations", "images"), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--epochs", type=positive_int, required=True)
    p.add_argument("--batch-size", type=positive_int, default=256)
    p.add_argument("--hidden", type=positive_int, default=256)
    p.add_argument("--lr", type=positive_float, default=1e-3)
    for name in ("evaluate", "imagine-check"):
        p = commands.add_parser(name)
        p.add_argument("--dataset", type=Path, required=True)
        p.add_argument("--checkpoint", type=Path, required=True, help="decoder.pt, not Dreamer latest.pt")
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--device", required=True)
        p.add_argument("--stride", type=positive_int, required=True)
        p.add_argument("--horizon", type=positive_int, required=True, help="Policy/action-repeat steps")
        p.add_argument("--max-starts", type=positive_int, required=True)
    p = commands.add_parser("prediction-check", help="Stage 3: held-out prediction only, no simulation rollout")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--batch-size", type=positive_int, default=256)
    args = parser.parse_args()
    if hasattr(args, "seed") and args.seed < 0:
        parser.error("seed must be non-negative")
    return args


def main():
    args = parse_args()
    if hasattr(args, "device"):
        validate_device(args.device)
    torch.set_num_threads(args.threads)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite experiment output: {args.output}")
    if args.command == "manifest":
        entries = make_manifest(
            args.root, args.output, args.tasks, args.seeds, args.distance_space, args.expected_count, args.modality
        )
        print(f"Manifest: {len(entries)} independent models -> {args.output}")
        return
    if hasattr(args, "manifest"):
        entry = select_entry(args.manifest, args.entry)
    if args.command == "checkpoint-check":
        config = load_config(entry)
        env = ProbeEnv(config.env, args.seed, include_image=observation_mode(config) == "vision")
        try:
            agent = load_agent(entry, env, args.device)
            write_json(
                args.output,
                {
                    "entry": entry,
                    "strict_load": True,
                    "trainable_parameters": sum(p.numel() for p in agent.parameters() if p.requires_grad),
                    "feature_dim": agent.rssm.feat_size,
                    "schema": env.schema(),
                },
            )
        finally:
            env.close()
    elif args.command == "restore-check":
        restore_check(entry, args.output, args.seed, args.warmup, args.steps, args.atol)
    elif args.command == "collect":
        collect(entry, args.output, args.seed, args.episodes, args.max_steps, args.device)
    elif args.command == "fit":
        fit(
            args.dataset,
            args.output,
            args.source,
            args.seed,
            args.epochs,
            args.batch_size,
            args.hidden,
            args.lr,
            args.device,
        )
    elif args.command == "prediction-check":
        prediction_check(args.dataset, args.checkpoint, args.output, args.device, args.batch_size)
    else:
        function = evaluate if args.command == "evaluate" else imagine_check
        function(args.dataset, args.checkpoint, args.output, args.device, args.stride, args.horizon, args.max_starts)
    print(f"Saved {args.command}: {args.output}")


if __name__ == "__main__":
    main()
